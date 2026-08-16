"""
OKX API v5 Client — 基于官方 python-okx SDK (v0.4.3) 封装。
支持合约 (SWAP) 的 REST API + WebSocket 实时数据推送。

相比自定义 HTTP 请求的优势：
  - 官方 SDK 内置 HTTP/2 支持，连接复用，延迟更低
  - 自动处理签名、时间戳同步、错误重试
  - WebSocket 毫秒级实时数据推送
  - 完整的异常处理和日志记录
"""

import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple

import requests

# 修复 P0-C: python-okx SDK 底层是 httpx.Client(HTTP/2)，网络异常是
# httpx.HTTPError 一族(ConnectError/TimeoutException/RemoteProtocolError...)，
# 它们 NOT 是内置 ConnectionError/TimeoutError/OSError 的子类。
# 旧代码 `except (ConnectionError, TimeoutError, OSError)` 守卫对网络异常永不触发，
# 导致"API 故障"被静默当作"无持仓/无挂单"处理 → 超限开仓(资金安全风险)。
try:
    import httpx
except ImportError:  # 理论不可达(SDK 强依赖 httpx)，防御
    httpx = None

from okx.Account import AccountAPI
from okx.Funding import FundingAPI
from okx.MarketData import MarketAPI
from okx.PublicData import PublicAPI
from okx.Trade import TradeAPI
from okx.TradingData import TradingDataAPI
from okx.consts import POST, PLACE_ALGO_ORDER, WITHDRAWAL_COIN


logger = logging.getLogger(__name__)


# 修复 P0-B (fail-closed): 查询类 API 失败时抛出/返回哨兵，
# 上层必须区分"查询失败"与"确无持仓"，杜绝把 API 故障当无持仓。
class OKXQueryError(Exception):
    """OKX 查询失败（fail-closed 信号）。"""


# 可重试的 OKX 业务错误码：限流 / 系统繁忙（网络类错误由 httpx.HTTPError 覆盖）
_RETRYABLE_CODES = {"50011", "50012", "50013", "50000"}
_RETRY_DELAYS = (0.5, 1.5, 4.0)

# 全局 API 限流令牌桶（PRO 模块可选启用；由 fvg_killer_pro.create_rate_limiter
# 在 OKXClient.__init__ 中赋值）。None = 未启用（dry_run 或开源核心版）。
_GLOBAL_RATE_LIMITER = None


def _call_sdk_retry(fn, *args, retries: int = 3, **kwargs):
    """调用 SDK 方法，对网络异常与限流/繁忙码做带退避重试。

    - httpx 网络异常：重试，耗尽后抛出原始异常
    - 业务错误(code != 0 且非可重试码)：不重试，直接返回结果 dict
    - 限流/繁忙码(50000/50011/...)：重试

    修复 P0-C: 交易端点(下单/撤单/平仓/设杠杆)此前零重试零退避，
    限流/网络故障时静默失败 → 保护单/平仓单丢失。
    """
    last_exc = None
    for attempt in range(retries):
        try:
            # 主动限流: 每次请求前从全局令牌桶取一个令牌（未启用时无操作）
            if _GLOBAL_RATE_LIMITER is not None:
                _GLOBAL_RATE_LIMITER.acquire()
            result = fn(*args, **kwargs)
        except httpx.HTTPError as e:
            last_exc = e
            logger.warning(f"网络异常 (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            continue
        if (isinstance(result, dict)
                and str(result.get("code")) in _RETRYABLE_CODES
                and attempt < retries - 1):
            logger.warning(f"OKX 限流/繁忙 code={result.get('code')} "
                           f"(attempt {attempt+1}/{retries})")
            time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            continue
        return result
    if last_exc is not None:
        raise last_exc
    return {"code": "retry_exhausted", "msg": "retry exhausted", "data": []}


class OKXClient:
    """OKX API v5 客户端，封装官方 SDK 的常用接口。
    
    保持与旧版相同的接口签名，确保 agent.py 无需修改。
    """

    def __init__(self, config: dict):
        cfg = config.get("okx", {})
        self.dry_run = config.get("agent", {}).get("dry_run", False)
        # 0 = live trading, 1 = demo trading (模拟交易)
        # 修复 2026-08-07: testnet 配置项未生效 — OKX v5 无独立 testnet,
        # 模拟盘即 demo trading(flag=1), testnet=true 与 demo=true 等效。
        self.flag = "1" if (cfg.get("demo", False) or cfg.get("testnet", False)) else "0"

        # 修复 2026-08-07: close_limit 参数未生效 — 限价平仓超时/滑点保护
        # 原硬编码 30s/±0.1%, 现读 risk.close_limit_wait_seconds 与
        # risk.close_market_fallback_slippage_pct (config 已配 10s/0.3%)。
        _risk_cfg = config.get("risk", {}) or {}
        try:
            self.close_limit_timeout_s: float = float(
                _risk_cfg.get("close_limit_wait_seconds", 10) or 10)
        except (TypeError, ValueError):
            self.close_limit_timeout_s = 10.0
        try:
            self.close_fallback_slippage_pct: float = float(
                _risk_cfg.get("close_market_fallback_slippage_pct", 0.3) or 0.3)
        except (TypeError, ValueError):
            self.close_fallback_slippage_pct = 0.3

        # 模拟交易使用独立的 API 密钥
        if self.flag == "1":
            self.api_key = cfg.get("demo_api_key") or cfg.get("api_key", "")
            self.api_secret = cfg.get("demo_api_secret") or cfg.get("api_secret", "")
            self.passphrase = cfg.get("demo_passphrase") or cfg.get("passphrase", "")
            # 模拟盘使用独立的 base_url，与实盘不同
            # 文档: https://www.okx.com/docs-v5/zh/#overview-demo-trading
            self.base_url = "https://www.okx.com"
        else:
            self.api_key = cfg.get("api_key", "")
            self.api_secret = cfg.get("api_secret", "")
            self.passphrase = cfg.get("passphrase", "")
            self.base_url = cfg.get("base_url", "https://www.okx.com").rstrip("/")

        # 代理配置（国内用户需要配置代理访问 OKX API）
        self.proxy = cfg.get("proxy", None)

        # 初始化官方 SDK 模块（传入 domain 和 proxy）
        sdk_kwargs = {"domain": self.base_url, "flag": self.flag}
        if self.proxy:
            sdk_kwargs["proxy"] = self.proxy

        self._market = MarketAPI(**sdk_kwargs)
        self._public = PublicAPI(**sdk_kwargs)
        self._trading_data = TradingDataAPI(**sdk_kwargs)

        # 需要认证的模块
        auth_kwargs = {
            "domain": self.base_url,
            "flag": self.flag,
        }
        if self.proxy:
            auth_kwargs["proxy"] = self.proxy

        self._account = AccountAPI(
            self.api_key, self.api_secret, self.passphrase, False,
            **auth_kwargs,
        )
        self._trade = TradeAPI(
            self.api_key, self.api_secret, self.passphrase, False,
            **auth_kwargs,
        )
        # 资金账户模块 (Royalty 分成提现/资金划转用)
        self._funding = FundingAPI(
            self.api_key, self.api_secret, self.passphrase, False,
            **auth_kwargs,
        )

        # 多空比缓存: inst_id -> (时间戳, 多空比)。多空比是低频统计(1H周期)，
        # 每轮扫描每币种请求会触发限流，缓存 10 分钟。
        self._lsr_cache: Dict[str, Tuple[float, float]] = {}
        self._lsr_cache_ttl = 600.0  # 10 分钟

        # 全局 API 限流令牌桶 (v3.3 / PRO 模块): 实盘模式按 okx.rate_limit
        # 启用主动 QPS 控制。开源核心版（无 fvg_killer_pro）不限流。
        global _GLOBAL_RATE_LIMITER
        _GLOBAL_RATE_LIMITER = None
        try:
            from fvg_killer_pro import create_rate_limiter
            _GLOBAL_RATE_LIMITER = create_rate_limiter(
                cfg.get("rate_limit") or {}, self.dry_run)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(v, default=0.0):
        """安全转换浮点数，处理 None、空字符串、非数字字符串。

        OKX API 某些字段可能返回空字符串 ""，float("") 会抛出 ValueError。
        该方法对 None / "" / 非数字字符串统一返回 default。
        """
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _format_price(price: float, tick_sz: float) -> str:
        """按 tick_sz 精度格式化价格。"""
        if tick_sz <= 0:
            tick_sz = 0.01
        # 使用 Decimal 精确计算小数位数
        from decimal import Decimal
        d = Decimal(str(tick_sz))
        exponent = d.as_tuple().exponent
        decimals = max(0, -exponent) if exponent < 0 else 0
        rounded = round(price / tick_sz) * tick_sz
        return f"{rounded:.{decimals}f}"

    def _gen_cl_ord_id(self, prefix: str = "fvg") -> str:
        """生成幂等 Client Order ID，格式: {prefix}{timestamp_ms}{random}

        OKX 支持通过 clOrdId / algoClOrdId 实现幂等下单：
        提交相同的 clOrdId 两次不会创建重复订单，有效防止网络重试导致的重复下单。

        修复: 不能含下划线 — OKX 实测拒绝带下划线的 clOrdId
        （sCode=51000 "Parameter clOrdId error" → "All operations failed"），
        此前导致所有下单失败。改为纯字母数字组合。
        """
        ts = int(time.time() * 1000)
        rand = uuid.uuid4().hex[:12]
        return f"{prefix}{ts}{rand}"

    # ------------------------------------------------------------------
    # 公开接口 — 市场数据
    # ------------------------------------------------------------------
    
    def get_tickers(self, inst_type: str = "SWAP") -> List[dict]:
        """获取所有合约 ticker。"""
        try:
            result = self._market.get_tickers(instType=inst_type)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_tickers failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_tickers exception: {e}")
            return []

    def get_candles(self, inst_id: str, bar: str = "1H",
                    limit: int = 100) -> List[list]:
        """获取 K 线数据。
        SDK 返回格式: [[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm], ...]
        时间倒序（最新在前）。
        """
        try:
            result = self._market.get_candlesticks(
                instId=inst_id, bar=bar, limit=str(limit)
            )
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_candles failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_candles exception: {e}")
            return []

    def get_candles_enhanced(self, inst_id: str, bar: str = "1H",
                             limit: int = 200) -> List[list]:
        """增强版 K 线获取 — 使用 history-candles 端点 + 限速重试。

        相比 get_candles 的优势：
          - 双端点回退: history-candles (深度历史) → candles (近期数据)
          - 分页遍历: 支持大量历史数据
          - 限速重试: 自动处理 429/5xx 错误
          - OHLC 验证: 过滤无效数据

        Args:
            inst_id: 合约 ID
            bar: K 线周期
            limit: 获取数量

        Returns:
            OKX 原始格式的 K 线列表
        """
        try:
            from okx_loader import fetch_candles_enhanced
            return fetch_candles_enhanced(
                inst_id=inst_id,
                bar=bar,
                limit=limit,
                proxy=self.proxy,
                use_history=True,
            )
        except ImportError:
            logger.debug("okx_loader not available, falling back to SDK")
            return self.get_candles(inst_id, bar, limit)
        except Exception as e:
            logger.warning(f"get_candles_enhanced failed: {e}, falling back to SDK")
            return self.get_candles(inst_id, bar, limit)

    def get_funding_rate(self, inst_id: str) -> Optional[float]:
        """获取当前资金费率。"""
        try:
            result = self._public.get_funding_rate(instId=inst_id)
            if result.get("code") == "0" and result.get("data"):
                v = result["data"][0].get("fundingRate", "0")
                return self._safe_float(v)
            return None
        except Exception as e:
            logger.error(f"get_funding_rate exception: {e}")
            return None

    def get_funding_info(self, inst_id: str) -> Optional[Tuple[float, float]]:
        """获取资金费率 + 下次结算时间戳。

        用于换仓结算保护: 距结算点过近时先收完资金费再换仓，避免白丢费率。

        Returns:
            (funding_rate, next_funding_ts) — 失败返回 None
        """
        try:
            result = self._public.get_funding_rate(instId=inst_id)
            if result.get("code") == "0" and result.get("data"):
                d = result["data"][0]
                rate = self._safe_float(d.get("fundingRate", "0"))
                ts = self._safe_float(d.get("fundingTime", "0"))
                if ts > 0:
                    return (rate, ts)
            return None
        except Exception as e:
            logger.error(f"get_funding_info exception: {e}")
            return None

    def get_order_book(self, inst_id: str, sz: int = 20) -> Optional[dict]:
        """获取订单簿深度快照。"""
        try:
            result = self._market.get_orderbook(instId=inst_id, sz=str(sz))
            if result.get("code") == "0" and result.get("data"):
                return result["data"][0]
            return None
        except Exception as e:
            logger.error(f"get_order_book exception: {e}")
            return None

    def get_order_book_depth_usd(
        self,
        inst_id: str,
        levels: int = 10,
        ct_val: float = 0.01,
    ) -> Optional[dict]:
        """计算前 N 档订单簿的 USDT 名义深度 (2026-08-14 流动性检查)。

        顶级交易所/做市商开仓前必看订单簿深度：名义仓位相对前 N 档深度
        过大时，平仓市价兜底会吃掉多档、滑点远超成本模型估算。本方法把
        前 levels 档的 bids/asks 各累加成 USDT 名义深度，供流动性门槛使用。

        OKX orderbook 每档结构: [px, sz, liquidatedOrders, numOrders]，
        sz 是合约张数，USDT 深度 = Σ px × sz × ctVal。

        Args:
            inst_id: 合约 ID
            levels: 档位数
            ct_val: 合约面值 (USDT)

        Returns:
            {"bids_usd": float, "asks_usd": float} 或 None (查询失败)
        """
        book = self.get_order_book(inst_id, sz=levels)
        if not book:
            return None

        def _sum_usd(entries):
            total = 0.0
            for row in (entries or []):
                try:
                    px = float(row[0])
                    sz = float(row[1])
                    if px > 0 and sz > 0:
                        total += px * sz * ct_val
                except (ValueError, IndexError, TypeError):
                    continue
            return total

        bids_usd = _sum_usd(book.get("bids"))
        asks_usd = _sum_usd(book.get("asks"))
        if bids_usd <= 0 and asks_usd <= 0:
            return None
        return {"bids_usd": bids_usd, "asks_usd": asks_usd}

    def get_open_interest(self, inst_id: str) -> Optional[float]:
        """获取当前未平仓合约量 (OI)。"""
        try:
            result = self._public.get_open_interest(
                instType="SWAP", instId=inst_id
            )
            if result.get("code") == "0" and result.get("data"):
                v = result["data"][0].get("oi", "0")
                return self._safe_float(v)
            return None
        except Exception as e:
            logger.error(f"get_open_interest exception: {e}")
            return None

    def get_long_short_ratio(self, inst_id: str, period: str = "5m") -> Optional[float]:
        """获取合约多空比。
        
        SDK 的 TradingDataAPI.get_long_short_ratio 返回格式: [[ts, ratio], ...]
        需要从 instId 中提取币种代码。
        """
        try:
            ccy = inst_id.split("-")[0] if "-" in inst_id else inst_id
            result = self._trading_data.get_long_short_ratio(
                ccy=ccy, period=period
            )
            if result.get("code") == "0":
                data = result.get("data", [])
                # SDK 返回 [[ts, ratio], [ts, ratio], ...] 格式
                if data and isinstance(data[0], list) and len(data[0]) >= 2:
                    v = data[0][1]
                    return self._safe_float(v)
                # 兼容 dict 格式
                if data and isinstance(data[0], dict):
                    v = data[0].get("longShortRatio", "0")
                    return self._safe_float(v)
            return None
        except Exception as e:
            logger.warning(f"get_long_short_ratio not available: {e}")
            return None

    def get_funding_rate_history(self, inst_id: str, limit: int = 24) -> List[dict]:
        """获取历史资金费率。"""
        try:
            result = self._public.funding_rate_history(
                instId=inst_id, limit=str(limit)
            )
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_funding_rate_history exception: {e}")
            return []

    def get_mark_price(self, inst_id: str) -> Optional[float]:
        """获取标记价格。"""
        try:
            result = self._public.get_mark_price(
                instType="SWAP", instId=inst_id
            )
            if result.get("code") == "0" and result.get("data"):
                v = result["data"][0].get("markPx", "0")
                return self._safe_float(v)
            return None
        except Exception as e:
            logger.error(f"get_mark_price exception: {e}")
            return None

    def get_long_short_ratio(self, inst_id: str, period: str = "1H") -> Optional[float]:
        """获取合约多空账户比（Rubik 公开统计接口，无需认证）。

        多空账户比 = 多头账户数 / 空头账户数。>1 偏多，<1 偏空。
        数据源: GET /api/v5/rubik/stat/contracts/long-short-account-ratio
        失败时返回 None，由上层降级为中性（不阻塞信号生成）。

        Args:
            inst_id: 合约 ID 如 "RSR-USDT-SWAP"
            period: 周期 "5m" | "1H" | "1D" 等

        Returns:
            最新多空比，或 None（无数据/失败）
        """
        if self.dry_run:
            return None
        # 缓存命中（10 分钟 TTL）
        _cached = self._lsr_cache.get(inst_id)
        if _cached and (time.time() - _cached[0]) < self._lsr_cache_ttl:
            return _cached[1]
        try:
            # SDK 该方法以 ccy（币种）为参数，从 instId 提取
            ccy = inst_id.split("-")[0]
            result = self._trading_data.get_long_short_ratio(ccy=ccy, period=period)
            if result.get("code") != "0" or not result.get("data"):
                logger.debug(f"get_long_short_ratio 无数据: {inst_id} {result.get('msg')}")
                return None
            ratio = self._safe_float(result["data"][0].get("longShortRatio"), 0.0)
            if ratio <= 0:
                return None
            # 缓存成功结果；失败不缓存（下一轮重试）
            self._lsr_cache[inst_id] = (time.time(), ratio)
            return ratio
        except Exception as e:
            logger.debug(f"get_long_short_ratio exception: {e}")
            return None

    def get_index_candles(self, inst_id: str, bar: str = "1D",
                          limit: int = 30) -> List[dict]:
        """获取指数 K 线。"""
        try:
            result = self._market.get_index_candlesticks(
                instId=inst_id, bar=bar, limit=str(limit)
            )
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_index_candles exception: {e}")
            return []

    def get_taker_volume(self, inst_id: str, bar: str = "1H",
                         limit: int = 24) -> List[dict]:
        """获取主动买卖成交量。
        
        SDK 的 TradingDataAPI.get_taker_volume 使用 ccy 参数（币种），
        需要从 instId 中提取币种代码。
        """
        try:
            ccy = inst_id.split("-")[0] if "-" in inst_id else inst_id
            result = self._trading_data.get_taker_volume(
                ccy=ccy, instType="SWAP", period=bar
            )
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.warning(f"get_taker_volume not available: {e}")
            return []

    # ------------------------------------------------------------------
    # 私有接口 — 账户 & 交易
    # ------------------------------------------------------------------
    
    def get_balance(self) -> Optional[float]:
        """获取 USDT 可用余额。"""
        try:
            result = self._account.get_account_balance()
            if result.get("code") == "0" and result.get("data"):
                for detail in result["data"][0].get("details", []):
                    if detail.get("ccy") == "USDT":
                        v = detail.get("availBal", "0")
                        return self._safe_float(v)
            return None
        except Exception as e:
            logger.error(f"get_balance exception: {e}")
            return None

    def get_total_equity(self) -> Optional[float]:
        """获取账户总权益 (USDT)。"""
        try:
            result = self._account.get_account_balance()
            if result.get("code") == "0" and result.get("data"):
                v = result["data"][0].get("totalEq", "0")
                return self._safe_float(v)
            return None
        except Exception as e:
            logger.error(f"get_total_equity exception: {e}")
            return None

    def get_positions(self, inst_id: str = "",
                      mgn_mode: str = "") -> Optional[List[dict]]:
        """获取当前持仓。

        GET /api/v5/account/positions

        Args:
            inst_id: 可选，按合约 ID 过滤（如 "BTC-USDT-SWAP"）
            mgn_mode: 可选，按保证金模式过滤 "isolated" | "cross"

        Returns:
            持仓列表（每项含 instId, posSide, pos, avgPx, upl, lever, mgnMode 等）；
            查询失败(网络异常/非 0 业务码)返回 **None**（fail-closed），
            上层必须区分"查询失败"与"确无持仓"。
        """
        try:
            kwargs = {}
            if inst_id:
                kwargs["instId"] = inst_id
            result = self._account.get_positions(**kwargs) if kwargs else self._account.get_positions()
            if result.get("code") == "0":
                data = result.get("data", [])
                # 修复 P1-6: SDK get_positions 无 mgnMode 参数（透传会 TypeError 被
                # 静默吞成 []），改为客户端侧过滤
                if mgn_mode:
                    data = [p for p in data if p.get("mgnMode") == mgn_mode]
                return data
            logger.error(f"get_positions failed: code={result.get('code')} "
                         f"msg={result.get('msg')}")
            return None
        except Exception as e:
            logger.error(f"get_positions exception: {e}")
            return None

    def get_pending_orders(self) -> Optional[List[dict]]:
        """获取挂单。

        Returns:
            挂单列表；查询失败返回 **None**（fail-closed，与 get_positions 一致）。
        """
        try:
            result = self._trade.get_order_list()
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_pending_orders failed: code={result.get('code')} "
                         f"msg={result.get('msg')}")
            return None
        except Exception as e:
            logger.error(f"get_pending_orders exception: {e}")
            return None

    def place_order(self, inst_id: str, side: str, pos_side: str,
                    sz: str, px: str = "", ord_type: str = "limit",
                    td_mode: str = "isolated",
                    tp_trigger: str = "", tp_price: str = "",
                    sl_trigger: str = "", sl_price: str = "",
                    reduce_only: bool = False) -> Optional[str]:
        """下单（通过 attachAlgoOrds 附加止盈止损）。"""
        # 参数校验
        try:
            sz_val = float(sz)
            if sz_val <= 0:
                logger.error(f"place_order: sz must be positive, got {sz}")
                return None
        except (ValueError, TypeError):
            logger.error(f"place_order: invalid sz value: {sz}")
            return None
        if px:
            try:
                px_val = float(px)
                if px_val <= 0:
                    logger.error(f"place_order: px must be positive, got {px}")
                    return None
            except (ValueError, TypeError):
                logger.error(f"place_order: invalid px value: {px}")
                return None

        if self.dry_run:
            logger.info(f"[DRY RUN] {side} {sz} {inst_id} @ {px} "
                        f"TP={tp_trigger} SL={sl_trigger}")
            return "dry_run_" + str(int(time.time() * 1000))

        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": ord_type,
            "sz": sz,
            # 修复 P2-1: clOrdId 在方法内生成一次，网络重试时复用同一 ID，
            # 交易所端幂等去重才真正生效（此前每次重试重新生成 → 去重失效，
            # 响应丢失后重试会重复下单）
            "clOrdId": self._gen_cl_ord_id("fvg"),
        }
        if px:
            body["px"] = px
        if reduce_only:
            body["reduceOnly"] = True

        # 通过 attachAlgoOrds 附加止盈止损（SDK 原生支持此参数）
        attach_algos = []
        if tp_trigger:
            attach_algos.append({
                "tpTriggerPx": tp_trigger,
                "tpOrdPx": tp_price if tp_price else "-1",
            })
        if sl_trigger:
            attach_algos.append({
                "slTriggerPx": sl_trigger,
                "slOrdPx": sl_price if sl_price else "-1",
            })
        if attach_algos:
            body["attachAlgoOrds"] = attach_algos

        try:
            # 修复 P0-C: 网络异常/限流退避重试（同一 clOrdId 重放 → 幂等）
            result = _call_sdk_retry(self._trade.place_order, **body)
            if result.get("code") == "0" and result.get("data"):
                ord_id = result["data"][0].get("ordId", "")
                logger.info(f"Order placed: {ord_id} — {side} {sz} {inst_id} @ {px} "
                            f"TP={tp_trigger} SL={sl_trigger}")
                return ord_id
            else:
                # 修复: 打印完整 result（含 data 里的 sCode/sMsg 详细拒因），
                # 便于区分权限/风控/参数问题
                logger.error(
                    f"place_order failed: msg={result.get('msg')} "
                    f"code={result.get('code')} "
                    f"data={json.dumps(result.get('data'), ensure_ascii=False)[:500]} "
                    f"req={json.dumps(body, ensure_ascii=False)[:400]}"
                )
                return None
        except httpx.HTTPError as e:
            logger.error(f"place_order 网络失败(重试后仍失败): {e}")
            return None
        except Exception as e:
            logger.error(f"place_order exception: {e}")
            return None

    def place_algo_order(self, inst_id: str, td_mode: str,
                         side: str, pos_side: str, sz: str,
                         ord_type: str,
                         tp_trigger_px: str = "",
                         tp_trigger_px_type: str = "last",
                         tp_ord_px: str = "",
                         sl_trigger_px: str = "",
                         sl_trigger_px_type: str = "last",
                         sl_ord_px: str = "",
                         reduce_only: bool = True,
                         **kwargs) -> Optional[str]:
        """创建策略订单（止盈止损 / 计划委托）。

        使用 POST /api/v5/trade/order-algo 端点，支持为已有持仓
        独立设置止盈止损，无需依赖开仓时的 attachAlgoOrds。
        文档: https://www.okx.com/docs-v5/zh/#order-book-trading-algo-trading

        Args:
            inst_id: 合约 ID
            td_mode: 交易模式 "isolated" | "cross"
            side: 订单方向 "buy" | "sell"
            pos_side: 持仓方向 "long" | "short"
            sz: 订单数量
            ord_type: 策略订单类型
                      "conditional" — 止盈止损
                      "oco" — 止盈止损 OCO
                      "trigger" — 计划委托
                      "move_order_stop" — 移动止盈止损
                      "iceberg" — 冰山委托
                      "twap" — 时间加权委托
            tp_trigger_px: 止盈触发价格
            tp_trigger_px_type: 止盈触发价格类型 "last" | "mark" | "index"
            tp_ord_px: 止盈委托价格（-1 表示市价）
            sl_trigger_px: 止损触发价格
            sl_trigger_px_type: 止损触发价格类型 "last" | "mark" | "index"
            sl_ord_px: 止损委托价格（-1 表示市价）
            reduce_only: 是否仅减仓（默认 True，防止幽灵订单开反向仓位）

        Returns:
            algo_id 或 None
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Algo order: {ord_type} {side} {sz} {inst_id} "
                        f"TP={tp_trigger_px} SL={sl_trigger_px}")
            return "dry_run_algo_" + str(int(time.time() * 1000))

        # ============ 关键修复 (TP 丢失根因) ============
        # OKX 实测: `ordType=conditional` 的订单同时传 TP+SL 时，
        # 服务器会**静默丢弃 TP，只保留 SL**（返回 code=0、failCode=0，无法从返回码发现）。
        # A/B/C/D/E/F/G/H/I 九组实测结论:
        #   - 仅 TP  → 正常
        #   - 仅 SL  → 正常
        #   - TP+SL 同时挂 conditional → TP 被丢弃
        #   - TP+SL 挂 `ordType=oco` → 双方全部正常落单（触发一方自动撤销另一方）
        # 因此当 TP 与 SL 同时提供时，自动改用 oco 单。
        has_tp = bool(tp_trigger_px)
        has_sl = bool(sl_trigger_px)
        if has_tp and has_sl and ord_type == "conditional":
            logger.info(f"place_algo_order: TP+SL 同时设置，自动改用 ordType=oco "
                        f"(conditional 单同时挂 TP+SL 会丢失 TP，实测验证)")
            ord_type = "oco"

        # 干净请求体 — 仅包含必需字段。
        # 修复: 绕过 SDK place_algo_order（其 attachAlgoOrds=[] 可变默认参数会无条件
        # 写入请求体，并携带 ccy/tag/triggerPx 等 20+ 空字符串字段），
        # 改用底层签名机制发送符合官方 API 契约的最小请求。
        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": ord_type,
            "sz": sz,
            "reduceOnly": "true" if reduce_only else "false",
            "algoClOrdId": self._gen_cl_ord_id("algo"),
        }
        if has_tp:
            body["tpTriggerPx"] = tp_trigger_px
            body["tpTriggerPxType"] = tp_trigger_px_type
            body["tpOrdPx"] = tp_ord_px if tp_ord_px else "-1"  # 空则市价
        if has_sl:
            body["slTriggerPx"] = sl_trigger_px
            body["slTriggerPxType"] = sl_trigger_px_type
            body["slOrdPx"] = sl_ord_px if sl_ord_px else "-1"  # 空则市价

        # 透传额外参数
        body.update(kwargs)

        try:
            # 修复: SDK 方法名为 place_algo_order（原 order_algo 不存在，导致止损单永远挂不上）
            # 但此处不用 SDK 封装，改走底层 _request 保证请求体干净。
            # 修复 P0-C: 加退避重试（网络异常/限流时保护单不再静默丢失）。
            result = _call_sdk_retry(
                self._trade._request, POST, PLACE_ALGO_ORDER, body)
            if result.get("code") == "0" and result.get("data"):
                algo_id = result["data"][0].get("algoId", "")
                logger.info(f"Algo order placed: {algo_id} — {ord_type} "
                            f"{side} {sz} {inst_id} "
                            f"TP={tp_trigger_px} SL={sl_trigger_px}")
                # 下单后核验 TP/SL 确实落单（OKX 会静默丢弃字段，必须主动复核）
                _ok, _detail = self._verify_algo_protection(
                    algo_id, inst_id, has_tp, has_sl
                )
                if _ok is False:
                    logger.error(f"place_algo_order 核验失败: {algo_id} {_detail}，"
                                 f"撤销并返回 None 交由上层补挂")
                    self.cancel_algo_order(algo_id, inst_id)
                    return None
                if _ok:
                    logger.info(f"Algo 核验通过: {algo_id} ({_detail})")
                return algo_id
            else:
                # 修复: TP 方向校验失败时降级 — 异常波动币种（FVG 信号基于缓存
                # K 线计算 TP），下单时刻实时价格可能已越过 TP，OKX 返回
                # sCode=51277/51279 (TP trigger price cannot be higher/lower
                # than the last price) 拒绝整个 OCO 单。此时保不住 TP 至少
                # 要保住 SL，降级为仅挂 SL 的 conditional 单。
                _sc = ""
                try:
                    _sc = str((result.get("data") or [{}])[0].get("sCode", ""))
                except (IndexError, TypeError, AttributeError):
                    _sc = ""
                if has_tp and has_sl and _sc in ("51277", "51279"):
                    logger.warning(
                        f"place_algo_order: TP 方向校验失败(sCode={_sc}) — "
                        f"实时价已越过 TP={tp_trigger_px}，降级仅挂 SL 保护"
                    )
                    return self._place_algo_sl_only(body, inst_id)
                logger.error(f"place_algo_order failed: {result.get('msg')} "
                             f"code={result.get('code')} "
                             f"data={json.dumps(result.get('data'), ensure_ascii=False)[:500]}")
                return None
        except httpx.HTTPError as e:
            logger.error(f"place_algo_order 网络失败(重试后仍失败): {e}")
            return None
        except Exception as e:
            logger.error(f"place_algo_order exception: {e}")
            return None

    def _place_algo_sl_only(self, oco_body: dict, inst_id: str) -> Optional[str]:
        """OCO 因 TP 方向校验失败时，降级为仅挂 SL 的 conditional 单。

        保留 SL 保护（价格已越过 TP 说明入场可能已不可达，止损优先），
        TP 留给后续 trailing 补挂（价格回到合理区间后再带上）。
        """
        sl_body = {k: v for k, v in oco_body.items()
                   if k not in ("tpTriggerPx", "tpOrdPx", "tpTriggerPxType")}
        sl_body["ordType"] = "conditional"
        try:
            result = _call_sdk_retry(
                self._trade._request, POST, PLACE_ALGO_ORDER, sl_body)
            if result.get("code") == "0" and result.get("data"):
                algo_id = result["data"][0].get("algoId", "")
                logger.warning(f"降级仅挂 SL 成功: {algo_id} — {inst_id} "
                               f"SL={sl_body.get('slTriggerPx')}")
                _ok, _detail = self._verify_algo_protection(
                    algo_id, inst_id, False, True
                )
                if _ok is False:
                    logger.error(f"降级 SL 核验失败: {algo_id} {_detail}，撤销")
                    self.cancel_algo_order(algo_id, inst_id)
                    return None
                return algo_id
            else:
                logger.error(f"降级仅挂 SL 失败: {result.get('msg')} "
                             f"data={json.dumps(result.get('data'), ensure_ascii=False)[:300]}")
                return None
        except Exception as e:
            logger.error(f"降级仅挂 SL 异常: {e}")
            return None

    def _verify_algo_protection(self, algo_id: str, inst_id: str,
                                need_tp: bool, need_sl: bool):
        """核验策略单的 TP/SL 是否真实落单（最多重试 3 次，间隔 1s）。

        Returns:
            (True, 详情)  — 全部落单
            (False, 缺失项) — 订单存在但字段缺失（真实故障，应撤销重挂）
            (None, 原因)  — 无法核验（查询失败，不阻塞下单流程）
        """
        for _attempt in range(3):
            try:
                d = self._trade.get_algo_order_details(algoId=algo_id)
                if d.get("code") != "0" or not d.get("data"):
                    time.sleep(1)
                    continue
                x = d["data"][0]
                got_tp = bool(x.get("tpTriggerPx"))
                got_sl = bool(x.get("slTriggerPx"))
                missing = []
                if need_tp and not got_tp:
                    missing.append("TP")
                if need_sl and not got_sl:
                    missing.append("SL")
                if missing:
                    return False, f"缺失 {', '.join(missing)}"
                return True, (f"TP={x.get('tpTriggerPx')} SL={x.get('slTriggerPx')} "
                              f"state={x.get('state')}")
            except Exception as e:
                time.sleep(1)
        return None, "查询失败(无法核验)"

    def place_plan_order(self, inst_id: str, td_mode: str,
                         side: str, pos_side: str, sz: str,
                         trigger_px: float, ord_px: float,
                         trigger_px_type: str = "last") -> Optional[str]:
        """计划委托 (conditional 触发单) — 价格触及 trigger_px 后以 ord_px 限价进场。

        2026-08-10 用户要求: 深挂 FVG 回补位不直接挂限价单（提前深挂易空转、
        被逆向选择），改挂触发单 — 价格先走到距回补位一个有效阈值窗口的
        触发位，触发后才以回补位限价进场，避免空转也不错过深回补机会。

        POST /api/v5/trade/order-algo, ordType=conditional (开仓, reduceOnly=false)

        Args:
            trigger_px: 触发价 (最新价触及后激活委托)
            ord_px: 触发后的委托价 (FVG 回补位)

        Returns:
            algo_id 或 None
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Plan order(conditional): {side} {sz} "
                        f"{inst_id} trigger={trigger_px} ord={ord_px}")
            return "dry_run_plan_" + str(int(time.time() * 1000))

        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": "conditional",
            "sz": sz,
            "triggerPx": f"{trigger_px}",
            "triggerPxType": trigger_px_type,
            "ordPx": f"{ord_px}",
            "reduceOnly": "false",
            "algoClOrdId": self._gen_cl_ord_id("plan"),
        }
        try:
            result = _call_sdk_retry(
                self._trade._request, POST, PLACE_ALGO_ORDER, body)
            if result.get("code") == "0" and result.get("data"):
                algo_id = result["data"][0].get("algoId", "")
                logger.info(f"Plan order(conditional) placed: {algo_id} — "
                            f"{side} {sz} {inst_id} "
                            f"trigger={trigger_px} ord={ord_px}")
                return algo_id
            logger.warning(
                f"place_plan_order failed: code={result.get('code')} "
                f"msg={result.get('msg')} ({inst_id} trigger={trigger_px})")
            return None
        except Exception as e:
            logger.error(f"place_plan_order exception: {e}")
            return None

    def cancel_algo_order(self, algo_id: str, inst_id: str = "") -> bool:
        """撤销策略订单。

        POST /api/v5/trade/cancel-algos

        Args:
            algo_id: 策略订单 ID
            inst_id: 合约 ID（可选）

        Returns:
            是否成功
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel algo {algo_id}")
            return True

        try:
            # 修复: SDK 的 cancel_algo_order 签名是 (orders_data=None, params=None)，
            # 原调用 **kwargs(algoId=...) 参数不匹配导致一直异常。
            # orders_data 为列表，每项含 algoId/instId（对应 POST /api/v5/trade/cancel-algos）
            orders_data = [{"algoId": algo_id}]
            if inst_id:
                orders_data[0]["instId"] = inst_id
            # 修复 P0-C: 撤保护单加退避重试（撤单失败 → 裸仓风险）
            result = _call_sdk_retry(
                self._trade.cancel_algo_order, orders_data=orders_data)
            if result.get("code") == "0":
                logger.info(f"Algo order cancelled: {algo_id}")
                return True
            else:
                logger.error(f"cancel_algo_order failed: {result.get('msg')} (code={result.get('code')})")
                return False
        except Exception as e:
            logger.error(f"cancel_algo_order exception: {e}")
            return False

    def get_algo_orders(self, algo_id: str = "", inst_type: str = "SWAP",
                        inst_id: str = "", ord_type: str = "",
                        state: str = "", limit: str = "20") -> List[dict]:
        """查询策略订单列表。

        GET /api/v5/trade/orders-algo-pending

        Args:
            algo_id: 策略订单 ID（可选）
            inst_type: 产品类型
            inst_id: 合约 ID（可选）
            ord_type: 策略订单类型（可选）
            state: 策略订单状态（可选）
            limit: 返回数量限制

        Returns:
            策略订单列表
        """
        try:
            kwargs = {"instType": inst_type, "limit": limit}
            if algo_id:
                kwargs["algoId"] = algo_id
            if inst_id:
                kwargs["instId"] = inst_id
            if ord_type:
                kwargs["ordType"] = ord_type
            if state:
                kwargs["state"] = state
            # 修复: SDK 方法名为 order_algos_list（原 get_algo_order_list 不存在）
            result = self._trade.order_algos_list(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_algo_orders exception: {e}")
            return []

    def get_algo_order_details(self, algo_id: str) -> Optional[dict]:
        """查询单个策略订单详情（含 TP/SL 字段，用于落单核验与 TP 保留）。

        GET /api/v5/trade/order-algo

        Args:
            algo_id: 策略订单 ID

        Returns:
            订单详情 dict 或 None
        """
        if not algo_id:
            return None
        try:
            result = self._trade.get_algo_order_details(algoId=algo_id)
            if result.get("code") == "0" and result.get("data"):
                return result["data"][0]
            return None
        except Exception as e:
            logger.error(f"get_algo_order_details exception: {e}")
            return None

    def cancel_order(self, inst_id: str, ord_id: str) -> bool:
        """撤销订单。"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel {ord_id}")
            return True

        try:
            # 修复 P0-C: 撤单加退避重试
            result = _call_sdk_retry(
                self._trade.cancel_order, instId=inst_id, ordId=ord_id)
            return result.get("code") == "0"
        except Exception as e:
            logger.error(f"cancel_order exception: {e}")
            return False

    def get_order(self, inst_id: str, ord_id: str) -> Optional[dict]:
        """查询订单状态。"""
        try:
            result = self._trade.get_order(instId=inst_id, ordId=ord_id)
            if result.get("code") == "0" and result.get("data"):
                return result["data"][0]
            return None
        except Exception as e:
            logger.error(f"get_order exception: {e}")
            return None

    def close_position(self, inst_id: str, pos_side: str,
                      mgn_mode: str = "isolated",
                      ccy: str = "", auto_cxl: str = "",
                      cl_ord_id: str = "") -> Optional[str]:
        """市价仓位全平。

        使用专用 POST /api/v5/trade/close-position 端点，
        根据持仓方向自动匹配平仓方向，无需手动指定 side 和 sz。
        文档: https://www.okx.com/docs-v5/zh/#order-book-trading-trade-post-close-position

        Args:
            inst_id: 合约 ID（如 "BTC-USDT-SWAP"）
            pos_side: 持仓方向 "long" | "short"（开平仓模式下必填）
            mgn_mode: 保证金模式 "isolated" | "cross"
            ccy: 保证金币种，仅单币种保证金模式下的全仓平仓时需要
            auto_cxl: 是否自动撤销触发市价全平的订单
            cl_ord_id: 客户端自定义订单 ID

        Returns:
            ord_id 或 None
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Close position {inst_id} {pos_side} "
                        f"mgn_mode={mgn_mode}")
            return "dry_run_close_" + str(int(time.time() * 1000))

        try:
            kwargs = {
                "instId": inst_id,
                "mgnMode": mgn_mode,
                "posSide": pos_side,
            }
            if ccy:
                kwargs["ccy"] = ccy
            if auto_cxl:
                kwargs["autoCxl"] = auto_cxl
            if cl_ord_id:
                kwargs["clOrdId"] = cl_ord_id

            # 修复: SDK 方法名为 close_positions（原 close_position 不存在，
            # 导致市价全平路径永远抛 AttributeError）
            # 修复 P0-C: 加退避重试（平仓单失败 → 仓位滞留风险）
            result = _call_sdk_retry(self._trade.close_positions, **kwargs)
            if result.get("code") == "0":
                ord_id = (result.get("data", [{}])[0].get("ordId", "")
                          if result.get("data") else "")
                if ord_id:
                    logger.info(f"Position closed: {ord_id} — "
                                f"{inst_id} {pos_side}")
                    return ord_id
                # 修复: OKX close-position 成功但响应无 ordId（极小仓实测），
                # 直接返回 None 会让上层误判平仓失败，走下方通用核验
                logger.warning(
                    f"close_position: {inst_id} {pos_side} code=0 但响应无 ordId，"
                    f"核验实际持仓..."
                )
            else:
                # 修复: OKX close-position 端点实测返回 51023 ("Position doesn't
                # exist") 但已成功平仓（极小仓市价全平实测: 返回 51023 后交易所
                # 持仓已清空）。返回前核验交易所实际持仓，已平则判定成功。
                logger.warning(
                    f"close_position: {result.get('code')} {result.get('msg')}, "
                    f"核验 {inst_id} {pos_side} 实际持仓..."
                )
            # 通用核验: 返回非 None 前确认交易所持仓已清空（覆盖 51023 竞态与
            # code=0 无 ordId 两种情况），避免上层误判平仓失败而重复平仓。
            # 修复 P0-B/P1-4: get_positions 失败返回 None 时不得据此判定成功；
            # 持仓判定用 abs()（兼容 cross 模式空头 pos 为负）。
            try:
                _pos = self.get_positions(inst_id=inst_id)
                if _pos is None:
                    logger.error(
                        f"close_position: 核验持仓查询失败，不能判定平仓成功，"
                        f"{inst_id} {pos_side}"
                    )
                else:
                    _left = [
                        p for p in _pos
                        if p.get("posSide") == pos_side
                        and abs(float(p.get("pos", 0))) > 0
                    ]
                    if not _left:
                        logger.warning(
                            f"close_position: {inst_id} {pos_side} 交易所已无持仓，"
                            f"判定平仓成功"
                        )
                        return "closed_pos"
            except Exception:
                pass
            if result.get("code") == "0":
                logger.error(f"close_position: {inst_id} {pos_side} code=0 但持仓未清空")
            else:
                logger.error(f"close_position failed: {result.get('msg')}")
            return None
        except Exception as e:
            logger.error(f"close_position exception: {e}")
            return None

    def get_close_order_pnl(self, inst_id: str, ord_id: str,
                            max_retries: int = 3,
                            retry_delay: float = 0.5) -> float:
        """查询平仓订单的交易所端已实现盈亏。

        修复: 不再使用全局权益差（会被其他持仓/资金费率污染），
        直接从 OKX 订单的 pnl 字段获取该笔平仓的精确盈亏。

        OKX API v5 订单返回的 pnl 字段:
          - 正数 = 盈利，负数 = 亏损
          - 仅在订单状态为 filled 时有值
          - 精确到该笔订单的已实现盈亏，不受其他持仓影响

        Args:
            inst_id: 合约 ID
            ord_id: 平仓订单 ID
            max_retries: 最大重试次数（订单可能尚未成交）
            retry_delay: 重试间隔（秒）

        Returns:
            已实现盈亏 (USDT)，获取失败返回 0.0
        """
        if self.dry_run or ord_id.startswith("dry_run"):
            return 0.0

        for attempt in range(max_retries):
            try:
                order_info = self.get_order(inst_id=inst_id, ord_id=ord_id)
                if order_info:
                    state = order_info.get("state", "")
                    pnl_str = order_info.get("pnl", "0")
                    pnl = float(pnl_str) if pnl_str else 0.0
                    if state == "filled" and pnl != 0.0:
                        logger.debug(f"[ClosePnl] {inst_id} ord={ord_id} pnl={pnl:.4f} USDT")
                        return pnl
                    if state == "filled" and pnl == 0.0:
                        # 订单已成交但 pnl 为 0（保本平仓），不再重试
                        return 0.0
                # 订单尚未成交，等待后重试
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"[ClosePnl] get_order failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.warning(f"[ClosePnl] 无法获取 {inst_id} ord={ord_id} 的 PnL，返回 0")
        return 0.0

    def close_position_limit(self, inst_id: str, pos_side: str,
                             pos_size: str, mark_price: float,
                             td_mode: str = "isolated",
                             timeout_sec: Optional[float] = None) -> Tuple[Optional[str], float]:
        """限价单平仓 — 竞争性限价策略，优先快速成交避免滑点。

        使用对手方最优报价（best_bid/best_ask）作为限价，确保高概率成交。
        若限价单超时未成交，自动降级为市价单兜底。
        修复 2026-08-07: 超时秒数与滑点保护改读 config
        (risk.close_limit_wait_seconds / close_market_fallback_slippage_pct),
        原硬编码 30s/±0.1% 导致配置项形同虚设。
        注意：使用对手方报价意味着以 Taker 身份成交（费率 0.05%），
        而非 Maker（费率 0.02%）。若需 Maker 策略，应将限价偏离 1 tick。

        Args:
            inst_id: 合约 ID
            pos_side: 持仓方向 "long" | "short"
            pos_size: 持仓张数（字符串）
            mark_price: 当前标记价格
            td_mode: 交易模式 "isolated" | "cross"
            timeout_sec: 限价单超时秒数，超时后降级市价单；None=用 config 值

        Returns:
            (ord_id, fill_price) — ord_id 为 None 表示失败
        """
        if timeout_sec is None:
            timeout_sec = self.close_limit_timeout_s

        if self.dry_run:
            logger.info(f"[DRY RUN] Limit close {inst_id} {pos_side} "
                        f"size={pos_size} mark={mark_price}")
            return "dry_run_limit_" + str(int(time.time() * 1000)), mark_price

        # 1. 获取订单簿
        order_book = self.get_order_book(inst_id, sz=1)
        if not order_book:
            logger.warning(f"[LimitClose] 无法获取订单簿 {inst_id}，降级市价单")
            ord_id = self.close_position(inst_id=inst_id, pos_side=pos_side,
                                         mgn_mode=td_mode)
            return ord_id, mark_price

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            logger.warning(f"[LimitClose] 订单簿为空 {inst_id}，降级市价单")
            ord_id = self.close_position(inst_id=inst_id, pos_side=pos_side,
                                         mgn_mode=td_mode)
            return ord_id, mark_price

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
        except (ValueError, IndexError, TypeError):
            logger.warning(f"Order book data anomaly for {inst_id}, falling back to market close")
            ord_id = self.close_position(inst_id=inst_id, pos_side=pos_side, mgn_mode=td_mode)
            return ord_id, mark_price

        # 2. 获取 tick_size
        try:
            info = self.get_instrument_info(inst_id)
            tick_sz = float(info.get("tickSz", "0.1")) if info else 0.1
        except Exception:
            tick_sz = 0.1

        # 3. 计算竞争性限价 — 修复 2026-08-07: 滑点保护读 config
        # (risk.close_market_fallback_slippage_pct), 原硬编码 ±0.1%。
        _slip = self.close_fallback_slippage_pct / 100.0
        if pos_side == "long":
            # 卖出平多：取 max(bid, mark×(1-slip)) 避免卖在异常低价
            limit_px = max(best_bid, mark_price * (1 - _slip))
            limit_side = "sell"
        else:
            # 买入平空：取 min(ask, mark×(1+slip)) 避免买在异常高价
            limit_px = min(best_ask, mark_price * (1 + _slip))
            limit_side = "buy"

        # tick_size 对齐
        if pos_side == "long":
            # 卖出平多 — 向下取整，确保 ≤ best_bid，快速成交
            limit_px = math.floor(limit_px / tick_sz) * tick_sz
        else:
            # 买入平空 — 向上取整，确保 ≥ best_ask，快速成交
            limit_px = math.ceil(limit_px / tick_sz) * tick_sz
        limit_px_str = self._format_price(limit_px, tick_sz)

        logger.info(
            f"[LimitClose] {inst_id} {pos_side} size={pos_size} "
            f"limit_px={limit_px_str} bid={best_bid} ask={best_ask} mark={mark_price}"
        )

        # 4. 下限价单
        limit_ord_id = self.place_order(
            inst_id=inst_id,
            side=limit_side,
            pos_side=pos_side,
            sz=pos_size,
            px=limit_px_str,
            ord_type="limit",
            td_mode=td_mode,
            reduce_only=True,
        )
        if not limit_ord_id:
            logger.error(f"[LimitClose] 限价单下单失败 {inst_id}，降级市价单")
            ord_id = self.close_position(inst_id=inst_id, pos_side=pos_side,
                                         mgn_mode=td_mode)
            return ord_id, mark_price

        # 5. 轮询等待成交
        poll_interval = 0.5
        start = time.time()
        while time.time() - start < timeout_sec:
            order_info = self.get_order(inst_id=inst_id, ord_id=limit_ord_id)
            if not order_info:
                time.sleep(poll_interval)
                continue
            state = order_info.get("state", "")
            if state == "filled":
                avg_px = self._safe_float(order_info.get("avgPx", "0"))
                fill_px = avg_px if avg_px > 0 else mark_price
                logger.info(
                    f"[LimitClose] {inst_id} 限价单已成交 "
                    f"ord={limit_ord_id} fill_px={fill_px}"
                )
                return limit_ord_id, fill_px
            if state == "partially_filled":
                acc_fill_sz = float(order_info.get("accFillSz", "0"))
                total_sz = float(order_info.get("sz", "1"))
                fill_ratio = acc_fill_sz / total_sz if total_sz > 0 else 0
                if fill_ratio >= 0.90:  # 90%+ filled, accept as done
                    avg_px = self._safe_float(order_info.get("avgPx", "0"))
                    fill_px = avg_px if avg_px > 0 else mark_price
                    self.cancel_order(inst_id=inst_id, ord_id=limit_ord_id)
                    logger.info(
                        f"Limit close {inst_id} {fill_ratio*100:.0f}% filled, "
                        f"cancelled remaining"
                    )
                    # 修复 P1-2: 残仓市价兜底 — 剩余 ≤10% 仓位可能仍挂在交易所，
                    # 直接市价清掉，避免残仓滞留导致换仓中断/状态错乱
                    _rem = self.close_position(inst_id=inst_id, pos_side=pos_side,
                                               mgn_mode=td_mode)
                    if _rem is None:
                        logger.warning(
                            f"[LimitClose] {inst_id} 残仓市价兜底失败"
                            f"（下一轮 pending_close 核验将重试）"
                        )
                    return limit_ord_id, fill_px
                # 等待剩余部分成交
            if state in ("canceled", "failed"):
                logger.warning(
                    f"[LimitClose] {inst_id} 限价单状态={state}，降级市价单"
                )
                break
            time.sleep(poll_interval)

        # 6. 超时，撤销限价单并市价兜底
        logger.warning(
            f"[LimitClose] {inst_id} 限价单 {limit_ord_id} "
            f"超时 ({timeout_sec}s)，撤销并降级市价单"
        )
        self.cancel_order(inst_id=inst_id, ord_id=limit_ord_id)
        ord_id = self.close_position(inst_id=inst_id, pos_side=pos_side,
                                     mgn_mode=td_mode)
        return ord_id, mark_price

    def close_position_safe(self, inst_id: str, pos_side: str,
                            pos_size: float = 0.0,
                            mgn_mode: str = "isolated",
                            max_slippage_pct: float = 0.005) -> Tuple[Optional[str], bool]:
        """限价单保护平仓 — 始终使用限价单，Maker 费率 + 零滑点。

        默认使用 close_position_limit 以限价单平仓，享受 Maker 费率（0.02%）
        和零滑点。若 30s 内未成交，自动降级为市价单兜底。
        保留 max_slippage_pct 参数以兼容旧调用方，实际不再使用。

        Args:
            inst_id: 合约 ID
            pos_side: 持仓方向 "long" | "short"
            pos_size: 持仓张数
            mgn_mode: 保证金模式
            max_slippage_pct: 保留参数（兼容性），不再使用

        Returns:
            (ord_id, is_limit_order) — 向后兼容
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Safe close position {inst_id} {pos_side}")
            return "dry_run_safe_" + str(int(time.time() * 1000)), True

        # 获取标记价格
        mark_price = self.get_mark_price(inst_id)
        if mark_price is None:
            logger.warning(f"[SafeClose] 无法获取标记价格 {inst_id}，降级市价单")
            ord_id = self.close_position(
                inst_id=inst_id, pos_side=pos_side, mgn_mode=mgn_mode,
            )
            return ord_id, False

        # 格式化 pos_size 为字符串，并对齐 lotSz
        try:
            info = self.get_instrument_info(inst_id)
            lot_sz = info.get("lotSz", "1") if info else "1"
        except Exception:
            lot_sz = "1"

        if lot_sz and float(lot_sz) > 0:
            ls = float(lot_sz)
            ps = float(pos_size)
            pos_size_aligned = math.floor(ps / ls) * ls
            if pos_size_aligned <= 0:
                logger.warning(f"Position size {pos_size} is below lotSz {lot_sz}")
                return None, False
            if "." in lot_sz:
                dec = len(lot_sz.split(".")[1])
            else:
                dec = 0
            sz_str = f"{pos_size_aligned:.{dec}f}"
        else:
            sz_str = f"{float(pos_size):g}"

        # 使用限价单平仓（30s 超时自动降级市价单）
        ord_id, _fill_px = self.close_position_limit(
            inst_id=inst_id,
            pos_side=pos_side,
            pos_size=sz_str,
            mark_price=mark_price,
            td_mode=mgn_mode,
        )
        return ord_id, True

    def set_leverage(self, inst_id: str, lever: int,
                     mgn_mode: str = "isolated",
                     pos_side: Optional[str] = None) -> bool:
        """设置合约杠杆倍数。

        OKX API v5 要求在下单前必须设置杠杆，否则使用默认 1x。
        POST /api/v5/account/set-leverage

        修复: 双向持仓模式下必须传 posSide，否则 API 返回
        "Parameter posSide error"。单向持仓模式下不需要。

        Args:
            inst_id: 合约 ID 或币种（如 "BTC-USDT-SWAP"）
            lever: 杠杆倍数
            mgn_mode: 保证金模式 "isolated" | "cross"
            pos_side: 持仓方向 "long" | "short" | None（单向持仓模式）

        Returns:
            是否成功
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Set leverage: {inst_id} {lever}x {mgn_mode}")
            return True

        try:
            # 构造参数，仅包含必要字段
            # 修复: 绕过 SDK 的 set_leverage() 方法，因为 SDK 默认 posSide=''
            # 和 ccy=''，始终将空字符串发送给 API，导致 "Parameter posSide error"
            params = {
                "instId": inst_id,
                "lever": str(lever),
                "mgnMode": mgn_mode,
            }
            if pos_side:
                params["posSide"] = pos_side

            # 修复 P0-C: 设杠杆加退避重试
            result = _call_sdk_retry(
                self._account._request_with_params,
                "POST", "/api/v5/account/set-leverage", params)
            if result.get("code") == "0":
                logger.info(f"Leverage set: {inst_id} {lever}x {mgn_mode}"
                            f"{' posSide=' + pos_side if pos_side else ''}")
                return True
            else:
                logger.error(f"set_leverage failed: {result.get('msg')}")
                return False
        except Exception as e:
            logger.error(f"set_leverage exception: {e}")
            return False

    def get_instrument_info(self, inst_id: str) -> Optional[dict]:
        """获取合约信息（面值、最小下单量等）。"""
        try:
            result = self._public.get_instruments(
                instType="SWAP", instId=inst_id
            )
            if result.get("code") == "0" and result.get("data"):
                return result["data"][0]
            return None
        except Exception as e:
            logger.error(f"get_instrument_info exception: {e}")
            return None

    def get_position_tiers(self, inst_id: str,
                           td_mode: str = "isolated") -> Optional[dict]:
        """获取合约杠杆档位（含维持保证金率 mmr / 最大杠杆 maxLever）。

        GET /api/v5/public/position-tiers

        修复 P0-A: 强平距离计算依赖档位 MMR。档位按名义价值分档，
        取第一档（小仓位）的 mmr 即可满足开仓前校验精度需求。

        修复 2026-08-09 (满倍率模式): SWAP 合约此接口必须传 instFamily
        (如 "RLS-USDT") 或 uly，只传 instId 返回 code=50015
        "Either parameter instFamily or uly is required"，导致
        resolve_full_leverage 恒回退信号杠杆，满倍率模式静默失效。

        Returns:
            档位 dict（含 mmr, maxLever 等），失败返回 None（fail-open：
            上层用保守默认 MMR 兜底）。
        """
        try:
            # SWAP 品种: instFamily = instId 去掉 "-SWAP" 后缀 (RLS-USDT-SWAP → RLS-USDT)
            _inst_family = inst_id[:-5] if inst_id.endswith("-SWAP") else ""
            result = self._public.get_position_tiers(
                instType="SWAP", tdMode=td_mode, instId=inst_id,
                instFamily=_inst_family
            )
            if result.get("code") == "0" and result.get("data"):
                return result["data"][0]
            logger.debug(
                f"get_position_tiers {inst_id} failed: code={result.get('code')} "
                f"msg={result.get('msg')}")
            return None
        except Exception as e:
            logger.debug(f"get_position_tiers exception: {e}")
            return None

    # ------------------------------------------------------------------
    # 聪明钱追踪 — Copy Trading 公开接口（无需鉴权）
    # ------------------------------------------------------------------
    # 参考: OKX V5 API → Copy Trading → Public endpoints
    # https://www.okx.com/docs-v5/zh/#order-book-trading-copy-trading

    def _copytrading_get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """调用 Copy Trading 公开接口（无需 API Key）。

        Args:
            path: API 路径（如 "/api/v5/copytrading/public-lead-traders"）
            params: 查询参数

        Returns:
            API 响应 data 字段，失败返回 None
        """
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(
                url, params=params or {},
                timeout=15,
                proxies={"http": self.proxy, "https": self.proxy} if self.proxy else None,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") == "0":
                return body.get("data", [])
            # 修复: 60004=Trader doesn't exist — 排行榜动态变化，榜单中的交易员
            # 可能已退榜/失效，属正常现象。降级为 debug 避免每次刷新刷 warning。
            if body.get("code") == "60004":
                logger.debug(
                    f"CopyTrading API trader-not-exist: {body.get('msg')} "
                    f"(code=60004) path={path}")
            else:
                logger.warning(f"CopyTrading API error: {body.get('msg')} (code={body.get('code')})")
            return None
        except requests.RequestException as e:
            logger.warning(f"CopyTrading API request failed: {e}")
            return None

    def get_top_lead_traders(
        self,
        inst_type: str = "SWAP",
        limit: int = 30,
        sort_type: str = "",
        min_assets: str = "",
    ) -> List[dict]:
        """获取 top 带单交易员排行（公开接口）。

        Args:
            inst_type: 产品类型，SWAP=永续合约
            limit: 返回数量上限
            sort_type: 排序方式（空=默认）
            min_assets: 最低资产（如 "10000"）

        Returns:
            交易员列表，每项含 uniqueCode/nickName/instType/总收益/AUM 等
        """
        params: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
        if sort_type:
            params["sortType"] = sort_type
        if min_assets:
            params["minAssets"] = min_assets

        result = self._copytrading_get("/api/v5/copytrading/public-lead-traders", params)
        if not isinstance(result, list) or not result:
            return []
        # 修复: OKX 返回 data = [{"dataVer": "...", "ranks": [交易员...]}] 包装结构，
        # 交易员列表在 data[0]["ranks"] 中，直接返回 data 会把包装对象当交易员，
        # 其 uniqueCode 为空导致 get_smart_money_coins 永远跳过所有人 → 0 共识币种。
        wrapper = result[0]
        if isinstance(wrapper, dict) and isinstance(wrapper.get("ranks"), list):
            return wrapper["ranks"]
        return result if isinstance(result, list) else []

    def get_trader_current_positions(self, unique_code: str) -> List[dict]:
        """获取指定交易员当前持仓（公开接口）。

        Args:
            unique_code: 交易员唯一码（从 get_top_lead_traders 获取）

        Returns:
            持仓列表，每项含 instId/posSide/lever/openAvgPx/subPos 等
        """
        result = self._copytrading_get(
            "/api/v5/copytrading/public-current-subpositions",
            {"uniqueCode": unique_code},
        )
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # 历史交易记录 — 账单 & 成交明细
    # ------------------------------------------------------------------

    def get_bills(self, inst_type: str = "SWAP", limit: int = 100,
                  begin: str = "", end: str = "",
                  bill_type: str = "") -> List[dict]:
        """获取账户账单流水（含已实现盈亏）。

        GET /api/v5/account/bills

        Args:
            inst_type: 产品类型，默认 SWAP
            limit: 返回条数上限（最大 100）
            begin: 起始时间戳（毫秒），空=不限
            end: 结束时间戳（毫秒），空=不限
            bill_type: 账单类型过滤（空=全部）

        Returns:
            账单列表，每项含 billId/instId/px/sz/pnl/billType/ts 等字段
        """
        try:
            kwargs: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
            if begin:
                kwargs["begin"] = begin
            if end:
                kwargs["end"] = end
            if bill_type:
                # 修复 2026-08-14: SDK v0.4.3 账单参数名为 `type`（非 billType），
                # 方法名为 `get_account_bills`（非 get_bills）。旧调用会 AttributeError
                # 被 except 吞掉返回 []，导致账单拉取永远为空。
                kwargs["type"] = bill_type
            result = self._account.get_account_bills(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_bills failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_bills exception: {e}")
            return []

    def get_bills_archive(self, inst_type: str = "SWAP", limit: int = 100,
                          begin: str = "", end: str = "",
                          bill_type: str = "") -> List[dict]:
        """获取三个月前的历史账单（归档数据）。

        GET /api/v5/account/bills-archive

        Args:
            inst_type: 产品类型
            limit: 返回条数上限
            begin: 起始时间戳
            end: 结束时间戳
            bill_type: 账单类型

        Returns:
            账单列表
        """
        try:
            kwargs: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
            if begin:
                kwargs["begin"] = begin
            if end:
                kwargs["end"] = end
            if bill_type:
                # 修复 2026-08-14: SDK v0.4.3 参数名为 `type`（非 billType）。
                kwargs["type"] = bill_type
            result = self._account.get_account_bills_archive(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_bills_archive failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_bills_archive exception: {e}")
            return []

    def get_fills(self, inst_type: str = "SWAP", limit: int = 100,
                  inst_id: str = "", ord_id: str = "",
                  begin: str = "", end: str = "") -> List[dict]:
        """获取成交明细（每笔成交的价格、数量、方向）。

        GET /api/v5/trade/fills

        Args:
            inst_type: 产品类型
            limit: 返回条数上限
            inst_id: 合约 ID（可选）
            ord_id: 订单 ID（可选）
            begin: 起始时间戳
            end: 结束时间戳

        Returns:
            成交明细列表，每项含 instId/tradeId/ordId/px/sz/side/ts 等字段
        """
        try:
            kwargs: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
            if inst_id:
                kwargs["instId"] = inst_id
            if ord_id:
                kwargs["ordId"] = ord_id
            if begin:
                kwargs["begin"] = begin
            if end:
                kwargs["end"] = end
            result = self._trade.get_fills(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_fills failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_fills exception: {e}")
            return []

    def get_orders_history(self, inst_type: str = "SWAP", limit: int = 100,
                           inst_id: str = "", state: str = "filled",
                           begin: str = "", end: str = "") -> List[dict]:
        """获取历史订单记录（最近 7 天）。

        GET /api/v5/trade/orders-history

        Args:
            inst_type: 产品类型
            limit: 返回条数上限
            inst_id: 合约 ID（可选）
            state: 订单状态，默认 filled（已成交）
            begin: 起始时间戳
            end: 结束时间戳

        Returns:
            订单列表，每项含 instId/ordId/px/sz/avgPx/side/state/cTime 等字段
        """
        try:
            kwargs: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
            if inst_id:
                kwargs["instId"] = inst_id
            if state:
                kwargs["state"] = state
            if begin:
                kwargs["begin"] = begin
            if end:
                kwargs["end"] = end
            result = self._trade.get_orders_history(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_orders_history failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_orders_history exception: {e}")
            return []

    def get_orders_history_archive(self, inst_type: str = "SWAP", limit: int = 100,
                                    inst_id: str = "", state: str = "filled",
                                    begin: str = "", end: str = "") -> List[dict]:
        """获取三个月前的历史订单（归档数据）。

        GET /api/v5/trade/orders-history-archive

        Args:
            inst_type: 产品类型
            limit: 返回条数上限
            inst_id: 合约 ID（可选）
            state: 订单状态
            begin: 起始时间戳
            end: 结束时间戳

        Returns:
            订单列表
        """
        try:
            kwargs: Dict[str, str] = {"instType": inst_type, "limit": str(limit)}
            if inst_id:
                kwargs["instId"] = inst_id
            if state:
                kwargs["state"] = state
            if begin:
                kwargs["begin"] = begin
            if end:
                kwargs["end"] = end
            result = self._trade.get_orders_history_archive(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            logger.error(f"get_orders_history_archive failed: {result.get('msg')}")
            return []
        except Exception as e:
            logger.error(f"get_orders_history_archive exception: {e}")
            return []

    def get_all_bills_paginated(
        self,
        inst_type: str = "SWAP",
        days_back: int = 90,
        bill_type: str = "",
        page_limit: int = 100,
    ) -> List[dict]:
        """分页获取所有历史账单（自动翻页，合并最近和归档数据）。

        Args:
            inst_type: 产品类型
            days_back: 回溯天数
            bill_type: 账单类型过滤
            page_limit: 每页条数

        Returns:
            所有账单列表，按时间戳降序
        """
        all_bills: List[dict] = []
        now_ms = int(time.time() * 1000)
        begin_ms = str(now_ms - days_back * 24 * 3600 * 1000)
        end_ms = str(now_ms)
        three_months_ago = str(now_ms - 90 * 24 * 3600 * 1000)

        # 先拉最近账单（近 3 个月）
        after = begin_ms
        while True:
            bills = self.get_bills(
                inst_type=inst_type,
                limit=page_limit,
                begin=after,
                end=end_ms,
                bill_type=bill_type,
            )
            if not bills:
                break
            all_bills.extend(bills)
            if len(bills) < page_limit:
                break
            # 下一页：从最早一条的时间戳往前
            new_after = str(int(bills[-1].get("ts", "0")))
            if new_after == after or new_after == "0":
                break
            after = new_after
            time.sleep(0.15)

        # 如果回溯超过 3 个月，拉归档数据
        if begin_ms < three_months_ago:
            after = begin_ms
            archive_end = three_months_ago
            while True:
                bills = self.get_bills_archive(
                    inst_type=inst_type,
                    limit=page_limit,
                    begin=after,
                    end=archive_end,
                    bill_type=bill_type,
                )
                if not bills:
                    break
                all_bills.extend(bills)
                if len(bills) < page_limit:
                    break
                new_after = str(int(bills[-1].get("ts", "0")))
                if new_after == after or new_after == "0":
                    break
                after = new_after
                time.sleep(0.15)

        # 按 ts 降序去重
        seen: set = set()
        unique: List[dict] = []
        for b in sorted(all_bills, key=lambda x: int(x.get("ts", "0")), reverse=True):
            bid = b.get("billId", "")
            if bid and bid not in seen:
                seen.add(bid)
                unique.append(b)
        return unique

    def get_smart_money_coins(
        self,
        top_n_traders: int = 20,
        min_traders_holding: int = 2,
    ) -> Dict[str, int]:
        """聚合 top 交易员的持仓币种，按「持有该币种的交易员数」降序排列。

        流程：
          1. 获取 top N 带单交易员
          2. 逐个查询当前持仓
          3. 统计每个币种被多少 top 交易员同时持有
          4. 返回 {instId: holder_count}，按持有者数降序

        Args:
            top_n_traders: 查询前 N 名交易员
            min_traders_holding: 最少持有者数阈值

        Returns:
            {inst_id: holder_count}，按 holder_count 降序
        """
        traders = self.get_top_lead_traders(
            inst_type="SWAP",
            limit=top_n_traders,
        )
        if not traders:
            logger.debug("[SmartMoney] 未获取到带单交易员数据")
            return {}

        # 按 uniqueCode 去重
        seen: set = set()
        coin_holders: Dict[str, int] = {}
        queried = 0

        for t in traders:
            code = t.get("uniqueCode", "")
            if not code or code in seen:
                continue
            seen.add(code)

            positions = self.get_trader_current_positions(code)
            if not positions:
                continue
            queried += 1

            for p in positions:
                inst_id = p.get("instId", "")
                v = p.get("subPos", "0")
                pos_sz = self._safe_float(v, 0.0)
                if inst_id and abs(pos_sz) > 0:
                    coin_holders[inst_id] = coin_holders.get(inst_id, 0) + 1

            # 速率限制：公开接口也要适当冷却
            time.sleep(0.15)

        # 过滤低于阈值的币种
        result = {
            k: v for k, v in coin_holders.items()
            if v >= min_traders_holding
        }
        # 按持有者数降序排列
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        logger.info(
            f"[SmartMoney] 查询 {queried}/{len(traders)} 个交易员，"
            f"发现 {len(result)} 个共识币种 (≥{min_traders_holding}人持有)"
        )
        return result

    # ------------------------------------------------------------------
    # 资金账户 / 链上提现 (Royalty 盈利分成用)
    # ------------------------------------------------------------------

    def get_withdrawal_fee_info(self, chain: str = "USDT-TRC20"
                                ) -> Optional[Tuple[float, float]]:
        """查询链上提现手续费与最小提现额。

        经 GET /api/v5/asset/currencies 获取指定链的 minFee / minWd。

        Returns:
            (fee, min_wd) 元组；查询失败或未找到链信息返回 None (fail-closed)
        """
        try:
            result = _call_sdk_retry(self._funding.get_currencies, ccy="USDT")
            for d in (result or {}).get("data", []) or []:
                if d.get("chain") == chain:
                    fee = self._safe_float(d.get("minFee"), 0.0)
                    min_wd = self._safe_float(d.get("minWd"), 0.0)
                    if fee > 0:
                        return (fee, min_wd)
            logger.error(f"get_withdrawal_fee_info: 未找到链 {chain} 的费率信息")
            return None
        except Exception as e:
            logger.error(f"get_withdrawal_fee_info exception: {e}")
            return None

    def get_funding_balance(self, ccy: str = "USDT") -> Optional[float]:
        """获取资金账户 (funding) 某币种可用余额。

        Returns:
            可用余额 (无该币种持仓时为 0.0)；查询失败返回 None (fail-closed,
            调用方必须区分"余额为 0"与"查询失败")
        """
        try:
            result = _call_sdk_retry(self._funding.get_balances, ccy=ccy)
            data = (result or {}).get("data", []) or []
            if not data:
                return 0.0
            d = data[0]
            return self._safe_float(d.get("availBal") or d.get("bal"), 0.0)
        except Exception as e:
            logger.error(f"get_funding_balance exception: {e}")
            return None

    def transfer_trading_to_funding(self, ccy: str, amt: float) -> bool:
        """交易账户 → 资金账户划转 (链上提现的前置步骤, 提现只能从资金账户发起)。

        Args:
            ccy: 币种, 如 USDT
            amt: 划转金额 (USDT)

        Returns:
            划转是否成功
        """
        if amt <= 0:
            return True
        try:
            # OKX 账户代码: 18=Trading account, 6=Funding account
            result = _call_sdk_retry(
                self._funding.funds_transfer, ccy, f"{amt:.2f}", "18", "6")
            ok = bool(result) and result.get("code") == "0"
            if not ok:
                logger.error(
                    f"transfer_trading_to_funding failed: "
                    f"{(result or {}).get('msg')}")
            return ok
        except Exception as e:
            logger.error(f"transfer_trading_to_funding exception: {e}")
            return False

    def submit_withdrawal(self, ccy: str, amt: float, fee: float,
                          dest: str, to_addr: str, chain: str) -> Optional[dict]:
        """发起链上提现 (Royalty 分成专用)。

        注: SDK v0.4.3 的 FundingAPI.withdrawal() 未暴露必填的 fee 参数,
        故复用 SDK 底层 _request_with_params (自动签名) 发送完整请求体。

        Args:
            ccy: 币种 (USDT)
            amt: 提现数量 (保守口径: 到账额, 总扣款 = amt + fee)
            fee: 链上手续费 (从 get_withdrawal_fee_info 实时获取)
            dest: "3"=链上提现
            to_addr: 收款地址 (TRC20)
            chain: 链名 (USDT-TRC20)

        Returns:
            OKX 响应 dict (含 code/msg/data[wdId]); 网络异常返回 None
        """
        params = {
            "ccy": ccy,
            "amt": f"{amt:.2f}",
            "fee": f"{fee:.2f}",
            "dest": dest,
            "toAddr": to_addr,
            "chain": chain,
        }
        try:
            return self._funding._request_with_params(POST, WITHDRAWAL_COIN, params)
        except Exception as e:
            logger.error(f"submit_withdrawal exception: {e}")
            return None

    def get_withdrawal_info(self, wd_id: str) -> Optional[dict]:
        """按 wdId 查询单笔提现状态 (核验 pending 提现到账)。

        Returns:
            OKX 响应 dict, data[0].state ∈ Pending/Review/Success/Failure...
        """
        if not wd_id:
            return None
        try:
            return _call_sdk_retry(
                self._funding.get_withdrawal_history,
                ccy="USDT", wdId=wd_id)
        except Exception as e:
            logger.error(f"get_withdrawal_info exception: {e}")
            return None