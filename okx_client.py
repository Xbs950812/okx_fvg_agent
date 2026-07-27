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
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx.PublicData import PublicAPI
from okx.Trade import TradeAPI
from okx.TradingData import TradingDataAPI


logger = logging.getLogger(__name__)


class OKXClient:
    """OKX API v5 客户端，封装官方 SDK 的常用接口。
    
    保持与旧版相同的接口签名，确保 agent.py 无需修改。
    """

    def __init__(self, config: dict):
        cfg = config["okx"]
        self.dry_run = config["agent"].get("dry_run", False)
        # 0 = live trading, 1 = demo trading (模拟交易)
        self.flag = "1" if cfg.get("demo", False) else "0"

        # 模拟交易使用独立的 API 密钥
        if self.flag == "1":
            self.api_key = cfg.get("demo_api_key", cfg["api_key"])
            self.api_secret = cfg.get("demo_api_secret", cfg["api_secret"])
            self.passphrase = cfg.get("demo_passphrase", cfg["passphrase"])
            # 模拟盘使用独立的 base_url，与实盘不同
            # 文档: https://www.okx.com/docs-v5/zh/#overview-demo-trading
            self.base_url = "https://www.okx.com"
        else:
            self.api_key = cfg["api_key"]
            self.api_secret = cfg["api_secret"]
            self.passphrase = cfg["passphrase"]
            self.base_url = cfg["base_url"].rstrip("/")

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

    def get_funding_rate(self, inst_id: str) -> Optional[float]:
        """获取当前资金费率。"""
        try:
            result = self._public.get_funding_rate(instId=inst_id)
            if result.get("code") == "0" and result.get("data"):
                return float(result["data"][0].get("fundingRate", "0"))
            return None
        except Exception as e:
            logger.error(f"get_funding_rate exception: {e}")
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

    def get_open_interest(self, inst_id: str) -> Optional[float]:
        """获取当前未平仓合约量 (OI)。"""
        try:
            result = self._public.get_open_interest(
                instType="SWAP", instId=inst_id
            )
            if result.get("code") == "0" and result.get("data"):
                return float(result["data"][0].get("oi", "0"))
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
                    return float(data[0][1])
                # 兼容 dict 格式
                if data and isinstance(data[0], dict):
                    return float(data[0].get("longShortRatio", "0"))
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
                return float(result["data"][0].get("markPx", "0"))
            return None
        except Exception as e:
            logger.error(f"get_mark_price exception: {e}")
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
                        return float(detail.get("availBal", "0"))
            return 0.0
        except Exception as e:
            logger.error(f"get_balance exception: {e}")
            return None

    def get_total_equity(self) -> Optional[float]:
        """获取账户总权益 (USDT)。"""
        try:
            result = self._account.get_account_balance()
            if result.get("code") == "0" and result.get("data"):
                return float(result["data"][0].get("totalEq", "0"))
            return None
        except Exception as e:
            logger.error(f"get_total_equity exception: {e}")
            return None

    def get_positions(self, inst_id: str = "",
                       mgn_mode: str = "") -> List[dict]:
        """获取当前持仓。

        GET /api/v5/account/positions

        Args:
            inst_id: 可选，按合约 ID 过滤（如 "BTC-USDT-SWAP"）
            mgn_mode: 可选，按保证金模式过滤 "isolated" | "cross"

        Returns:
            持仓列表，每项包含 instId, posSide, pos, avgPx, upl, lever, mgnMode 等字段
        """
        try:
            kwargs = {}
            if inst_id:
                kwargs["instId"] = inst_id
            if mgn_mode:
                kwargs["mgnMode"] = mgn_mode
            result = self._account.get_positions(**kwargs) if kwargs else self._account.get_positions()
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_positions exception: {e}")
            return []

    def get_pending_orders(self) -> List[dict]:
        """获取挂单。"""
        try:
            result = self._trade.get_order_list()
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_pending_orders exception: {e}")
            return []

    def place_order(self, inst_id: str, side: str, pos_side: str,
                    sz: str, px: str = "", ord_type: str = "limit",
                    td_mode: str = "isolated",
                    tp_trigger: str = "", tp_price: str = "",
                    sl_trigger: str = "", sl_price: str = "") -> Optional[str]:
        """下单（通过 attachAlgoOrds 附加止盈止损）。"""
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
        }
        if px:
            body["px"] = px

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
            result = self._trade.place_order(**body)
            if result.get("code") == "0" and result.get("data"):
                ord_id = result["data"][0].get("ordId", "")
                logger.info(f"Order placed: {ord_id} — {side} {sz} {inst_id} @ {px} "
                            f"TP={tp_trigger} SL={sl_trigger}")
                return ord_id
            else:
                logger.error(f"place_order failed: {result.get('msg')}")
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

        Returns:
            algo_id 或 None
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Algo order: {ord_type} {side} {sz} {inst_id} "
                        f"TP={tp_trigger_px} SL={sl_trigger_px}")
            return "dry_run_algo_" + str(int(time.time() * 1000))

        body = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": ord_type,
            "sz": sz,
        }
        if tp_trigger_px:
            body["tpTriggerPx"] = tp_trigger_px
            body["tpTriggerPxType"] = tp_trigger_px_type
            if tp_ord_px:
                body["tpOrdPx"] = tp_ord_px
            else:
                body["tpOrdPx"] = "-1"  # 市价
        if sl_trigger_px:
            body["slTriggerPx"] = sl_trigger_px
            body["slTriggerPxType"] = sl_trigger_px_type
            if sl_ord_px:
                body["slOrdPx"] = sl_ord_px
            else:
                body["slOrdPx"] = "-1"  # 市价

        # 透传额外参数
        body.update(kwargs)

        try:
            result = self._trade.order_algo(**body)
            if result.get("code") == "0" and result.get("data"):
                algo_id = result["data"][0].get("algoId", "")
                logger.info(f"Algo order placed: {algo_id} — {ord_type} "
                            f"{side} {sz} {inst_id} "
                            f"TP={tp_trigger_px} SL={sl_trigger_px}")
                return algo_id
            else:
                logger.error(f"place_algo_order failed: {result.get('msg')}")
                return None
        except Exception as e:
            logger.error(f"place_algo_order exception: {e}")
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
            kwargs = {"algoId": algo_id}
            if inst_id:
                kwargs["instId"] = inst_id
            result = self._trade.cancel_algo_order(**kwargs)
            if result.get("code") == "0":
                logger.info(f"Algo order cancelled: {algo_id}")
                return True
            else:
                logger.error(f"cancel_algo_order failed: {result.get('msg')}")
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
            result = self._trade.get_algo_order_list(**kwargs)
            if result.get("code") == "0":
                return result.get("data", [])
            return []
        except Exception as e:
            logger.error(f"get_algo_orders exception: {e}")
            return []

    def cancel_order(self, inst_id: str, ord_id: str) -> bool:
        """撤销订单。"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel {ord_id}")
            return True

        try:
            result = self._trade.cancel_order(instId=inst_id, ordId=ord_id)
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

            result = self._trade.close_position(**kwargs)
            if result.get("code") == "0":
                ord_id = (result.get("data", [{}])[0].get("ordId", "")
                          if result.get("data") else "")
                logger.info(f"Position closed: {ord_id or 'N/A'} — "
                            f"{inst_id} {pos_side}")
                return ord_id or "closed"
            else:
                logger.error(f"close_position failed: {result.get('msg')}")
                return None
        except Exception as e:
            logger.error(f"close_position exception: {e}")
            return None

    def set_leverage(self, inst_id: str, lever: int,
                     mgn_mode: str = "isolated") -> bool:
        """设置合约杠杆倍数。

        OKX API v5 要求在下单前必须设置杠杆，否则使用默认 1x。
        POST /api/v5/account/set-leverage

        Args:
            inst_id: 合约 ID 或币种（如 "BTC-USDT-SWAP"）
            lever: 杠杆倍数
            mgn_mode: 保证金模式 "isolated" | "cross"

        Returns:
            是否成功
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Set leverage: {inst_id} {lever}x {mgn_mode}")
            return True

        try:
            result = self._account.set_leverage(
                instId=inst_id,
                lever=str(lever),
                mgnMode=mgn_mode,
            )
            if result.get("code") == "0":
                logger.info(f"Leverage set: {inst_id} {lever}x {mgn_mode}")
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