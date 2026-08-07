"""
WebSocket Ticker Cache — 通过 OKX WebSocket 公共频道实时缓存行情。
替代 REST get_tickers() 轮询，延迟从 200-500ms → < 50ms。
"""
import json
import logging
import threading
import copy
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import websocket
except ImportError:
    websocket = None
    logger.warning("websocket-client not installed; WebSocket ticker cache disabled")

# OKX 建议每 30 秒内发送一次应用层心跳。实测 (2026-08-07) 心跳格式必须是
# 纯文本 "ping"（期望服务器回 "pong"）；JSON {"op":"ping"} 会被服务器
# 以 60012 "Illegal request" 拒绝。无心跳时服务器约 30s 强制断开，
# 曾导致每 ~30s 断线重连一次。取 25s 留出余量避免边界触发断线。
PING_INTERVAL_SEC = 25.0
# 单连接单次 subscribe 的频道数上限（OKX 公共频道限制，超出报错）
MAX_SUBSCRIBE_ARGS = 100


class WsTickerCache:
    """OKX WebSocket Ticker 实时缓存。
    
    订阅 SWAP 全量 ticker，维护本地内存缓存。
    主循环从缓存读取，无需 REST 轮询。
    """
    
    def __init__(self, proxy: Optional[str] = None, inst_provider: Optional[callable] = None):
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self.proxy = proxy
        # 修复: OKX V5 tickers 频道不支持 instType 模糊订阅 (实测返回 60018
        # "Wrong URL or channel:tickers,instType:SWAP doesn't exist")。
        # 必须按 instId 逐币订阅。inst_provider 回调返回要订阅的 instId 列表
        # (通常来自 REST get_tickers 的 Top N 币种)。
        self._inst_provider = inst_provider
        self._connected = threading.Event()
        self._last_update: float = 0.0
        # 修复: 上次心跳发送时间戳，用于按 PING_INTERVAL_SEC 周期发 ping
        self._last_ping: float = 0.0
    
    def start(self):
        """启动 WebSocket 连接（后台线程）。"""
        if websocket is None:
            logger.warning("websocket-client not installed; WebSocket ticker cache disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="WsTicker")
        self._thread.start()
        logger.info("WebSocket ticker cache starting...")
    
    def _run(self):
        """WebSocket 主循环 — 连接 + 订阅 + 心跳 + 解析 + 重连。"""
        if websocket is None:
            logger.warning("websocket-client not available; WS ticker cache thread exit")
            return
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self._running:
            try:
                self._ws = websocket.create_connection(
                    url,
                    timeout=10,
                    **self._get_proxy_config(),
                )
                # 修复: tickers 频道必须按 instId 订阅 (instType 模糊订阅实测报 60018)。
                # 从 inst_provider 获取 Top N 币种列表逐币订阅，订阅成功后缓存即有数据
                inst_ids = []
                if self._inst_provider is not None:
                    try:
                        inst_ids = list(self._inst_provider())
                    except Exception as e:
                        logger.warning(f"inst_provider 获取订阅列表失败: {e}")
                if inst_ids:
                    # 单连接频道数上限内订阅 (top 100)
                    args = [
                        {"channel": "tickers", "instId": i}
                        for i in inst_ids[:MAX_SUBSCRIBE_ARGS]
                    ]
                    sub_msg = json.dumps({"op": "subscribe", "args": args})
                    self._ws.send(sub_msg)
                    logger.info(f"WebSocket ticker subscribed: {len(args)} insts")
                    # 修复: 仅在真正发送订阅后标记已连接；
                    # 订阅错误事件/断连时会被清除，避免误报"连接正常"
                    self._connected.set()
                else:
                    self._connected.clear()
                    logger.warning("WebSocket ticker 无可用 instId，订阅跳过 (主循环将回退 REST)")
                logger.info("WebSocket ticker connected")
                self._last_ping = time.time()
                
                while self._running:
                    try:
                        raw = self._ws.recv()
                        # 修复: 服务端主动关闭时 recv 返回 None/空串，
                        # 直接判定断开退出内层循环，避免 json.loads(None)
                        # 抛错被吞后空转一个超时周期
                        if not raw:
                            logger.warning("WebSocket closed by remote (empty recv)")
                            break
                        self._on_message(raw)
                    except websocket.WebSocketTimeoutException:
                        # 空闲超时 — 正常，进入心跳检查
                        pass
                    except Exception as e:
                        if self._running:
                            logger.warning(f"WS recv error: {e}")
                        break
                    
                    # 修复: OKX 心跳保活。格式为纯文本 "ping"（服务器回 "pong"），
                    # 不是 JSON {"op":"ping"}（实测报 60012 Illegal request）。
                    # 无心跳时服务器约 30s 强制断开，曾导致每 ~30s 断线重连一次
                    now = time.time()
                    if now - self._last_ping >= PING_INTERVAL_SEC:
                        try:
                            self._ws.send("ping")
                            self._last_ping = now
                        except Exception as e:
                            logger.warning(f"WS ping failed: {e}")
                            break
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}, retrying in 5s...")
                self._connected.clear()
            
            if self._running:
                time.sleep(5)
    
    def _get_proxy_config(self):
        """返回 websocket-client 兼容的代理配置。"""
        if not self.proxy:
            return {}
        from urllib.parse import urlparse
        parsed = urlparse(self.proxy if "://" in self.proxy else f"http://{self.proxy}")
        config = {
            "http_proxy_host": parsed.hostname,
            "http_proxy_port": parsed.port or (443 if parsed.scheme == "https" else 80),
        }
        if parsed.scheme in ("socks5", "socks5h", "socks4"):
            config["proxy_type"] = parsed.scheme
        return config
    
    @staticmethod
    def _safe_float(val, default=0.0):
        """安全转换 float，处理空字符串和 None。"""
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _on_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            # 修复: OKX 心跳应答为纯文本 "pong"（非 JSON），属预期控制消息，
            # 直接忽略，避免每次心跳都记录解析错误
            logger.debug(f"WS non-JSON control message: {raw[:64]!r}")
            return
        try:
            # 修复: 解析事件消息 — 订阅确认/错误。此前 {"event":"error"} (如 60018)
            # 被静默忽略，订阅失败无法诊断；错误时清除 connected 标志。
            # 仅当错误携带 arg (订阅上下文) 时才视为连接异常清除 connected；
            # 无 arg 的错误 (如心跳格式被拒 60012) 属于良性，不影响已建立的连接
            if "event" in msg and "data" not in msg:
                if msg.get("event") == "error":
                    logger.error(
                        f"WS subscribe error: code={msg.get('code')} "
                        f"msg={msg.get('msg')} arg={msg.get('arg')}"
                    )
                    if msg.get("arg"):
                        self._connected.clear()
                return
            data = msg.get("data")
            if not data:
                return
            with self._lock:
                for item in data:
                    inst_id = item.get("instId", "")
                    if inst_id:
                        # 解析关键字段
                        self._cache[inst_id] = {
                            "instId": inst_id,
                            "last": self._safe_float(item.get("last")),
                            "bidPx": self._safe_float(item.get("bidPx")),
                            "askPx": self._safe_float(item.get("askPx")),
                            # 修复: 补充 open24h 字段。MarketGuard 计算 BTC 24h 收益
                            # 依赖它，缺失导致 WS 路径下收益恒为 0，熔断评估失真
                            "open24h": self._safe_float(item.get("open24h")),
                            "high24h": self._safe_float(item.get("high24h")),
                            "low24h": self._safe_float(item.get("low24h")),
                            "vol24h": self._safe_float(item.get("vol24h")),
                            "ts": int(item.get("ts") or 0),
                        }
                # 修复: 仅在收到真实行情数据时刷新新鲜度时间戳，
                # 空 data / 事件消息不计入，避免 is_fresh 误判
                self._last_update = time.time()
        except Exception as e:
            logger.debug(f"WS parse error: {e}")
    
    def get(self, inst_id: str, max_age_sec: Optional[float] = None) -> Optional[dict]:
        """获取单个币种 ticker。
        
        Args:
            inst_id: 合约 ID。
            max_age_sec: 可选。超过该秒数未更新视为过期返回 None（基于交易所 ts）。
                         为 None 时始终返回缓存条目（保持原语义）。
        """
        with self._lock:
            entry = self._cache.get(inst_id)
            if entry is None:
                return None
            if max_age_sec is not None:
                ts = entry.get("ts", 0)
                # 修复: 币种停牌/退市后 last 永久残留；按交易所时间戳判断过期
                if ts and (time.time() * 1000 - ts) > max_age_sec * 1000:
                    return None
            # 修复: 返回浅拷贝，防止调用方修改内部缓存条目污染共享状态
            return dict(entry)
    
    def get_all(self) -> Dict[str, dict]:
        """获取所有 ticker 快照。"""
        with self._lock:
            return copy.deepcopy(self._cache)
    
    def get_top_by_volume(self, n: int = 100) -> list:
        """获取成交量前 N 的合约。"""
        with self._lock:
            sorted_coins = sorted(
                self._cache.values(),
                key=lambda x: x.get("vol24h", 0),
                reverse=True,
            )
            return [dict(c) for c in sorted_coins[:n]]
    
    def is_fresh(self, max_age_sec: float = 5.0) -> bool:
        """检查缓存是否新鲜。"""
        with self._lock:
            last = self._last_update
        return (time.time() - last) < max_age_sec
    
    @property
    def connected(self) -> bool:
        return self._connected.is_set()
    
    def stop(self):
        """停止 WebSocket 连接。"""
        self._running = False
        self._connected.clear()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("WebSocket ticker cache stopped")
