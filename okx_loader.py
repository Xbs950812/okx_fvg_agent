"""
OKX 历史 K 线加载器 — 借鉴 Vibe-Trading (23.6k⭐) 的健壮实现。

相比原版 get_candles 的改进：
  - 双端点回退: history-candles (深度历史) → candles (近期数据)
  - 分页遍历: 支持多年历史数据，自动翻页直到覆盖完整时间范围
  - 限速重试: retry_with_budget + check_budget 机制，处理 429/5xx 和业务错误
  - 代理支持: 环境变量 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
  - OHLC 验证: validate_ohlc 确保数据质量
  - 本地缓存: 可选的 parquet 缓存，减少重复 API 调用

HunHeng_OS_V1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BASE_URL = "https://www.okx.com/api/v5"
CANDLES_PATH = f"{BASE_URL}/market/candles"
HISTORY_CANDLES_PATH = f"{BASE_URL}/market/history-candles"
_MAX_PER_PAGE = 300
_RECENT_ONLY_DAYS = 400

# 从环境变量读取超时和预算参数
try:
    _OKX_TIMEOUT = int(os.getenv("OKX_TIMEOUT_S", "20"))
except (TypeError, ValueError):
    _OKX_TIMEOUT = 20
try:
    _OKX_FETCH_BUDGET_S = float(os.getenv("OKX_FETCH_BUDGET_S", "90.0"))
except (TypeError, ValueError):
    _OKX_FETCH_BUDGET_S = 90.0
try:
    _OKX_PROBE_TIMEOUT = int(os.getenv("OKX_PROBE_TIMEOUT_S", "8"))
except (TypeError, ValueError):
    _OKX_PROBE_TIMEOUT = 8

# 默认退避策略
DEFAULT_BACKOFF: Tuple[float, ...] = (0.5, 1.5, 4.0)
DEFAULT_MAX_RETRIES = 3

# 缓存开关
_CACHE_ENABLED = os.getenv("OKX_CACHE_ENABLED", "").lower() in ("1", "true", "yes", "on")
_CACHE_ROOT = Path(os.getenv("OKX_CACHE_ROOT", str(Path.home() / ".okx_fvg" / "cache")))


# ---------------------------------------------------------------------------
# 代理配置
# ---------------------------------------------------------------------------

def _first_proxy_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _proxy_config(proxy_str: str = "") -> Dict[str, str]:
    """构建代理配置，优先使用传入的 proxy 字符串，其次使用环境变量。"""
    if proxy_str:
        return {"https": proxy_str, "http": proxy_str}

    all_proxy = _first_proxy_env("ALL_PROXY", "all_proxy")
    http_proxy = _first_proxy_env("HTTP_PROXY", "http_proxy") or all_proxy
    https_proxy = _first_proxy_env("HTTPS_PROXY", "https_proxy") or all_proxy or http_proxy
    proxies: Dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies


def _create_session(proxy_str: str = "") -> requests.Session:
    session = requests.Session()
    proxies = _proxy_config(proxy_str)
    if proxies:
        session.proxies.update(proxies)
    return session


# ---------------------------------------------------------------------------
# 重试 / 预算机制
# ---------------------------------------------------------------------------

def check_budget(deadline: float, label: str, budget_s: float | None = None) -> None:
    """超时检查：如果单调时钟已超过 deadline，抛出 TimeoutError。"""
    if time.monotonic() > deadline:
        suffix = f" exceeded {budget_s:.0f}s budget" if budget_s is not None else " exceeded budget"
        raise TimeoutError(f"{label}{suffix}")


def retry_with_budget(
    fn,
    *,
    transient: type[BaseException] | tuple[type[BaseException], ...],
    deadline: float,
    label: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: Tuple[float, ...] = DEFAULT_BACKOFF,
):
    """带预算的重试机制：在 deadline 内对 transient 异常重试。"""
    if len(backoff) < max_retries:
        raise ValueError(f"backoff has {len(backoff)} entries; need >= max_retries ({max_retries})")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except transient as exc:
            remaining = deadline - time.monotonic()
            if attempt == max_retries or remaining <= 0:
                raise TimeoutError(f"{label} failed after {attempt + 1} attempt(s): {exc}") from exc
            time.sleep(min(backoff[attempt], max(0.0, remaining)))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# OHLC 验证
# ---------------------------------------------------------------------------

def validate_ohlc(frame: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """验证 OHLC 数据质量：high < low、非正价格等。"""
    required = ("open", "high", "low", "close")
    if frame.empty or not all(col in frame.columns for col in required):
        return frame

    open_, high, low, close = (frame[c] for c in required)
    structural = (high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)
    nonpositive = (open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)
    invalid = structural | nonpositive
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return frame

    if strategy == "raise":
        raise ValueError(f"{n_invalid} bar(s) violate OHLC invariants")
    if strategy == "warn":
        logger.warning("OHLC validation: %d bar(s) violate invariants (kept)", n_invalid)
        return frame
    logger.warning("OHLC validation: dropping %d invalid bar(s)", n_invalid)
    return frame[~invalid]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _bar_to_ms(bar: str) -> int:
    """K 线周期转毫秒。"""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1H": 3_600_000,
        "2H": 7_200_000,
        "4H": 14_400_000,
        "6H": 21_600_000,
        "12H": 43_200_000,
        "1D": 86_400_000,
        "1W": 604_800_000,
        "1M": 2_592_000_000,
    }
    if bar in mapping:
        return mapping[bar]
    # 尝试解析数字+单位格式，如 "3D", "2W"
    import re
    m = re.match(r"^(\d+)([a-zA-Z]+)$", bar)
    if m:
        num = int(m.group(1))
        unit = m.group(2).lower()
        unit_ms = {
            "m": 60_000,
            "h": 3_600_000,
            "d": 86_400_000,
            "w": 604_800_000,
        }
        if unit in unit_ms:
            return num * unit_ms[unit]
    logger.warning("Unknown bar format '%s', falling back to 1H (3600000 ms)", bar)
    return 3_600_000


# ---------------------------------------------------------------------------
# 本地缓存
# ---------------------------------------------------------------------------

def _cache_key(
    inst_id: str,
    bar: str,
    start_ts: int,
    end_ts: int,
) -> str:
    """构建缓存键。"""
    bar_ms = _bar_to_ms(bar)
    aligned_end_ts = (end_ts // bar_ms) * bar_ms
    payload = json.dumps({
        "inst_id": inst_id,
        "bar": bar,
        "start_ts": start_ts,
        "end_ts": aligned_end_ts,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_path(inst_id: str, bar: str, start_ts: int, end_ts: int) -> Path:
    key = _cache_key(inst_id, bar, start_ts, end_ts)
    return _CACHE_ROOT / key[:2] / f"{key}.parquet"


def _cache_read(inst_id: str, bar: str, start_ts: int, end_ts: int) -> pd.DataFrame | None:
    if not _CACHE_ENABLED:
        return None
    path = _cache_path(inst_id, bar, start_ts, end_ts)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            logger.debug("Cache hit: %s %s (%d rows)", inst_id, bar, len(df))
            return df
    except Exception as e:
        logger.debug("Cache read failed: %s", e)
    return None


def _cache_write(df: pd.DataFrame, inst_id: str, bar: str, start_ts: int, end_ts: int):
    if not _CACHE_ENABLED:
        return
    path = _cache_path(inst_id, bar, start_ts, end_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
        logger.debug("Cache write: %s %s (%d rows)", inst_id, bar, len(df))
    except Exception as e:
        logger.debug("Cache write failed: %s", e)


# ---------------------------------------------------------------------------
# OKX 历史 K 线加载器
# ---------------------------------------------------------------------------

class OKXHistoryLoader:
    """OKX 历史 K 线加载器 — 健壮的 history-candles 端点 + 限速重试。

    用法:
        loader = OKXHistoryLoader(proxy="http://127.0.0.1:7890")
        df = loader.fetch("BTC-USDT-SWAP", bar="1H", limit=500)
        # 或按日期范围
        df = loader.fetch_range("BTC-USDT-SWAP", "2024-01-01", "2024-06-30", bar="4H")
    """

    def __init__(self, proxy: str = ""):
        self._session = _create_session(proxy)
        self._proxy = proxy

    def is_available(self) -> bool:
        """探测 OKX API 是否可用。"""
        try:
            resp = self._session.get(
                CANDLES_PATH,
                params={"instId": "BTC-USDT", "bar": "1D", "limit": "1"},
                timeout=_OKX_PROBE_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("OKX probe HTTP %s", resp.status_code)
                return False
            data = resp.json()
            return data.get("code") == "0" and bool(data.get("data"))
        except Exception as e:
            logger.warning("OKX probe failed: %s", e)
            return False

    def fetch(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 200,
        *,
        use_history: bool = True,
        keep_extra: bool = False,
    ) -> pd.DataFrame:
        """获取最近 limit 根 K 线。

        Args:
            inst_id: 合约 ID，如 "BTC-USDT-SWAP"
            bar: K 线周期，"1m"/"5m"/"15m"/"30m"/"1H"/"4H"/"1D"
            limit: 获取数量
            use_history: 是否优先使用 history-candles 端点

        Returns:
            OHLCV DataFrame，索引为时间戳
        """
        end_ts = int(time.time() * 1000)
        # 估算 start_ts：假设每根 K 线间隔
        bar_ms = _bar_to_ms(bar)
        start_ts = end_ts - limit * bar_ms

        return self.fetch_range(
            inst_id=inst_id,
            start_ts=start_ts,
            end_ts=end_ts,
            bar=bar,
            use_history=use_history,
            keep_extra=keep_extra,
        )

    def fetch_range(
        self,
        inst_id: str,
        start_ts: int,
        end_ts: int,
        bar: str = "1H",
        *,
        use_history: bool = True,
        keep_extra: bool = False,
    ) -> pd.DataFrame:
        """按时间范围获取 K 线。

        Args:
            inst_id: 合约 ID
            start_ts: 起始时间戳 (ms)
            end_ts: 结束时间戳 (ms)
            bar: K 线周期
            use_history: 是否优先使用 history-candles

        Returns:
            OHLCV DataFrame
        """
        # 尝试缓存（仅普通模式）
        if not keep_extra:
            cached = _cache_read(inst_id, bar, start_ts, end_ts)
            if cached is not None:
                return cached

        # 根据 bar 调整最大翻页数
        max_pages = self._max_pages_for_bar(bar)

        df = self._fetch_candles(
            inst_id=inst_id,
            start_ts=start_ts,
            end_ts=end_ts,
            bar=bar,
            max_pages=max_pages,
            use_history=use_history,
            keep_extra=keep_extra,
        )

        if df is not None and not df.empty:
            _cache_write(df, inst_id, bar, start_ts, end_ts)

        return df if df is not None else pd.DataFrame()

    def fetch_date_range(
        self,
        inst_id: str,
        start_date: str,
        end_date: str,
        bar: str = "1H",
        *,
        use_history: bool = True,
    ) -> pd.DataFrame:
        """按日期范围获取 K 线。

        Args:
            inst_id: 合约 ID
            start_date: 起始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            bar: K 线周期
            use_history: 是否优先使用 history-candles

        Returns:
            OHLCV DataFrame
        """
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int((pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp() * 1000)

        # 如果起始日期早于 400 天前，强制使用 history-candles
        try:
            start = pd.Timestamp(start_date)
            age_days = (pd.Timestamp.now(tz="UTC").tz_localize(None) - start).days
            use_history = use_history or (age_days > _RECENT_ONLY_DAYS)
        except Exception:
            pass

        return self.fetch_range(inst_id, start_ts, end_ts, bar, use_history=use_history)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _max_pages_for_bar(bar: str) -> int:
        """根据 K 线周期返回最大翻页数。

        修复 Bug 45: 1m/5m 的 200 页 × 退避重试时间可能远超 90s 预算，
        调低到 80 页以确保能在合理预算内完成。
        """
        if bar in ("1m", "5m"):
            return 80
        elif bar in ("15m", "30m"):
            return 60
        else:
            return 40

    def _fetch_candles(
        self,
        inst_id: str,
        start_ts: int,
        end_ts: int,
        bar: str = "1H",
        max_pages: int = 40,
        *,
        use_history: bool = True,
        keep_extra: bool = False,
    ) -> pd.DataFrame | None:
        """核心：双端点分页获取 K 线。"""
        endpoints: list[str] = (
            [HISTORY_CANDLES_PATH, CANDLES_PATH]
            if use_history
            else [CANDLES_PATH, HISTORY_CANDLES_PATH]
        )

        last_error: Exception | None = None
        for endpoint in endpoints:
            try:
                df = self._paginate(
                    endpoint=endpoint,
                    inst_id=inst_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    bar=bar,
                    max_pages=max_pages,
                    keep_extra=keep_extra,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OKX %s failed for %s: %s — trying next endpoint",
                    endpoint.rsplit("/", 1)[-1],
                    inst_id,
                    exc,
                )

        if last_error is not None:
            logger.warning("OKX empty/failed for %s: %s", inst_id, last_error)
        return None

    def _paginate(
        self,
        endpoint: str,
        inst_id: str,
        start_ts: int,
        end_ts: int,
        bar: str,
        max_pages: int,
        *,
        keep_extra: bool = False,
    ) -> pd.DataFrame | None:
        """分页遍历获取全量 K 线数据。"""
        all_rows: list = []
        after = str(end_ts)
        deadline = time.monotonic() + _OKX_FETCH_BUDGET_S
        label = f"OKX fetch for {inst_id} via {endpoint.rsplit('/', 1)[-1]}"

        for _ in range(max_pages):
            check_budget(deadline, label, budget_s=_OKX_FETCH_BUDGET_S)

            params = {
                "instId": inst_id,
                "bar": bar,
                "limit": str(_MAX_PER_PAGE),
                "after": after,
            }

            def _do_request():
                resp = self._session.get(endpoint, params=params, timeout=_OKX_TIMEOUT)
                # 瞬态错误 → 抛出供 retry_with_budget 重试
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"OKX HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise requests.RequestException(
                        f"OKX non-JSON response HTTP {resp.status_code}"
                    ) from exc
                code = str(data.get("code", ""))
                if code != "0":
                    msg = data.get("msg") or data.get("error_message") or code
                    raise requests.RequestException(f"OKX API code={code} msg={msg}")
                return data

            data = retry_with_budget(
                _do_request,
                transient=(requests.RequestException, TimeoutError),
                deadline=deadline,
                label=label,
            )

            raw_rows = data.get("data") or []
            if not raw_rows:
                break

            # 只保留已确认的 K 线 (confirm == "1")
            confirmed = [r for r in raw_rows if len(r) > 8 and str(r[8]) == "1"]
            if confirmed:
                rows = confirmed
            else:
                logger.warning("No confirmed candles in response for %s, falling back to unconfirmed data", inst_id)
                rows = list(raw_rows)
            all_rows.extend(rows)

            try:
                oldest_ts = int(rows[-1][0])
            except (TypeError, ValueError, IndexError):
                logger.warning("Paginate: invalid timestamp in rows[-1][0], breaking")
                break
            if oldest_ts <= start_ts or len(raw_rows) < _MAX_PER_PAGE:
                break
            new_after = str(oldest_ts)
            if new_after == after:
                logger.warning("Paginate: oldest_ts unchanged at %s, breaking to avoid infinite loop", new_after)
                break
            after = new_after

        if not all_rows:
            return None

        # 构建 DataFrame
        columns = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
        normalized = []
        for r in all_rows:
            row = list(r) + [""] * (len(columns) - len(r))
            normalized.append(row[:len(columns)])

        df = pd.DataFrame(normalized, columns=columns)
        df["trade_date"] = pd.to_datetime(
            pd.to_numeric(df["ts"], errors="coerce"),
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)

        df = df.dropna(subset=["trade_date"]).set_index("trade_date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # 时间范围裁剪
        start_dt = pd.Timestamp(start_ts, unit="ms")
        end_dt = pd.Timestamp(end_ts, unit="ms")
        df = df[(df.index >= start_dt) & (df.index < end_dt)]

        if keep_extra:
            extra_cols = ["volCcy", "volCcyQuote", "confirm"]
            for c in extra_cols:
                if c in df.columns:
                    df[c] = df[c].fillna("0")
            base_cols = ["open", "high", "low", "close", "volume"]
            keep_cols = base_cols + [c for c in extra_cols if c in df.columns]
            df = df[keep_cols].dropna(subset=["open", "high", "low", "close"])
        else:
            df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])

        # OHLC 验证
        df = validate_ohlc(df, strategy="drop")

        return df if not df.empty else None


# ---------------------------------------------------------------------------
# 便捷函数：与现有 okx_client 接口兼容
# ---------------------------------------------------------------------------

def fetch_candles_enhanced(
    inst_id: str,
    bar: str = "1H",
    limit: int = 200,
    proxy: str = "",
    use_history: bool = True,
) -> List[list]:
    """增强版 K 线获取，返回与现有 get_candles 兼容的格式。

    返回格式: [[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm], ...]
    时间倒序（最新在前），与 OKX SDK 保持一致。

    Args:
        inst_id: 合约 ID
        bar: K 线周期
        limit: 获取数量
        proxy: 代理地址
        use_history: 是否使用 history-candles 端点

    Returns:
        OKX 原始格式的 K 线列表
    """
    loader = OKXHistoryLoader(proxy=proxy)
    df = loader.fetch(inst_id=inst_id, bar=bar, limit=limit, use_history=use_history, keep_extra=True)

    if df.empty:
        return []

    # 转换为 OKX 原始格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    # 时间倒序
    df = df.sort_index(ascending=False)
    rows = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp() * 1000)
        rows.append([
            str(ts),
            str(row["open"]),
            str(row["high"]),
            str(row["low"]),
            str(row["close"]),
            str(row["volume"]),
            str(row.get("volCcy", "0")),
            str(row.get("volCcyQuote", "0")),
            str(row.get("confirm", "1")),
        ])

    return rows[:limit]