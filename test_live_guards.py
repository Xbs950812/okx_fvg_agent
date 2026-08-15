# -*- coding: utf-8 -*-
"""
实盘守卫生效性测试 — 限流令牌桶 + 订单簿流动性检查 (2026-08-15)。

模拟实盘极端行情环境（薄订单簿山寨币 + 高频请求），验证两项新守卫：

  A. 全局 API 限流令牌桶 (okx_client._TokenBucket / _GLOBAL_RATE_LIMITER)
     - 突发容量内即时放行、超出后按 QPS 节流（计时验证）
     - _call_sdk_retry 每次尝试前取令牌（含重试）
     - OKXClient 实盘模式(dry_run=false)初始化全局桶；纸面不初始化
  B. 订单簿深度检查 (okx_client.get_order_book_depth_usd + executor.check_order_book_liquidity)
     - USDT 名义深度计算: Σ px×sz×ctVal（OKX orderbook sz 是张数）
     - 极端薄书: 名义/对侧深度 > 5% → 拒绝；深书放行
     - 方向语义: 平多吃 bids / 平空吃 asks
     - 查询失败 fail-open（增强门不是核心闸）
     - execute_signal 接线: 薄书在 set_leverage 之前被拦截

注意: 行情类直连 SDK 调用（get_order_book 等）不经过 _call_sdk_retry，
不受令牌桶约束（由 coin_tracker 批间隔节流）— 本测试固化该边界。

运行: python -m pytest test_live_guards.py -v
"""

import time
from types import SimpleNamespace

import pytest

import okx_client
from okx_client import OKXClient, _call_sdk_retry, _TokenBucket
from executor import check_order_book_liquidity, execute_signal


@pytest.fixture(autouse=True)
def _reset_global_limiter():
    """每个测试后恢复全局令牌桶为未启用，防止测试间污染。"""
    yield
    okx_client._GLOBAL_RATE_LIMITER = None


def thin_book(bids_usd_target, asks_usd_target, px=100.0, ct_val=0.01):
    """构造名义深度恰好等于目标的订单簿（每档 px×sz×ctVal 均摊）。"""
    def rows(total_usd, n_levels=10):
        per = total_usd / n_levels
        return [[f"{px + i * 0.1:.1f}", f"{per / ((px + i * 0.1) * ct_val):.0f}",
                 "0", "1"] for i in range(n_levels)]
    return {"bids": rows(bids_usd_target), "asks": rows(asks_usd_target)}


# ===========================================================================
# A. 限流令牌桶
# ===========================================================================

class TestTokenBucket:

    def test_burst_capacity_instant(self):
        """容量内突发即时放行（不阻塞）。"""
        tb = _TokenBucket(rate=5.0, capacity=5.0)
        t0 = time.perf_counter()
        for _ in range(5):
            tb.acquire()
        assert time.perf_counter() - t0 < 0.05, "容量内获取不应阻塞"

    def test_throttle_to_rate(self):
        """超出容量后按 QPS 节流: rate=20/s, cap=2, 共取 6 个 →
        前 2 个免费, 后 4 个需 4/20 = 0.2s（断言 ≥0.15s 留余量）。"""
        tb = _TokenBucket(rate=20.0, capacity=2.0)
        t0 = time.perf_counter()
        for _ in range(6):
            tb.acquire()
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.15, f"6 次获取应至少耗时 0.2s, 实际 {elapsed:.3f}s"
        assert elapsed < 3.0, "节流不应卡死"

    def test_invalid_params_floored(self):
        """非法参数兜底: rate<0.1 → 0.1, capacity<1 → 1（构造不炸）。"""
        tb = _TokenBucket(rate=-5, capacity=0)
        assert tb.rate == pytest.approx(0.1)
        assert tb.capacity == pytest.approx(1.0)
        tb.acquire()  # 不抛异常


class TestRateLimiterWiring:

    def test_call_sdk_retry_acquires_per_attempt(self):
        """_call_sdk_retry 每次尝试前取一个令牌（含限流重试的每次）。

        计数用包装 acquire 实现 — 退避睡眠(0.5/1.5s)期间令牌会回填，
        期末消耗差值不可测。
        """
        tb = _TokenBucket(rate=1000.0, capacity=1000.0)
        acquire_count = [0]
        _orig = tb.acquire

        def _counting_acquire():
            acquire_count[0] += 1
            _orig()

        tb.acquire = _counting_acquire
        okx_client._GLOBAL_RATE_LIMITER = tb
        calls = {"n": 0}

        def flaky_fn():
            calls["n"] += 1
            return {"code": "50011", "msg": "rate limit"}  # 可重试码

        _call_sdk_retry(flaky_fn, retries=3)
        assert calls["n"] == 3
        assert acquire_count[0] == 3, \
            f"令牌获取次数应 == 尝试次数(每次尝试前取令牌), 实际 {acquire_count[0]}"

    def test_throttled_call_sdk_retry(self):
        """端到端节流: 低 QPS 桶下连续调用耗时符合 QPS 约束。"""
        okx_client._GLOBAL_RATE_LIMITER = _TokenBucket(rate=25.0, capacity=2.0)
        ok_fn = lambda: {"code": "0", "data": []}  # noqa: E731
        t0 = time.perf_counter()
        for _ in range(5):
            _call_sdk_retry(ok_fn)
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.10, f"5 次调用(容量2)应 ≥0.12s, 实际 {elapsed:.3f}s"

    def test_client_init_live_mode_enables_bucket(self):
        """实盘模式(dry_run=false)按配置初始化全局桶。"""
        cfg = {
            "agent": {"dry_run": False},
            "okx": {"rate_limit": {"enabled": True, "max_qps": 7, "burst_capacity": 11}},
        }
        OKXClient(cfg)
        lim = okx_client._GLOBAL_RATE_LIMITER
        assert lim is not None, "实盘模式应初始化全局令牌桶"
        assert lim.rate == pytest.approx(7.0)
        assert lim.capacity == pytest.approx(11.0)

    def test_client_init_dry_run_no_bucket(self):
        """纸面模式(dry_run=true)不初始化（行情端点已被批间隔节流）。"""
        cfg = {
            "agent": {"dry_run": True},
            "okx": {"rate_limit": {"enabled": True, "max_qps": 7, "burst_capacity": 11}},
        }
        OKXClient(cfg)
        assert okx_client._GLOBAL_RATE_LIMITER is None

    def test_client_init_disabled_no_bucket(self):
        """rate_limit.enabled=false 显式关闭。"""
        cfg = {
            "agent": {"dry_run": False},
            "okx": {"rate_limit": {"enabled": False}},
        }
        OKXClient(cfg)
        assert okx_client._GLOBAL_RATE_LIMITER is None


# ===========================================================================
# B. 订单簿深度检查
# ===========================================================================

class FakeDepthClient:
    """只实现 get_order_book_depth_usd 的假客户端（可注入异常）。"""

    def __init__(self, result=None, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc
        self.calls = []

    def get_order_book_depth_usd(self, inst_id, levels=10, ct_val=0.01):
        self.calls.append((inst_id, levels, ct_val))
        if self.raise_exc:
            raise self.raise_exc
        return self.result


class TestGetOrderBookDepthUsd:

    def _client(self):
        return OKXClient({"agent": {"dry_run": True}, "okx": {}})

    def test_depth_computation_usd(self):
        """深度 = Σ px×sz×ctVal（sz 是张数，非币数）。"""
        c = self._client()
        c.get_order_book = lambda inst_id, sz=20: {
            "bids": [["100.0", "50", "0", "1"], ["101.0", "25", "0", "1"]],
            "asks": [["102.0", "40", "0", "1"]],
        }
        d = c.get_order_book_depth_usd("X-USDT-SWAP", levels=5, ct_val=0.01)
        # bids: 100×50×0.01 + 101×25×0.01 = 50 + 25.25 = 75.25
        assert d["bids_usd"] == pytest.approx(75.25, abs=1e-9)
        assert d["asks_usd"] == pytest.approx(102.0 * 40 * 0.01, abs=1e-9)

    def test_malformed_rows_skipped(self):
        """坏行（空串/非数字/零价/负量）跳过不炸。ct_val=1.0。"""
        c = self._client()
        c.get_order_book = lambda inst_id, sz=20: {
            "bids": [["", "50"], ["abc", "1"], ["0", "99"], ["100.0", "10"]],
            "asks": [["100.0", "-5"], ["100.0", "10"]],
        }
        d = c.get_order_book_depth_usd("X-USDT-SWAP", levels=5, ct_val=1.0)
        assert d["bids_usd"] == pytest.approx(100.0 * 10 * 1.0)   # 仅最后一行有效
        assert d["asks_usd"] == pytest.approx(1000.0)

    def test_none_book_returns_none(self):
        c = self._client()
        c.get_order_book = lambda inst_id, sz=20: None
        assert c.get_order_book_depth_usd("X-USDT-SWAP") is None

    def test_all_zero_depth_returns_none(self):
        c = self._client()
        c.get_order_book = lambda inst_id, sz=20: {"bids": [], "asks": []}
        assert c.get_order_book_depth_usd("X-USDT-SWAP") is None


class TestCheckOrderBookLiquidity:

    CFG = {"risk": {"order_book_depth": {
        "enabled": True, "max_notional_depth_ratio": 0.05, "depth_levels": 10}}}

    def test_extreme_thin_book_rejected(self):
        """极端行情: 山寨币崩盘薄书 — 名义 8400 / 深度 100000 = 8.4% > 5% 拒绝。"""
        client = FakeDepthClient(result={"bids_usd": 100_000.0, "asks_usd": 100_000.0})
        ok, reason = check_order_book_liquidity(
            client, "CRASH-USDT-SWAP", "long", 8400.0, 0.01, self.CFG)
        assert not ok
        assert "8.4%" in reason or "拒绝" in reason
        assert client.calls == [("CRASH-USDT-SWAP", 10, 0.01)]

    def test_deep_book_passes(self):
        """主流币深书: 8400 / 500000 = 1.7% ≤ 5% 放行。"""
        client = FakeDepthClient(result={"bids_usd": 500_000.0, "asks_usd": 500_000.0})
        ok, _ = check_order_book_liquidity(
            client, "BTC-USDT-SWAP", "long", 8400.0, 0.01, self.CFG)
        assert ok

    def test_side_semantics_long_bids_short_asks(self):
        """方向语义: 平多吃 bids、平空吃 asks — 只有所查侧决定结果。"""
        client = FakeDepthClient(result={"bids_usd": 1_000_000.0, "asks_usd": 50_000.0})
        ok_long, _ = check_order_book_liquidity(
            client, "X", "long", 8400.0, 0.01, self.CFG)
        assert ok_long, "多单查 bids(深)应放行"
        ok_short, reason = check_order_book_liquidity(
            client, "X", "short", 8400.0, 0.01, self.CFG)
        assert not ok_short, "空单查 asks(薄, 16.8%)应拒绝"

    def test_query_failure_fail_open(self):
        """查询抛异常 → fail-open 放行（增强门不阻塞）。"""
        client = FakeDepthClient(raise_exc=ConnectionError("proxy down"))
        ok, _ = check_order_book_liquidity(
            client, "X", "long", 8400.0, 0.01, self.CFG)
        assert ok

    def test_none_depth_fail_open(self):
        client = FakeDepthClient(result=None)
        ok, _ = check_order_book_liquidity(client, "X", "long", 8400.0, 0.01, self.CFG)
        assert ok

    def test_disabled_config_passes(self):
        client = FakeDepthClient(result={"bids_usd": 1.0, "asks_usd": 1.0})
        cfg = {"risk": {"order_book_depth": {"enabled": False}}}
        ok, _ = check_order_book_liquidity(client, "X", "long", 8400.0, 0.01, cfg)
        assert ok
        assert client.calls == [], "关闭时不应发起查询"

    def test_boundary_ratio_exactly_threshold_passes(self):
        """ratio == 阈值(严格 > 判定)放行: 8400/168000 = 5.0%。"""
        client = FakeDepthClient(result={"bids_usd": 168_000.0, "asks_usd": 168_000.0})
        ok, _ = check_order_book_liquidity(
            client, "X", "long", 8400.0, 0.01, self.CFG)
        assert ok


# ===========================================================================
# execute_signal 接线: 薄书在 set_leverage 之前被拦截
# ===========================================================================

class FakeLiveClient:
    """execute_signal 依赖面最小实现 + 调用记录。

    TP/SL 挂单段及之后的核验用 __getattr__ 兜底 stub（记录调用并返回
    truthy 值），保证流经闸门后能走完主路径。
    """

    def __init__(self, bids_usd, asks_usd):
        self.depth = {"bids_usd": bids_usd, "asks_usd": asks_usd}
        self.set_leverage_calls = []
        self.place_order_calls = []
        self.other_calls = []

    def get_position_tiers(self, inst_id):
        return {"maxLever": 50, "mmr": 0.005}

    def get_order_book_depth_usd(self, inst_id, levels=10, ct_val=0.01):
        return self.depth

    def set_leverage(self, inst_id, lever, mgn_mode, pos_side=None):
        self.set_leverage_calls.append((inst_id, lever, mgn_mode, pos_side))
        return True

    def place_order(self, **kwargs):
        self.place_order_calls.append(kwargs)
        return "ord123"

    def place_algo_order(self, **kwargs):
        self.other_calls.append(("place_algo_order", kwargs))
        return "algo123"

    def get_algo_order_details(self, algo_id):
        self.other_calls.append(("get_algo_order_details", algo_id))
        return {"algoId": algo_id, "state": "live",
                "tpTriggerPx": "110.0", "slTriggerPx": "98.0"}

    def __getattr__(self, name):
        """未显式实现的方法: 记录并返回 truthy stub（流经闸门后的次要路径）。"""
        def _stub(*args, **kwargs):
            self.other_calls.append((name, args, kwargs))
            return "stub_ok"
        return _stub


LIVE_RISK_CFG = {
    "risk_per_trade_pct": 30.0,
    "position_sizing": "margin",
    "margin_pct": 30,
    "max_leverage": 10,
    "max_position_leverage": 0,
    "margin_mode": "isolated",
    "liq_check_fail_closed": False,
    "default_mmr": 0.005,
    "liq_safety_factor": 0.5,
    "enforce_risk_cap": True,
    "order_book_depth": {"enabled": True,
                         "max_notional_depth_ratio": 0.05,
                         "depth_levels": 10},
}


def _live_signal():
    # entry=100, sl=98.5 → 止损距离 1.5%; 满杠杆50 → LiqCheck 降杠杆至 28
    # (爆仓距离 0.0307×0.5=0.0154 > 0.015 覆盖止损), 名义 ≈ 300×28 = 8400 USDT
    return SimpleNamespace(
        inst_id="CRASH-USDT-SWAP", position_side="long",
        entry_price=100.0, stop_loss=98.5, take_profit=110.0,
        leverage=50, score=0.8, reason="test",
        signal_id="lg1", entry_trigger_px=0.0,
    )


def _instrument_info():
    return {"ctVal": "0.01", "minSz": "1", "lotSz": "1", "tickSz": "0.1"}


class TestExecuteSignalLiquidityGate:

    def test_thin_book_blocks_before_leverage(self):
        """极端薄书(100k 深度 vs 8400 名义 = 8.4%): 在 set_leverage 前拦截。"""
        client = FakeLiveClient(bids_usd=100_000.0, asks_usd=100_000.0)
        cfg = {"risk": dict(LIVE_RISK_CFG)}
        ord_id = execute_signal(client, _live_signal(), 1000.0, cfg,
                                _instrument_info())
        assert ord_id is None, "薄书应拒绝开仓"
        assert client.set_leverage_calls == [], "流动性拦截必须发生在设杠杆之前"
        assert client.place_order_calls == [], "不得有任何下单"

    def test_deep_book_proceeds_past_gate(self):
        """深书(2M): 通过流动性闸门 → 进入设杠杆（证明闸门放行且不误杀）。"""
        client = FakeLiveClient(bids_usd=2_000_000.0, asks_usd=2_000_000.0)
        cfg = {"risk": dict(LIVE_RISK_CFG)}
        sig = _live_signal()
        execute_signal(client, sig, 1000.0, cfg, _instrument_info())
        assert len(client.set_leverage_calls) == 1, \
            "深书应通过流动性检查并推进到设杠杆"
        # LiqCheck 降杠杆语义同时被验证: 50 → 28 (止损先于爆仓)
        assert client.set_leverage_calls[0][1] == 28
        # 名义 ≈ 8400, 对侧深度 2M → ratio 0.42% ≤ 5%
        assert client.place_order_calls, "通过闸门后应实际下单"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
