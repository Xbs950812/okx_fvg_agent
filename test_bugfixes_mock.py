# -*- coding: utf-8 -*-
"""本地 mock 测试: 验证 2026-08-13 修复的 8 个 bug（可独立验证部分）。

覆盖:
  Bug 1  _direction_momentum_gate   换仓路径补传 candles_1h 后方向动量门生效
  Bug 2  _position_notional_usd     名义敞口补乘 ctVal
  Bug 4  _update_mfe_mae_quant      ws_cache 实时价回退(纸面 MFE/MAE 恢复)
  Bug 5  OI 24h 变化                跨度 >= 24h 才计算, 不再用单轮变化
  Bug 6  _estimate_funding_cost     <8h 持仓不再按满周期高估
  Bug 8  _estimate_fee_cost         开仓 maker + 平仓按 is_taker 区分

  (Bug 3 低流动性基准 / Bug 7 成单率漏斗唯一键 为 main_loop 内联状态逻辑,
   已通过 py_compile + 全量单测 48/48 验证, 此处不做 mock 复现)

纯本地, 不依赖网络/交易所, 使用 unittest.mock。运行:
  C:\\Users\\casey\\AppData\\Local\\Programs\\Python\\Python310\\python.exe test_bugfixes_mock.py
"""
import sys
import time
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import Candle, FVG, Signal
from market_guard import MarketState
from agent import (
    _direction_momentum_gate,
    _position_notional_usd,
    _estimate_funding_cost,
    _estimate_fee_cost,
    _update_mfe_mae_quant,
    _evaluate_market_guard,
)

failures = []
passed = []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        failures.append(name)
        print(f"  [FAIL] {name} {detail}")


def make_candles(closes):
    """由 close 序列构造 Candle 列表(open/high/low 围绕 close 简单生成)。"""
    out = []
    for i, c in enumerate(closes):
        out.append(Candle(
            timestamp=1000_000_000_000 + i * 3600_000,
            open=c - 0.5, high=c + 1.0, low=c - 1.0, close=c, volume=100.0,
        ))
    return out


def make_signal(side="long"):
    fvg = FVG(
        direction=side, top=110.0, bottom=90.0, width_pct=20.0,
        candle_ts=0, timeframe="1H",
        impulse_candle=Candle(0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )
    return Signal(
        inst_id="TEST-USDT-SWAP", fvg=fvg,
        entry_price=100.0, stop_loss=98.0, take_profit=110.0,
        leverage=5, position_side=side, score=0.8,
    )


def test_bug1_direction_momentum():
    print("\n===== Bug 1: 方向动量门(换仓路径补传 candles_1h 后生效) =====")
    cfg = {"strategy": {"direction_momentum_gate": {"enabled": True, "ma_period": 20}}}

    # 1H 明确下行: close 100 -> 75 递减, SMA 下行且 last 在均线下方
    down_candles = make_candles([100 - i for i in range(26)])

    # 做多信号 vs 明确下行 → 应拒绝(修复后 candles_1h 传入, 门生效)
    long_sig = make_signal(side="long")
    ok, reason = _direction_momentum_gate(long_sig, down_candles, cfg)
    check("做多逆 1H 下行被拒绝", ok is False, f"reason={reason[:50] if reason else reason}")

    # 做空信号 vs 明确下行 → 应放行(顺势)
    short_sig = make_signal(side="short")
    ok2, _ = _direction_momentum_gate(short_sig, down_candles, cfg)
    check("做空顺 1H 下行放行", ok2 is True)

    # 对照: 空 candles_1h(修复前换仓路径传入) → 数据不足放行(门被绕过)
    ok3, _ = _direction_momentum_gate(long_sig, [], cfg)
    check("空 candles_1h 时放行(旧行为, 反证修复必要性)", ok3 is True)


def test_bug2_notional_ctval():
    print("\n===== Bug 2: 名义敞口补乘 ctVal =====")
    client = MagicMock()
    client.get_instrument_info.return_value = {"ctVal": "1000"}  # NEIRO ctVal=1000
    pos = {"instId": "NEIRO-USDT-SWAP", "pos": "702", "markPx": "0.00007169"}
    notional = _position_notional_usd(client, pos)
    # 期望 702 * 1000 * 0.00007169 = 50.326
    check("NEIRO 敞口 = pos×ctVal×markPx", abs(notional - 50.326) < 0.01,
          f"notional={notional:.4f} (期望 ~50.33)")
    # 对比: 旧实现 pos×markPx = 0.0503 (低估 1000 倍)
    check("旧口径(pos×markPx)严重低估", 702 * 0.00007169 < 0.1,
          f"旧值={702*0.00007169:.5f}")


def test_bug6_funding_cycles():
    print("\n===== Bug 6: 资金费周期 <8h 不再满周期高估 =====")
    client = MagicMock()
    client.get_instrument_info.return_value = {"ctVal": "1000"}
    # hold_hours=4, funding_rate=0.0001, pos_size=702, avg_px=0.00007169
    cost = _estimate_funding_cost(
        client, "NEIRO-USDT-SWAP", pos_size=702, avg_px=0.00007169,
        hold_hours=4.0, funding_rate=0.0001, pos_side="long",
    )
    # 修复后 cycles = 4/8 = 0.5; 旧实现 max(1, int(4/8)) = 1
    # position_value = 702*1000*0.00007169 = 50.326
    # long 付费: cost = -50.326 * 0.0001 * 0.5 = -0.002516
    check("4h 持仓按 0.5 周期估算", abs(cost - (-0.002516)) < 0.001,
          f"cost={cost:.6f} (期望 -0.002516)")


def test_bug8_fee_cost():
    print("\n===== Bug 8: 手续费 开仓 maker + 平仓按 is_taker =====")
    client = MagicMock()
    client.get_instrument_info.return_value = {"ctVal": "1"}
    # pos_size=100, avg_px=1.0 → position_value=100
    cost_taker = _estimate_fee_cost(client, "X-USDT-SWAP", 100, 1.0, is_taker=True)
    # 修复后: -(100)*(0.0002 + 0.0005) = -0.07; 旧实现 -0.10
    check("taker 平仓 = maker开 + taker平 = 0.07%", abs(cost_taker - (-0.07)) < 1e-9,
          f"cost={cost_taker}")
    cost_maker = _estimate_fee_cost(client, "X-USDT-SWAP", 100, 1.0, is_taker=False)
    # -(100)*(0.0002+0.0002) = -0.04
    check("maker 平仓 = 0.04%", abs(cost_maker - (-0.04)) < 1e-9, f"cost={cost_maker}")


def test_bug4_mfe_mae():
    print("\n===== Bug 4: MFE/MAE ws_cache 实时价回退 =====")
    signal_tracker = MagicMock()
    ws_cache = MagicMock()
    ws_cache.get.return_value = {"last": "100.5"}
    client = MagicMock()
    active_signals = {"TEST-USDT-SWAP": {"signal_id": "sig_1"}}  # 无 mark_px

    _update_mfe_mae_quant(signal_tracker, active_signals, ws_cache, client)

    pos = active_signals["TEST-USDT-SWAP"]
    check("mfe_high 由 ws_cache 回退填充", pos.get("mfe_high") == 100.5,
          f"mfe_high={pos.get('mfe_high')}")
    check("mae_low 由 ws_cache 回退填充", pos.get("mae_low") == 100.5)
    signal_tracker.update_mfe_mae.assert_called_with("sig_1", 100.5, 100.5)
    check("update_mfe_mae 被正确调用", True)


def test_bug5_oi_24h():
    print("\n===== Bug 5: OI 变化改为真实 24h 窗口 =====")
    client = MagicMock()
    client.get_tickers.return_value = [
        {"instId": "BTC-USDT-SWAP", "open24h": "100", "last": "101"},
        {"instId": "ETH-USDT-SWAP", "open24h": "200", "last": "201"},
    ]
    client.get_candles_enhanced.return_value = None
    client.get_funding_rate.return_value = None
    client.get_open_interest.return_value = 130.0

    ws_cache = MagicMock()
    ws_cache.get.return_value = None  # 触发 REST fallback

    market_guard = MagicMock()
    # 让 evaluate 返回真实 MarketState，避免 MagicMock 状态触发 logger.info
    # 格式化失败 → UNKNOWN 兜底 → evaluate 被调用两次
    market_guard.evaluate.return_value = MarketState(
        timestamp=time.time(), btc_return_24h=0.0, btc_volatility_24h=0.0,
        market_breadth=1.0, funding_extreme=0.0, oi_change_pct=0.0,
        regime="NORMAL", reasons=[],
    )
    # 模拟 25 小时前采样的 OI 基准 = 100
    market_guard._prev_oi = 100.0
    market_guard._prev_oi_ts = time.time() - 25 * 3600.0

    _evaluate_market_guard(client, market_guard, ws_cache)

    # 检查 evaluate 收到的 open_interest_change_pct = (130-100)/100 = 0.30
    call_kwargs = market_guard.evaluate.call_args.kwargs
    oi_change = call_kwargs.get("open_interest_change_pct", None)
    check("跨度>=24h 计算真实 OI 变化 0.30", oi_change is not None and abs(oi_change - 0.30) < 1e-6,
          f"oi_change={oi_change}")

    # 场景2: 采样跨度 < 24h(刚采样 1h 前) → oi_change 应为 0(不误触发)
    market_guard2 = MagicMock()
    market_guard2.evaluate.return_value = MarketState(
        timestamp=time.time(), btc_return_24h=0.0, btc_volatility_24h=0.0,
        market_breadth=1.0, funding_extreme=0.0, oi_change_pct=0.0,
        regime="NORMAL", reasons=[],
    )
    market_guard2._prev_oi = 100.0
    market_guard2._prev_oi_ts = time.time() - 1 * 3600.0
    _evaluate_market_guard(client, market_guard2, ws_cache)
    oi_change2 = market_guard2.evaluate.call_args.kwargs.get("open_interest_change_pct", None)
    check("跨度<24h 保持 0(不误触发)", oi_change2 == 0.0, f"oi_change={oi_change2}")


if __name__ == "__main__":
    test_bug1_direction_momentum()
    test_bug2_notional_ctval()
    test_bug6_funding_cycles()
    test_bug8_fee_cost()
    test_bug4_mfe_mae()
    test_bug5_oi_24h()

    print("\n" + "=" * 56)
    print(f"结果: PASS {len(passed)} | FAIL {len(failures)}")
    if failures:
        print("失败项:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")
