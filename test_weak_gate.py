# -*- coding: utf-8 -*-
"""本地单测: 弱信号多指标共振审核 _weak_signal_multi_gate。

验证点:
  1. 强信号 (score≥0.45 且 conf≥0.50) → 直接放行, 不触发指标检查
  2. 弱信号 + 5 项指标全部顺向 → 放行
  3. 弱信号 + 顺向不足 (0-2/5) → 拒绝 (赌单拦截)
  4. 弱信号 + 数据不足 (可用指标 < min_confluence) → 降级放行
  5. 配置 enabled=false → 全放行
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import Candle, Signal, FVG
from agent import _weak_signal_multi_gate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_weak_gate")

WCFG = {
    "enabled": True,
    "score_threshold": 0.45,
    "confidence_threshold": 0.50,
    "min_confluence": 3,
    "volume_ratio_min": 1.2,
    "lsr_long_min": 1.05,
    "lsr_short_max": 0.95,
    "turnover_min": 2.0,
    "trend_ma": 20,
}


def base_config():
    return {"strategy": {"weak_signal_gate": dict(WCFG)}}


class FakeAnalysis:
    def __init__(self, conf):
        self.final_confidence = conf


class FakeClient:
    """可控指标: lsr / funding / oi / (默认 None=未判定)"""

    def __init__(self, lsr=None, funding=None, oi=None):
        self._lsr = lsr
        self._funding = funding
        self._oi = oi

    def get_long_short_ratio(self, inst_id, period="1H"):
        return self._lsr

    def get_funding_rate(self, inst_id):
        return self._funding

    def get_open_interest(self, inst_id):
        return self._oi


def make_candles(n=32, trend=+0.0, vol=1.0, vol_spike=False):
    """构造 K 线。vol_spike: 最后 3 根放量 2x。"""
    candles = []
    px = 100.0
    for i in range(n):
        px *= (1 + trend * 0.002)
        v = vol * (2.0 if (vol_spike and i >= n - 3) else 1.0)
        candles.append(Candle(
            timestamp=1700000000000 + i * 3600000,
            open=px, high=px * 1.001, low=px * 0.999,
            close=px, volume=v,
        ))
    return candles


def make_signal(score, side="short"):
    fvg = FVG(direction=side, top=101.0, bottom=99.0, width_pct=2.0,
              candle_ts=0, timeframe="1H", impulse_candle=None,
              fvg_index=0, is_abnormal=True, sigma=2.0, volume_ratio=3.0)
    return Signal(inst_id="TEST-USDT-SWAP", fvg=fvg,
                  entry_price=100.0, stop_loss=101.5 if side == "short" else 98.5,
                  take_profit=97.5 if side == "short" else 102.5,
                  leverage=5, position_side=side, score=score,
                  reason="test")


def run_case(name, score, conf, client, candles, side="short", wcfg=None):
    cfg = base_config()
    if wcfg is not None:
        cfg["strategy"]["weak_signal_gate"] = wcfg
    sig = make_signal(score, side)
    ana = FakeAnalysis(conf)
    ok, reason = _weak_signal_multi_gate(cfg, client, sig, ana, candles)
    log.info(f"[{name}] ok={ok} reason={reason!r}")
    return ok


def main():
    # 1. 强信号 → 放行 (score/conf 均达标, 即使指标全部逆向)
    ok = run_case("strong-score", 0.60, 0.60,
                  FakeClient(lsr=0.5, funding=-0.001, oi=1e9), make_candles())
    assert ok is True
    ok = run_case("strong-conf", 0.30, 0.90,
                  FakeClient(), make_candles())
    assert ok is True
    log.info("  强信号放行 OK")

    # 2. 弱信号 + 全顺向 → 放行
    #    short: lsr 0.8(顺向), funding +0.0005(吃正费率), vol spike(量能顺向),
    #    trend 下降(顺向), oi 小换手高(顺向)
    c_short = make_candles(trend=-0.02, vol=1.0, vol_spike=True)
    ok = run_case("weak-all-ok", 0.30, 0.40,
                  FakeClient(lsr=0.8, funding=0.0005, oi=50.0), c_short, side="short")
    assert ok is True, "全顺向应放行"
    log.info("  弱信号全顺向放行 OK")

    # 3. 弱信号 + 顺向不足 → 拒绝
    #    long 信号: trend 上升(顺向) 但 lsr 0.6(逆向)、funding +0.0005(逆向)、
    #    无量能放大(逆向)、换手低(逆向) → 1/5 顺向 → 拒绝
    c_long = make_candles(trend=+0.02, vol=1.0, vol_spike=False)
    ok = run_case("weak-blocked", 0.30, 0.40,
                  FakeClient(lsr=0.6, funding=0.0005, oi=1e9), c_long, side="long")
    assert ok is False, "顺向不足应拒绝"
    log.info("  弱信号共振不足拒绝 OK")

    # 4. 弱信号 + 数据不足 (< min_confluence) → 降级放行
    #    funding=None, oi=None, lsr=None → 仅量能+趋势可用 (2 < 3) → 放行
    ok = run_case("weak-no-data", 0.30, 0.40,
                  FakeClient(), make_candles(trend=-0.02, vol_spike=True))
    assert ok is True, "数据不足应降级放行"
    log.info("  数据不足降级放行 OK")

    # 5. enabled=false → 全放行
    ok = run_case("disabled", 0.30, 0.40,
                  FakeClient(lsr=0.6, funding=0.0005, oi=1e9), make_candles(),
                  wcfg={**WCFG, "enabled": False})
    assert ok is True, "disabled 应放行"
    log.info("  enabled=false 放行 OK")

    # 6. min_confluence=1 → 只需 1 项顺向
    #    long: trend 上升(顺向), 其他逆向 → 1/5 ≥ 1 → 放行
    ok = run_case("min-confl-1", 0.30, 0.40,
                  FakeClient(lsr=0.6, funding=0.0005, oi=1e9),
                  make_candles(trend=+0.02),
                  side="long", wcfg={**WCFG, "min_confluence": 1})
    assert ok is True, "min_confluence=1 应放行"
    log.info("  min_confluence=1 放行 OK")

    log.info("=" * 50)
    log.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
