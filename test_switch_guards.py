# -*- coding: utf-8 -*-
"""本地单测: 方案A(结算保护) + 方案D(目标费率顺向) 换仓守卫。

验证 _switch_funding_guards 的两分支 + _switch_cost_surcharge 成本附加分。
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import _switch_cost_surcharge, _switch_funding_guards

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("test_switch_guards")

# ---- 模拟 client: 固定 next_funding_ts ----
class FakeClient:
    def __init__(self, wait_hours):
        self._next_ts = time.time() + wait_hours * 3600.0

    def get_funding_info(self, inst_id):
        return (0.0005, self._next_ts)


def base_config(**kw):
    cfg = {"strategy": {
        "switch_funding_lockout_hours": 3,
        "switch_funding_direction_required": True,
        "switch_round_trip_cost_pct": 0.3,
        "switch_cost_edge_pct": 1.5,
        "funding_confluence_min_abs": 0.0003,
    }}
    cfg["strategy"].update(kw)
    return cfg


def run_case(name, cfg, client, old_inst, rate, side):
    ok, reason = _switch_funding_guards(cfg, client, old_inst, rate, side)
    log.info(f"[{name}] ok={ok} reason={reason!r}")
    return ok


def main():
    log.info("=" * 60)
    log.info("验证 方案B: _switch_cost_surcharge")
    c1 = _switch_cost_surcharge(base_config())
    assert abs(c1 - 0.3 / 1.5) < 1e-9, c1
    log.info(f"  默认配置 cost_surcharge = {c1:.4f} (期望 0.2000)")
    c2 = _switch_cost_surcharge(base_config(switch_round_trip_cost_pct=0))
    assert c2 == 0.0
    log.info("  cost_pct=0 → 0.0 (退化基础门槛) OK")
    c3 = _switch_cost_surcharge(base_config(switch_cost_edge_pct=-1))
    assert c3 == 0.0
    log.info("  edge<=0 → 0.0 OK")

    log.info("=" * 60)
    log.info("验证 方案D: 目标费率顺向过滤 (不做空负费率/不做多正费率)")
    # 做空 + 负费率 → 拦截
    ok = run_case("D-short-neg", base_config(), None, "X-USDT-SWAP", -0.0005, "short")
    assert ok is False
    # 做空 + 正费率 → 放行
    ok = run_case("D-short-pos", base_config(), None, "X-USDT-SWAP", 0.0005, "short")
    assert ok is True
    # 做多 + 正费率 → 拦截
    ok = run_case("D-long-pos", base_config(), None, "X-USDT-SWAP", 0.0005, "long")
    assert ok is False
    # 做多 + 负费率 → 放行
    ok = run_case("D-long-neg", base_config(), None, "X-USDT-SWAP", -0.0005, "long")
    assert ok is True
    # 费率不足 min_abs (0.0003): -0.0002 不构成负向明显 → 放行(中性区)
    ok = run_case("D-short-smallneg", base_config(), None, "X-USDT-SWAP", -0.0002, "short")
    assert ok is True
    log.info("  方案D 全分支通过")

    log.info("=" * 60)
    log.info("验证 方案A: 结算保护 (距结算<3h拦截, ≥3h放行)")
    # 距结算 2h < 3h → 拦截
    fc_near = FakeClient(wait_hours=2.0)
    ok = run_case("A-near", base_config(), fc_near, "OLD-USDT-SWAP", 0.0005, "short")
    assert ok is False
    # 距结算 5h ≥ 3h → 放行
    fc_far = FakeClient(wait_hours=5.0)
    ok = run_case("A-far", base_config(), fc_far, "OLD-USDT-SWAP", 0.0005, "short")
    assert ok is True
    # lockout=0 → 关闭结算保护
    ok = run_case("A-disabled", base_config(switch_funding_lockout_hours=0),
                  fc_near, "OLD-USDT-SWAP", 0.0005, "short")
    assert ok is True
    # 查询失败(client 抛异常) → 放行(不阻塞)
    class BoomClient:
        def get_funding_info(self, inst_id):
            raise RuntimeError("timeout")
    ok = run_case("A-client-error", base_config(), BoomClient(),
                  "OLD-USDT-SWAP", 0.0005, "short")
    assert ok is True
    log.info("  方案A 全分支通过")

    log.info("=" * 60)
    log.info("ALL PASS ✓")

if __name__ == "__main__":
    main()
