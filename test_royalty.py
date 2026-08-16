"""
Royalty 盈利分成模块单元测试。

覆盖:
  1. TRC20 地址格式校验 (合法/非法/边界)
  2. record_profit 记账 (盈利/亏损/保本/禁用/纸面模式不写状态)
  3. 状态文件持久化往返 + 原子写
  4. maybe_withdraw 阈值/节流/纸面模式
  5. 提现金额保守口径 (pool - fee, 向下取整到分)
  6. 最小提现额拦截
  7. 资金账户余额不足 → 先划转再提现
  8. 划转失败安全退出 (池保留)
  9. 提现成功 → 池清零 + wdId 落账 + pending 核验
  10. 权限被拒 → permission_denied 降级 + 1h 冷却
  11. 费率查询失败 → fail-closed 跳过
  12. 非法钱包地址 → 禁用不崩溃
"""

import json
import os
import time

import pytest

from royalty import (
    DEFAULT_ROYALTY_WALLET,
    RoyaltyManager,
    is_valid_trc20_address,
)


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------

class FakeClient:
    """OKXClient 资金接口桩 — 记录调用供断言。"""

    _UNSET = object()  # 区分"未传 wd_resp 用默认成功响应"与"显式传 None 模拟网络异常"

    def __init__(self, fee=1.0, min_wd=10.0, funding_bal=100.0,
                 wd_resp=_UNSET, transfer_ok=True, fee_ok=True, bal_ok=True):
        self.fee = fee
        self.min_wd = min_wd
        self.funding_bal = funding_bal
        self.wd_resp = ({"code": "0", "msg": "",
                         "data": [{"wdId": "123456", "cId": ""}]}
                        if wd_resp is FakeClient._UNSET else wd_resp)
        self.transfer_ok = transfer_ok
        self.fee_ok = fee_ok
        self.bal_ok = bal_ok
        self.calls = {"transfer": [], "withdraw": [], "fee": 0, "bal": 0}

    def get_withdrawal_fee_info(self, chain="USDT-TRC20"):
        self.calls["fee"] += 1
        if not self.fee_ok:
            return None
        return (self.fee, self.min_wd)

    def get_funding_balance(self, ccy="USDT"):
        self.calls["bal"] += 1
        if not self.bal_ok:
            return None
        return self.funding_bal

    def transfer_trading_to_funding(self, ccy, amt):
        self.calls["transfer"].append((ccy, amt))
        return self.transfer_ok

    def submit_withdrawal(self, ccy, amt, fee, dest, to_addr, chain):
        self.calls["withdraw"].append(
            {"ccy": ccy, "amt": amt, "fee": fee, "dest": dest,
             "to_addr": to_addr, "chain": chain})
        return self.wd_resp

    def get_withdrawal_info(self, wd_id):
        return {"code": "0", "data": [{"wdId": wd_id, "state": "Success"}]}


def make_mgr(tmp_path, enabled=True, paper=False, dry_run=False,
             pool=0.0, wallet="", rate_pct=10.0, min_wd_usdt=20.0,
             **extra_cfg):
    cfg = {"royalty": {"enabled": enabled, "rate_pct": rate_pct,
                       "min_withdraw_usdt": min_wd_usdt, **extra_cfg}}
    if wallet:
        cfg["royalty"]["wallet_address"] = wallet
    m = RoyaltyManager(cfg, state_dir=str(tmp_path),
                       dry_run=dry_run, paper=paper)
    if pool > 0:
        m.state["pool_usdt"] = pool
        m.state["cumulative_royalty_usdt"] = pool
    # 允许立即尝试 (绕过失败冷却)
    m.state["last_attempt_ts"] = 0.0
    return m


# ---------------------------------------------------------------------------
# 1. 地址校验
# ---------------------------------------------------------------------------

def test_address_valid():
    assert is_valid_trc20_address(DEFAULT_ROYALTY_WALLET)
    assert is_valid_trc20_address("TEf5qnzpBziem4myejR4uUkgyZ2jUEuz9r")


def test_address_invalid_cases():
    assert not is_valid_trc20_address("")                       # 空
    assert not is_valid_trc20_address("TEf5qnzpBziem4myejR4uUkgyZ2jUEuz9")  # 33 位
    assert not is_valid_trc20_address("TEf5qnzpBziem4myejR4uUkgyZ2jUEuz9rr")  # 35 位
    assert not is_valid_trc20_address("1Ef5qnzpBziem4myejR4uUkgyZ2jUEuz9r")  # 非 T 开头
    assert not is_valid_trc20_address("TEf5qnzpBziem4myejR4uUkgyZ2jUEuz0r".replace("9r", "0r"))  # 含 0
    assert not is_valid_trc20_address("TEf5qnzpBziem4myejR4uUkgyZ2jUElu9r")  # 含 l
    assert not is_valid_trc20_address(12345)                    # 非字符串


def test_invalid_wallet_disables_manager(tmp_path):
    m = make_mgr(tmp_path, wallet="0xbtc_eth_invalid_address_xxxxxxxxxx")
    assert m.enabled is False


# ---------------------------------------------------------------------------
# 2. 记账
# ---------------------------------------------------------------------------

def test_record_profit_positive(tmp_path):
    m = make_mgr(tmp_path)
    m.record_profit(10.0, "BTC-USDT-SWAP")
    assert m.state["pool_usdt"] == pytest.approx(1.0)       # 10%
    assert m.state["cumulative_royalty_usdt"] == pytest.approx(1.0)
    m.record_profit(25.0, "ETH-USDT-SWAP")
    assert m.state["pool_usdt"] == pytest.approx(3.5)


def test_record_profit_loss_and_zero(tmp_path):
    m = make_mgr(tmp_path)
    m.record_profit(-10.0, "BTC-USDT-SWAP")   # 亏损不计
    m.record_profit(0.0, "ETH-USDT-SWAP")     # 保本不计
    m.record_profit(None, "X-USDT-SWAP")      # None 容错
    assert m.state["pool_usdt"] == 0.0


def test_record_profit_disabled(tmp_path):
    m = make_mgr(tmp_path, enabled=False)
    m.record_profit(100.0, "BTC-USDT-SWAP")
    assert m.state["pool_usdt"] == 0.0


def test_record_profit_paper_mode_no_state(tmp_path):
    """paper/dry_run 模式只打日志, 不写状态 (防虚拟池污染实盘)。"""
    m = make_mgr(tmp_path, paper=True)
    m.record_profit(100.0, "BTC-USDT-SWAP")
    assert m.state["pool_usdt"] == 0.0
    # 状态文件也不应被写入池金额
    sf = os.path.join(str(tmp_path), "royalty_state.json")
    if os.path.exists(sf):
        with open(sf, encoding="utf-8") as f:
            assert json.load(f).get("pool_usdt", 0.0) == 0.0


def test_record_profit_exception_swallowed(tmp_path):
    """内部异常必须被吞掉 — 分成功能绝不拖垮主交易循环。"""
    m = make_mgr(tmp_path)
    m.state = None  # 破坏内部状态
    m.record_profit(10.0, "BTC-USDT-SWAP")  # 不应抛出


# ---------------------------------------------------------------------------
# 3. 持久化
# ---------------------------------------------------------------------------

def test_state_persistence_roundtrip(tmp_path):
    m = make_mgr(tmp_path)
    m.record_profit(50.0, "BTC-USDT-SWAP")
    m2 = make_mgr(tmp_path)
    assert m2.state["pool_usdt"] == pytest.approx(5.0)
    assert m2.state["cumulative_royalty_usdt"] == pytest.approx(5.0)


def test_corrupted_state_file_uses_default(tmp_path):
    sf = os.path.join(str(tmp_path), "royalty_state.json")
    with open(sf, "w", encoding="utf-8") as f:
        f.write("{broken json!!")
    m = make_mgr(tmp_path)
    assert m.state["pool_usdt"] == 0.0
    assert m.state["withdrawals"] == []


# ---------------------------------------------------------------------------
# 4. maybe_withdraw 守卫
# ---------------------------------------------------------------------------

def test_withdraw_below_threshold_no_attempt(tmp_path):
    m = make_mgr(tmp_path, pool=19.99)  # 阈值 20
    c = FakeClient()
    m.maybe_withdraw(c)
    assert c.calls["fee"] == 0          # 未达阈值不发任何请求


def test_withdraw_paper_mode_never_calls(tmp_path):
    m = make_mgr(tmp_path, pool=100.0, paper=True)
    c = FakeClient()
    m.maybe_withdraw(c)
    assert c.calls["fee"] == 0
    assert c.calls["withdraw"] == []


def test_withdraw_dry_run_mode_never_calls(tmp_path):
    m = make_mgr(tmp_path, pool=100.0, dry_run=True)
    c = FakeClient()
    m.maybe_withdraw(c)
    assert c.calls["withdraw"] == []


def test_withdraw_cooldown_blocks_retry(tmp_path):
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient()
    m.state["last_attempt_ts"] = time.time() - 10  # 10s 前 (< 300s 冷却)
    m.maybe_withdraw(c)
    assert c.calls["fee"] == 0


# ---------------------------------------------------------------------------
# 5-8. 提现执行路径
# ---------------------------------------------------------------------------

def test_withdraw_conservative_amount_and_success(tmp_path):
    """pool=25, fee=1 → 提现 24.00 (到账口径), 总扣 25 = pool。"""
    m = make_mgr(tmp_path, pool=25.0)
    c = FakeClient(fee=1.0, min_wd=10.0, funding_bal=100.0)
    m.maybe_withdraw(c)
    assert len(c.calls["withdraw"]) == 1
    w = c.calls["withdraw"][0]
    assert w["amt"] == pytest.approx(24.0)
    assert w["fee"] == pytest.approx(1.0)
    assert w["to_addr"] == DEFAULT_ROYALTY_WALLET
    assert w["dest"] == "3"
    assert w["chain"] == "USDT-TRC20"
    # 池清零, 流水落账
    assert m.state["pool_usdt"] == pytest.approx(0.0)
    assert m.state["withdrawal_count"] == 1
    assert m.state["withdrawals"][0]["wd_id"] == "123456"
    assert m.state["withdrawals"][0]["state"] in ("pending", "Success")
    assert m.state["fees_paid_usdt"] == pytest.approx(1.0)


def test_withdraw_triggers_transfer_when_funding_short(tmp_path):
    """资金账户余额不足 → 先 trading→funding 划转再提现。"""
    m = make_mgr(tmp_path, pool=25.0)
    c = FakeClient(fee=1.0, funding_bal=5.0)   # 需提 24+1, 缺 20+0.5 缓冲
    m.maybe_withdraw(c)
    assert len(c.calls["transfer"]) == 1
    ccy, amt = c.calls["transfer"][0]
    assert ccy == "USDT"
    assert amt == pytest.approx(20.5, abs=0.01)
    assert len(c.calls["withdraw"]) == 1


def test_withdraw_no_transfer_when_funding_sufficient(tmp_path):
    m = make_mgr(tmp_path, pool=25.0)
    c = FakeClient(fee=1.0, funding_bal=100.0)
    m.maybe_withdraw(c)
    assert c.calls["transfer"] == []       # 余额足够, 不划转
    assert len(c.calls["withdraw"]) == 1


def test_withdraw_transfer_failure_keeps_pool(tmp_path):
    """划转失败 → 安全退出, 池保留至下轮。"""
    m = make_mgr(tmp_path, pool=25.0)
    c = FakeClient(fee=1.0, funding_bal=0.0, transfer_ok=False)
    m.maybe_withdraw(c)
    assert c.calls["withdraw"] == []       # 未发提现
    assert m.state["pool_usdt"] == pytest.approx(25.0)
    assert m.state["withdrawal_count"] == 0


def test_withdraw_blocked_by_min_wd(tmp_path):
    """amt < 交易所最小提现额 → 拒绝发起。pool=11, fee=1 → amt=10 = minWd 放行;
    minWd=15 时 amt=10 < 15 拦截。"""
    m = make_mgr(tmp_path, pool=11.0, min_wd_usdt=5.0)
    c = FakeClient(fee=1.0, min_wd=15.0)
    m.maybe_withdraw(c)
    assert c.calls["withdraw"] == []
    assert m.state["pool_usdt"] == pytest.approx(11.0)


def test_withdraw_fee_query_fail_closed(tmp_path):
    """费率查询失败 (None) → 跳过本轮, 不发提现。"""
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(fee_ok=False)
    m.maybe_withdraw(c)
    assert c.calls["withdraw"] == []
    assert m.state["pool_usdt"] == pytest.approx(100.0)


def test_withdraw_balance_query_fail_closed(tmp_path):
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(bal_ok=False)
    m.maybe_withdraw(c)
    assert c.calls["withdraw"] == []


def test_withdraw_residual_stays_in_pool(tmp_path):
    """分位以下残差滚存: pool=25.999, fee=1 → amt=24.99, 残差 0.009 留池。"""
    m = make_mgr(tmp_path, pool=25.999)
    c = FakeClient(fee=1.0)
    m.maybe_withdraw(c)
    w = c.calls["withdraw"][0]
    assert w["amt"] == pytest.approx(24.99)
    assert 0.0 < m.state["pool_usdt"] < 0.01


# ---------------------------------------------------------------------------
# 9-10. 失败处理
# ---------------------------------------------------------------------------

def test_permission_denied_degrades(tmp_path):
    """50101 无提现权限 → permission_denied=True, 不崩溃。"""
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(wd_resp={"code": "50101",
                            "msg": "Apikey does not have authority to withdraw",
                            "data": []})
    m.maybe_withdraw(c)
    assert m.state["permission_denied"] is True
    assert m.state["pool_usdt"] == pytest.approx(100.0)   # 池保留
    assert "50101" in m.state["last_error"]


def test_permission_denied_cooldown_then_retry(tmp_path):
    """权限被拒后 1h 冷却, 冷却内不重试, 过期后允许真实重试。"""
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(wd_resp={"code": "50101", "msg": "no authority", "data": []})
    m.maybe_withdraw(c)
    assert m.state["permission_denied"] is True
    # 冷却内 (59min 后): 不发请求
    m.state["last_attempt_ts"] = time.time() - 59 * 60
    c2 = FakeClient()
    m.maybe_withdraw(c2)
    assert c2.calls["fee"] == 0
    # 冷却过期 (61min 后): 真实重试 (这次成功)
    m.state["last_attempt_ts"] = time.time() - 61 * 60
    c3 = FakeClient()
    m.maybe_withdraw(c3)
    assert len(c3.calls["withdraw"]) == 1


def test_generic_error_retries_after_cooldown(tmp_path):
    """普通失败 (非权限): 冷却 300s 后重试。"""
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(wd_resp={"code": "59000", "msg": "misc error", "data": []})
    m.maybe_withdraw(c)
    assert m.state["permission_denied"] is False
    assert m.state["pool_usdt"] == pytest.approx(100.0)
    # 4min 后 (< 300s): 不重试
    m.state["last_attempt_ts"] = time.time() - 240
    c2 = FakeClient()
    m.maybe_withdraw(c2)
    assert c2.calls["fee"] == 0
    # 6min 后 (> 300s): 重试
    m.state["last_attempt_ts"] = time.time() - 360
    c3 = FakeClient()
    m.maybe_withdraw(c3)
    assert len(c3.calls["withdraw"]) == 1


def test_network_exception_none_resp(tmp_path):
    """submit_withdrawal 返回 None (网络异常) → 按失败处理不崩溃。"""
    m = make_mgr(tmp_path, pool=100.0)
    c = FakeClient(wd_resp=None)
    m.maybe_withdraw(c)
    assert m.state["pool_usdt"] == pytest.approx(100.0)


def test_maybe_withdraw_exception_isolated(tmp_path):
    """client 完全损坏 → 异常被隔离, 不向主循环传播。"""
    class BrokenClient:
        def get_withdrawal_fee_info(self, chain):
            raise ConnectionError("network down")

        def get_withdrawal_info(self, wd_id):
            raise ConnectionError("network down")

    m = make_mgr(tmp_path, pool=100.0)
    m.maybe_withdraw(BrokenClient())  # 不应抛出
    assert m.state["pool_usdt"] == pytest.approx(100.0)


def test_pending_verification_updates_state(tmp_path):
    """pending 提现核验 → 状态更新为 Success。"""
    m = make_mgr(tmp_path, pool=100.0)
    m.state["withdrawals"] = [
        {"wd_id": "w1", "amt": 24.0, "fee": 1.0, "ts": 1.0, "state": "pending"}]
    c = FakeClient()
    m._verify_pending(c)
    assert m.state["withdrawals"][0]["state"] == "Success"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
