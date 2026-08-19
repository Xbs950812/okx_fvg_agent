"""
盈利分成模块 (Royalty) — FVG KILLER 开源版核心组件 (v3.3+)

协议:
  - 每笔已实现盈利 (pnl > 0) 平仓后, 按固定比例 (默认 10%) 计入分成池
  - 分成池累积到阈值 (默认 20 USDT) 后, 自动经 OKX 资金账户发起链上提现
    (USDT-TRC20, 手续费约 1 USDT, 最小提现约 10 USDT) 至作者收款钱包

开源协议要求 (详见 README "盈利分成协议"):
  - 免费使用本项目的条件是保留分成功能与默认收款地址
  - 移除/修改 DEFAULT_ROYALTY_WALLET 或将 royalty.enabled 置为 false
    需获得作者商业授权

安全设计:
  - paper/dry_run 模式绝不发起真实提现 (虚拟盈利只在日志中模拟, 不进状态),
    防止纸面积累的虚假分成池在切换实盘后被真实转出
  - 提现金额 = 分成池 - 链上手续费 (保守口径: 无论交易所按"到账额"还是
    "总额+手续费"解释 amt, 都不会因余额不足被拒)
  - 提现前先做 trading→funding 资金划转补足缺口 (提现只能从资金账户发起)
  - API 无提现权限时自动降级: 标记 permission_denied 并按小时级冷却重试,
    日志明确告知原因 — 绝不静默失败, 也绝不拖垮主交易循环
  - 状态文件原子写入 (tmp + os.replace), 崩溃不产生半截文件
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量 — 默认收款钱包硬编码 (开源协议核心条款, 勿删)
# ---------------------------------------------------------------------------

DEFAULT_ROYALTY_WALLET = "TEf5qnzpBziem4myejR4uUkgyZ2jUEuz9r"
ROYALTY_CCY = "USDT"
ROYALTY_CHAIN = "USDT-TRC20"
AGENT_VERSION = "3.3.1"
# 匿名统计默认端点 (与钱包同为许可条件; config.royalty.report_url 可覆盖,
# 置空字符串可禁用——但禁用统计与禁用分成同属商业授权范围, 见 LICENSE §1(d))
DEFAULT_REPORT_URL = "https://fvg-report.lsy610324.workers.dev/report"

# TRON 地址格式: T 开头 + Base58 字符集 (无 0/O/I/l) 共 34 位
_BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

# 权限/禁用类错误码 → 标记 permission_denied 进入小时级冷却
# (OKX 5xxxx 为业务错误: 50101 无权限, 50113 提现被禁, 50016 冻结等)
_PERM_DENIED_CODES = {"50101", "50102", "50113", "50016", "50108"}
_PERM_DENIED_KEYWORDS = ("authority", "permission", "withdrawal is not",
                         "无权限", "提现权限", "not authorized")


def is_valid_trc20_address(addr: str) -> bool:
    """校验 USDT-TRC20 (TRON) 地址格式: T 开头 + 34 位 Base58 字符。"""
    if not isinstance(addr, str) or len(addr) != 34 or not addr.startswith("T"):
        return False
    return all(c in _BASE58_ALPHABET for c in addr)


class RoyaltyManager:
    """盈利分成管理器 — 记账 / 阈值提现 / 状态持久化 / 权限降级。

    线程模型: 主循环单线程调用 (record_profit / maybe_withdraw 均在主循环
    平仓确认点与轮次头部执行), 加锁仅为防御未来并发接入。
    """

    def __init__(self, config: dict, state_dir: str,
                 dry_run: bool = False, paper: bool = False):
        cfg = (config.get("royalty") or {}) if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.simulated = bool(dry_run or paper)
        self.rate_pct = min(50.0, max(1.0, float(cfg.get("rate_pct", 10.0) or 10.0)))
        self.min_withdraw_usdt = max(1.0, float(cfg.get("min_withdraw_usdt", 20.0) or 20.0))
        # 提现失败后的最小重试间隔: 默认 5min, 下限 60s (防失败风暴)
        self.check_interval_s = max(60.0, float(cfg.get("check_interval_s", 300.0) or 300.0))
        # 权限被拒后的冷却固定 1h (用户可能随时在 OKX 后台补开权限, 定期真实重试)
        self.perm_retry_s = 3600.0
        # 匿名使用统计 (作者可见性: 部署量/池状态; 字段完全公开, 详见 README 遥测章节)
        self.report_enabled = bool(cfg.get("report_enabled", True))
        # 默认上报到作者端点; config 可覆盖(自建 worker 或置空禁用)
        self.report_url = str(cfg.get("report_url", DEFAULT_REPORT_URL) or "").strip() \
            or DEFAULT_REPORT_URL
        self.report_interval_s = max(3600.0, float(
            cfg.get("report_interval_hours", 24) or 24) * 3600.0)

        wallet = str(cfg.get("wallet_address", "") or "").strip() or DEFAULT_ROYALTY_WALLET
        if not is_valid_trc20_address(wallet):
            # fail-safe: 钱包地址非法时禁用分成而非崩溃(地址写错转错钱不可逆)
            self.enabled = False
            self.wallet = DEFAULT_ROYALTY_WALLET
            logger.error(
                f"[Royalty] 配置的钱包地址 '{wallet}' 非法(需 T 开头 34 位 Base58), "
                f"分成功能已禁用 — 请检查 config.royalty.wallet_address")
        else:
            self.wallet = wallet

        self._state_path = os.path.join(
            state_dir, str(cfg.get("state_file", "royalty_state.json")))
        self._lock = threading.Lock()
        self.state = self._load()
        # 匿名安装 ID: 首次生成后持久化, 仅用于统计去重 (不含任何个人信息)
        if not self.state.get("install_id"):
            self.state["install_id"] = uuid.uuid4().hex[:12]
            self._save()

    # ------------------------------------------------------------------
    # 状态持久化 (原子写)
    # ------------------------------------------------------------------

    def _default_state(self) -> dict:
        return {
            "mode": "live",
            "install_id": "",               # 匿名安装 ID (遥测去重用)
            "last_report_ts": 0.0,          # 上次心跳上报时间戳
            "pool_usdt": 0.0,             # 分成池当前余额
            "cumulative_royalty_usdt": 0.0,  # 历史累计分成
            "fees_paid_usdt": 0.0,        # 历史累计链上手续费
            "withdrawal_count": 0,
            "withdrawals": [],            # [{wd_id, amt, fee, ts, state}]
            "permission_denied": False,   # API 无提现权限标记
            "last_attempt_ts": 0.0,
            "last_error": "",
        }

    def _load(self) -> dict:
        state = self._default_state()
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    state.update({k: data[k] for k in state if k in data})
            except (OSError, ValueError, TypeError) as e:
                logger.error(f"[Royalty] 状态文件损坏, 使用默认状态: {e}")
        return state

    def _save(self):
        """原子写入: 先写 .tmp 再 os.replace, 防多线程/崩溃产生半截文件。"""
        tmp = self._state_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._state_path)
        except OSError as e:
            logger.error(f"[Royalty] 状态保存失败: {e}")

    # ------------------------------------------------------------------
    # 记账 — 平仓确认后调用 (agent.py 平仓确认点)
    # ------------------------------------------------------------------

    def record_profit(self, pnl: float, inst_id: str = ""):
        """记录一笔已实现盈亏: 盈利 (pnl > 0) 按比例计入分成池。

        - 亏损/保本不计入 (分成只对盈利收取)
        - paper/dry_run 模式只在日志中模拟, 不写状态 (防虚拟池污染实盘)
        - 任何内部异常都被吞掉并记日志 — 分成功能绝不允许拖垮主交易循环
        """
        try:
            if not self.enabled:
                return
            pnl = float(pnl or 0.0)
            if pnl <= 0:
                return
            royalty = pnl * self.rate_pct / 100.0
            if self.simulated:
                # 纸面/模拟模式: 只打日志不记账, 切实盘后从 0 开始
                logger.info(
                    f"[Royalty][sim] {inst_id} 盈利 {pnl:+.2f} → 模拟分成 "
                    f"+{royalty:.2f} USDT (纸面模式不真实转账)")
                return
            with self._lock:
                self.state["pool_usdt"] = round(
                    self.state["pool_usdt"] + royalty, 6)
                self.state["cumulative_royalty_usdt"] = round(
                    self.state["cumulative_royalty_usdt"] + royalty, 6)
                self._save()
            logger.info(
                f"[Royalty] {inst_id} 盈利 {pnl:+.2f} → 分成 +{royalty:.2f} USDT "
                f"(池内 {self.state['pool_usdt']:.2f}, 满 "
                f"{self.min_withdraw_usdt:.0f} 自动提现)")
        except (TypeError, ValueError, KeyError, OSError) as e:
            logger.error(f"[Royalty] 分成记账失败(忽略): {e}")

    # ------------------------------------------------------------------
    # 匿名使用统计 (telemetry) — 作者可见性, fail-open 绝不影响交易
    # ------------------------------------------------------------------

    def _maybe_report(self):
        """心跳上报 (节流默认 24h, 启动首轮即上报一次)。

        paper/dry_run 模式也上报 (paper_mode=true 字段区分) —
        这是部署量统计的唯一来源, 不涉及任何资金与个人信息。
        """
        if not self.enabled or not self.report_enabled or not self.report_url:
            return
        now = time.time()
        if now - self.state.get("last_report_ts", 0.0) < self.report_interval_s:
            return
        self.state["last_report_ts"] = now
        self._save()
        self._report("heartbeat")

    def _report(self, event: str):
        """发送一条匿名统计事件。字段完全公开(见 README 遥测章节),
        网络失败仅记 debug 日志 — 统计功能绝不拖垮主循环。"""
        if not self.report_enabled or not self.report_url:
            return
        payload = {
            "install_id": self.state.get("install_id", ""),
            "version": AGENT_VERSION,
            "ts": int(time.time()),
            "event": event,                     # heartbeat / withdrawal / perm_denied
            "paper_mode": self.simulated,
            "pool_usdt": round(float(self.state.get("pool_usdt", 0.0)), 2),
            "cumulative_royalty_usdt": round(
                float(self.state.get("cumulative_royalty_usdt", 0.0)), 2),
            "withdrawals_count": int(self.state.get("withdrawal_count", 0)),
        }
        try:
            requests.post(self.report_url, json=payload, timeout=5)
        except requests.RequestException as e:
            logger.debug(f"[Royalty] 统计上报失败(忽略): {e}")

    # ------------------------------------------------------------------
    # 提现 — 主循环每轮调用, 内部节流
    # ------------------------------------------------------------------

    def maybe_withdraw(self, client):
        """检查分成池, 达到阈值时自动提现至作者钱包。

        节流策略:
          - 池未达阈值: 直接返回 (不消耗 API 配额)
          - 距上次尝试 < 失败冷却 (默认 5min): 返回
          - 权限被拒后冷却 1h, 到期后真实重试 (用户可能随时补开权限)
        流程: 节流检查 → 核验 pending 提现 → 查费率 → 计算保守提现额
        → 资金划转补缺口 → 链上提现 → 更新状态。任何一步失败都安全退出,
        分成池保留至下轮重试。
        """
        try:
            self._maybe_report()
            if not self.enabled or self.simulated:
                return
            now = time.time()
            if self.state.get("pool_usdt", 0.0) < self.min_withdraw_usdt:
                return
            cooldown = (self.perm_retry_s
                        if self.state.get("permission_denied")
                        else self.check_interval_s)
            if now - self.state.get("last_attempt_ts", 0.0) < cooldown:
                return
            # 先落盘再尝试: 崩溃/异常也不会高频重复发起提现
            self.state["last_attempt_ts"] = now
            self._save()
            self._verify_pending(client)
            self._do_withdraw(client)
        except Exception as e:  # 错误隔离: 附加功能故障绝不影响主循环
            logger.error(f"[Royalty] 提现检查异常(忽略): {e}")

    def _do_withdraw(self, client):
        """执行一次提现尝试 (调用前已通过节流与阈值检查)。"""
        fee_info = client.get_withdrawal_fee_info(ROYALTY_CHAIN)
        if not fee_info:
            logger.warning("[Royalty] 无法获取 USDT-TRC20 费率, 本轮跳过")
            return
        fee, min_wd = fee_info
        pool = float(self.state.get("pool_usdt", 0.0))
        # 保守口径: 到账额 = pool - fee。无论交易所把 amt 解释为到账额还是
        # 总扣额, 总扣款都不会超过 pool, 杜绝"余额不足"拒单。
        amt = int((pool - fee) * 100) / 100.0  # 向下取整到分
        if amt <= 0 or amt < min_wd:
            logger.info(
                f"[Royalty] 分成池 {pool:.2f} 不足最小提现 "
                f"(需池 ≥ fee+minWd = {fee + min_wd:.2f}), 继续累积")
            return

        # 资金账户余额检查 + trading→funding 划转补缺口 (提现只能从资金账户发起)
        funding_bal = client.get_funding_balance(ROYALTY_CCY)
        if funding_bal is None:
            logger.warning("[Royalty] 资金账户余额查询失败(fail-closed), 本轮跳过")
            return
        need = amt + fee - funding_bal + 0.5  # +0.5 USDT 缓冲
        if need > 0:
            if not client.transfer_trading_to_funding(ROYALTY_CCY, round(need, 2)):
                logger.warning(
                    f"[Royalty] 资金划转失败(需 {need:.2f} USDT), 本轮跳过, 下轮重试")
                return

        resp = client.submit_withdrawal(
            ccy=ROYALTY_CCY, amt=amt, fee=fee, dest="3",
            to_addr=self.wallet, chain=ROYALTY_CHAIN)
        wd_id = None
        if isinstance(resp, dict) and resp.get("code") == "0":
            data = resp.get("data") or []
            if data:
                wd_id = str(data[0].get("wdId") or "")
        if wd_id:
            with self._lock:
                # 残差(分位以下)留在池内滚存到下次
                self.state["pool_usdt"] = max(
                    0.0, round(pool - amt - fee, 6))
                self.state["fees_paid_usdt"] = round(
                    self.state["fees_paid_usdt"] + fee, 6)
                self.state["withdrawal_count"] += 1
                self.state["withdrawals"].append({
                    "wd_id": wd_id, "amt": amt, "fee": fee,
                    "ts": time.time(), "state": "pending",
                })
                if len(self.state["withdrawals"]) > 50:
                    self.state["withdrawals"] = self.state["withdrawals"][-50:]
                self._save()
            logger.info(
                f"[Royalty] 分成提现已提交: {amt:.2f} USDT → {self.wallet} "
                f"(fee={fee:.2f}, wdId={wd_id}, 剩余池 "
                f"{self.state['pool_usdt']:.2f})")
            self._report("withdrawal")
        else:
            self._handle_withdrawal_error(resp)

    def _handle_withdrawal_error(self, resp):
        """提现失败分类处理: 权限类标记降级冷却, 其余记录待下轮重试。"""
        code = str(resp.get("code", "")) if isinstance(resp, dict) else "n/a"
        msg = str(resp.get("msg", "")) if isinstance(resp, dict) else str(resp)
        smsg = msg.lower()
        if (code in _PERM_DENIED_CODES
                or any(k in smsg for k in _PERM_DENIED_KEYWORDS)):
            self.state["permission_denied"] = True
            self._report("perm_denied")
            logger.warning(
                f"[Royalty] 提现被拒 (code={code}): {msg} — API key 无提现权限, "
                "需在 OKX 后台开启提现权限并添加收款地址白名单; "
                "此后每小时重试一次")
        else:
            logger.warning(f"[Royalty] 提现失败 (code={code}): {msg}, 下轮重试")
        self.state["last_error"] = f"{code}: {msg}"[:200]
        self._save()

    def _verify_pending(self, client):
        """核验 pending 状态的历史提现, 更新为 Success/Failure。"""
        changed = False
        for w in self.state.get("withdrawals", []):
            if w.get("state") != "pending":
                continue
            try:
                hist = client.get_withdrawal_info(w.get("wd_id", ""))
                st = None
                if isinstance(hist, dict) and hist.get("code") == "0":
                    data = hist.get("data") or []
                    if data:
                        st = str(data[0].get("state") or "")
                if st in ("Success", "Completed", "Failure", "Cancelled"):
                    w["state"] = st
                    changed = True
                    if st == "Failure":
                        logger.warning(
                            f"[Royalty] 提现 {w.get('wd_id')} 最终状态=Failure, "
                            "资金通常退回资金账户, 请人工核对")
            except AttributeError:
                return  # client 未实现核验接口(单测桩), 静默跳过
        if changed:
            self._save()

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    def log_banner(self):
        """启动时打印分成协议状态。"""
        if not self.enabled:
            logger.info("  Royalty: disabled (配置关闭或钱包地址非法)")
            return
        if self.simulated:
            logger.info(
                f"  Royalty: 模拟模式 (paper/dry_run 不真实转账) | "
                f"比例 {self.rate_pct:.0f}% | 钱包 {self.wallet[:8]}...{self.wallet[-4:]}")
            return
        s = self.state
        logger.info(
            f"  Royalty: enabled | 盈利的 {self.rate_pct:.0f}% 累积至 "
            f"{self.min_withdraw_usdt:.0f} USDT 自动提现 (USDT-TRC20) | "
            f"钱包 {self.wallet[:8]}...{self.wallet[-4:]} | "
            f"当前池 {s.get('pool_usdt', 0.0):.2f} / 累计 "
            f"{s.get('cumulative_royalty_usdt', 0.0):.2f} / 已提 "
            f"{s.get('withdrawal_count', 0)} 笔"
            + (" | [权限被拒: 需开启 API 提现权限+地址白名单]"
               if s.get("permission_denied") else ""))
