"""
执行模块 — 仓位计算、下单执行、持仓监控、提现检查。

职责：
  - 根据信号和风控参数计算精确仓位
  - 通过 OKXClient 执行限价单（含止盈止损）
  - 监控已开仓位状态，检测止盈止损触发
  - 钱包翻倍时触发提现提醒
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from okx_client import OKXClient
from strategy import Signal


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 状态管理
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Agent 运行状态，持久化到 JSON 文件。"""
    initial_equity: float = 0.0          # 本轮初始权益
    highest_equity: float = 0.0          # 本轮最高权益
    total_pnl: float = 0.0               # 累计盈亏
    last_equity: float = 0.0             # 上一轮权益（用于计算轮间已实现盈亏）
    positions_opened: int = 0            # 本轮已开仓数
    positions_closed: int = 0            # 本轮已平仓数
    winning_trades: int = 0              # 盈利笔数
    losing_trades: int = 0               # 亏损笔数
    last_withdrawal_equity: float = 0.0  # 上次提现时的权益
    withdrawal_count: int = 0            # 提现次数
    daily_loss: float = 0.0              # 当日累计已实现亏损
    daily_date: str = ""                 # 当日日期 (YYYY-MM-DD)
    active_signals: Dict[str, dict] = field(default_factory=dict)  # 活跃信号 {ord_id: signal_info}


class StateManager:
    """状态持久化管理器。"""

    def __init__(self, state_path: str):
        self.state_path = state_path
        self.state = self._load()

    def _load(self) -> AgentState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AgentState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, starting fresh")
        return AgentState()

    def save(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state.__dict__, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_equity(self, equity: float):
        """更新权益追踪。"""
        if self.state.initial_equity == 0.0:
            self.state.initial_equity = equity
            self.state.highest_equity = equity
            self.state.last_withdrawal_equity = equity
        if equity > self.state.highest_equity:
            self.state.highest_equity = equity
        self.state.total_pnl = equity - self.state.initial_equity

    def reset_daily_if_new_day(self):
        """跨日重置当日亏损。"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.state.daily_date:
            self.state.daily_loss = 0.0
            self.state.daily_date = today

    def check_withdrawal(self, equity: float, withdrawal_pct: float) -> bool:
        """检查是否触发提现条件。

        触发条件: 权益 >= 上次提现权益 * (1 + withdrawal_pct/100)
        例如 withdrawal_pct=25 → 权益增长 25% 触发
        """
        if self.state.last_withdrawal_equity <= 0:
            return False
        if withdrawal_pct <= 0:
            return False
        return equity >= self.state.last_withdrawal_equity * (1.0 + withdrawal_pct / 100.0)

    def record_withdrawal(self, equity: float, withdrawal_pct: float = 25.0):
        """记录提现事件。"""
        self.state.last_withdrawal_equity = equity
        self.state.withdrawal_count += 1
        self.save()
        withdraw_amount = equity * withdrawal_pct / 100.0
        logger.info(f"=== 提现提醒 #{self.state.withdrawal_count} === "
                    f"当前权益: {equity:.2f} USDT, "
                    f"建议提现: {withdraw_amount:.2f} USDT ({withdrawal_pct:.0f}%)")


# ---------------------------------------------------------------------------
# 仓位计算
# ---------------------------------------------------------------------------

def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
    leverage: int,
    margin_pct: float,
    contract_value: float = 1.0,
    min_sz: float = 1.0,
    sz_precision: int = 0,
) -> Tuple[float, float]:
    """计算仓位大小。

    公式推导：
      risk_amount = equity * risk_pct / 100
      stop_distance_pct = |entry_price - stop_loss| / entry_price
      position_value = risk_amount / stop_distance_pct
      margin = position_value / leverage
      sz = position_value / (entry_price * contract_value)

    Args:
        equity: 账户权益
        entry_price: 入场价
        stop_loss: 止损价
        risk_pct: 单笔风险比例 (1.0 = 1%)
        leverage: 杠杆倍数
        margin_pct: 最大保证金比例 (30 = 30%)
        contract_value: 每张合约面值 (USDT 本位默认为 1)
        min_sz: 最小下单量
        sz_precision: 数量精度（小数位数）

    Returns:
        (sz, margin_used) 合约张数和占用保证金
    """
    if equity <= 0 or entry_price <= 0:
        return 0.0, 0.0

    risk_amount = equity * risk_pct / 100.0
    stop_distance_pct = abs(entry_price - stop_loss) / entry_price

    if stop_distance_pct < 1e-10:
        return 0.0, 0.0

    # 仓位价值 = 风险金额 / 止损距离
    position_value = risk_amount / stop_distance_pct

    # 保证金 = 仓位价值 / 杠杆
    margin = position_value / leverage

    # 保证金上限检查
    max_margin = equity * margin_pct / 100.0
    if margin > max_margin:
        margin = max_margin
        position_value = margin * leverage
        logger.debug(f"Position capped by margin limit: "
                     f"margin={margin:.2f} (max={max_margin:.2f})")

    # 合约张数
    sz = position_value / (entry_price * contract_value)
    sz = round(sz, sz_precision)

    if sz < min_sz:
        logger.debug(f"Position size {sz} < min {min_sz}, skipping")
        return 0.0, 0.0

    return sz, margin


# ---------------------------------------------------------------------------
# 订单执行
# ---------------------------------------------------------------------------

def execute_signal(
    client: OKXClient,
    signal: Signal,
    equity: float,
    config: dict,
    instrument_info: Optional[dict] = None,
) -> Optional[str]:
    """执行交易信号 — 下单并设置止盈止损。

    Args:
        client: OKX 客户端
        signal: 交易信号
        equity: 当前权益
        config: 完整配置
        instrument_info: 合约信息 (ctVal, lotSz, minSz, tickSz 等)

    Returns:
        ord_id 或 None
    """
    risk_cfg = config["risk"]

    # ---- 确定合约参数 ----
    # USDT 本位永续合约面值因币种而异，默认 0.01（如 BTC ctVal=0.01）
    # 当 instrument_info 获取失败时，用 0.01 比 1.0 更安全
    ct_val = 0.01
    min_sz = 1.0
    sz_precision = 0

    if instrument_info:
        ct_val = float(instrument_info.get("ctVal", "0.01"))
        min_sz = float(instrument_info.get("minSz", "1"))
        # 从 lotSz 推断精度
        lot_sz = instrument_info.get("lotSz", "1")
        if "." in lot_sz:
            sz_precision = len(lot_sz.split(".")[1])
        else:
            sz_precision = 0

    # ---- 计算仓位 ----
    sz, margin = calculate_position_size(
        equity=equity,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        risk_pct=risk_cfg["risk_per_trade_pct"],
        leverage=signal.leverage,
        margin_pct=risk_cfg["margin_pct"],
        contract_value=ct_val,
        min_sz=min_sz,
        sz_precision=sz_precision,
    )

    if sz <= 0:
        logger.warning(f"Calculated sz=0 for {signal.inst_id}, skip")
        return None

    # ---- 确定 side 和 posSide ----
    if signal.position_side == "long":
        side = "buy"
        pos_side = "long"
    else:
        side = "sell"
        pos_side = "short"

    # ---- 格式化价格精度 ----
    # OKX 要求价格和数量为字符串
    tick_sz = "0.1"
    if instrument_info:
        tick_sz = instrument_info.get("tickSz", "0.1")

    px_precision = 0
    if "." in tick_sz:
        px_precision = len(tick_sz.split(".")[1])

    entry_px = f"{signal.entry_price:.{px_precision}f}"
    tp_px = f"{signal.take_profit:.{px_precision}f}"
    sl_px = f"{signal.stop_loss:.{px_precision}f}"
    sz_str = f"{sz:.{sz_precision}f}"

    logger.info(
        f"\n{'='*60}\n"
        f"  SIGNAL: {signal.inst_id} {signal.position_side.upper()}\n"
        f"  Entry:  {entry_px}  |  TP: {tp_px}  |  SL: {sl_px}\n"
        f"  Size:   {sz_str}  |  Leverage: {signal.leverage}x  |  Margin: {margin:.2f}\n"
        f"  Score:  {signal.score:.2f}  |  {signal.reason}\n"
        f"{'='*60}"
    )

    # ---- 设置杠杆 ----
    # OKX 合约交易必须先设置杠杆，否则使用默认 1x
    leverage_ok = client.set_leverage(
        inst_id=signal.inst_id,
        lever=signal.leverage,
        mgn_mode=risk_cfg["margin_mode"],
    )
    if not leverage_ok:
        logger.warning(f"Failed to set leverage for {signal.inst_id}, "
                       f"using default (may cause margin mismatch)")

    # ---- 下单 ----
    ord_id = client.place_order(
        inst_id=signal.inst_id,
        side=side,
        pos_side=pos_side,
        sz=sz_str,
        px=entry_px,
        ord_type="limit",
        td_mode=risk_cfg["margin_mode"],
        tp_trigger=tp_px,
        tp_price="-1",      # 市价止盈
        sl_trigger=sl_px,
        sl_price="-1",      # 市价止损
    )

    return ord_id


# ---------------------------------------------------------------------------
# 持仓监控
# ---------------------------------------------------------------------------

def monitor_positions(
    client: OKXClient,
    state_manager: StateManager,
    config: dict,
) -> Dict[str, dict]:
    """监控当前持仓状态。

    检查：
      - 持仓盈亏
      - 止盈止损是否触发
      - 更新状态追踪

    Returns:
        {inst_id: position_info}
    """
    positions = client.get_positions()
    risk_cfg = config["risk"]
    result = {}

    for pos in positions:
        inst_id = pos.get("instId", "")
        pos_side = pos.get("posSide", "")
        pos_sz = float(pos.get("pos", "0"))
        avg_px = float(pos.get("avgPx", "0"))
        mark_px = float(pos.get("markPx", "0"))
        upl = float(pos.get("upl", "0"))          # 未实现盈亏
        upl_ratio = float(pos.get("uplRatio", "0"))
        margin = float(pos.get("margin", "0"))
        lever = float(pos.get("lever", "0"))

        result[inst_id] = {
            "pos_side": pos_side,
            "size": pos_sz,
            "avg_px": avg_px,
            "mark_px": mark_px,
            "upl": upl,
            "upl_ratio_pct": upl_ratio * 100,
            "margin": margin,
            "leverage": lever,
        }

        if abs(pos_sz) > 0:
            logger.info(
                f"[POS] {inst_id} {pos_side} | "
                f"Sz={pos_sz} | Avg={avg_px:.2f} | Mark={mark_px:.2f} | "
                f"PnL={upl:.2f} ({upl_ratio*100:.2f}%) | "
                f"Margin={margin:.2f} | Lever={lever}x"
            )

    # 获取当前权益
    equity = client.get_total_equity() or 0.0

    # 检查平仓变动，更新胜率统计
    prev_active = len(state_manager.state.active_signals)
    current_active = len([p for p in positions if abs(float(p.get("pos", "0"))) > 0])

    if prev_active > 0 and current_active < prev_active:
        # 有仓位平掉了，通过权益变化近似判断
        closed_count = prev_active - current_active
        if equity and state_manager.state.last_equity > 0:
            pnl_change = equity - state_manager.state.last_equity
            if pnl_change > 0:
                state_manager.state.winning_trades += closed_count
            elif pnl_change < 0:
                state_manager.state.losing_trades += closed_count
        state_manager.state.positions_closed += closed_count

    state_manager.state.active_signals = {
        p.get("instId", ""): p for p in positions
        if abs(float(p.get("pos", "0"))) > 0
    }

    # 每日亏损检查 — 使用已实现亏损（权益变化），而非未实现盈亏
    # 仅在权益下降时累加亏损
    if equity and state_manager.state.last_equity > 0:
        realized_pnl = equity - state_manager.state.last_equity
        if realized_pnl < 0:
            state_manager.state.daily_loss += realized_pnl

    state_manager.state.last_equity = equity

    max_daily_loss = equity * risk_cfg["max_daily_loss_pct"] / 100.0 if equity else 0
    if abs(state_manager.state.daily_loss) >= max_daily_loss and max_daily_loss > 0:
        logger.warning(
            f"!!! DAILY LOSS LIMIT REACHED !!! "
            f"Loss: {state_manager.state.daily_loss:.2f} >= {max_daily_loss:.2f}"
        )

    return result


# ---------------------------------------------------------------------------
# 挂单管理
# ---------------------------------------------------------------------------

def manage_pending_orders(
    client: OKXClient,
    current_price: float,
    signal: Optional[Signal] = None,
    stale_minutes: int = 60,
) -> int:
    """管理挂单：取消过期订单，避免重复下单。

    Args:
        client: OKX 客户端
        current_price: 当前市场价格（用于价格偏离检测）
        signal: 如果有新信号，检查是否已存在相同方向的挂单
        stale_minutes: 超过此时间的挂单视为过期

    Returns:
        取消的订单数
    """
    orders = client.get_pending_orders()
    cancelled = 0
    now = time.time() * 1000  # ms

    for order in orders:
        ord_id = order.get("ordId", "")
        inst_id = order.get("instId", "")
        pos_side = order.get("posSide", "")
        px = float(order.get("px", "0"))
        create_time = int(order.get("cTime", "0"))

        # 检查是否过期
        age_minutes = (now - create_time) / 60000.0 if create_time > 0 else 0
        if age_minutes > stale_minutes:
            logger.info(f"Cancelling stale order {ord_id} ({inst_id}, age={age_minutes:.0f}m)")
            if client.cancel_order(inst_id, ord_id):
                cancelled += 1
            continue

        # 如果有新信号，检查冲突
        if signal and inst_id == signal.inst_id and pos_side == signal.position_side:
            logger.info(f"Similar pending order exists for {inst_id}, skip new signal")
            return cancelled

        # 如果价格远离挂单价超过阈值，取消
        if px > 0 and current_price > 0:
            deviation = abs(current_price - px) / px
            if deviation > 0.05:  # 5% 偏离
                logger.info(f"Cancelling deviated order {ord_id} ({inst_id}, "
                            f"deviation={deviation*100:.2f}%)")
                if client.cancel_order(inst_id, ord_id):
                    cancelled += 1

    return cancelled


# ---------------------------------------------------------------------------
# 获取合约标的列表
# ---------------------------------------------------------------------------

def get_tradable_coins(
    client: OKXClient,
    config: dict,
) -> List[dict]:
    """获取可交易的 USDT 本位永续合约列表，按成交量排序。

    Returns:
        [{"instId": "BTC-USDT-SWAP", "last": 50000, "vol24h": 1e9, ...}, ...]
    """
    tickers = client.get_tickers(inst_type="SWAP")
    limit = config["agent"].get("coin_scan_limit", 100)

    # 只保留 USDT 本位永续合约
    usdt_swaps = []
    for t in tickers:
        inst_id = t.get("instId", "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue

        vol24h = float(t.get("volCcy24h", "0"))
        if vol24h < config["strategy"]["min_volume_24h_usd"]:
            continue

        usdt_swaps.append({
            "instId": inst_id,
            "last": float(t.get("last", "0")),
            "vol24h": vol24h,
            "bidPx": float(t.get("bidPx", "0")),
            "askPx": float(t.get("askPx", "0")),
            "high24h": float(t.get("high24h", "0")),
            "low24h": float(t.get("low24h", "0")),
        })

    # 按成交量降序
    usdt_swaps.sort(key=lambda x: x["vol24h"], reverse=True)
    return usdt_swaps[:limit]


# ---------------------------------------------------------------------------
# 价差计算
# ---------------------------------------------------------------------------

def calculate_spread(bid: float, ask: float) -> float:
    """计算买卖价差百分比。"""
    if ask <= 0 or bid <= 0:
        return 0.0
    return (ask - bid) / ask * 100.0


# ---------------------------------------------------------------------------
# 打印统计摘要
# ---------------------------------------------------------------------------

def print_summary(state: AgentState, equity: float):
    """打印运行统计。"""
    if state.initial_equity <= 0:
        return

    total_pnl_pct = (equity - state.initial_equity) / state.initial_equity * 100
    total_trades = state.winning_trades + state.losing_trades
    win_rate = state.winning_trades / total_trades * 100 if total_trades > 0 else 0

    logger.info(
        f"\n{'='*50}\n"
        f"  SUMMARY\n"
        f"  Equity:     {equity:.2f} USDT\n"
        f"  PnL:        {equity - state.initial_equity:+.2f} ({total_pnl_pct:+.2f}%)\n"
        f"  Trades:     {total_trades} (W:{state.winning_trades} L:{state.losing_trades})\n"
        f"  Win Rate:   {win_rate:.1f}%\n"
        f"  Withdrawals:{state.withdrawal_count}\n"
        f"{'='*50}"
    )