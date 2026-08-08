# 更新日志

本项目维护变更记录，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [2026-08-08] FVG Hunter 硬门禁（只吃确定性极端行情）

### 策略 Alpha 升级（源自 4 小时横盘监控复盘）
ADA 入场后 4h 横盘 0.1987~0.2001（±0.35%），触发不了任何退出逻辑，空耗保证金。
结论：横盘折磨行情不该入场，FVG Hunter 只吃确定性极端行情（上涨下跌都很大的）。

### 实现
- `_extreme_move_reject_reason`：per-symbol 硬门禁，`generate_signal` 过滤器链第 0.75 位
  - ADX(14) ≥ 25：趋势强度（Wilder 1978 行业标准，与 `adx_trend_threshold` 同值，
    趋势市 FVG 回补有效 / 震荡市假回补风险高）
  - ATR(14)/现价 ≥ 2%：单根 K 线平均振幅（"上涨下跌都很大"）
  - 任一不满足 → `[ExtremeMove] 横盘/低波动拒绝`，信号直接否决
  - K 线不足自动 fail-open（防新币阻塞，与 ATRGrade 同策略）
- 参数可配置：`strategy.extreme_move_min_adx` / `extreme_move_min_atr_pct`（0=关闭）

### 验证
- 新增回归 `test_extreme_move_gate`：横盘(低ADX)拒绝 / 强趋势+高波动放行 /
  数据不足放行 / 门禁关闭放行
- 回归 8/8 + `tests/` 48/48 全过

## [2026-08-08] 纸面监控批次修复（满杠杆测试 + 4 项监控发现）

源自 30 USDT 纸面监控会话（1x 挡位 → 满杠杆 30% 仓位）的修复。

### 满杠杆测试配置
- `max_position_leverage` 1→5：恢复"满杠杆"档位（杠杆由 25/止损距离% 预算公式给出，5x 专业封顶）
- `risk_per_trade_pct` 1→2：允许 30% 保证金仓位在常见止损距离（SL≤6.6%）下足额使用，
  仍在社区铁律 1-2% 上限内，`enforce_risk_cap` 机制保持生效
- 其余 70% 余额自动作为保证金缓冲（isolated 模式可用保证金）

### 监控发现修复（4 项）
- **纸面移动止损缺失**：paper_trading 新增 `_update_trailing`（ATR(14) 动态激活 0.5x/
  追踪 0.75x，无 ATR 回退固定百分比），纸面 SL 与实盘 trailing 同步收紧、只松不紧、
  状态持久化于 pos.extra；与实盘 `optimization.TrailingStop` 同参同逻辑
- **纸面重复执行噪音**：主循环在 execute_signal 前用 `paper_engine.has_position()`
  源头去重 — dry-run 下 `get_pending_orders` 恒空导致同一信号每轮重复走假单路径，
  `positions_opened` 虚增
- **纸面杠杆口径**（续）：纸面开仓应用 `max_position_leverage` 封顶后计算仓位（上批
  已修，本批补回归测试）
- **体制切换日志无币种**：`EnhancedRegimeDetector` 增加 `symbol` 参数，切换日志
  前缀 `[币种]`（per-symbol 实例可区分）；coin_tracker/主循环创建点透传 inst_id

### 验证
- 回归测试 7/7 通过（新增 `test_paper_trailing_moves_sl`：ATR 激活后 SL 上移且不放松）
- `tests/` 48 项单测全部通过
- 变更文件：agent.py / paper_trading.py / regime_detector.py / coin_tracker.py / config.json / test_production_fixes.py

## [2026-08-07] 生产级审计修复批次（P0 全部 + P1 全部 + 关键 P2）

基于完整生产审计（覆盖执行层/资金管理/状态恢复/策略/回测）落地的修复。
**审计背景**：确认 3 个结构性爆仓级缺口（无强平距离校验、API 故障被静默当作"无持仓"、
平仓记账在重启后断裂），全部在此批次修复。

### P0 级（可致爆仓/资金损失）
- **P0-A 强平距离校验（新增）**：`execute_signal` 开仓前硬校验
  `|entry−SL| < 强平距离×安全系数`。强平距离 ≈ `1/杠杆 − MMR`（MMR 从
  OKX position-tiers 档位获取，失败用保守默认 0.5%）。50x+3% 止损会被直接拒单。
  此前全系统无强平价计算，跳空时止损单先于强平失效。
- **P0-B API 故障 fail-closed**：`get_positions`/`get_pending_orders` 失败返回 `None`
  （原静默返回 `[]`），调用方区分"查询失败"与"确无持仓"。根因是 python-okx SDK
  底层为 httpx，其异常**不是**内置 `ConnectionError/OSError` 子类，旧重抛守卫永不触发。
  已实测复核确认并全链路修复（monitor_positions 抛 `OKXQueryError`、risk_gate 敞口检查
  拦截、close_position 核验、挂单管理跳过）。
- **P0-C 交易端点退避重试**：`_call_sdk_retry` 助手覆盖 place_order /
  place_algo_order / close_position / cancel_order / cancel_algo_order / set_leverage，
  对 httpx 网络异常与限流码（50000/50011…）做 0.5s/1.5s/4s×3 重试。
- **P0-D 平仓记账持久化**：`_pending_close` 拆出可序列化元数据
  `pending_close_meta` 持久化，重启后重建续跑平仓确认（已实现盈亏/日亏限额不再丢）；
  崩溃-重启路径补 `state_manager.save()`；两处无 try 的 `_refresh_positions()` 改为受保护刷新。
- **P0-E 保护单登记校验**：trailing 登记已有保护单前校验 `posSide` 一致 +
  挂单时间晚于开仓时间，过期孤儿单自动撤销重挂（此前重启后可能误登记旧保护单
  → 新仓裸奔或错误价位触发）。

### P1 级
- 杠杆封顶 `max_position_leverage` 现在同步作用于 `set_leverage`（此前仅作用仓位计算）
- 限价平仓 ≥90% 部分成交后残仓市价兜底（此前残仓滞留）
- 平仓确认后联动撤销该币残留 oco/conditional 保护单（防止孤儿单）
- 全链路持仓判定统一 `abs()`（兼容 cross 模式空头 pos 为负）
- 换仓/反手/平仓后开新仓路径统一查 `_risk_breaker_triggered`（日亏限额 + 自适应暂停，
  此前换仓在 risk_gate 之前执行可绕过）
- 金字塔加仓前查聚合敞口上限（`_exposure_cap_allows_add`，fail-closed）
- `get_positions` 移除 SDK 不支持的 `mgnMode` 透传（改为客户端侧过滤）

### P2 级（关键项）
- `clOrdId` 方法内生成一次、重试复用（幂等去重真正生效）
- `market_guard.reduce_position_factor`（WARNING 减半仓）接线生效
- 绝对回撤断路器补 `pause_until`（24h 冷却，日志与恢复语义一致）
- 弱信号共振审核异常改 fail-closed（拒绝开仓）
- 纸面模式平仓确认后清理 active_signals 条目；`paper_state.json` 补 fsync + .bak 恢复 +
  数值字段强转；记忆文件改原子写

### 验证
- 新增回归测试 `test_production_fixes.py`（5 项：强平校验拒/放/档位MMR、pending_close
  元数据往返、monitor_positions fail-closed）全部通过
- `tests/` 目录 48 项单测全部通过；6 个改动文件 VS Code 诊断 0 错误
- 注：`test_weak_gate.py`/`test_gala_lessons_gate.py` 为改动前已存在的测试夹具问题
  （stash 对照确认），非本次回归

## [2026-08-07] WebSocket 行情缓存修复与心跳协议变更

### 变更原因（心跳协议）

OKX WebSocket 要求客户端定期发送应用层心跳以维持连接。原实现**未发送任何心跳**，
导致服务器约每 30 秒强制断开连接（实测日志：`agent.log.2026-08-01` 中每 ~30s 出现
`WS recv error: Connection to remote host was lost.` 并触发重连）。

首次修复尝试使用 `{"op":"ping"}` 格式，经 **2026-08-07 实网验证被 OKX 服务器拒绝**，
返回 `60012 Illegal request: {"op": "ping"}`。查证 OKX 官方行为后确认：
应用层心跳格式为**纯文本 `"ping"`**，服务器应答纯文本 `"pong"`（直连协议验证通过）。

最终方案：每 25 秒发送一次纯文本 `"ping"`（`PING_INTERVAL_SEC = 25.0`），实网 70 秒
连续运行验证连接持续存活、无 30s 级断线、行情持续推送。

### 修复内容

#### ws_ticker_cache.py
- **心跳保活**：新增每 25s 发送纯文本 `"ping"`，消除周期性断线重连
- **连接标志**：`connected` 仅在订阅消息真正发送成功后置位；仅携带 `arg` 的订阅错误事件
  才清除该标志，无 `arg` 的良性错误（如心跳格式被拒 60012）不再误清
- **断开检测**：`recv()` 返回空（服务端主动关闭）时立即退出并重连，不再空转一个超时周期
- **错误诊断**：`{"event":"error"}`（如 60018）显式记录日志，此前被静默吞掉
- **数据完整性**：缓存补充 `open24h` 字段（`_evaluate_market_guard` 计算 BTC 24h 收益依赖，
  缺失时 WS 路径下收益恒为 0）
- **并发安全**：`get()` / `get_top_by_volume()` 返回拷贝，防止调用方污染内部缓存
- **新鲜度语义**：仅真实行情数据到达才刷新 `_last_update`；空 data / 事件消息不再误刷新
- **过期机制**：`get(inst_id, max_age_sec=None)` 可选参数，基于交易所 `ts` 判断单币种
  陈旧数据（停牌/退市币不再永久残留）
- **防御加固**：`websocket` 未安装时线程入口守卫；非 JSON 控制消息（纯文本 `"pong"`）
  静默忽略，不再每次心跳记录解析错误

#### agent.py
- **纸面行情键错位修复**：`_paper_market_data` 原使用 `bid_px` 读取 WS 缓存，实际字段名为
  `bidPx`，导致 WS 买价永远取不到、纸面价格优先级降级；已改为 `bidPx`

### 验证
- 单元测试 15/15 通过（拷贝隔离、过期判定、error/pong 处理、新鲜度语义等）
- 实网 70s 保活测试通过：`connected=True` 全程保持、无 60012、无 30s 级断线
- 心跳协议定向验证：发送 `"ping"` 收到 `"pong"`（纯文本）
- 两文件 VS Code 语言诊断 0 错误
