# 故障排除 — OKX FVG Agent v3.3

> 配套文档：[使用方法](USAGE.md) · [更新手册](UPDATE_MANUAL.md)

## 网络与代理

### 症状：`httpx.ConnectError` / `EOF occurred in violation of protocol (_ssl.c:1007)`

- 代理节点抖动（常见于 Clash 端口 7897）。Agent 内置重试会自动恢复，偶发可忽略
- 持续出现：确认 `config.json → okx.proxy` 指向的代理进程存活，浏览器能否打开 okx.com
- **WebSocket 不走系统代理**，必须在 `okx.proxy` 显式配置

### 症状：启动即 `Cannot get account equity`

- 权益查询失败重试 5 次后仍继续启动（7x24 设计），主循环会再重试；持续失败检查代理/密钥

### 症状：`429 Too Many Requests` / OKX code `50011/50013`

- 实盘模式已内置全局令牌桶（默认 10 QPS）主动削峰 + 被动退避重试
- 仍触发：调低 `okx.rate_limit.max_qps`（如 6），或增大 coin_tracker 的 `batch_cooldown_seconds`

## Python 环境

### 症状：`python` 命令报"拒绝访问"/"not a valid application"

- Windows Store 版 shim 损坏。使用完整路径：
  `C:\Users\<用户>\AppData\Local\Programs\Python\Python310\python.exe`
- 装包：`python.exe -m pip install ...`（不要复制 python.exe，缺 DLL）

### 症状：cmd 里 `cd d:\123` 不生效

- 跨盘符必须 `cd /d d:\123`

## 交易执行

### 症状：`[Liquidity] ... 订单簿过薄，拒绝开仓`

- 名义仓位超过对侧前 10 档深度的 5%——崩盘薄书保护，属预期行为
- 确需交易主流币仍被拒：检查仓位是否过大，或适度放宽 `risk.order_book_depth.max_notional_depth_ratio`（不建议 >0.10）

### 症状：`sCode=51277/51279`（TP 触发价方向错误）

- 异常波动币实时价已越过 TP。系统自动降级为仅挂 SL 的 conditional 单，TP 由 trailing 后续补挂——无需干预

### 症状：日志反复出现"联动撤销孤儿保护单"

- 限价单超时被撤后清理残留保护单，属正常自愈；频繁出现说明挂单距离过深，检查 `FillFunnel` 告警

### 症状：`[FillFunnel] 成单率 x% < 20%`

- 挂单长期不成交（深挂空转）。收窄 `strategy.liquidity_extension_pct` 或降低 `entry_distance_atr_mult`

### 症状：`get_positions failed (fail-closed)，跳过本轮`

- 行情/持仓查询失败时**不会**被当作"无持仓"（防超限开仓），下一轮自动恢复
- 持续出现按"网络与代理"排查

### 症状：`[LiqCheck] 降杠杆止损优先: 50x → 28x`

- 满杠杆下止损距离 ≥ 爆仓安全距离，自动降杠杆使止损先于爆仓触发——预期行为，不是 bug

## 风控与状态

### 症状：`[RollingKelly] 风险上限 30.0% → x%`

- 滚动 Kelly 在压缩单笔风险（负边压到 1%，正边按 1/4 或 1/2 Kelly）
- 想看统计依据：日志同行的 f\*/样本/胜率/档位
- **关闭方法**：`risk.rolling_kelly.enabled=false`（不建议，见更新手册的蒙特卡洛结论）

### 症状：`[Breaker] ... 禁止开仓`

- 统一断路器命中：日亏限额 / 每日交易上限 / 连亏暂停 / 回撤断路器之一
- 日亏与每日笔数跨 UTC 日自动重置；连亏暂停按 `optimization.loss_pause_hours` 到期

### 症状：状态文件疑似损坏

- 写入为原子操作（tmp + os.replace），正常不会损坏
- 恢复：删除 `agent_state.json` 即完全重置（滚动 Kelly 样本清零重新积累）

## 测试

### 症状：`fixture 'xxx' not found` 收集错误

- pytest 会把 `test_*` 函数的参数当 fixture。项目内 `test_entry_logic.py` 已修复此模式
- 新写测试：需要实参的辅助函数**不要**用 `test_` 前缀；入口函数写成无参或用 pytest fixture

### 症状：`verify_kelly_monte_carlo.py` 抛 `快速Kelly与生产函数不一致`

- 交叉核对机制在保护你：说明 `hyperopt.rolling_kelly_risk_pct` 与模拟实现发生了漂移
- 先跑 `python -m pytest test_rolling_kelly.py -q` 确认生产函数测试仍绿，再对比两边的窗口/EWMA 语义

### 症状：git 终端中文乱码 / PSReadLine 报错

- PowerShell 控制台 GBK/UTF-8 渲染问题，仅影响显示，不影响提交内容
- 查看提交信息用 `git show HEAD --stat` 或在 GitHub 网页端

## 日志维护

```powershell
# 保留最近 1000 行
Get-Content agent.log -Tail 1000 | Set-Content agent.log -Encoding UTF8

# 查看关键事件
Select-String -Path agent.log -Pattern "\[RollingKelly\]|\[Liquidity\]|\[MarketGuard\]|\[Reconcile\]"
```

## 仍然解决不了

1. 跑全量测试确认基线：见 [USAGE.md 第 7 节](USAGE.md#7-测试与验证)
2. 收集：完整报错栈 + 前后 50 行日志 + `config.json` 中**非密钥**的段
3. 密钥问题只在 OKX 后台重建密钥解决（日志/截图先打码）
