# 使用方法 — FVG KILLER（公允价值缺口杀手）v3.3

> 配套文档：[README.md](../README.md)（项目概览） · [故障排除](TROUBLESHOOTING.md) · [更新手册](UPDATE_MANUAL.md)

## 1. 环境要求

| 项 | 要求 |
|----|------|
| Python | **3.10+**（推荐 3.10.11，numpy/pandas/scipy/okx/websocket 全兼容） |
| 系统 | Windows / Linux / macOS（启动脚本 `start_agent.bat` 仅 Windows） |
| 网络 | 可直连或代理访问 `www.okx.com`（国内需 HTTP 代理） |

> Windows 注意：Microsoft Store 版 Python 的 `AppData\Local\Python\bin` shim 可能损坏
> （报"拒绝访问"/"not a valid application"），请使用完整安装路径，例如
> `C:\Users\<用户>\AppData\Local\Programs\Python\Python310\python.exe`。

## 2. 安装

```bash
cd okx_fvg_agent
pip install -r requirements.txt

# python-okx 单独安装失败时
pip install "python-okx>=0.4.0"
```

## 3. 配置

```bash
# 复制模板（config.json 含密钥，已被 .gitignore 排除，绝不入库）
cp config.example.json config.json
```

必填项（`okx` 段）：

```json
{
  "okx": {
    "api_key": "你的API_KEY",
    "api_secret": "你的API_SECRET",
    "passphrase": "你的PASSPHRASE",
    "proxy": "http://127.0.0.1:7897"
  }
}
```

- **API 权限**：仅需 交易(Trade) + 读取(Read)，**不要勾选提现(Withdraw)**
- **代理**：国内必填；WebSocket 不读系统代理，必须显式配置才会走代理
- **模拟盘**：OKX App → 模拟交易 → API 单独创建密钥，`demo: true` + 填 `demo_*` 三项

## 4. 运行模式

| 模式 | 配置 | 行为 |
|------|------|------|
| **纸面模式**（推荐先用） | `agent.dry_run=true` + `paper.enabled=true` | 虚拟余额 + 实时行情，模拟限价回补成交/止盈止损/爆仓封顶，**绝不下真实单** |
| **演练模式** | `agent.dry_run=true`（无 paper） | 只输出信号日志，不模拟持仓 |
| **实盘模式** | `agent.dry_run=false` | 真实下单。限流令牌桶/订单簿流动性检查/启动三方对账自动生效 |

启动方式：

```bash
# Windows 一键
start_agent.bat

# 通用命令行
python agent.py

# 常用参数
python agent.py --演练 --单轮        # 演练一轮退出
python agent.py --演练 --轮次 10     # 演练 10 轮
python agent.py --配置文件 my.json   # 指定配置
```

## 5. 研判挡位（agent.aggressiveness）

| 挡位 | 名称 | 频率 | 说明 |
|------|------|------|------|
| 1 | 激进 | 每天必找一币 | 大幅降阈值，无信号时强制选最优；ML 阈值放宽 |
| 2 | 均衡 | 2-3 天一笔 | 适中 |
| 3 | 保守 | 低频 | 严格门禁（默认） |

## 6. 日志监控关键字

盯盘时按这些前缀过滤 `agent.log`：

| 关键字 | 含义 |
|--------|------|
| `[RollingKelly]` | 滚动 Kelly 压低了单笔风险上限（含 f\*/样本/胜率/档位/ewmaλ） |
| `[VolTarget]` | 波动率目标缩放了保证金（ATR% 偏离目标） |
| `[Liquidity]` | 订单簿过薄拒单（名义仓/对侧深度超 5%） |
| `[RateLimit]` | 全局令牌桶启用（实盘启动时一行） |
| `[Reconcile]` | 启动三方对账结果（清理残留/登记保护单/孤儿单告警） |
| `[FundingReconcile]` | 资金费率实际 vs 估算对账偏差 |
| `[Slippage]` | 平仓实际滑点回填 |
| `[MarketGuard]` | 市场熔断状态（CRISIS 禁开仓 / WARNING 减仓） |
| `[FillFunnel]` | 成单率漏斗（挂单空转告警） |
| `[Paper]` | 纸面引擎成交/平仓/撤单事件 |
| `[TS]` | Trailing Stop 保护单状态 |
| `[LiqCheck]` | 止损-爆仓距离校验（降杠杆止损优先） |

## 7. 测试与验证

```bash
# 全量单元测试（169 项，~11s）
python -m pytest test_live_guards.py test_paper_slippage.py test_rolling_kelly.py \
  test_production_fixes.py test_extreme_scenario.py test_bugfixes_mock.py \
  test_switch_guards.py test_winrate_fixes.py test_atr_stop.py test_entry_logic.py \
  test_gala_lessons_gate.py test_paper_assist_rr.py test_roi_tp_floor.py \
  test_weak_gate.py tests/ -q

# 滚动 Kelly 档位切换验证（60 笔逐笔喂入）
python verify_rolling_kelly_transition.py

# 蒙特卡洛（300 路径 × 1000 笔，含与生产函数交叉核对）
python verify_kelly_monte_carlo.py                 # 静态边 A/B 场景
python verify_kelly_monte_carlo.py --drift 0.5 0.4 # 边漂移: 衰减 20%
python verify_kelly_monte_carlo.py --drift 0.5 0.25 # 边消失: 跌破保本点
python verify_kelly_monte_carlo.py --curves          # 5 条样本资金曲线追踪
```

## 8. 状态文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `agent_state.json` | 主状态 | 权益/累计盈亏/recent_pnl(滚动Kelly输入)/daily_trades/滑点样本 |
| `paper_state.json` | 纸面状态 | 虚拟余额/持仓/历史交易 |
| `night_watch_state.json` | 夜间守望 | — |
| `agent.log` | 日志 | 建议定期清理（见故障排除） |

全部为原子写入（先写 `.tmp` 再 `os.replace`），意外断电不会产生半截文件。
删除状态文件 = 完全重置（滚动 Kelly 样本清零、纸面余额回到初始值）。

## 9. 切换实盘检查单

1. `docs/UPDATE_MANUAL.md` 通读一遍（行为变化清单）
2. 纸面模式连续跑 ≥2 周且滚动期望值为正
3. `config.json`：`dry_run=false`，`max_daily_trades` 设 6~10
4. 首次实盘建议最小资金（≥ 最小下单量的 50 倍）
5. 启动后确认日志出现 `[RateLimit] 全局令牌桶已启用` 与 `[Reconcile]`
