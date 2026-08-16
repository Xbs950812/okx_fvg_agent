# 更新手册 — FVG KILLER v3.2 → v3.3

> 2026-08-14~15 增强：**实盘守卫体系 + 滚动 Kelly 翻倍协议**
> 2026-08-16 追加：**盈利分成模块（Royalty）— 开源分成协议**
> 配套文档：[使用方法](USAGE.md) · [故障排除](TROUBLESHOOTING.md)

## 1. 变更总览

| 类别 | 变更 | 生效时机 |
|------|------|---------|
| 守卫 | 全局 API 限流令牌桶（10 QPS/突发 20） | **仅实盘**（dry_run 自动关闭） |
| 守卫 | 开仓前订单簿流动性检查（5% 阈值拒单） | **仅实盘**（execute_signal 下单段） |
| 守卫 | 波动率目标仓位（ATR% 缩放 margin_pct ∈[0.5,1.5]） | 纸面 + 实盘 |
| 守卫 | 每日最大交易次数（跨 UTC 日重置） | 全模式（默认 0=不限制） |
| 守卫 | 实际滑点回填（实盘平仓取 avgPx 对账） | **仅实盘** |
| 守卫 | 启动三方对账 + 资金费率 bills 对账 | **仅实盘** |
| 协议 | 滚动分数 Kelly 风险上限（探索→利用） | 全模式（样本 ≥10 笔起） |
| 协议 | EWMA 输入端平滑（λ=0.97，防窗口边界跳变） | 全模式 |
| 分成 | 盈利分成 Royalty（盈利 10% 计池 → 20 USDT 自动 TRC20 提现） | **仅实盘**（paper/dry_run 只记日志） |
| 修复 | okx_client.get_bills 调用不存在的 SDK 方法（账单恒空） | 影响 trade_analyzer |
| 测试 | 197 项单测（含 test_royalty.py 28 项）+ 蒙特卡洛验证资产 | — |

## 1a. 盈利分成模块（2026-08-16 追加）

**新文件**：`royalty.py`（开源核心组件）+ `royalty_state.json`（运行时状态，已入 .gitignore）

**行为**：
- 平仓确认点（`[Close]` 日志处）对**已实现盈利**（pnl>0）计 `rate_pct`（默认 10%）入池
- 池 ≥ `min_withdraw_usdt`（默认 20）且池-手续费 ≥ 交易所最小提现额时自动提现：
  查费率（`/asset/currencies`）→ trading→funding 划转补缺口 → 链上提现
  （`/asset/withdrawal`，含 SDK 未暴露的 fee 参数，走底层请求）→ 记 wdId 流水
- **权限降级**：API key 无 Withdraw 权限（code 50101 等）→ 小时级冷却重试 + 日志提醒，交易不受任何影响
- **模拟模式安全**：paper/dry_run 绝不真实转账、不写池状态（防虚拟盈利污染实盘分成池）

**配置迁移**：`config.json` 无 `royalty` 段时代码用硬编码默认值（enabled=true、
内置作者钱包、10%、20 USDT、300s 重试）。`config.example.json` 已含完整段。

**用户须知**：自动提现需在 OKX 后台开启 API 的 Withdraw 权限并将收款地址加入
提现白名单；不开启则仅记账累积。关闭分成或改地址需作者商业授权（见 README）。

## 2. 滚动 Kelly 翻倍协议（核心行为变化）

**机制**：从 `agent_state.json → recent_pnl`（最近 100 笔已实现盈亏）滚动计算
分数 Kelly，作为单笔风险上限：

```
eff_risk = min(自适应风险, 滚动Kelly上限)
```

| decisive 样本数（pnl≠0） | 档位 | 上限 |
|--------------------------|------|------|
| < 10 | 不约束 | 维持现有 eff_risk（30%） |
| 10 ~ 49 | 探索档 | f\* × 1/4 |
| ≥ 50 | 利用档 | f\* × 1/2 |

- **负边**（f\*≤0）：上限落到 1% 下限（暂停由期望值门禁负责，仓位不会归零）
- **EWMA λ=0.97**：有效记忆 ≈33 笔；消除窗口边界驱逐跳变、加快体制切换响应；`ewma_lambda≤0` 回退平权窗口

**蒙特卡洛结论**（300 路径 × 1000 笔，`verify_kelly_monte_carlo.py`）：

- 边衰减 0.5→0.4：回撤 83.1% vs 固定 30% 的 100%（中位）
- 边消失 0.5→0.25：滚动版 +27 次翻倍存活，固定 30% 湮灭（2^−21）
- 离散档位跳变（第 50 笔 1/4→1/2）：对长期统计**不可见**（maxDD 差 0.0%）

**代价认知**：边稳定为正时分数 Kelly 牺牲约 20% 增长换回撤浅 15~20 个百分点。
这是设计目标（小资金翻倍协议 = 探索攒样本 → 利用爬风险），不是 bug。

## 3. 配置迁移

`config.example.json` 已含全部新键。旧 `config.json` **不补也能跑**（代码默认值
与模板一致），但建议显式补齐以下三段便于审计（完整字段见模板）：

```json
{
  "okx": { "rate_limit": { "enabled": true, "max_qps": 10, "burst_capacity": 20 } },
  "risk": {
    "max_daily_trades": 0,
    "rolling_kelly": { "enabled": true, "window": 100, "min_samples": 10,
                        "sample_full_kelly": 50, "min_risk_pct": 1.0,
                        "max_risk_pct": 0, "ewma_lambda": 0.97 },
    "order_book_depth": { "enabled": true, "max_notional_depth_ratio": 0.05, "depth_levels": 10 },
    "vol_targeting": { "enabled": true, "target_vol_pct": 2.0, "min_scale": 0.5, "max_scale": 1.5 }
  },
  "agent": { "reconciliation": { "startup_enabled": true, "funding_fee_enabled": true,
                                  "funding_fee_interval_rounds": 6 } }
}
```

约定：`max_daily_trades=0` / `max_risk_pct=0` = 不限制/用基准值（与
`max_position_leverage` 惯例一致）。

## 4. 升级步骤

```bash
git pull                       # 或解压新版本覆盖
pip install -r requirements.txt  # 依赖无变化，幂等
python -m pytest test_rolling_kelly.py test_live_guards.py -q   # 快速自检
python agent.py --演练 --单轮    # 演练确认启动无报错
```

**旧状态文件完全兼容**：`recent_pnl`/`daily_trades`/`slippage_samples` 为新增
字段，缺失时从零开始积累，不影响已有权益/持仓记录。

## 5. 切实盘时"醒来"的三件套

以下功能纸面模式自动关闭，`dry_run=false` 后无需改配置即生效：

1. **限流令牌桶** — 启动日志出现 `[RateLimit] 全局令牌桶已启用`
2. **订单簿流动性检查** — 薄书开仓被拒时日志出现 `[Liquidity]`
3. **启动三方对账** — 启动日志出现 `[Reconcile]`（清理本地残留/登记保护单/孤儿单告警）

## 6. 回滚

```bash
git log --oneline                    # 找到 v3.3 合并提交的前一个提交
git checkout <v3.2_commit> -- .      # 恢复代码（保留工作区配置与状态文件）
python -m pytest test_production_fixes.py -q
```

单独禁用新守卫（不回滚代码）：

| 禁用项 | 配置 |
|--------|------|
| 滚动 Kelly | `risk.rolling_kelly.enabled=false` |
| 订单簿检查 | `risk.order_book_depth.enabled=false` |
| 波动率目标 | `risk.vol_targeting.enabled=false` |
| 限流令牌桶 | `okx.rate_limit.enabled=false` |
| 启动对账 | `agent.reconciliation.startup_enabled=false` |

## 7. 验证资产清单

| 资产 | 数量 | 覆盖 |
|------|------|------|
| `test_rolling_kelly.py` | 41 | Kelly 数学/档位契约/EWMA 语义/窗口等价 |
| `test_live_guards.py` | 21 | 令牌桶节流计时/订单簿拒单/execute_signal 闸门 |
| `test_paper_slippage.py` | 8 | 极端跳空 8 场景/爆仓封顶/记账守恒 |
| `test_production_fixes.py` 等 11 个 | 99 | 既有回归 |
| `verify_rolling_kelly_transition.py` | 13 | 60 笔逐笔档位切换 |
| `verify_kelly_monte_carlo.py` | 6+4+2 | 静态边/边漂移/曲线追踪（含每 50 笔与生产函数交叉核对） |

> 交叉核对机制：模拟中的快速 Kelly 实现与生产 `hyperopt.rolling_kelly_risk_pct`
> 每 50 笔比对一次，任何一侧语义漂移会立即抛异常——修改 Kelly 相关代码后
> 必跑 `python verify_kelly_monte_carlo.py`。
