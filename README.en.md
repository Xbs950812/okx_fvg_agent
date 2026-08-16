# FVG KILLER v3.3

**English** | [简体中文](README.md)

[![Tests](https://img.shields.io/badge/tests-197%20passed-brightgreen)](docs/USAGE.md#7-测试与验证)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Exchange](https://img.shields.io/badge/exchange-OKX%20Perpetual%20Swaps-black)](https://www.okx.com)
[![License](https://img.shields.io/badge/license-PolyForm%20Shield%20%2B%2010%25%20Royalty-orange)](LICENSE)
[![Discussions](https://img.shields.io/badge/discussions-welcome-9cf)](https://github.com/Xbs950812/okx_fvg_agent/discussions)

An automated Fair Value Gap (FVG) trading agent for OKX perpetual futures, distilling the best ideas from the top open-source quant projects on GitHub.

> **⚠️ Important — Source-available, not open source.** Free for any purpose
> (including live trading for profit) under one condition: keep the built-in
> **10% royalty on profits** feature intact. See [Royalty](#royalty-10-profit-sharing)
> and [LICENSE](LICENSE). Removing or disabling the royalty requires a
> commercial license.
>
> **Not financial advice. Leveraged futures trading can lose your entire capital.**

## What It Does

1. **Anomaly detection** — flags candles with ≥3σ price moves + 5× volume spikes
2. **FVG detection** — standard ICT three-candle gap detection on 1H / 4H
3. **Top-100 tracking** — background thread researches the top 100 swap contracts continuously
4. **Five-channel analysis** — price action, market structure, fund flow, sentiment, macro
5. **Multi-agent debate** (TradingAgents) — 6 analysts debate → structured verdict
6. **Regime detection** (Vibe-Trading) — causal hysteresis state machine
7. **Alpha Zoo** — 461 built-in factors (Alpha101, GTJA191, Qlib158, academic)
8. **FreqAI-style online learning** — signal quality prediction
9. **Three aggressiveness levels** — aggressive / balanced / conservative
10. **Limit entries** at FVG boundaries; TP at 50% gap fill; SL beyond the gap edge
11. **Trailing stop** — activates at 50% profit, ATR-based distance
12. **Rolling fractional Kelly** — risk cap from your last 100 trades (EWMA-smoothed, explore → exploit tiers)
13. **Live guards (v3.3)** — global API rate limiter, order-book depth check, volatility targeting, daily trade cap, slippage feedback loop, 3-way startup reconciliation, funding-fee reconciliation

## Monte Carlo — Rolling Kelly vs Fixed Risk

![Monte Carlo equity curves](docs/images/montecarlo_curves.png)

1000 trades × 8 paired paths (win rate 50%, payoff 2.5×). Same random
sequences for both strategies; the chart is generated from the exact same
code path as the verification script (`verify_kelly_monte_carlo.py`,
300 paths × 1000 trades, including edge-decay robustness scenarios — see
[USAGE.md](docs/USAGE.md#7-测试与验证)).

## Royalty — 10% Profit Sharing

**You pay only when the bot wins.** After each **closed profitable trade**,
10% of the realized profit is accrued to a royalty pool; when the pool
reaches 20 USDT, it is automatically withdrawn on-chain (USDT-TRC20,
~1 USDT fee) to the author's wallet.

```
Your profit +100 USDT → pool +10 USDT → pool ≥ 20 USDT → auto-withdraw (TRC20)
Your losses           → no royalty charged, ever
```

**Compliance is automatic out of the box.** The default config already
satisfies every license condition — you do not need to change anything:

| # | Condition (LICENSE Additional Terms §1) | Default satisfies? |
|---|------------------------------------------|--------------------|
| 1 | `royalty.enabled` stays `true` | ✅ |
| 2 | Built-in wallet `DEFAULT_ROYALTY_WALLET` in `royalty.py` not removed/replaced | ✅ |
| 3 | `royalty.rate_pct` ≥ `10.0` | ✅ |

**When money actually moves** — all must hold: live mode (paper/dry-run
**never** transfers), position **closed** with **positive realized PnL**
(floating PnL, losses and break-evens don't count), pool full ≥ 20 USDT.

**Not a violation** — if your OKX API key lacks withdrawal authority or the
wallet is not whitelisted, the bot just accrues and logs hourly reminders;
keeping the feature intact satisfies the license (§2).

**Requires a commercial license** — removing/disabling the royalty module,
changing the wallet address, or setting `rate_pct` < 10 (§3). Contact:
https://github.com/Xbs950812

## Quick Start

```bash
pip install -r requirements.txt
cp config.example.json config.json   # fill in your OKX API keys
python agent.py
```

- **Paper mode first** (recommended): `"agent": {"dry_run": true}, "paper": {"enabled": true}` — virtual balance, real market data, zero real orders
- **Demo trading**: `"okx": {"demo": true}` with demo API keys from the OKX app
- API key needs **Trade + Read**. Withdrawal authority is only needed for the
  royalty auto-withdraw (optional — see above)
- Users in mainland China: set `"okx": {"proxy": "http://127.0.0.1:7890"}`

Aggressiveness levels: `1` = aggressive (forces a position daily),
`2` = balanced, `3` = conservative (default).

Full configuration reference: [README.md](README.md#配置说明) (Chinese) ·
Usage guide: [docs/USAGE.md](docs/USAGE.md) · Troubleshooting:
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Project Structure

```
okx_fvg_agent/
├── agent.py            # Main loop entry (v3.3)
├── strategy.py         # FVG detection + anomaly + signal generation
├── executor.py         # Position sizing + order execution + order-book checks
├── okx_client.py       # OKX API v5 client (proxy/demo/rate limiter)
├── paper_trading.py    # Paper engine (limit-fill semantics, liquidation caps)
├── royalty.py          # Royalty module (10% pool → threshold auto-withdraw)
├── multi_channel.py    # Five-channel analysis engine
├── debate_engine.py    # TradingAgents multi-agent debate
├── hyperopt.py         # freqtrade-style hyperopt + Kelly + FreqAI
├── alpha_zoo.py        # Vibe-Trading factor zoo + regime detection
├── factor_zoo/         # 461 factors (Alpha101/GTJA191/Qlib158/academic)
├── coin_tracker.py     # Background top-100 research thread
├── fvg_killer_pro.py   # v3.3 guards (rate limiter, book depth, Kelly, reconciliation)
├── test_*.py           # 197 unit tests
├── verify_*.py         # Monte Carlo / tier-transition verification scripts
└── docs/               # Usage / troubleshooting / update manual (Chinese)
```

## Testing

```bash
python -m pytest -q                       # 197 tests, ~11s
python verify_kelly_monte_carlo.py        # Monte Carlo cross-validation
python verify_kelly_monte_carlo.py --drift 0.5 0.4   # edge decay robustness
```

## Influences

| Project | Stars | Contributions |
|---------|-------|---------------|
| [freqtrade](https://github.com/freqtrade/freqtrade) | 52k⭐ | Hyperopt, Edge analysis, Trailing Stop, FreqAI, Kelly sizing |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 86k⭐ | Multi-agent debate, analyst reputation, decision reflection |
| [Vibe-Trading](https://github.com/vibe-trading/vibe-trading) | 23.6k⭐ | Alpha Zoo, causal regime detection, memory lifecycle |

## Risk Warning

- This strategy is based on historical price patterns; future returns are not guaranteed
- Always test with small capital first
- High leverage can wipe out your entire margin
- Read and understand the code before going live

## License

[PolyForm Shield 1.0.0](LICENSE) + Author's Additional Terms (Royalty
Provision). Any use is free — including running it live for profit — as long
as the 10% royalty feature and default wallet remain intact; competing
products built from this software are not permitted. Removing the royalty
requires a commercial license: https://github.com/Xbs950812
