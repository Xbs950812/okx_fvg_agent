# 贡献指南 / Contributing

感谢关注 FVG KILLER / Thanks for your interest in FVG KILLER.

## 行为准则 / Ground Rules

- **绝不提交任何密钥** / **Never commit any secrets** — API key、secret、passphrase、钱包私钥一律不得出现在 issue、PR、日志粘贴中（请用 `****` 脱敏）
- 提交前跑全量测试 / Run the full test suite before submitting:
  ```bash
  python -m pytest -q   # 197 项应全绿 / all 197 must pass
  ```
- 新功能必须带单测 / New features ship with unit tests
- Bug 修复请附复现步骤或日志片段（脱敏）/ Bug fixes need repro steps or sanitized logs

## 许可提示 / License Note

本项目采用 [PolyForm Shield + 作者附加条款](LICENSE)。提交 PR 即表示你同意
贡献内容按该协议授权（含 10% 盈利分成条款对代码的适用）。

This project uses [PolyForm Shield + Author's Additional Terms](LICENSE).
By submitting a PR you agree your contribution is licensed under the same
terms (including the 10% royalty provision).

## 如何开始 / Good First Steps

1. 看 [docs/USAGE.md](docs/USAGE.md) 跑通纸面模式 / Get paper mode running
2. 从带 `good first issue` 标签的 issue 入手 / Pick issues labeled `good first issue`
3. 讨论/提问优先发 [Discussions](https://github.com/Xbs950812/okx_fvg_agent/discussions)
