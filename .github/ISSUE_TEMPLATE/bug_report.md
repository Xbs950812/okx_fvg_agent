name: Bug 报告 / Bug report
description: 报告运行问题 / Report a problem running the agent
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        ## ⚠️ 提交前必读 / Read before posting

        **绝对不要粘贴你的 API key / secret / passphrase / 钱包私钥！**
        **NEVER paste your API key / secret / passphrase / wallet private key!**

        如果日志里包含密钥，请先打码（`****`）再提交。
        Mask any credentials with `****` before posting logs.

  - type: textarea
    id: what-happened
    attributes:
      label: 问题描述 / What happened
      description: 发生了什么、期望是什么 / A clear description of the actual vs expected behavior
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: 相关日志（脱敏后）/ Relevant logs (sanitized)
      description: |
        复制 agent.log 中的关键片段（已脱敏）。
        Key lines from agent.log with credentials masked.
      render: shell

  - type: input
    id: version
    attributes:
      label: 版本 / Version
      description: 如 v3.3.0（git log -1 或 README 版本号）
      placeholder: v3.3.0

  - type: dropdown
    id: mode
    attributes:
      label: 运行模式 / Mode
      options:
        - Paper (dry_run + paper.enabled)
        - dry_run only
        - Demo trading (okx.demo=true)
        - Live trading
    validations:
      required: true

  - type: textarea
    id: config
    attributes:
      label: 相关配置（脱敏后）/ Relevant config (sanitized)
      description: 只贴相关段，删除密钥字段 / Only relevant sections, strip credentials
