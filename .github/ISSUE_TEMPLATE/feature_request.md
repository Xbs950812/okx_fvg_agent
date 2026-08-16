name: 功能建议 / Feature request
description: 建议新功能或改进 / Suggest a new feature or improvement
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: 解决什么问题 / What problem does it solve
      description: 这个功能帮助完成什么场景 / Describe the use case this feature would serve
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: 期望的方案 / Proposed solution
      description: 你希望它怎么工作 / How you'd expect it to work

  - type: textarea
    id: alternatives
    attributes:
      label: 替代方案 / Alternatives considered
