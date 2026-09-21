# Jev Decision Lab 设计文档

**日期：** 2026-09-21  
**状态：** 已完成讨论，等待书面复核  
**项目位置：** `03-research/jev-decision-lab`  
**项目性质：** 独立、公开、离线优先的 Python 决策模型实验台

## 1. 背景与研究问题

Jev 面向有限答案空间的决策任务，返回类型化选择、评分或概率，而不是自由文本。它对 Agent Runtime 中的路由、风险判断和是否转人工等节点具有潜在价值，但厂商公开的速度、成本和准确率数据不能直接替代本地任务验证。

本项目不预设 Jev 优于规则或生成式模型，而是建立一套可审计的实验系统，回答：

- 在固定答案空间的 Agent 路由任务上，Jev 的决策质量如何；
- 不输出面向人的推理文本，是否影响边界样本、歧义样本和高风险样本的正确性；
- Jev 相对确定性规则和 DeepSeek 结构化输出，在准确率、校准、延迟和成本上有什么差异；
- 低置信度拒答后，自动化覆盖率与剩余错误风险如何变化；
- Jev 的潜在速度和成本优势，是否足以覆盖新增供应商、校准、监控与回退成本。

实验台负责产生证据，`agent-engineering-notes` 网站负责发布经过脱敏和人工审阅的研究结论。两个仓库不共享构建依赖、密钥或原始运行记录。

## 2. 目标与非目标

### 2.1 第一版目标

- 建立可复现的规则、DeepSeek、Jev 三方对照实验；
- 无网络、无 API Key 时仍可验证数据、运行规则基线、计算指标并生成报告；
- 用 100 条中文合成或脱敏样本研究 Agent 请求路由；
- 严格隔离开发、校准和保留测试集；
- 记录模型输出、拒答、延迟、成本依据和失败类型；
- 生成完整本地报告以及允许公开的脱敏摘要；
- 提供 Jev 访问申请、密钥配置和真实 smoke test 指南；
- 让网站结论能够引用固定数据集版本、代码 commit 和实验运行标识。

### 2.2 非目标

- 不实现通用 Agent Harness 或 Agent Loop；
- 不执行模型选择的工具或业务动作；
- 不使用公司请求、个人数据或投资决策作为第一版数据；
- 不把 mock 或录制响应写成真实 Jev 成绩；
- 不复现或猜测未公开的 Jev 内部架构；
- 不在缺少本地证据时宣传 Jev 更快、更便宜或更准确；
- 不让实验台自动修改或部署公开网站；
- 不把 Jev 或任何模型单独用于高风险动作审批。

## 3. 技术路线

第一版使用轻量、透明的 Python 技术栈：

- Pydantic：数据集和运行结果契约；
- Typer：命令行界面；
- pytest：测试；
- 标准 Python 模块或小型数值库：指标与报告生成；
- Provider 适配器：隔离规则、DeepSeek、Jev 和录制响应。

第一版不引入通用评测框架或 Agent Harness。原因不是这些框架没有价值，而是本阶段的核心学习目标是看清决策输入、Provider 输出、阈值、拒答和指标之间的关系。接口应保持清晰，以便未来接入 Inspect AI、DeepEval 或其他评测基础设施。

## 4. 总体架构

```text
版本化样本
   ↓
数据契约校验与按样本族分区
   ↓
┌──────────┬──────────────┬──────────┐
│ 规则基线 │ DeepSeek基线 │ Jev候选  │
└──────────┴──────────────┴──────────┘
   ↓
统一 DecisionResult
   ↓
正确率 / 混淆矩阵 / 校准 / 延迟 / 成本 / Coverage
   ↓
失败案例 + 完整报告 + 可公开脱敏摘要
```

这是一个明确、可审计的 experiment loop，不是 Agent Loop。它不会根据模型输出自主调用工具、扩展计划或改变目标。

## 5. 数据集设计

### 5.1 第一项任务

第一项任务是 Agent 请求路由，固定答案空间为：

- `search`：需要检索外部资料；
- `code`：需要阅读、修改或调试代码；
- `database`：需要查询或检查数据库；
- `human_review`：信息不足、风险过高或必须由人工判断。

### 5.2 规模与切分

第一版包含 100 条中文样本：

- 60 条开发集：完善标签定义、规则和提示；
- 20 条校准集：选择拒答阈值并观察概率校准；
- 20 条保留测试集：冻结配置后进行正式评估。

样本按 `family_id` 组织。同一个原始任务的改写、模糊版、对抗版必须位于同一个分区，禁止近重复内容跨分区泄漏。数据切分必须在调参前固定。

### 5.3 样本契约

```yaml
sample_id: route-code-001
family_id: route-code-debug
input: "定位这个 Java 接口偶发 500 的原因"
expected: code
acceptable:
  - code
risk: low
difficulty: clear
rationale: "需要阅读和调试程序，不是检索资料"
source: synthetic
split: dev
```

样本至少覆盖：

- 清晰的单一路由；
- 两个能力交叉的请求；
- 信息不足；
- 高风险操作；
- 提示注入或诱导性措辞；
- 应转 `human_review` 的请求。

初始标注必须附理由和可接受备选。存在真实分歧的样本进入 `disputed` 集，不计入主准确率，直至完成人工复核。

## 6. Provider 与统一结果契约

### 6.1 Provider

- `rules`：完全离线且确定性，是最低成本基线；
- `deepseek`：使用严格结构化输出；缺少 Key 时跳过；
- `jev`：通过独立适配器封装官方 SDK；缺少访问资格时不阻塞其他流程；
- `recorded`：读取经过脱敏的固定响应，验证指标与报告管线，不能作为实时模型成绩。

### 6.2 DecisionResult

所有 Provider 统一返回：

```text
DecisionResult
├── label
├── probabilities
├── abstained
├── latency_ms
├── estimated_cost_usd
├── provider
├── model_version
├── request_status
└── error
```

Provider 不返回可比较概率时，`probabilities` 必须为空，不能人为制造置信度。成本来源不明确时，成本字段为空并记录原因。

远程 Provider 需要超时、有限重试和明确错误分类。正式比较中不得悄悄切换到其他 Provider；服务失败必须作为原 Provider 的运行结果记录。生产回退策略与实验统计是两个不同概念。

### 6.3 运行模式

报告必须区分：

- `offline-development`：规则、数据和本地逻辑；
- `recorded-replay`：固定录制响应回放；
- `live-evaluation`：真实远程模型调用。

任何页面和报告都不得把前三者混为一谈。

## 7. 指标与判定

### 7.1 决策质量

- accuracy；
- confusion matrix；
- 每类 precision / recall；
- 高风险 false approval；
- 漏掉的 `human_review`。

### 7.2 选择性决策

- coverage；
- 拒答后的 accuracy；
- 不同阈值下 coverage 与错误率的关系。

不能通过把大量样本转人工来制造虚假的高准确率。

### 7.3 概率质量

- Brier Score；
- calibration buckets / curve。

这些指标只对真实且语义可比较的概率计算。缺失概率的 Provider 不参与该项排名。

### 7.4 工程性质

- 请求成功率；
- P50 / P95 端到端延迟；
- 单样本估算成本；
- 成本价格表、币种和计算日期；
- 超时、限额、坏响应等失败分布。

## 8. CLI 与实验流程

第一版提供以下命令语义：

```bash
jev-lab dataset validate
jev-lab run --provider rules --split dev
jev-lab run --provider deepseek --split dev
jev-lab run --provider jev --split dev
jev-lab evaluate --run <run-id>
jev-lab report --run <run-id>
```

推荐实验顺序：

```text
数据验证
→ 规则基线
→ 开发集迭代
→ 校准集确定阈值
→ 冻结规则、提示、问题和阈值
→ 保留测试集只运行一次正式评分
→ Shadow Mode
→ 仅为高置信、低风险节点考虑自动化
```

每个运行记录：

- 代码 commit；
- 数据集版本与哈希；
- Provider 和模型版本；
- 非敏感配置摘要；
- 阈值；
- 时间与运行模式；
- 是否为开发、校准、正式测试或版本回归。

## 9. 产物、安全与网站发布

运行产物分为：

```text
runs/       本地原始运行记录，默认不提交
reports/    完整报告，发布前必须审查
public/     脱敏且允许网站引用的摘要
```

仓库公开以下内容：

- 源代码；
- 合成或充分脱敏的数据；
- 测试；
- 数据集卡；
- 不包含敏感信息的聚合报告。

不得提交：

- `.env` 和 API Key；
- 未脱敏原始响应；
- 私有请求、个人信息或公司数据；
- 可用于还原敏感输入的完整 trace；
- 不必要的 request id；
- 本地缓存与完整运行日志。

网站同步保留人工发布门：

```text
生成 public 摘要
→ 敏感信息扫描
→ 人工审阅证据与措辞
→ 更新 agent-engineering-notes
→ 网站测试
→ 发布
```

第一版不允许实验台自动修改或部署网站。

## 10. 项目结构

```text
jev-decision-lab/
├── pyproject.toml
├── README.md
├── .env.example
├── src/jev_lab/
│   ├── cli.py
│   ├── contracts.py
│   ├── datasets/
│   ├── providers/
│   │   ├── rules.py
│   │   ├── deepseek.py
│   │   ├── jev.py
│   │   └── recorded.py
│   ├── runner/
│   ├── metrics/
│   ├── reports/
│   └── security/
├── datasets/
│   ├── routing-v1.yaml
│   └── dataset-card.md
├── tests/
├── runs/
├── reports/
├── public/
└── docs/
    ├── getting-jev-access.md
    └── superpowers/
```

## 11. 测试策略

实现过程遵循 TDD。至少覆盖：

- 数据契约、非法枚举和缺失字段；
- `family_id` 不得跨分区；
- 重复 `sample_id`；
- Provider 输出标准化；
- 缺少 Key、超时、坏响应与概率缺失；
- 指标在已知小样本上的精确结果；
- 拒答、coverage 和零覆盖边界；
- 录制响应的确定性回归；
- 敏感信息扫描和公开摘要脱敏；
- CLI 端到端离线运行；
- 无网络、无 Key 环境下的完整基础流程。

远程 API 测试默认不在普通测试中执行。真实 smoke test 使用显式标记，并要求本地环境具备相应凭据。

## 12. 实施阶段

1. 项目骨架、契约、数据校验和离线 CLI；
2. 100 条带理由、分组和分区的路由数据集；
3. 规则 Provider、指标系统和离线报告；
4. DeepSeek Provider 与真实调用验证；
5. Jev Provider 契约、录制响应测试和访问申请指南；
6. 获得 Jev Key 后完成真实 smoke test 与三方保留集比较；
7. 审阅并把脱敏结论更新到 `notes.ironmao.com`。

## 13. 第一版完成标准

在全新环境、没有网络和任何 API Key 的情况下，用户能够：

1. 安装项目；
2. 验证数据集；
3. 运行规则基线；
4. 计算指标；
5. 生成包含失败和缺失项说明的报告；
6. 运行全部离线测试。

缺少远程凭据时，DeepSeek 和 Jev 必须明确显示 `skipped` 或配置错误，并给出下一步指引，不能阻塞离线流程，也不能生成伪造的远程模型结果。

第二阶段只有在获得 Jev 访问资格并完成真实运行后，才可以把专题中的“实验尚未开始”更新为“已完成本地复现”。
