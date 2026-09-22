# Jev Decision Lab

一个离线优先、证据可追溯的有限答案空间决策模型实验台，用来比较确定性规则、DeepSeek 结构化决策和 Jev。当前已形成第一版离线研究系统，但尚未使用有效 Jev Key 完成本地真实调用，因此不存在可发布的 Jev 性能结论。

## 安装

需要 Python 3.11 或更新版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## 无 Key 的完整流程

```bash
.venv/bin/jev-lab dataset validate
.venv/bin/jev-lab run --provider rules --split dev
.venv/bin/jev-lab evaluate --run runs/<run-id>
.venv/bin/jev-lab report --run runs/<run-id>
.venv/bin/python -m pytest -m 'not live_deepseek and not live_jev'
```

远程 Provider 可通过 `--threshold 0.75` 固定拒答阈值；该值会写入 manifest、完整报告和公开摘要。阈值只影响 coverage 与 selective accuracy，不改写未筛选的 baseline accuracy 或 Brier Score。

运行模式必须如实解释：

- `offline-development`：规则和本地逻辑，不包含模型调用；
- `recorded-replay`：合成或脱敏后的固定响应，只验证管线；
- `live-evaluation`：真实、具备凭据的远程请求。

`recorded-replay` 不能被描述成 Jev 实测结果。当前 evidence status 是：规则基线可离线运行；DeepSeek 和 Jev 适配器已具备契约测试；真实 Jev 实验尚未开始。

## 远程 Provider

DeepSeek 使用本地环境变量 `DEEPSEEK_API_KEY`。Jev 的访问申请、`TYPESAFE_API_KEY`、可选 SDK 和显式 live smoke test 见 [Jev 访问指南](docs/getting-jev-access.md)。密钥缺失时，Provider 返回明确的 `skipped`，不会阻塞离线流程。

## 专用分类器基线

分类器依赖是可选的，不影响默认离线流程：

```bash
.venv/bin/python -m pip install -e '.[dev,classifier]'
.venv/bin/jev-lab train \
  --provider tfidf-logreg \
  --split dev \
  --model-id routing-v1-s42
.venv/bin/jev-lab run \
  --provider tfidf-logreg \
  --model artifacts/routing-v1-s42 \
  --split calibration
```

训练命令内部使用按 `family_id` 分组的三折交叉验证，只把它作为开发稳定性证据。分类器随后使用全部 `dev` 样本训练；`calibration` 只用于选择拒答阈值。规则、特征、模型、Prompt、Criteria 和阈值全部冻结后，才能运行保留 `test`。开发集交叉验证、校准结果和保留测试结果不得混为同一种证据。

## 产物边界

- `runs/`：本地原始运行记录，默认被 Git 忽略；
- `reports/`：完整报告，公开前必须人工审阅；
- `public/`：不包含样本输入和原始响应的聚合摘要，可供网站人工引用。

API Key、私有样本、原始 live 响应、request ID 和完整 trace 不得提交。公开摘要必须通过递归敏感信息检查。

## 研究与安全声明

本项目是研究和模拟工具，不构成投资建议，不连接券商，不自动下单，也不执行模型选择的代码、数据库、搜索或任何其他工具。它不是通用 Agent Harness，也不是生产授权系统；高风险动作必须由人工复核。
