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

## 开源 Laya 本地基线

Laya 是独立的本地决策模型，不调用 Jev 或 DeepSeek API。可在需要时额外安装其运行环境；默认依赖和离线 CI 不包含 Laya、PyTorch 或 Transformers：

```bash
.venv/bin/python -m pip install -e '.[dev,laya]'
.venv/bin/jev-lab run --provider laya --split dev
.venv/bin/jev-lab run --provider laya --split calibration
```

实验 provider 固定使用 `convaiinnovations/laya-multilingual`，checkpoint revision 为 `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`，并把这组来源写入 run manifest。首次运行若本机缓存没有权重，会从 Hugging Face 下载；之后可使用本地缓存。需要断网重跑时，请先确认该 revision 已缓存。

`latency_ms` 记录的是已加载模型后的单样本推理时间，不含首次下载和 checkpoint 加载；它是本机结果，不能直接与远端 API 的端到端延迟比较。Laya 没有按次 API 账单，但本实验不估算硬件折旧和电力成本，因此报告中的美元成本留空，不能解读成总运行成本为零。适配器只消费路线类别概率，不使用 `act_probability` 或 entropy `confidence` 作为自动拒答门槛。与其他候选一样，只在冻结实验协议前使用开发集和校准集；保留测试集继续保持封存。

## 产物边界

- `runs/`：本地原始运行记录，默认被 Git 忽略；
- `reports/`：完整报告，公开前必须人工审阅；
- `public/`：不包含样本输入和原始响应的聚合摘要，可供网站人工引用。

API Key、私有样本、原始 live 响应、request ID 和完整 trace 不得提交。公开摘要必须通过递归敏感信息检查。

## 研究与安全声明

本项目是研究和模拟工具，不构成投资建议，不连接券商，不自动下单，也不执行模型选择的代码、数据库、搜索或任何其他工具。它不是通用 Agent Harness，也不是生产授权系统；高风险动作必须由人工复核。

## 研究协议

- [路由标签变化成本实验](docs/research/classifier-change-cost-protocol.md)
- [Jev 与分类器专题源稿](docs/research/jev-vs-classifier-draft.md)

这些文件是 Agent 工程笔记网站的人工发布素材，不是自动部署通道。网站上的实测数据仍需逐项核对运行 ID、代码 commit 和数据哈希。
