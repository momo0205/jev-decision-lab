# 获取并配置 Jev 访问

## 官方 early access

先通过 TypeSafe AI 官方渠道申请 Jev early access。获得密钥后，只在本地被 Git 忽略的 `.env` 或当前 shell 中配置：

```bash
read -s TYPESAFE_API_KEY
export TYPESAFE_API_KEY
.venv/bin/python -m pip install -e '.[jev]'
```

不要把真实值写入源码、Markdown、截图或命令历史。`TYPESAFE_API_KEY` 缺失时，Jev Provider 返回 `skipped/missing_credentials`；未安装可选 SDK 时返回 `skipped/dependency_missing`。

## 显式 live smoke test

只有在主动提供密钥和网络时，才运行：

```bash
.venv/bin/python -m pytest -m live_jev -q
```

检查生成的 manifest、状态、模型版本与聚合指标，不打印原始响应或密钥。普通测试和 CI 明确排除 `live_jev`。

只有成功的 `live-evaluation` 运行、完整来源记录和人工审阅通过后，才可以把网站上的“实验尚未开始”改成“已完成本地复现”。录制回放和合成 fixture 永远不能触发该更新。

Vercel AI Gateway 的 `typesafe-ai/jev` 是另一条访问路径，但当前实验台尚未实现对应适配器；不能把官方 SDK 适配器直接当成 Gateway 已受支持。
