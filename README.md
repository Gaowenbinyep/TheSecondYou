# The second you 🤖

面向微信陪伴聊天场景风格化对齐的中文对话与强化学习训练流水线，让模型成为第二个你。
项目包含数据示例、SFT 微调、ASPO 强化训练、Agent 工具链、vLLM 本地推理部署，接口遵循 OpenAI API 格式，可零改动接入各类前端或 Bot 框架。

## 🚀 Features

- 📌 **微调（SFT）**：基于 LLaMA-Factory + LoRA，支持自定义数据或开源语料扩展（如风格对齐后的 LCCC），适配轻量训练与快速实验。
- 🧠 **ASPO 强化学习**：多维奖励（风格/关联/简洁/角色一致性），结合 Qwen3-plus/max 批量打分；使用 Z-Score 相对奖励 + n-gram 重复抑制 + 前后缀相似度惩罚，提升自然度与多样性，降低 reward hacking 风险。
- 🔧 **Agent 支持**：基于 LangGraph，内置搜索/天气/时间/奶茶推荐等示例工具，可快速构建风格化陪伴 Agent。
- ⚡ **轻量推理**：支持 vLLM 部署，本地即可启动 API 服务器，对接所有前端应用。


## 🌟 核心亮点

- 🧠 **多维奖励设计**
  - 同时考察 风格对齐 / 内容关联 / 简洁度 / 角色一致性
  - 由 Qwen3-plus/Max 批量调用作为 RM 生成 JSON 结构化得分
  - 提供显式目标分布，模型训练方向可控可解释

- 📈 **动态加权奖励机制**
  - 根据当前 batch 表现动态调整维度权重
  - 若风格偏离多，则风格reward权重↑
  - 若啰嗦、无情绪承接，则 conciseness/relevance 权重↑

- 🔥 **反 reward-hacking 机制**
  - Z-Score Normalize reward，避免模型靠绝对高分“刷奖励”
  - n-gram重复惩罚（whole-sentence repetition control）
  - edge-ngram前后缀重复惩罚（抑制模板化开头/结尾）
  - 有效延缓训练中 熵坍缩，输出更自然多样而非僵化背诵

- 🧬 **ASPO（非对称重要性采样策略优化）**
  - 基于 TRL-PPO 的算法级改造，引入 token 级更新过滤与优势感知更新
  - 正优势下采用逆 ratio + stop-gradient，训练更稳定
  - 使用可导软裁剪替代硬裁剪，减少梯度截断

- 🧊 **Reference Model EMA（稳定策略学习）**
  - 支持 EMA 软更新 Ref 模型（step=k 时做一次平滑吸收）
  - 类似 Dr.GRPO 中 reference refresh 的思想 → 更平稳、更难发散
---

## 🚀 快速上手

- 环境：`conda create -n TheSecondYou python=3.10 -y && conda activate TheSecondYou && pip install -r requirements.txt`
- 替换环境中 TRL 的 PPO 实现（ASPO 核心步骤）：本项目对 `trl/trainer/ppo_trainer.py` 进行了算法级修改，以支持基于 LoRA 的 ASPO 训练。将 `Train/ASPO/aspo_trainer.py` 整体内容复制并替换你当前 Python 环境中的 `<your_env>/lib/python3.10/site-packages/trl/trainer/ppo_trainer.py`，⚠️确保安装的 TRL 版本与本项目验证版本一致，否则可能存在接口不兼容问题。
- 复制一份环境变量模板：将 `.env.example` 复制为 `.env` 并按需填写（`.env` 已在 `.gitignore` 中忽略，避免误提交）。
- 下载基础模型：`bash ./LLM_download.sh`（默认下载到 `Models/Base_models/Qwen3-8B`，依赖 ModelScope）
- 部署推理：`bash ./model_deploy.sh`（vLLM，默认端口 8888，可在脚本中改模型路径/参数，日志默认写入 `Logs/`）
- 体验对话：`python start.py`，如需异步示例可调用 `async_single_chat`
- Agent 工作流：`python Agent/main.py`，读取环境变量配置的主模型与路由模型
- 先设密钥再跑脚本，例如 `setx DASHSCOPE_API_KEY "<your_key>"`（PowerShell）或 `export DASHSCOPE_API_KEY=<your_key>`（Linux/macOS）；如使用智能搜索工具，还需设置 `QIANFAN_API_KEY`

---

## 🗂️ 目录速览

```
TheSecondYou/
|- Agent/                # LangGraph 工作流、工具、提示词
|- Train/                # 训练与评测
|  |- ASPO/              # ASPO 训练代码与脚本
|  |- Data/              # 数据集与数据处理脚本
|  |- Evaluation/        # 评测脚本与结果
|  |- LLaMA-Factory/     # LLaMA-Factory 源码与脚本
|- Models/
|  |- Base_models/       # 基础模型缓存（需手动下载）
|  |- Saved_models/      # 已训练/合并产物
|- Logs/                 # 日志
|- .env.example          # 环境变量示例（勿提交 .env）
|- .gitignore
|- LLM_download.sh       # ModelScope 下载
|- model_deploy.sh       # vLLM 部署
|- start.py              # 本地聊天示例
|- requirements.txt
```

---

## 💬 聊天 / ⚙️ 推理 / 🧭 Agent

- 本地推理：`model_deploy.sh` 通过 vLLM 暴露 OpenAI 兼容端口，按需改路径。
- 聊天示例：`start.py` 支持同步与异步调用，默认 `OPENAI_API_BASE=http://localhost:8888/v1`、`OPENAI_API_KEY=EMPTY`；运行前确保推理服务已启动。
- Agent：`Agent/main.py` 基于 LangGraph，内置天气/搜索/时间/奶茶点单等工具。可用环境变量切换模型：
  - `OPENAI_API_BASE` / `OPENAI_API_KEY` / `WECHATROBOT_MODEL_PATH`
  - `ROUTER_OPENAI_API_BASE` / `ROUTER_OPENAI_API_KEY` / `ROUTER_MODEL_NAME`
  - `QIANFAN_API_KEY`（如启用 `web_search` 工具）
  - `BGE_MODEL_PATH` / `DRINK_DB_PATH` / `DRINKS_JSON_PATH`（奶茶点单向量库/Embedding 路径可选配置）

---

## 📚 数据与训练

- 数据：`Train/Data/` 目录提供示例 json/jsonl，可直接替换。若需清洗/筛选，请自行编写脚本，注意勿提交私密数据。
- ASPO 训练：`Train/ASPO/train.py` 的 `RunCfg` 里定义了路径与超参（默认用 DeepSpeed）。路径支持相对项目根目录或绝对路径。运行前：
  1) 确认基础模型与 LoRA 适配器路径（默认示例：`Models/Base_models/Qwen3-8B`、`Models/Saved_models/...`）；
  2) 设置 `DASHSCOPE_API_KEY`（奖励模型/评测调用 DashScope 兼容接口）；
  3) 使用 `deepspeed Train/ASPO/train.py` 启动，或参考 `Train/ASPO/aspo.sh`。
- 其他训练/合并：可使用 `Train/LLaMA-Factory/*.sh` 进行 SFT/DPO/RM/PPO 等训练与导出；数据路径配置位于 `Train/LLaMA-Factory/data/dataset_info.json`（已改为相对 `Train/Data/`）。

---

## ✅ 评测

- `Train/Evaluation/model_test.py`：轻量测试入口，可自定义待测文件与模型，生成批量的对话测试结果，可由 `Train/Evaluation/llm_eva.py` 评分。
- `Train/Evaluation/llm_eva.py`：读取 `test_result_your_model_name.json`，写入 `scored_result/result_your_model_name.json`；运行前需设置 `DASHSCOPE_API_KEY` 并确认 Base URL 可访问。


---

## 🔑 必备环境变量

| 变量 | 作用 |
| --- | --- |
| `OPENAI_API_BASE` / `OPENAI_API_KEY` | 主聊天/Agent 的 OpenAI 兼容地址与密钥 |
| `WECHATROBOT_MODEL_PATH` | Agent/聊天使用的模型路径（vLLM 或本地权重） |
| `ROUTER_OPENAI_API_BASE` / `ROUTER_OPENAI_API_KEY` / `ROUTER_MODEL_NAME` | Agent 路由模型配置（默认 DashScope Qwen） |
| `DASHSCOPE_API_KEY` | 奖励模型与评测脚本密钥 |
| `QIANFAN_API_KEY` | Agent 工具 `web_search`（百度千帆 AppBuilder）密钥 |
| `BGE_MODEL_PATH` | （可选）奶茶点单向量库使用的 Embedding 模型路径 |
| `DRINK_DB_PATH` / `DRINKS_JSON_PATH` | （可选）奶茶点单向量库与原始数据路径 |

> 建议参考 `.env.example`；注意本项目不会自动加载 `.env`，需要在运行前手动导出环境变量或写进启动脚本。

---

## 💡 小贴士

- 日志默认写入 `Logs/`，可在脚本中调整路径。
- 更换模型/端口时，先改 `model_deploy.sh`、`OPENAI_API_BASE` 等配置，保持接口兼容即可。
- 准备开源或上传 GitHub 前，再次排查是否有明文密钥或个人数据。
