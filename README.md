# WechatRobot 🤖

面向微信陪伴聊天场景的中文对话与强化学习训练流水线。项目包含数据示例、SFT 微调、PPO 强化训练、Agent 工具链、vLLM 本地推理部署，接口遵循 OpenAI API 格式，可零改动接入各类前端或 Bot 框架。

## 🚀 Features

- 📌 **微调（SFT）**：基于 LLaMA-Factory + LoRA，支持自定义数据或开源语料扩展（如风格对齐后的 LCCC），适配轻量训练与快速实验。
- 🧠 **PPO 强化学习**：多维奖励（风格/关联/简洁/角色一致性），结合 Qwen3-plus/max 批量打分；使用 Z-Score 相对奖励 + n-gram 重复抑制 + 前后缀相似度惩罚，提升自然度与多样性，降低 reward hacking 风险。
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

🧊 **Reference Model EMA（稳定策略学习）**
  - 支持 EMA 软更新 Ref 模型（step=k 时做一次平滑吸收）
  - 类似 Dr.GRPO 中 reference refresh 的思想 → 更平稳、更难发散

---

## 🚀 快速上手

- 环境：`conda create -n wechatrobot python=3.10 -y && conda activate wechatrobot && pip install -r requirements.txt`
- 下载基础模型：`bash ./LLM_download.sh`（默认至 `Base_models/Qwen3-8B`，依赖 ModelScope）
- 部署推理：`bash ./model_deploy.sh`（vLLM，默认端口 8888，可在脚本中改模型路径/参数）
- 体验对话：`python start.py`，如需异步示例可调用 `async_single_chat`
- Agent 工作流：`python Agent/main.py`，读取环境变量配置的主模型与路由模型
- 先设密钥再跑脚本，例如 `setx DASHSCOPE_API_KEY "<your_key>"`（PowerShell）或 `export DASHSCOPE_API_KEY=<your_key>`（Linux/macOS）

---

## 🗂️ 目录速览

```
WechatRobot/
|- Agent/                # LangGraph 工作流、工具、提示词
|- Base_models/          # 基础模型缓存（需手动下载）
|- LLaMA-Factory/        # 训练框架与脚本
|- PPO/                  # PPO 训练代码与脚本
|- Saved_models/         # 已训练/合并产物
|- data/                 # 示例数据
|- evaluation/           # 评测脚本与结果
|- logs/                 # 日志
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

---

## 📚 数据与训练

- 数据：`data/` 目录提供示例 jsonl，可直接替换。若需清洗/筛选，请自行编写脚本（路径不限），注意勿提交私密数据。
- PPO 训练：`PPO/train.py` 的 `RunCfg` 里定义了路径与超参（默认用 DeepSpeed）。运行前：
  1) 修改 `policy_base`、`lora_adapter_path`、`output_dir` 为本地路径；
  2) 设置 `DASHSCOPE_API_KEY`（奖励模型调用 DashScope 兼容接口）；
  3) 用 `torchrun` 或 `deepspeed` 启动，例如 `torchrun --nproc_per_node=2 PPO/train.py`，或参考 `PPO/ppo.sh`。
- 其他训练/合并：可复用 `LLaMA-Factory` 的脚本做 SFT/DPO 等，产物建议放入 `Saved_models/`。

---

## ✅ 评测

- `evaluation/model_test.py`：轻量测试入口，可自定义待测文件与模型，生成批量的对话测试结果，可由 `evaluation/llm_eva.py` 评分。
- `evaluation/llm_eva.py`：读取 `test_result_your_model_name.json`，写入 `scored_result/result_your_model_name.json`；运行前需设置 `DASHSCOPE_API_KEY` 并确认 Base URL 可访问。


---

## 🔑 必备环境变量

| 变量 | 作用 |
| --- | --- |
| `OPENAI_API_BASE` / `OPENAI_API_KEY` | 主聊天/Agent 的 OpenAI 兼容地址与密钥 |
| `WECHATROBOT_MODEL_PATH` | Agent/聊天使用的模型路径（vLLM 或本地权重） |
| `ROUTER_OPENAI_API_BASE` / `ROUTER_OPENAI_API_KEY` / `ROUTER_MODEL_NAME` | Agent 路由模型配置（默认 DashScope Qwen） |
| `DASHSCOPE_API_KEY` | 奖励模型与评测脚本密钥 |

> 建议只在环境或启动脚本中设置密钥，避免写入代码或提交到仓库。

---

## 💡 小贴士

- 日志默认写入 `logs/`，可在脚本中调整路径。
- 更换模型/端口时，先改 `model_deploy.sh`、`OPENAI_API_BASE` 等配置，保持接口兼容即可。
- 准备开源或上传 GitHub 前，再次排查是否有明文密钥或个人数据。
