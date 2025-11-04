# TheSecondYou

微信恋爱陪伴场景下的中文对话机器人与强化学习流水线，开箱即用地提供数据处理、微调、PPO、Agent 工具链，并预置本地 vLLM 推理接口。仓库现已内置三份零依赖入口脚本，帮助初学者无需深入整个项目结构即可快速体验核心功能。

---

## 🚀 快速上手

| 功能 | 命令 | 说明 |
| --- | --- | --- |
| 监督微调 (SFT) | `python run_finetune.py` | 调用 LLaMA-Factory 默认配置，对 `data/v1.0/Single_train.json` 进行全量微调。 |
| PPO 训练 | `python run_ppo.py` | 启动自定义 PPO 训练器，日志输出到 `logs/ppo_train.log`。 |
| 智能 Agent | `python run_agent.py` | 打开 LangGraph 工作流 Agent 交互式会话（默认连接本地 vLLM）。 |

> 所有脚本都支持 `--help` 查看可选参数，例如 `python run_finetune.py --help`。

---

## 🗂️ 目录概览

```
WechatRobot
├── Agent/                    # LangGraph 工作流、工具与 prompts
├── Base_models/              # 预训练模型（自备）
├── Saved_models/             # 微调/强化学习产物
├── data/                     # 数据集与脚本输出
├── evaluation/               # 评测脚本与结果
├── LLaMA-Factory/            # LLaMA-Factory 配置与训练脚本
├── PPO/                      # PPO 训练代码
├── logs/                     # 训练/推理日志
├── model_deploy.sh           # vLLM 部署脚本
├── run_finetune.py           # SFT 快速入口
├── run_ppo.py                # PPO 快速入口
├── run_agent.py              # Agent 快速入口
└── requirements.txt
```

---

## ⚙️ 环境准备

```bash
conda create -n wechatrobot python=3.10 -y
conda activate wechatrobot
pip install -r requirements.txt
# 可选：加速推理
pip install vllm
```

- 将基础模型放入 `Base_models/` 对应目录。
- 需要 RLHF/SFT 产物时，放入 `Saved_models/`。
- 奶茶点单工具依赖向量库，可在 `Agent/tools/drink_ordering/utils.py` 中调用 `build_drink_db()` 生成。

---

## 🧪 三大入口脚本详解

### 1. run_finetune.py
- 默认配置：`Base_models/Qwen3-1.7B` + `data/v1.0/Single_train.json`。
- 输出目录：`Saved_models/sft/demo_output`。
- 可选参数：
  - `--base-model PATH`
  - `--dataset PATH`
  - `--output-dir PATH`
  - `--epochs INT`
- 自动检查文件是否存在，避免新手踩坑。

### 2. run_ppo.py
- 调用 `PPO/train.py`，保持项目原始的奖励计算与 DeepSpeed 设置。
- 支持环境变量覆盖最大训练步数：`--max-steps`（内部传递给 `PPO_MAX_STEPS`）。
- 指定 GPU：`--cuda-devices "0,1"` 等。
- 日志默认写入 `logs/ppo_train.log`，可用 `--log-file` 修改。

### 3. run_agent.py
- 启动 LangGraph Agent，默认连接 `http://localhost:8888/v1`（需提前运行 `model_deploy.sh`）。
- 可通过 CLI/环境变量切换模型：
  - `--model-base` → `OPENAI_API_BASE`
  - `--api-key` → `OPENAI_API_KEY`
  - `--model-path` → `WECHATROBOT_MODEL_PATH`
- 支持在本地或云端模型之间自由切换。

---

## 📦 深入功能模块

### 数据与预处理
- `data_process.py`：核心数据清洗、打分、筛选流水线（异步调用模型）。
- `data/data_gen.py`：根据已有对话生成增强样本。
- `data/data_select.py` 与 `data/eval/data_select.py`：基于 SentenceTransformer 筛选高质量样本。
- `data/LCCC/format_trans.py`：将 LCCC 数据转换为 ShareGPT 格式。

### 训练脚本
- `LLaMA-Factory/sft_rag.sh`、`dpo.sh`、`ppo.sh`：完整的 SFT / DPO / PPO 训练 shell。
- `LLaMA-Factory/Lora_merge.sh`：将 LoRA adapter 与基座模型合并输出。
- `PPO/train.py`：自定义 PPO Trainer，支持多维奖励、动态权重、DeepSpeed。

### 推理与 Agent
- `model_deploy.sh`：以 vLLM 部署微调模型（默认端口 8888）。
- `start.py`：简单的同步/异步聊天示例。
- `Agent/graph_nodes/nodes.py`：LangGraph 状态机节点定义（现已支持环境变量配置 API）。
- `Agent/tools/tools.py`：天气、搜索、时间、奶茶点单等工具集合。

### 评测
- `evaluation/model_test.py`：异步评测脚本，输出测评 JSON。
- `evaluation/sim_eva.py`：语义相似度评分工具，可训练自定义打分模型。
- `async_evaluation.py`：基于 DashScope 的并发评估管线。

---

## 🔧 常见配置

| 变量/参数 | 说明 |
| --- | --- |
| `OPENAI_API_BASE`, `OPENAI_API_KEY` | 本地 vLLM 或 OpenAI 兼容服务地址与密钥。 |
| `WECHATROBOT_MODEL_PATH` | Agent 使用的模型路径，可通过 `run_agent.py --model-path` 设置。 |
| `ROUTER_OPENAI_API_*` | Agent 路由模型（默认 Qwen DashScope）凭证，可改为其他模型。 |
| `PPO_MAX_STEPS` | PPO 训练步数限制（由 `run_ppo.py --max-steps` 自动注入）。 |
| `DASHSCOPE_API_KEY` | 与阿里云 DashScope 通信时的密钥。 |

---

## ❓ 常见问题

1. **模型路径不存在**  
   - 确认基础模型/微调模型已放入 `Base_models/` 和 `Saved_models/`，或使用入口脚本的参数指向正确目录。

2. **没有 GPU 怎么办？**  
   - SFT/PPO 默认需要 GPU。任意脚本运行前请确保 CUDA、驱动、DeepSpeed 等环境已配置；如无 GPU，可考虑改为 CPU 模式并调整 batch size（需手动修改底层脚本）。

3. **Agent 启动报错连接失败**  
   - 先运行 `bash model_deploy.sh` 启动本地 vLLM 服务，或修改 `run_agent.py` 的 `--model-base` 指向可用的 OpenAI 兼容接口。

4. **日志/输出在哪儿？**  
   - 训练相关的日志默认写在 `logs/`，模型输出目录可通过入口脚本或原始 shell 脚本的参数查看。

---

## 🤝 贡献指南

欢迎提出 Issue 或提交 PR，尤其是：
- 新的数据处理脚本或评测方法；
- 支持更多 Agent 工具、RAG 组件；
- 针对 PPO 奖励或训练稳定性的改进建议。

---

## 📄 License

仓库未明确开源协议，默认遵循项目作者约定。若需商用或二次分发，请务必事先沟通确认。

---

祝使用愉快，欢迎分享你的实战经验！🍵🤖
