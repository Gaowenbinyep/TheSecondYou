
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import DeepSpeedPlugin
from openai import OpenAI
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

os.environ.setdefault("DASHSCOPE_API_KEY", "REPLACE_WITH_DASHSCOPE_KEY")

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
current_device = torch.device(f"cuda:{local_rank}")
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

BASE_DIR = Path(__file__).resolve().parents[1]



@dataclass
class RunCfg:
    policy_base: str = str(BASE_DIR / "Base_models" / "Qwen3-4B")
    lora_adapter_path: str = str(BASE_DIR / "Saved_models" / "sft" / "4B_lora_V2")
    output_dir: str = str(BASE_DIR / "Saved_models" / "rlhf" / "4B_lora_PPO_V4")
    system_prompt: str = """你是高文彬，你要回复你女朋友的消息。
        你要参照示例学习高文彬作为男朋友的语言风格，保持风格个性化且有回应性。
        ---
        【语言风格示例】（请模仿此风格）：
        1. 诶嘿 好想你哦
        2. 记得吃点甜甜的缓一缓叭
        3. 小打工仔今天很乖呐
        4. 不准再不开心啦🙅🏻‍♂️
        5. 要是我在就给你揉揉头啦
        6. 忙完奖励自己点好吃的哦！
        7. 我的宝藏女孩今天也好棒呀！
        8. 哈哈哈哈你也太可爱了吧
        9. 我滴宝藏宝宝😊
        10. 好叭好叭你说了算！"""

    # precision & perf
    bf16: bool = True
    gradient_checkpointing: bool = True

    # quantization (QLoRA)
    load_policy_in_4bit: bool = False

    # deepspeed / distributed
    use_deepspeed: bool = True
    zero_stage: int = 2
    grad_accum_steps: int = 8

    # LoRA cfg
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    max_training_steps: int = 150

    # PPO hyper-params
    ppo_batch_size: int = 16
    ppo_mini_batch_size: int = 1
    ppo_epochs: int = 3
    learning_rate: float = 5e-7
    kl_coef: float = 0.1
    target_kl: float = 0.2
    init_kl_coef: float = 0.05
    max_grad_norm: float = 0.5

    # generation
    max_prompt_len: int = 256
    max_gen_len: int = 128
    temperature: float = 0.0

    # reproducibility
    seed: int = 42


cfg = RunCfg()
max_steps_override = os.getenv("PPO_MAX_STEPS")
if max_steps_override:
    try:
        cfg.max_training_steps = int(max_steps_override)
    except ValueError as exc:
        raise ValueError("PPO_MAX_STEPS must be an integer") from exc
os.makedirs(cfg.output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(cfg.output_dir, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

dtype = torch.bfloat16 if cfg.bf16 else torch.float16
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(
    cfg.policy_base, 
    use_fast=True,
    trust_remote_code=True,
    padding_side='left'
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading policy with value head…")
quant_cfg = None
if cfg.load_policy_in_4bit:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

# 先加载基础模型（不带value head）
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.policy_base,
    torch_dtype=dtype,
    device_map={"": local_rank},
    quantization_config=quant_cfg,
    trust_remote_code=True,
    use_cache=False
)

# 应用LoRA适配器
if os.path.isdir(cfg.lora_adapter_path):
    base_model = PeftModel.from_pretrained(base_model, cfg.lora_adapter_path, device_map={"": local_rank})
else:
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    base_model = get_peft_model(base_model, lora_cfg)

# 添加value head
policy_vhead = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map={"": local_rank}
)

if hasattr(policy_vhead, "v_head"):
    torch.nn.init.zeros_(policy_vhead.v_head.summary.weight) if hasattr(policy_vhead.v_head, "summary") else None
    # 常见结构：Linear
    if hasattr(policy_vhead.v_head, "weight"):
        torch.nn.init.xavier_uniform_(policy_vhead.v_head.weight)
    if hasattr(policy_vhead.v_head, "bias") and policy_vhead.v_head.bias is not None:
        torch.nn.init.zeros_(policy_vhead.v_head.bias)

if cfg.gradient_checkpointing and hasattr(policy_vhead, "gradient_checkpointing_enable"):
    policy_vhead.gradient_checkpointing_enable()
    policy_vhead.config.use_cache = False

# Freeze base, train only LoRA
for n, p in policy_vhead.named_parameters():
    p.requires_grad = ("lora_" in n) or ("lora" in n) or ("v_head" in n)

# -------------------- reference model: reload & freeze (no deepcopy) --------------------
print("Creating reference model (reload & freeze)…")  # <<< CHANGED block
# 先加载基础模型
ref_base = AutoModelForCausalLM.from_pretrained(
    cfg.policy_base,
    torch_dtype=dtype,
    device_map={"": local_rank},
    quantization_config=quant_cfg,
    trust_remote_code=True,
    use_cache=False
)

# 加载相同的LoRA适配器
if os.path.isdir(cfg.lora_adapter_path):
    ref_model = PeftModel.from_pretrained(ref_base, cfg.lora_adapter_path, device_map={"": local_rank}, use_cache=False)
else:
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    ref_model = get_peft_model(ref_base, lora_cfg)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False


def call_qwen(prompt):
    """ 调用 Qwen3-325B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score



# PPO + Accelerate / DeepSpeed
deepspeed_plugin = None
if cfg.use_deepspeed:
    ds_config = {
        "zero_optimization": {
            "stage": cfg.zero_stage
        },
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "train_batch_size": cfg.ppo_batch_size * 1,  # world_size由accelerate接管
        "gradient_clipping": 1.0,
    }

    if cfg.bf16:
        ds_config["bf16"] = {"enabled": True}
    else:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,           # 动态loss scale（关键）
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    accelerator_kwargs = {
        "mixed_precision": "bf16" if cfg.bf16 else "fp16",
        "deepspeed_plugin": deepspeed_plugin
    }

ppo_config = PPOConfig(
    model_name=cfg.policy_base,
    learning_rate=cfg.learning_rate,
    batch_size=cfg.ppo_batch_size,
    mini_batch_size=cfg.ppo_mini_batch_size,
    ppo_epochs=cfg.ppo_epochs,
    gradient_accumulation_steps=cfg.grad_accum_steps,
    init_kl_coef=cfg.init_kl_coef,
    target_kl=cfg.target_kl,
    adap_kl_ctrl=True,
    log_with=None,
    accelerator_kwargs=accelerator_kwargs,
    seed=cfg.seed,
    max_grad_norm=cfg.max_grad_norm,
    force_accelerator_device=True
)

class JsonConversationDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                convs = data["conversations"]
                user_msg = ""
                assistant_msg = ""
                for msg in convs:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_msg = msg["content"]

                if user_msg and assistant_msg:
                    self.samples.append({
                        "query": user_msg,
                        "response": assistant_msg
                    })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query = sample["query"]
        response = sample["response"]
        return {"query": query, "response": response}

train_ds = JsonConversationDataset(
    path=str(BASE_DIR / "data" / "PPO" / "Single_train.json"),
    tokenizer=tokenizer,
    max_length=128
)

# 奖励打分Prompt函数（确定性：固定样本顺序）
def rm_score_prompt(query: str, response: str) -> str:
    style_samples = [
        "嘿 好想我宝宝~",
        "要吃点甜甜的缓一缓嘛",
        "打工仔今天很听话呐",
        "不准不开心啦🙅🏻‍♂️",
        "抱抱宝贝，不生气啦",
        "男朋友来接宝贝下班啦！",
        "我的傻丫头今天就是厉害嘛",
        "可恶💢 我又想女朋友了",
        "我滴宝藏宝宝😊",
        "好叭好叭傻丫头说了算！",
        "宝贝不生气！男朋友错啦",
        "我们打工仔这么优秀，一定没问题的！",
        "报告🙋‍ 我想女朋友啦",
        "傻丫头不许喝太多酒啦",
        "没醋硬吃😗",
        "我知道错了嘛😗",
    ]
    samples = random.sample(style_samples, 5)

    prompt = f"""
你是一名严格的语言风格评审员。请根据下列维度对候选回答进行评分，仅输出 JSON，不要解释。

[评分维度定义]  
1. Style Alignment(风格对齐度) —— 回答是否贴近示例风格。  
   - 0–2分: 完全不符合，风格生硬或偏离严重  
   - 2–4分: 部分符合，存在一定风格元素但不够自然  
   - 4–5分: 高度符合，整体自然且风格贴合示例  

2. Relevance(关联度) —— 回答与用户输入的相关性。  
   - 0–2分: 大部分内容无关或偏离主题  
   - 2–4分: 有一定相关性，但存在冗余或偏题内容  
   - 4–5分: 高度相关，内容紧扣用户输入  

3. Conciseness(简洁度) —— 回答是否简洁明了。  
   - 0–2分: 冗长、啰嗦，包含重复表达或过多无效信息 
   - 2–4分: 基本简洁，但仍有无效信息  
   - 4–5分: 非常简洁，直观明了  

4. Role Cognition Clarity (角色认知清晰度)  
   - 0-2分: 严重偏离对方“男朋友”角色定位，把自己当成“女朋友”或其他人
   - 2-4分: 基本符合对方“男朋友”角色，但存在偶尔模糊或轻微角色混乱
   - 4-5分: 完全以对方“男朋友”角色定位回复消息，自称和语气一致无误

[风格示例(5条)]  
1) {samples[0]}  
2) {samples[1]}  
3) {samples[2]}  
4) {samples[3]}  
5) {samples[4]}  

[本轮对话]  
User: {query}  
Assistant(candidate): {response}  

[输出JSON格式]  
{{
    "style_alignment": 0-5,
    "relevance": 0-5,
    "conciseness": 0-5,
    "role_cognition_clarity": 0-5
}}  
仅输出以上JSON，禁止多余文本。
"""
    return prompt

# 维护历史统计量
reward_stats = {
    "style_alignment": {"mean": 4.15, "std": 1.0, "var": 1.0},
    "relevance": {"mean": 4.5, "std": 1.0, "var": 1.0},
    "conciseness": {"mean": 4.5, "std": 1.0, "var": 1.0},
    "role_cognition_clarity": {"mean": 4.5, "std": 1.0, "var": 1.0},
}
alpha = 0.03  # 指数滑动更新率

def update_stats(dim: str, values: List[float]):
    """更新均值和标准差 (滑动平均)"""
    if not values:
        return
    mean = np.mean(values)
    var = np.var(values) + 1e-6
    reward_stats[dim]["mean"] = (1 - alpha) * reward_stats[dim]["mean"] + alpha * mean
    reward_stats[dim]["var"] = (1 - alpha) * reward_stats[dim]["var"] + alpha * var
    reward_stats[dim]["std"] = np.sqrt(reward_stats[dim]["var"])


def parse_json_with_retry(text: str, retries: int = 1):
    """防御性解析，最多重试 N 次"""
    json_pat = re.compile(r"\{[\s\S]*\}")
    for _ in range(retries):
        try:
            m = json_pat.search(text)
            js = json.loads(m.group(0) if m else text)
            return {
                "style_alignment": float(js.get("style_alignment", np.nan)),
                "relevance": float(js.get("relevance", np.nan)),
                "conciseness": float(js.get("conciseness", np.nan)),
                "role_cognition_clarity": float(js.get("role_cognition_clarity", np.nan)),
            }
        except Exception:
            continue
    # 彻底失败时返回 NaN
    return {
        "style_alignment": np.nan,
        "relevance": np.nan,
        "conciseness": np.nan,
        "role_cognition_clarity": np.nan,
    }


def compute_rewards(prompts: List[str], responses: List[str]) -> torch.Tensor:
    # 固定目标值 - 基于SFT模型表现和期望设定
    TARGETS = {
        "style_alignment": 4.8,
        "relevance": 4.8,
        "conciseness": 4.9,
        "role_cognition_clarity": 4.9
    }
    
    # 基础权重 - 反映各维度相对重要性
    BASE_WEIGHTS = {
        "style_alignment": 0.3,
        "relevance": 0.2,
        "conciseness": 0.2,
        "role_cognition_clarity": 0.3
    }
    
    K = 0.1
    
    rm_prompts = [rm_score_prompt(q, r) for q, r in zip(prompts, responses)]
    reward_device = current_device
    
    scores = [call_qwen(p) for p in rm_prompts]
    parsed = [parse_json_with_retry(d) for d in scores]

    dim_scores = {k: [] for k in TARGETS.keys()}

    for js in parsed:
        for k in TARGETS.keys():
            js[k] = js.get(k, reward_stats[k]["mean"])
            if np.isnan(js[k]):
                js[k] = reward_stats[k]["mean"]
            dim_scores[k].append(js[k])
    
    # 计算当前批次各维度的平均得分
    batch_means = {k: np.mean(dim_scores[k]) for k in TARGETS.keys()}
    
    # 计算动态权重 (基于与目标值的差距)
    dynamic_weights = {}
    for k in TARGETS.keys():
        # 计算当前得分与目标值的差距
        gap = TARGETS[k] - batch_means[k]
        # 动态调整权重: 基础权重 + 系数×差距
        dynamic_weights[k] = BASE_WEIGHTS[k] + K * max(0, gap)
        # dynamic_weights[k] = BASE_WEIGHTS[k]
    
    # 归一化权重
    total_weight = sum(dynamic_weights.values())
    normalized_weights = {k: w / total_weight for k, w in dynamic_weights.items()}
    

    logger.info(f"Dynamic weights: {normalized_weights}")

    # 奖励计算 (Z-score归一化 + 动态权重)
    rewards = []
    for i in range(len(parsed)):
        total = 0
        for k in TARGETS.keys():
            mean, std = reward_stats[k]["mean"], reward_stats[k]["std"]
            # Z-score归一化
            z_score = (dim_scores[k][i] - mean) / std
            # 应用动态权重
            total += z_score * normalized_weights[k]
        rewards.append(total)
    # 更新历史统计量 (用于Z-score归一化)
    for k in TARGETS.keys():
        update_stats(k, dim_scores[k])
    rewards = torch.tensor(rewards, dtype=torch.float32, device=reward_device)
    
    return rewards, normalized_weights




print("Initializing PPOTrainer…")
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_vhead,
    ref_model=ref_model,    # <<< CHANGED: our frozen reload
    tokenizer=tokenizer,
    dataset=train_ds
)

generation_kwargs = {
        "max_new_tokens": cfg.max_gen_len,
        "do_sample": True,  # 改为采样模式
        "temperature": 0.7,  # 添加适度的温度
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "past_key_values": None,
        "use_cache": False,
        "return_prompt": False
    }


print("Starting PPO training…")
policy_vhead.train()

total_steps = 0
if local_rank == 0:
    total_steps = cfg.max_training_steps if cfg.max_training_steps is not None else len(ppo_trainer.dataloader)


for step, batch in enumerate(ppo_trainer.dataloader):
    prompts: List[str] = batch["query"]
    current_device = torch.device(f"cuda:{local_rank}")

    
    formatted_prompts = [
        f"<|im_start|>system\n{cfg.system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{q}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for q in prompts
    ]
    
    query_tensors = [
        tokenizer(
            fq,
            add_special_tokens=False,
            max_length=cfg.max_prompt_len,
            return_tensors="pt"
        ).input_ids[0].to(current_device)
        for fq in formatted_prompts  # 遍历格式化后的prompt列表
    ]
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

    decoded_responses = []
    for query, response in zip(query_tensors, response_tensors):
        # 把 prompt 部分 decode 出来
        prompt_text = tokenizer.decode(query, skip_special_tokens=True)
        full_text = tokenizer.decode(response, skip_special_tokens=True)
        # 去掉前面的prompt，只保留assistant生成部分
        if full_text.startswith(prompt_text):
            answer = full_text[len(prompt_text):].strip()
        else:
            answer = full_text.strip()
        answer = answer.split("</think>")[-1].strip()
        decoded_responses.append(answer)
    rewards, norm_weights = compute_rewards(prompts, decoded_responses)
    rewards = rewards.to(current_device)

    rewards_list = [r for r in rewards.detach()]
    
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
    if step % 1 == 0 and local_rank == 0:
        progress = (step + 1) / total_steps * 100 if total_steps > 0 else 0.0
        metrics = {
            "step": step,
            "progress": round(progress, 2),
            "style_alignment": float(reward_stats["style_alignment"]["mean"]),
            "relevance": float(reward_stats["relevance"]["mean"]),
            "conciseness": float(reward_stats["conciseness"]["mean"]),
            "role_cognition_clarity": float(reward_stats["role_cognition_clarity"]["mean"]),
            "mean_reward": float(rewards.mean()),
            "value_loss": float(stats.get("ppo/loss/value", 0.0)),
            "entropy": float(stats.get("objective/entropy", 0.0)),
            "kl": float(stats.get("objective/kl", 0.0))
        }
        print(metrics)
        
    if (step + 1) % 50 == 0:
        ckpt_dir = os.path.join(cfg.output_dir, f"ckpt_step_{step+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        policy_vhead.save_pretrained(ckpt_dir)
    
    if cfg.max_training_steps is not None and (step + 1) >= cfg.max_training_steps:
        logger.info(f"已达到最大训练步数 {cfg.max_training_steps}，终止训练。")
        break

print("Training complete. Saving final adapter…")
policy_vhead.save_pretrained(os.path.join(cfg.output_dir, "final_adapter"))


if cfg.use_deepspeed:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

print("Done.")
