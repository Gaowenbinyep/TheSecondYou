
import os
import json
import random
import re
import logging
from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from accelerate import DeepSpeedPlugin
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from utils import (
    JsonConversationDataset,
    call_qwen_batch,
    batch_repetition_penalty,
    batch_edge_ngram_penalty,
    update_ref_adapter_ema,
    update_ref_adapter_full
)


local_rank = int(os.environ.get("LOCAL_RANK", "0"))
current_device = torch.device(f"cuda:{local_rank}")
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)


@dataclass
class RunCfg:
    policy_base: str = "/media/a822/82403B14403B0E83/志明/RAG/model/LLM/Qwen3-8B"
    lora_adapter_path: str = "/media/a822/82403B14403B0E83/志明/WechatRobot/Saved_models/sft/8B_lora_Baseline"
    output_dir: str = "/media/a822/82403B14403B0E83/志明/WechatRobot/Saved_models/rl/8B_lora_PPO_v1"
    system_prompt: str = """
        你是志明，你要回复你女朋友的消息。
    """

    

    # precision & perf
    bf16: bool = True
    gradient_checkpointing: bool = False

    # quantization (QLoRA)
    load_policy_in_4bit: bool = False

    # deepspeed / distributed
    use_deepspeed: bool = True
    zero_stage: int = 2
    grad_accum_steps: int = 2
    

    # LoRA cfg
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    max_training_steps: int = 300

    # PPO hyper-params
    ppo_batch_size: int = 64
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 2
    learning_rate: float = 5e-7
    kl_coef: float = 0.01
    target_kl: float = 50     # target_kl*2触发早停
    init_kl_coef: float = 0.01
    max_grad_norm: float = 0.5

    # reference adapter EMA
    ref_ema: bool = True
    ref_ema_step: int = 5
    ref_ema_decay: float = 0.002
    
    # repetition penalty
    repetition_penalty_n: int = 2
    repetition_penalty_strength: float = 0.5
    edge_length: int = 4

    # generation
    max_prompt_len: int = 128
    max_gen_len: int = 64
    temperature: float = 0.0

    # reproducibility
    seed: int = 42


cfg = RunCfg()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(cfg.output_dir, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
if local_rank != 0:
    logging.getLogger().setLevel(logging.WARNING)



os.makedirs(cfg.output_dir, exist_ok=True)

dtype = torch.bfloat16 if cfg.bf16 else torch.float16
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

print("Loading tokenizer…") if local_rank == 0 else None
tokenizer = AutoTokenizer.from_pretrained(
    cfg.policy_base, 
    use_fast=True,
    trust_remote_code=True,
    padding_side='left'
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading policy with value head…") if local_rank == 0 else None
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
    peft_model = PeftModel.from_pretrained(
        base_model, 
        cfg.lora_adapter_path, 
        device_map={"": local_rank},
        adapter_name="policy")
else:
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.add_adapter("policy", lora_cfg)

# 再加载一份adapter，用于ref
peft_model.load_adapter(cfg.lora_adapter_path, adapter_name="ref")
peft_model.set_adapter("policy")

# 添加value head
policy_vhead = AutoModelForCausalLMWithValueHead.from_pretrained(
    peft_model,
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

# 只训练 train 适配器 + v_head
for n, p in policy_vhead.named_parameters():
    if "lora_" in n or "lora" in n:
        if "ref" in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    elif "v_head" in n:
        p.requires_grad = True
    else:
        p.requires_grad = False



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


train_ds = JsonConversationDataset(
    path="/media/a822/82403B14403B0E83/志明/WechatRobot/data/PPO/Single_train.json",
    tokenizer=tokenizer,
    max_length=128
)

# 奖励打分Prompt函数
def rm_score_prompt(query: str, response: str) -> str:
    style_samples = [
        "嘿 好想我宝宝~",
        "要吃点甜甜的缓一缓嘛",
        "春娇今天很听话呐",
        "不准不开心啦🙅🏻‍♂️",
        "抱抱宝贝，不生气啦",
        "男朋友来接宝贝下班啦！",
        "我的傻丫头今天就是厉害嘛",
        "可恶💢 我又想女朋友了",
        "我滴宝藏宝宝😊",
        "好叭好叭傻丫头说了算！",
        "宝贝不生气！男朋友错啦",
        "我们春娇这么优秀，一定没问题的！",
        "报告🙋‍ 我想女朋友啦",
        "傻丫头不许喝太多酒啦",
        "没醋硬吃😗",
        "我知道错了嘛😗",
    ]
    samples = random.sample(style_samples, 5)

    prompt = f"""
        你是一名**严格的语言风格评审员**。请根据下列维度对候选回答进行打分，仅输出 JSON，不要解释。

        【总体要求】
        - 回答需要像微信里“宠溺、暖心、话不多但很到位的男朋友”在说话。
        - 语言更偏口语化、短句化，类似微信对话：**1～2 句话为最佳**，少用长句和复杂书面表达。
        - 风格应与下面给出的**示例风格高度对齐**：亲昵自然，像真实聊天而不是作文，有一点幽默感，会真正在意并接住对方的情绪。

        【打分精度要求（非常重要）】
        - 所有分数必须是 **0.0–5.0 之间的浮点数**，且必须是 **一位小数**。
        - **禁止**输出整数（如 3、4、5 不合法；3.0、4.5 合法）。
        - 请根据细微差异给出精准分数，而非粗略等级。

        [评分维度]

        1. Style Alignment（风格对齐度）
        - 0.0–2.0 分：语气生硬、像路人或官方客服，几乎看不出示例中的宠溺风格。
        - 2.0–4.0 分：大致像男朋友说话，有一定宠溺和调侃，但不够自然，或有点模板化。
        - 4.0–5.0 分：整体语气、情绪、亲昵程度与示例高度接近，自然、有画面感，像真人在聊天。

        2. Relevance（关联度）
        - 0.0–2.0 分：几乎不回应用户说的内容，只是泛泛安慰或套话。
        - 2.0–4.0 分：有接住主题，但忽略了用户的重要信息/情绪，只做了浅层回应。
        - 4.0–5.0 分：清楚接住用户的重点（事件、情绪、细节），并有针对性地安慰、调侃或承接话题。

        3. Conciseness（简洁度）
        - 0.0–2.0 分：非常啰嗦，句子过长或超过 3 句，重复大量无意义语气词/表情，信息密度很低。
        - 2.0–4.0 分：整体还算简洁，但存在一些赘述或明显可以删减的部分。
        - 4.0–5.0 分：在 1～2 句话内表达清晰、有温度，没有明显废话或重复。

        4. Role Cognition Clarity（角色认知清晰度）
        - 0.0–2.0 分：角色混乱，例如把自己当成女朋友或第三者，称呼前后矛盾，难以判断视角。
        - 2.0–4.0 分：整体是男朋友视角，但个别用词让人略微困惑。
        - 4.0–5.0 分：始终以“我”作为男朋友，对“你”在说话，自称和对对方的称呼前后一致。

        [风格示例（只用于参考语气，禁止逐字模仿）]
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
            "style_alignment": 0.0,
            "relevance": 0.0,
            "conciseness": 0.0,
            "role_cognition_clarity": 0.0
        }}
        仅输出以上 JSON，禁止多余文本。
    """
    return prompt

# 维护历史统计量
reward_stats = {
    "style_alignment": {"mean": 2.35, "std": 1.0, "var": 1.0},
    "relevance": {"mean": 3.18, "std": 1.0, "var": 1.0},
    "conciseness": {"mean": 2.47, "std": 1.0, "var": 1.0},
    "role_cognition_clarity": {"mean": 3.54, "std": 1.0, "var": 1.0},
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


def parse_reward_json(text: str):
    """防御性解析"""
    json_pat = re.compile(r"\{[\s\S]*\}")
    try:
        if isinstance(text, dict):
            js = text
        else:
            m = json_pat.search(text)
            js = json.loads(m.group(0) if m else text)
        return {
            "style_alignment": float(js.get("style_alignment", np.nan)),
            "relevance": float(js.get("relevance", np.nan)),
            "conciseness": float(js.get("conciseness", np.nan)),
            "role_cognition_clarity": float(js.get("role_cognition_clarity", np.nan)),
        }
    except Exception:
        return {
            "style_alignment": np.nan,
            "relevance": np.nan,
            "conciseness": np.nan,
            "role_cognition_clarity": np.nan,
        }
    
    


def compute_rewards(prompts: List[str], responses: List[str]) -> torch.Tensor:
    # 固定目标值 - 基于SFT模型表现和期望设定
    TARGETS = {
        "style_alignment": 5,
        "relevance": 4.8,
        "conciseness": 4.9,
        "role_cognition_clarity": 4.9
    }
    
    # 基础权重 - 反映各维度相对重要性
    BASE_WEIGHTS = {
        "style_alignment": 0.3,
        "relevance": 0.2,
        "conciseness": 0.2,
        "role_cognition_clarity": 0.2
    }
    
    K = 0.1
    
    rm_prompts = [rm_score_prompt(q, r) for q, r in zip(prompts, responses)]
    reward_device = current_device

    
    scores = call_qwen_batch(rm_prompts, max_workers=16)
    parsed = [parse_reward_json(d) for d in scores]

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
    
    # 归一化权重
    total_weight = sum(dynamic_weights.values())
    normalized_weights = {k: w / total_weight for k, w in dynamic_weights.items()}

    # 计算重复惩罚
    rep_penalty = batch_repetition_penalty(
        responses,
        n=cfg.repetition_penalty_n,
        min_len=6,
        strength=cfg.repetition_penalty_strength,
        device=reward_device,
    )
    edge_penalty = batch_edge_ngram_penalty(
        responses,
        n=cfg.repetition_penalty_n,
        edge_len=cfg.edge_length,
        min_len=10,
        strength=cfg.repetition_penalty_strength,
        device=reward_device,
    )
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
    rewards += (rep_penalty + edge_penalty)
    return rewards, normalized_weights, rep_penalty+edge_penalty


print("Initializing PPOTrainer…") if local_rank == 0 else None
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_vhead,
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
        "use_cache": True,
        "return_prompt": False
    }



print("Starting PPO training…") if local_rank == 0 else None

policy_vhead.train()
step = 0
max_steps = cfg.max_training_steps

while step < max_steps:
    for batch in ppo_trainer.dataloader:
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
        rewards, norm_weights, rep_penalty = compute_rewards(prompts, decoded_responses)
        rewards = rewards.to(current_device)

        rewards_list = [r for r in rewards.detach()]
        
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)

        # # 指数移动平均更新参考模型
        if cfg.ref_ema:
            update_ref_adapter_ema(policy_vhead, alpha=cfg.ref_ema_decay) if (step + 1) % cfg.ref_ema_step == 0 and local_rank == 0 else None

        if step % 1 == 0 and local_rank == 0:
            progress = (step + 1) / max_steps * 100 if max_steps > 0 else 0.0
            metrics = {
                "step": step,
                "progress": round(progress, 2),
                "style_alignment": float(reward_stats["style_alignment"]["mean"]),
                "relevance": float(reward_stats["relevance"]["mean"]),
                "conciseness": float(reward_stats["conciseness"]["mean"]),
                "role_cognition_clarity": float(reward_stats["role_cognition_clarity"]["mean"]),
                "rep_penalty": float(rep_penalty.mean()),
                "mean_reward": float(rewards.mean()),
                "value_loss": float(stats.get("ppo/loss/value", 0.0)),
                "entropy": float(stats.get("objective/entropy", 0.0)),
                "kl": float(stats.get("objective/kl", 0.0))
            }
            logger.info(f"TRAIN_STEP | {json.dumps(metrics, ensure_ascii=False)}")
            
        if (step + 1) % 100 == 0 and local_rank == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"ckpt_step_{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)

            if cfg.use_deepspeed:
                base_model_to_save = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
            else:
                base_model_to_save = policy_vhead
            peft_model = getattr(base_model_to_save, "pretrained_model", base_model_to_save)
            peft_model.save_pretrained(ckpt_dir)
            


            



        step += 1
        if cfg.max_training_steps is not None and step >= cfg.max_training_steps:
            logger.info(f"已达到最大训练步数 {cfg.max_training_steps}，终止训练。")
            break
        

print("Training complete. Saving final adapter…") if local_rank == 0 else None
if local_rank == 0:
    save_dir = os.path.join(cfg.output_dir, "final_adapter")
    os.makedirs(save_dir, exist_ok=True)
    if cfg.use_deepspeed:
        base_model_to_save = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    else:
        base_model_to_save = policy_vhead
    peft_model = getattr(base_model_to_save, "pretrained_model", base_model_to_save)
    
    peft_model.save_pretrained(save_dir)

if cfg.use_deepspeed:
    ppo_trainer.accelerator.wait_for_everyone()

print("Done.") if local_rank == 0 else None
