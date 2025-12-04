import torch
from torch.utils.data import Dataset
import os
import re
import json
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, BadRequestError
from peft import get_peft_model_state_dict, set_peft_model_state_dict


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

def update_ref_adapter_full(
        policy_vhead,
        policy_adapter_name: str = "policy",
        ref_adapter_name: str = "ref",
    ):
    """
    将同一个 peft_model 里的 'policy' 适配器权重
    拷贝到 'ref' 适配器上。
    """
    base = getattr(policy_vhead, "pretrained_model", policy_vhead)
    policy_sd = get_peft_model_state_dict(base, adapter_name=policy_adapter_name)
    set_peft_model_state_dict(base, policy_sd, adapter_name=ref_adapter_name)


def update_ref_adapter_ema(
    policy_vhead,
    alpha: float = 0.02,
    policy_adapter_name: str = "policy",
    ref_adapter_name: str = "ref",
):
    """
    将同一个 peft_model 里的 'policy' LoRA 适配器参数
    以指数滑动平均(EMA)的方式更新到 'ref' 适配器上。

    ref <- (1 - alpha) * ref + alpha * policy

    Args:
        policy_vhead: 通常是 AutoModelForCausalLMWithValueHead
        alpha: EMA 系数, 越小 ref 越“钝”，越大 ref 跟得越紧
        policy_adapter_name: 策略适配器名称
        ref_adapter_name: 参考适配器名称
    """
    base = getattr(policy_vhead, "pretrained_model", policy_vhead)
    with torch.no_grad():
        policy_sd = get_peft_model_state_dict(base, adapter_name=policy_adapter_name)
        try:
            ref_sd = get_peft_model_state_dict(base, adapter_name=ref_adapter_name)
            has_ref = True
        except Exception:
            ref_sd = {}
            has_ref = False
        if not has_ref or len(ref_sd) == 0:
            set_peft_model_state_dict(base, policy_sd, adapter_name=ref_adapter_name)
            return
        ema_ref_sd = {}
        for name, p_param in policy_sd.items():
            r_param = ref_sd.get(name, None)
            ema_ref_sd[name] = (1.0 - alpha) * r_param + alpha * p_param
        set_peft_model_state_dict(base, ema_ref_sd, adapter_name=ref_adapter_name)


# 全局懒加载的 client，避免每次重建
_qwen_client = None

def get_qwen_client() -> OpenAI:
    """懒加载并复用全局 Qwen 客户端"""
    global _qwen_client
    if _qwen_client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")

        _qwen_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _qwen_client


def _call_qwen_once(prompt: str, client: OpenAI) -> str:
    """内部单次调用封装，供单条和并行复用"""
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        return {
            "style_alignment": np.nan,
            "relevance": np.nan,
            "conciseness": np.nan,
            "role_cognition_clarity": np.nan,
        }
    except Exception as e:
        print(f"Qwen 调用失败: {e}")
        return {
            "style_alignment": np.nan,
            "relevance": np.nan,
            "conciseness": np.nan,
            "role_cognition_clarity": np.nan,
        }

def call_qwen(prompt: str) -> str:
    """
    单条调用版本：保持你原来的签名不变，但复用 client。
    """
    client = get_qwen_client()
    return _call_qwen_once(prompt, client)


def call_qwen_batch(prompts: List[str], max_workers: int = 4) -> List[str]:
    """
    批量并行调用 Qwen：用于 PPO 里一整个 batch。
    - prompts: 一批 RM prompt
    - max_workers: 并发线程数（可根据 QPS/限流调小调大）
    """
    if not prompts:
        return []
    client = get_qwen_client()
    results = []
    # 用线程池并行发请求，隐藏网络延迟
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call_qwen_once, p, client) for p in prompts]
        for fut in futures:
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "style_alignment": np.nan,
                    "relevance": np.nan,
                    "conciseness": np.nan,
                    "role_cognition_clarity": np.nan,
                }
            results.append(r)

    return results

def _normalize_text(text: str) -> str:
    # 非常简单的归一化：去掉多余空格，统一一下符号
    text = text.strip()
    text = re.sub(r"\s+", "", text)  # 中文为主的话，干脆全去掉空格
    return text

def _char_ngrams(s: str, n: int = 3):
    # 字符级 n-gram，适合中文
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

def batch_repetition_penalty(
    responses: List[str],
    n: int = 3,
    min_len: int = 6,
    strength: float = 0.3,
    device=None,
) -> torch.Tensor:
    """
    对一个 batch 的回复做“相似度惩罚”，返回一个形状为 (B,) 的 penalty 向量（负数）。
    - n: n-gram 长度，中文建议 2~4
    - min_len: 太短的句子不惩罚
    - strength: 惩罚强度系数
    """
    B = len(responses)
    if B == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    norms = [_normalize_text(r or "") for r in responses]
    ngram_sets = []
    valid = []

    for s in norms:
        if len(s) < min_len:
            ngram_sets.append(set())
            valid.append(False)
        else:
            ngram_sets.append(_char_ngrams(s, n=n))
            valid.append(True)

    # 计算 pairwise Jaccard，相似度高说明模板化严重
    sims = np.zeros((B, B), dtype=np.float32)

    for i in range(B):
        if not valid[i]:
            continue
        for j in range(i+1, B):
            if not valid[j]:
                continue
            A, Bset = ngram_sets[i], ngram_sets[j]
            if not A or not Bset:
                continue
            inter = len(A & Bset)
            union = len(A | Bset)
            if union == 0:
                continue
            jacc = inter / union
            sims[i, j] = jacc
            sims[j, i] = jacc

    # 对于每个样本，取“与其它样本的平均相似度”
    mean_sim = sims.sum(axis=1) / np.maximum(
        (np.array(valid, dtype=np.float32).sum() - 1), 1.0
    )

    # 把相似度映射成负奖励：越像别人，惩罚越大
    # 例如 mean_sim ~ 0.5 -> penalty ~ -0.15（取决于 strength）
    penalties = -strength * mean_sim.astype(np.float32)

    if device is None:
        raise ValueError("device 必须指定")
    return torch.tensor(penalties, dtype=torch.float32, device=device)

def batch_edge_ngram_penalty(
    responses: List[str],
    n: int = 2,
    edge_len: int = 4,
    min_len: int = 4,
    strength: float = 0.5,
    device=None,
) -> torch.Tensor:
    """
    只看【前 edge_len 个字符】和【后 edge_len 个字符】，
    分别算 n-gram Jaccard 相似度，然后做平均，作为“边缘相似度惩罚”。

    - n: n-gram 长度，中文建议 2 或 3
    - edge_len: 前缀/后缀截取长度（字符数）
    - min_len: 回复长度太短则不参与
    - strength: 惩罚强度系数，最终 penalty ≈ -strength * edge_mean_sim
    """
    B = len(responses)
    if B == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if device is None:
        raise ValueError("device 必须指定")

    # 1) 标准化文本
    norms = [_normalize_text(r or "") for r in responses]

    # 2) 提取前缀 / 后缀 的 n-gram 集合
    prefix_sets = []
    suffix_sets = []
    valid = []

    for s in norms:
        if len(s) < min_len:
            prefix_sets.append(set())
            suffix_sets.append(set())
            valid.append(False)
        else:
            prefix = s[:edge_len]
            suffix = s[-edge_len:]
            prefix_sets.append(_char_ngrams(prefix, n=n))
            suffix_sets.append(_char_ngrams(suffix, n=n))
            valid.append(True)

    # 3) 计算 pairwise Jaccard（前缀和后缀分别算，再取平均）
    sims = np.zeros((B, B), dtype=np.float32)

    for i in range(B):
        if not valid[i]:
            continue
        for j in range(i + 1, B):
            if not valid[j]:
                continue

            # 前缀 n-gram Jaccard
            A_p, B_p = prefix_sets[i], prefix_sets[j]
            if A_p and B_p:
                inter_p = len(A_p & B_p)
                union_p = len(A_p | B_p)
                jacc_p = inter_p / union_p if union_p > 0 else 0.0
            else:
                jacc_p = 0.0

            # 后缀 n-gram Jaccard
            A_s, B_s = suffix_sets[i], suffix_sets[j]
            if A_s and B_s:
                inter_s = len(A_s & B_s)
                union_s = len(A_s | B_s)
                jacc_s = inter_s / union_s if union_s > 0 else 0.0
            else:
                jacc_s = 0.0

            # 前缀 / 后缀相似度做平均作为“边缘相似度”
            edge_sim = 0.5 * (jacc_p + jacc_s)

            sims[i, j] = edge_sim
            sims[j, i] = edge_sim

    # 4) 对每个样本，取“与其它样本的平均边缘相似度”
    valid_count = np.array(valid, dtype=np.float32).sum()
    if valid_count <= 1:
        # 基本没人有效，不惩罚
        return torch.zeros(B, dtype=torch.float32, device=device)

    mean_sim = sims.sum(axis=1) / np.maximum((valid_count - 1.0), 1.0)

    # 5) 映射成负奖励：越像别人，惩罚越大
    penalties = -strength * mean_sim.astype(np.float32)
    return torch.tensor(penalties, dtype=torch.float32, device=device)