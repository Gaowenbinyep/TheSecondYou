#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path "${PROJECT_ROOT}/Models/Base_models/Qwen3-8B" \
    --adapter_name_or_path "${PROJECT_ROOT}/Models/Saved_models/rl/8B_lora_PPO/final_adapter/policy" \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir "${PROJECT_ROOT}/Models/Saved_models/rl/8B_lora_PPO/merged" \
    --export_size 2 \
    --export_legacy_format False
