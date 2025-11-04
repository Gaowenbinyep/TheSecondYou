#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path "${PROJECT_ROOT}/Base_models/Qwen3-4B" \
    --adapter_name_or_path "${PROJECT_ROOT}/Saved_models/rlhf/4B_lora_PPO_V3/final_adapter" \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir "${PROJECT_ROOT}/Saved_models/rlhf/4B_lora_PPO_V3/merged" \
    --export_size 2 \
    --export_legacy_format False
