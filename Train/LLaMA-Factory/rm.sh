#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# 选择物理 GPU
export CUDA_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

# 启动分布式训练
nohup torchrun --nproc_per_node=2 src/train.py \
    --stage rm \
    --finetuning_type lora \
    --do_train \
    --model_name_or_path "${PROJECT_ROOT}/Models/Base_models/Qwen3-4B" \
    --dataset wechat_robot_rm \
    --template qwen3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --cutoff_len 2048 \
    --output_dir "${PROJECT_ROOT}/Models/Saved_models/rl/4B_lora_ppo/rm" \
    --save_safetensors False \
    --deepspeed deepspeed/ds_z3_config.json \
    --bf16 True \
    --overwrite_output_dir \
    --logging_steps 10 \
    --save_steps 1000 \
    --plot_loss \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --trust_remote_code \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --ddp_timeout 180000000 \
    > "${PROJECT_ROOT}/Logs/model_train.log" 2>&1
