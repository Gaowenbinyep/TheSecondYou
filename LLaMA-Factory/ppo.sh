#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 选择物理 GPU
export CUDA_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

# 启动分布式训练
nohup torchrun --nproc_per_node=2 src/train.py \
    --stage ppo \
    --finetuning_type lora \
    --do_train \
    --model_name_or_path ../Saved_models/sft/4B_lora \
    --adapter_name_or_path ../Saved_models/rlhf/4B_lora_ppo/checkpoint-2000 \
    --reward_model ../Saved_models/rlhf/4B_lora_ppo/rm/adapter \
    --reward_model_type lora \
    --dataset wechat_robot_ppo \
    --template qwen3 \
    --deepspeed "${SCRIPT_DIR}/deepspeed/ds_z2_config.json" \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 2 \
    --ddp_find_unused_parameters True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --cutoff_len 512 \
    --output_dir ../Saved_models/rlhf/4B_lora_ppo \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 2000 \
    --plot_loss \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --trust_remote_code \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --ddp_timeout 180000000 \
    --report_to tensorboard \
    > ../logs/model_train.log 2>&1 &
