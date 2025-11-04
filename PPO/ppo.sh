#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 使用DeepSpeed启动训练，并将输出重定向到项目 logs 目录
deepspeed train.py \
  --num_gpus 2 \
  > "${PROJECT_ROOT}/logs/ppo_train.log" 2>&1
