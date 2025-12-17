#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# 使用DeepSpeed启动训练，并将输出重定向到日志文件
deepspeed train.py \
--num_gpus 2 \
> "${PROJECT_ROOT}/Logs/aspo_train.log" 2>&1
