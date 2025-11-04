#!/bin/bash

set -euo pipefail

export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/../RAG/llm/RAG_1.7B_sft_v2" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/model_output.log"

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL_ROOT}" \
    --tensor-parallel-size 1 \
    --port 8888 \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser hermes \
    --max-model-len 30000 \
    > "${LOG_FILE}" 2>&1 &
