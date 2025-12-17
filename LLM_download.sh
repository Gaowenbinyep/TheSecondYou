#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

mkdir -p "${PROJECT_ROOT}/Models/Base_models" "${PROJECT_ROOT}/Logs"

modelscope download \
    --model Qwen/Qwen3-8B \
    --local_dir "${PROJECT_ROOT}/Models/Base_models/Qwen3-8B" \
    > "${PROJECT_ROOT}/Logs/nohup.txt" 2>&1 &
