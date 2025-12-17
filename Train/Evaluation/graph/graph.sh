#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${SCRIPT_DIR}"
python graph.py \
  --log_path_1 "${PROJECT_ROOT}/Models/Saved_models/rl/4B_lora_PPO_v4/training.log" \
  --log_path_2 "${PROJECT_ROOT}/Models/Saved_models/rl/4B_lora_PPO_v5/training.log" \
  --output_dir ppo_compare_plots \
  --label_1 "PPO_v3" \
  --label_2 "PPO_v4"
