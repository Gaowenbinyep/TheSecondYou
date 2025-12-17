export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
mkdir -p "${PROJECT_ROOT}/Logs"

MODEL_PATH="${1:-${WECHATROBOT_MODEL_PATH:-./your_model_path}}"

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size 1 \
    --port 8888 \
    --served-model-name your_model_name \
    --disable-log-requests \
    --max-model-len 30000 \
    > "${PROJECT_ROOT}/Logs/model_output.log" 2>&1 &
