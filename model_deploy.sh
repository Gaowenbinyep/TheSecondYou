export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

CUDA_VISIBLE_DEVICES=0 \
vllm serve ./your_model_path \
    --tensor-parallel-size 1 \
    --port 8888 \
    --served-model-name your_model_name \
    --disable-log-requests \
    --max-model-len 30000 \
    > ./logs/model_output.log 2>&1 &