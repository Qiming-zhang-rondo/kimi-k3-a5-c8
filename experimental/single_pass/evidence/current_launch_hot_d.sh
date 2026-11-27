#!/bin/bash

# ============================================================
# launch_hot_d.sh — D 节点（带 BF16 热窗口 recent128）
# 热窗口 export 必须放在 source ~/.bashrc 之后（否则被覆盖）
# 用法: docker exec -d mxllm_glm5.3c4_z00957521 bash /home/z00957521/launch_hot_d.sh
# ============================================================

# ---- 先执行原有环境（与 launch_d.sh 完全一致）----
source ~/.bashrc 2>/dev/null || true
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DEEPGEMM_MEGAMOE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_TOKENIZERS_V1_COEXIST=1
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_BUFFSIZE=1200
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export VLLM_DISABLE_COMPILE_CACHE=0
export MAX_NUM_SEQS=500
export MAX_MODEL_LEN=204800
export MAX_NUM_BATCHED_TOKENS=32
export GPU_MEMORY_UTILIZATION=0.9

export VLLM_TARGET_DEVICE=npu
export VLLM_PLUGINS=ascend_vllm,modelarts_ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,modelarts_general_plugin_patch

export MODEL_PATH=/data/models/GLM-5.3-w4a4c4-mxfp4-0902
export MODEL_NAME=glm

export LOCAL_IP=172.18.73.170
export NETWORK_CARD_NAME=enp39s0f3
export DP_ADDRESS=172.18.73.170

export DP_SIZE=8
export DP_SIZE_LOCAL=8
export TP_SIZE=1
export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

export SERVE_PORT=8667
export MAX_MODEL_LEN=150000
export MAX_NUM_BATCHED_TOKENS=32
export MAX_NUM_SEQS=8
export GPU_MEMORY_UTILIZATION=0.95
export NUM_SPECULATIVE_TOKENS=3

export KV_PORT=30300
export ENGINE_ID=3
export KV_PREFILL_DP_SIZE=1
export KV_PREFILL_TP_SIZE=8
export KV_DECODE_DP_SIZE=8
export KV_DECODE_TP_SIZE=1

export LOG_PATH=/home/z00957521/logs
export VLLM_PROFILER_DIR=/home/z00957521/profiling

# ---- 热窗口（recent128）— 必须在 bashrc 之后！----
export PYTHONPATH=/tmp/mxllm-split4-experiment:${PYTHONPATH}

bash /home/z00957521/a5_1p1d/start_d.sh D0
