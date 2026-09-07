#!/usr/bin/env bash
# Launch one pilot-parity 43M run on a single NPU inside the mzy-mindspeed container.
# Usage: run_one.sh <condition> <seed> <npu_logical_id> <master_port> [train_iters]
set -eo pipefail

CONDITION=$1
SEED=$2
NPU_ID=$3
MASTER_PORT=$4
TRAIN_ITERS=${5:-3072}

MZY=/mnt/models/CODE/mzy
MSLLM=/mnt/models/CODE/MindSpeed-LLM-v2.3.0
POOL=$MZY/umeko_reports/data/materialized_100m_all_sources
ORDER=$MZY/bilingual_npu_cluster/training/order/order_${CONDITION}_s${SEED}.npz
MG_INIT=$MZY/bilingual_npu_cluster/training/init_mg/init-s${SEED}
RUN_NAME=${RUN_NAME:-${CONDITION}-s${SEED}}
RUN_DIR=$MZY/bilingual_npu_cluster/training/runs/${RUN_NAME}
mkdir -p "$RUN_DIR"

source /usr/local/Ascend/driver/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh

export PYTHONPATH=$MSLLM:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID

PY=/root/miniconda3/envs/ms/bin/python

$PY -m torch.distributed.run --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    "$MZY/bilingual_npu_cluster/training/pretrain_bilingual_p0.py" \
    --pilot-pool-root "$POOL" \
    --pilot-order-file "$ORDER" \
    --use-mcore-models \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --num-layers 12 \
    --hidden-size 384 \
    --num-attention-heads 6 \
    --ffn-hidden-size 1024 \
    --kv-channels 64 \
    --group-query-attention \
    --num-query-groups 2 \
    --qk-layernorm \
    --seq-length 1023 \
    --max-position-embeddings 1024 \
    --position-embedding-type rope \
    --rotary-base 100000 \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "$MZY/Haidass1.5-143M" \
    --micro-batch-size "${MICRO_BATCH:-32}" \
    --global-batch-size 32 \
    --train-iters "$TRAIN_ITERS" \
    --dataloader-type single \
    --num-workers 2 \
    --optimizer adam \
    --lr 1.5e-3 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-iters 31 \
    --lr-decay-iters 3072 \
    --weight-decay 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --clip-grad 2.0 \
    --fp16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
    --loss-scale-window 2000 \
    --hysteresis 1 \
    --load "$MG_INIT" \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --save "$RUN_DIR/ckpt" \
    --save-interval 192 \
    --no-save-optim \
    --no-save-rng \
    --ckpt-format torch \
    --eval-iters 0 \
    --eval-interval 1000000 \
    --log-interval 1 \
    --tensorboard-dir "$RUN_DIR/tb" \
    --seed "$SEED" \
    2>&1 | tee "$RUN_DIR/train.log"
