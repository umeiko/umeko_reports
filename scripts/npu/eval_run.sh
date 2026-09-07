#!/usr/bin/env bash
# Evaluate one finished run: dev BPB at steps 0,192,...,3072 + test BPB at the
# final step, replicating the pilot's offline evaluation exactly.
# Usage: eval_run.sh <run_name> <seed> <npu_logical_id>
set -eo pipefail

RUN_NAME=$1
SEED=$2
NPU_ID=${3:-0}

MZY=/mnt/models/CODE/mzy
MSLLM=/mnt/models/CODE/MindSpeed-LLM-v2.3.0
source "$MZY/bilingual_npu_cluster/training/container_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID

RUN=$MZY/bilingual_npu_cluster/training/runs/$RUN_NAME
POOL=$MZY/umeko_reports/data/materialized_100m_all_sources
INIT_HF=$MZY/bilingual_npu_cluster/training/init_hf/init-s$SEED
EVAL_DIR=$RUN/evals
mkdir -p "$EVAL_DIR"

eval_one() {  # <hf_dir> <step> <split>
    local HF=$1 STEP=$2 SPLIT=$3
    local OUT=$EVAL_DIR/${SPLIT}_step$(printf %05d "$STEP").json
    if [ -f "$OUT" ]; then echo "skip $OUT (exists)"; return 0; fi
    $PY "$MZY/bilingual_npu_cluster/training/eval_bpb.py" \
        --hf-model-dir "$HF" --pool-root "$POOL" --split "$SPLIT" --out "$OUT"
}

# step 0: the raw HF init, no conversion needed
eval_one "$INIT_HF" 0 dev

# final step gets the test split too
MAX_STEP=$(ls -d "$RUN"/ckpt/iter_* | sed 's/.*iter_0*//' | sort -n | tail -1)

for CKPT in $(ls -d "$RUN"/ckpt/iter_* | sort); do
    STEP=$(basename "$CKPT" | sed 's/iter_0*//')
    VIEW=$RUN/ckpt_view
    HF_TMP=$RUN/hf_tmp_$STEP
    rm -rf "$VIEW" "$HF_TMP"
    mkdir -p "$VIEW"
    ln -s "$CKPT" "$VIEW/$(basename "$CKPT")"
    echo "$STEP" > "$VIEW/latest_checkpointed_iteration.txt"
    (cd "$MSLLM" && $PY convert_ckpt_v2.py \
        --load-model-type mg --save-model-type hf \
        --load-dir "$VIEW" --save-dir "$HF_TMP" --model-type-hf qwen3 \
        --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 >/dev/null 2>&1)
    cp "$INIT_HF/config.json" "$INIT_HF/generation_config.json" "$HF_TMP/"
    eval_one "$HF_TMP" "$STEP" dev
    if [ "$STEP" = "$MAX_STEP" ]; then
        eval_one "$HF_TMP" "$STEP" test
    fi
    rm -rf "$VIEW" "$HF_TMP"
done
echo "eval done for $RUN_NAME"
