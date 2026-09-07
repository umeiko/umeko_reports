#!/usr/bin/env bash
# Train all 6 conditions for one seed on one NPU, 2 concurrent runs at a time
# (43M model, ~17.6GB reserved each -> 2 concurrent is safe on 64GB),
# then run the BPB evals 2 at a time.
# Usage: run_wave.sh <npu_logical_id> <seed> <master_port_base>
set -o pipefail

NPU=$1
SEED=$2
PORT_BASE=$3
TRAIN=/mnt/models/CODE/mzy/bilingual_npu_cluster/training
CONDS=(iid_50 en_only zh_only en_first zh_first alternating)

i=0
for C in "${CONDS[@]}"; do
    echo "=== [$(date '+%F %T')] card $NPU: launch train ${C}-s${SEED} (port $((PORT_BASE+i))) ==="
    bash "$TRAIN/run_one.sh" "$C" "$SEED" "$NPU" $((PORT_BASE+i)) 3072 &
    i=$((i+1))
    if (( i % 2 == 0 )); then wait; fi
done
wait

i=0
for C in "${CONDS[@]}"; do
    echo "=== [$(date '+%F %T')] card $NPU: launch eval ${C}-s${SEED} ==="
    bash "$TRAIN/eval_run.sh" "${C}-s${SEED}" "$SEED" "$NPU" &
    i=$((i+1))
    if (( i % 2 == 0 )); then wait; fi
done
wait

echo "=== [$(date '+%F %T')] card $NPU seed $SEED ALL DONE ==="
for C in "${CONDS[@]}"; do
    CKPT=/mnt/models/CODE/mzy/bilingual_npu_cluster/training/runs/${C}-s${SEED}/ckpt/iter_0003072
    EV=/mnt/models/CODE/mzy/bilingual_npu_cluster/training/runs/${C}-s${SEED}/evals/test_step03072.json
    [ -d "$CKPT" ] && [ -f "$EV" ] && echo "OK  ${C}-s${SEED}" || echo "MISSING  ${C}-s${SEED}"
done
