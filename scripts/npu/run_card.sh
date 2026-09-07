#!/usr/bin/env bash
# Serially train+eval all 6 conditions for one seed on one NPU.
# Usage: run_card.sh <npu_logical_id> <seed> <master_port>
set -eo pipefail
NPU=$1
SEED=$2
PORT=$3
TRAIN=/mnt/models/CODE/mzy/bilingual_npu_cluster/training

for CONDITION in iid_50 en_only zh_only en_first zh_first alternating; do
    echo "=== [$(date '+%F %T')] card $NPU: train ${CONDITION}-s${SEED} ==="
    bash "$TRAIN/run_one.sh" "$CONDITION" "$SEED" "$NPU" "$PORT" 3072
    echo "=== [$(date '+%F %T')] card $NPU: eval ${CONDITION}-s${SEED} ==="
    bash "$TRAIN/eval_run.sh" "${CONDITION}-s${SEED}" "$SEED" "$NPU"
done
echo "=== [$(date '+%F %T')] card $NPU seed $SEED ALL DONE ==="
