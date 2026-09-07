#!/usr/bin/env bash
# Run a list of 143M training runs on one NPU, then their evals.
# Usage: run_wave_143m.sh <npu_logical_id> <master_port_base> <condition:seed> [<condition:seed> ...]
# Env: CONCURRENCY (default 1) runs trained concurrently on this card;
#      MICRO_BATCH forwarded to run_one_143m.sh.
set -o pipefail

NPU=$1
PORT_BASE=$2
shift 2
TRAIN=/mnt/models/CODE/mzy/bilingual_npu_cluster/training
CONCURRENCY=${CONCURRENCY:-1}

i=0
for CS in "$@"; do
    C=${CS%:*}; S=${CS#*:}
    echo "=== [$(date '+%F %T')] card $NPU: launch train ${C}-s${S} (port $((PORT_BASE+i))) ==="
    bash "$TRAIN/run_one_143m.sh" "$C" "$S" "$NPU" $((PORT_BASE+i)) 10112 &
    i=$((i+1))
    if (( i % CONCURRENCY == 0 )); then wait; fi
done
wait

for CS in "$@"; do
    C=${CS%:*}; S=${CS#*:}
    echo "=== [$(date '+%F %T')] card $NPU: launch eval ${C}-s${S} ==="
    bash "$TRAIN/eval_run_143m.sh" "${C}-s${S}" "$S" "$NPU"
done

echo "=== [$(date '+%F %T')] card $NPU ALL DONE ==="
for CS in "$@"; do
    C=${CS%:*}; S=${CS#*:}
    CKPT=$TRAIN/runs_143m/${C}-s${S}/ckpt/iter_0010112
    EV=$TRAIN/runs_143m/${C}-s${S}/evals/test_step10112.json
    [ -d "$CKPT" ] && [ -f "$EV" ] && echo "OK  ${C}-s${S}" || echo "MISSING  ${C}-s${S}"
done
