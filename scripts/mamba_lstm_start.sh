#!/bin/bash
# Start (or, after a reboot or crash, resume) the Mamba-LSTM seed queues in detached tmux sessions.
#
# Usage:
#   scripts/mamba_lstm_start.sh "<gpus>:<seeds>" ["<gpus>:<seeds>" ...]
# Examples:
#   scripts/mamba_lstm_start.sh "1:0 1 2"          # all three seeds, one after another, on GPU 1
#   scripts/mamba_lstm_start.sh "1:0 2" "0:1"      # seeds 0,2 on GPU 1 and seed 1 on GPU 0, in parallel
#   scripts/mamba_lstm_start.sh "0,1:0 1 2"        # each seed on both GPUs (DDP), one after another
#
# One tmux session per queue, named mamba_gpu<gpus> (e.g. mamba_gpu1). Idempotent:
#   * a queue whose tmux session already exists is left alone (no duplicates);
#   * finished seeds (DONE) are skipped, unfinished ones resume from their latest
#     verified checkpoint; a seed already being trained is refused by its run-dir lock;
#   * each queue waits until its GPU(s) have no other compute processes.
# Nothing is installed to start at boot: after a server restart, run this same
# command again. Each seed must appear in only one queue.
#
# Environment passed through: RUN_ROOT, EXTRA_ARGS, GPU_WAIT_SECONDS (see mamba_lstm_queue.sh).

set -u
if [ $# -lt 1 ]; then
    sed -n '2,20p' "$0"
    exit 1
fi
PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
cd "$PROJECT_DIR" || exit 1

seen=" "
for spec in "$@"; do
    for s in ${spec#*:}; do
        case "$seen" in *" $s "*) echo "Seed $s appears in more than one queue; refusing."; exit 1 ;; esac
        seen="$seen$s "
    done
done

for spec in "$@"; do
    GPUS="${spec%%:*}"
    SEEDS="${spec#*:}"
    SESSION="${SESSION_PREFIX:-mamba}_gpu${GPUS//,/_}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session $SESSION already exists; not starting another (attach: tmux attach -t $SESSION)"
        continue
    fi
    ROOT="${RUN_ROOT:-experiments/mamba_lstm}"
    QLOG="$ROOT/queue_gpu${GPUS//,/_}.log"
    mkdir -p "$ROOT"
    # The session ends when its queue ends; the queue's messages are kept in $QLOG.
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
        "env GPUS='$GPUS' RUN_ROOT='$ROOT' EXTRA_ARGS='${EXTRA_ARGS:-}' GPU_WAIT_SECONDS='${GPU_WAIT_SECONDS:-60}' scripts/mamba_lstm_queue.sh $SEEDS 2>&1 | tee -a '$QLOG'"
    echo "Started tmux session $SESSION: seeds $SEEDS on GPU(s) $GPUS (queue log: $QLOG)"
done
tmux ls 2>/dev/null | grep "^${SESSION_PREFIX:-mamba}_" || true
