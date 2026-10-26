#!/bin/bash
# Run Mamba-LSTM seeds one after another on one GPU (or one GPU pair), resuming unfinished ones.
#
# Usage:
#   GPUS=1 scripts/mamba_lstm_queue.sh <seed> [<seed> ...]
#
# For each seed in order:
#   * skip it if <run-root>/seed<N>/DONE exists;
#   * wait until every GPU in $GPUS is free of other compute processes (and has
#     < 2 GB in use) -- utilisation is never used as the "free" signal;
#   * run scripts/run_mamba_lstm.sh, which resumes from the latest verified
#     checkpoint or starts the seed if it has none.
# The queue stops at the first seed that does not finish cleanly (stop request,
# out of memory, error), so nothing keeps running unattended after a problem.
# Running it again later continues where it stopped.
#
# Environment: GPUS [1], RUN_ROOT [experiments/mamba_lstm], GPU_WAIT_SECONDS [60],
#              EXTRA_ARGS (extra train_mamba_lstm.py arguments; smoke tests only)
# Normally started inside tmux by scripts/mamba_lstm_start.sh.

set -u
if [ $# -lt 1 ]; then
    sed -n '2,21p' "$0"
    exit 1
fi

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
GPUS="${GPUS:-1}"
RUN_ROOT="${RUN_ROOT:-experiments/mamba_lstm}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
cd "$PROJECT_DIR" || exit 1

ts() { date '+%F %T'; }

gpu_busy_reason() {
    # Prints why a GPU in $GPUS cannot be used now; prints nothing when all are free.
    local id uuid used procs
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "nvidia-smi is not responding (driver not ready?)"
        return
    fi
    # Our own Mamba trainers count as soon as they start: they spend about a minute
    # building datasets before the GPU shows any compute process.
    local pid devs
    for pid in $(pgrep -u "$(id -u)" -f "scripts/train_mamba_lstm.py"); do
        devs=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
        for id in ${GPUS//,/ }; do
            case ",$devs," in *",$id,"*)
                echo "GPU $id is assigned to Mamba trainer PID $pid ($(tr '\0' ' ' < /proc/$pid/cmdline | grep -o -- '--run-dir [^ ]*'))"
                return ;;
            esac
        done
    done
    for id in ${GPUS//,/ }; do
        uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$id" '$1==i{print $2}')
        if [ -z "$uuid" ]; then
            echo "GPU $id is not visible"
            return
        fi
        procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u{print $2}' | tr '\n' ' ')
        if [ -n "$procs" ]; then
            echo "GPU $id has compute processes: $(for p in $procs; do ps -o user=,pid=,args= -p "$p" 2>/dev/null | cut -c1-80; done | tr '\n' ';')"
            return
        fi
        used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v i="$id" '$1==i{print $2}')
        if [ "$used" -gt 2048 ]; then
            echo "GPU $id has ${used} MiB in use"
            return
        fi
    done
}

for SEED in "$@"; do
    RUN_DIR="$RUN_ROOT/seed$SEED"
    if [ -f "$RUN_DIR/DONE" ]; then
        echo "[$(ts)] seed $SEED already finished: $(cat "$RUN_DIR/DONE")"
        continue
    fi
    while true; do
        reason=$(gpu_busy_reason)
        [ -z "$reason" ] && break
        echo "[$(ts)] seed $SEED waiting for GPU(s) $GPUS: $reason"
        sleep "$GPU_WAIT_SECONDS"
    done
    echo "[$(ts)] seed $SEED starting/resuming on GPU(s) $GPUS -> $RUN_DIR"
    # shellcheck disable=SC2086
    GPUS="$GPUS" RUN_DIR="$RUN_DIR" scripts/run_mamba_lstm.sh "$SEED" $EXTRA_ARGS
    CODE=$?
    if [ "$CODE" -ne 0 ]; then
        echo "[$(ts)] seed $SEED ended with code $CODE; stopping this queue. Rerun the same command to continue."
        exit "$CODE"
    fi
done
echo "[$(ts)] queue finished: seeds $* on GPU(s) $GPUS"
