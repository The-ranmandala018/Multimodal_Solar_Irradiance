#!/bin/bash
# Train one Mamba-LSTM seed with a crash watchdog.
#
# Usage:
#   scripts/run_mamba_lstm.sh <seed> [extra train_mamba_lstm.py args...]
#
# Environment (defaults in brackets):
#   GPUS     comma-separated GPU ids [0,1]; one id = single-GPU run, two = DDP via torchrun
#   RUN_DIR  output folder [experiments/mamba_lstm/seed<seed>]
#   ENV_PREFIX  conda env [/storage2/CV_Irradiance/.envs/mamba_lstm]
#   MAX_RESTARTS  crash restarts before giving up [20]
#
# Examples:
#   scripts/run_mamba_lstm.sh 0                       # seed 0 on GPUs 0+1 (DDP, 4 per GPU)
#   GPUS=1 scripts/run_mamba_lstm.sh 1                # seed 1 on GPU 1 only (batch 8)
#
# Safe to run again after a crash or server restart: it resumes from
# <run-dir>/checkpoints/last.pt, or exits if <run-dir>/DONE exists.
# Clean stop: touch <run-dir>/STOP (state is saved within ~25 steps).
# Restarts only on crashes; not after a clean stop (75), out of memory (76),
# non-finite loss (77) or a configuration/checkpoint error (78).
# Output: terminal and <run-dir>/logs/train.log (progress lines every 60 s, full tracebacks).

set -u

if [ $# -lt 1 ]; then
    sed -n '2,24p' "$0"
    exit 1
fi

SEED="$1"; shift
PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
ENV_PREFIX="${ENV_PREFIX:-/storage2/CV_Irradiance/.envs/mamba_lstm}"
GPUS="${GPUS:-0,1}"
RUN_DIR="${RUN_DIR:-experiments/mamba_lstm/seed$SEED}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

cd "$PROJECT_DIR" || exit 1

if [ -f "$RUN_DIR/DONE" ]; then
    echo "$RUN_DIR already finished: $(cat "$RUN_DIR/DONE")"
    exit 0
fi
if pgrep -f "train_mamba_lstm.py.*--run-dir $RUN_DIR( |$)" > /dev/null; then
    echo "Training for $RUN_DIR is already running."
    exit 1
fi

mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train.log"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=4

if [ "$NGPU" -gt 1 ]; then
    CMD=("$ENV_PREFIX/bin/torchrun" --standalone --nproc_per_node "$NGPU" scripts/train_mamba_lstm.py)
else
    CMD=("$ENV_PREFIX/bin/python" -u scripts/train_mamba_lstm.py)
fi
CMD+=(--seed "$SEED" --run-dir "$RUN_DIR" "$@")

echo "Seed $SEED on GPU(s) $GPUS -> $RUN_DIR. Log: $PROJECT_DIR/$LOG"
restarts=0
while true; do
    echo "[$(date '+%F %T')] launch: ${CMD[*]}" >> "$LOG"
    rm -f "$RUN_DIR/last_exit"
    # Output goes to the terminal/tmux pane and to the persistent log.
    "${CMD[@]}" 2>&1 | tee -a "$LOG"
    CODE=${PIPESTATUS[0]}
    # torchrun reports any worker failure as 1; the trainer records the real reason.
    if [ -f "$RUN_DIR/last_exit" ]; then
        CODE=$(head -1 "$RUN_DIR/last_exit")
    elif [ "$CODE" -eq 0 ]; then
        CODE=1  # a clean finish always writes last_exit
    fi
    case "$CODE" in
        0)  echo "[$(date '+%F %T')] finished: $(cat "$RUN_DIR/DONE" 2>/dev/null)" | tee -a "$LOG"; exit 0 ;;
        75) echo "[$(date '+%F %T')] stopped on request (resumable). Not restarting." | tee -a "$LOG"; exit 75 ;;
        76) echo "[$(date '+%F %T')] CUDA out of memory. Not restarting; see $LOG." | tee -a "$LOG"; exit 76 ;;
        77) echo "[$(date '+%F %T')] repeated non-finite loss. Not restarting; see $LOG." | tee -a "$LOG"; exit 77 ;;
        78) echo "[$(date '+%F %T')] configuration/checkpoint error. Not restarting; see $LOG." | tee -a "$LOG"; exit 78 ;;
    esac
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        echo "[$(date '+%F %T')] exit code $CODE; $MAX_RESTARTS restarts used. Giving up." | tee -a "$LOG"
        exit "$CODE"
    fi
    echo "[$(date '+%F %T')] exit code $CODE (crash). Restart $restarts/$MAX_RESTARTS in 60 s." | tee -a "$LOG"
    sleep 60
done
