#!/bin/bash
# Wait for a training run to finish, then evaluate its best checkpoint on the
# FULL chronological 2016 test set (single GPU, so predictions stay in time order).
#
# Usage:
#   scripts/wait_and_evaluate.sh <experiment_name> <gpu_id>
# Example (in tmux so it survives disconnects):
#   tmux new -d -s eval_seed1 "bash scripts/wait_and_evaluate.sh cnn_lstm_seed1 0"
#
# * Waits until experiments/<name>/DONE exists (train_with_watchdog.sh writes it
#   only when training exits normally).
# * Skips if evaluation/test_metrics.csv already exists, so it never overwrites
#   a finished evaluation.
# * evaluate_test.py saves per-rank progress, so a crashed evaluation resumes
#   where it stopped; this script retries up to 5 times.
# Log: experiments/<name>/logs/wait_and_evaluate.log
#      experiments/<name>/logs/test_evaluation.log

set -u

if [ $# -lt 2 ]; then
    sed -n '2,17p' "$0"
    exit 1
fi

EXP_NAME="$1"
GPU_ID="$2"
CHECK_SECONDS="${CHECK_SECONDS:-600}"
MAX_ATTEMPTS=5

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
PYTHON="$HOME/miniconda3/envs/solar_gpu/bin/python"
EXP_DIR="experiments/$EXP_NAME"
EVAL_DIR="$EXP_DIR/evaluation"
CKPT="$EXP_DIR/checkpoints/best_model.pt"
LOG="$EXP_DIR/logs/wait_and_evaluate.log"

cd "$PROJECT_DIR" || exit 1
if [ ! -d "$EXP_DIR" ]; then
    echo "No such experiment: $EXP_DIR"
    exit 1
fi
mkdir -p "$EXP_DIR/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo "$(date '+%F %T') | $*"; }

log "===== waiting to evaluate $EXP_NAME on GPU $GPU_ID ====="

if [ -f "$EVAL_DIR/test_metrics.csv" ]; then
    log "$EVAL_DIR/test_metrics.csv already exists -- nothing to do"
    exit 0
fi

while [ ! -f "$EXP_DIR/DONE" ]; do
    log "training not finished yet (no DONE); checking again in ${CHECK_SECONDS}s"
    sleep "$CHECK_SECONDS"
done
log "DONE found ($(cat "$EXP_DIR/DONE"))"
grep -E "Early stopping triggered|Training finished" "$EXP_DIR/logs/auto_training.log" | tail -2

if [ ! -f "$CKPT" ]; then
    log "missing $CKPT -- stopping"
    exit 1
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    log "evaluation attempt $attempt/$MAX_ATTEMPTS on GPU $GPU_ID (full test set)"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT_DIR" \
        "$PYTHON" -u scripts/evaluate_test.py \
        --checkpoint "$CKPT" \
        --output-dir "$EVAL_DIR" \
        --batch-size 8 --num-workers 4 \
        >> "$EXP_DIR/logs/test_evaluation.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -f "$EVAL_DIR/test_metrics.csv" ]; then
        log "evaluation finished"
        cat "$EVAL_DIR/test_metrics.csv"
        exit 0
    fi
    log "evaluation exited with code $code; retrying in 60 s (progress is resumed)"
    sleep 60
done

log "evaluation failed $MAX_ATTEMPTS times -- see $EXP_DIR/logs/test_evaluation.log"
exit 1
