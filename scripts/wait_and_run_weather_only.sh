#!/bin/bash
# Wait for a free GPU, then:
#   1. evaluate weather_only_seed1 on the 2016 test set
#   2. train weather_only_seed2 (same recipe, seed 2) with the crash watchdog
#
# "Free" = utilisation <= 5 % and memory <= 1000 MB on 3 checks in a row,
# 2 minutes apart (same thresholds as wait_for_gpu.sh). Checks GPU 1 first,
# then GPU 0. Never touches other users' processes. Skips a step that is
# already done (evaluation/test_metrics.csv exists, or the seed-2 folder exists).
#
# Run in tmux so it survives disconnects:
#   tmux new -d -s wo_wait "bash scripts/wait_and_run_weather_only.sh"
# Log: experiments/weather_only_seed1/logs/wait_and_run.log

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
PYTHON="$HOME/miniconda3/envs/solar_gpu/bin/python"
MAX_UTIL="${MAX_UTIL:-5}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-1000}"
REQUIRED_FREE_CHECKS=3
CHECK_SECONDS="${CHECK_SECONDS:-120}"
DRY_RUN="${DRY_RUN:-0}"   # 1 = only print what would run (for testing)

EVAL_DIR="experiments/weather_only_seed1/evaluation"
SEED2_DIR="experiments/weather_only_seed2"
LOG="experiments/weather_only_seed1/logs/wait_and_run.log"
[ "$DRY_RUN" = 1 ] && LOG="/dev/null"

cd "$PROJECT_DIR" || exit 1
exec > >(tee -a "$LOG") 2>&1

log() { echo "$(date '+%F %T') | $*"; }

gpu_is_free() {
    local util mem
    util=$(nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    mem=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    log "GPU $1 | util ${util}% | memory ${mem} MB"
    [ "$util" -le "$MAX_UTIL" ] && [ "$mem" -le "$MAX_MEMORY_MB" ]
}

# Wait until some GPU is free 3 checks in a row; print its index on fd 3.
wait_for_free_gpu() {
    local free1=0 free0=0
    while true; do
        if gpu_is_free 1; then free1=$((free1 + 1)); else free1=0; fi
        if gpu_is_free 0; then free0=$((free0 + 1)); else free0=0; fi
        if [ "$free1" -ge "$REQUIRED_FREE_CHECKS" ]; then echo 1 >&3; return; fi
        if [ "$free0" -ge "$REQUIRED_FREE_CHECKS" ]; then echo 0 >&3; return; fi
        log "waiting (free streak: GPU1 $free1/$REQUIRED_FREE_CHECKS, GPU0 $free0/$REQUIRED_FREE_CHECKS)"
        sleep "$CHECK_SECONDS"
    done
}

log "===== watcher started ====="

# Step 1: seed 1 test evaluation
if [ -f "$EVAL_DIR/test_metrics.csv" ]; then
    log "seed 1 evaluation already done ($EVAL_DIR/test_metrics.csv) -- skipping"
    GPU=""
else
    GPU=$( { wait_for_free_gpu 3>&1 1>&2; } )
    log "GPU $GPU confirmed free -> evaluating weather_only_seed1"
    if [ "$DRY_RUN" = 1 ]; then log "DRY RUN: would run evaluate_test.py on GPU $GPU"; exit 0; fi
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT_DIR" \
        "$PYTHON" -u scripts/evaluate_test.py \
        --checkpoint experiments/weather_only_seed1/checkpoints/best_model.pt \
        --output-dir "$EVAL_DIR" \
        > experiments/weather_only_seed1/logs/test_evaluation.log 2>&1
    code=$?
    log "seed 1 evaluation exited with code $code"
    grep -E "GHI_clipped_target_(RMSE|MAE)_Wm2" experiments/weather_only_seed1/logs/test_evaluation.log
    if [ "$code" -ne 0 ]; then
        log "evaluation failed -- see experiments/weather_only_seed1/logs/test_evaluation.log. Stopping."
        exit 1
    fi
fi

# Step 2: seed 2 training
if [ -e "$SEED2_DIR" ]; then
    log "$SEED2_DIR already exists -- not starting seed 2"
    exit 0
fi
# Reuse the evaluation GPU only if nobody else has taken it meanwhile.
if [ -z "$GPU" ] || ! gpu_is_free "$GPU"; then
    GPU=$( { wait_for_free_gpu 3>&1 1>&2; } )
fi
log "GPU $GPU -> starting weather_only_seed2 (seed 2) with the training watchdog"
MODEL=weather_only GPU_ID="$GPU" bash scripts/train_with_watchdog.sh weather_only_seed2 2 30 5
log "seed 2 watchdog exited with code $?"
log "===== watcher finished ====="
