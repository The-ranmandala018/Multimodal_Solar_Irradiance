#!/bin/bash
# Watchdog for the cloud-channel CNN+LSTM seed-0 trial run.
# New script (copy of auto_train_watchdog.sh), NOT an edit of the original.
# Adds a configurable NUM_WORKERS and single-threaded BLAS/OpenCV env vars
# to test whether CPU-side preprocessing (cloud-feature lookups, image
# decode) was the training bottleneck.
#
# If training exits, restart it automatically with --resume.

set -u

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
CONDA_ENV="solar_gpu"
GPU_ID="1"

EXP_DIR="$PROJECT_DIR/experiments/cnn_cloudchan_seed0"
LOG_DIR="$EXP_DIR/logs"
CHECKPOINT="$EXP_DIR/checkpoints/last_checkpoint.pt"
MID_CHECKPOINT="$EXP_DIR/checkpoints/mid_epoch_checkpoint.pt"
DONE="$EXP_DIR/DONE"

NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

echo "============================================================"
echo "Cloud-Channel CNN+LSTM Seed 0 Training Watchdog"
echo "GPU: $GPU_ID  NUM_WORKERS: $NUM_WORKERS"
echo "OMP/OPENBLAS/MKL/OPENCV threads: 1"
echo "============================================================"

if [ -f "$DONE" ]; then
    echo "cnn_cloudchan_seed0 already finished on $(cat "$DONE"). Nothing to do."
    exit 0
fi

while true; do
    if pgrep -f "python scripts/train_gpu.py.*cnn_cloudchan_seed0" > /dev/null; then
        sleep 60
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training is not running."

    if [ -f "$MID_CHECKPOINT" ] || [ -f "$CHECKPOINT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint found. Resuming training."
        RESUME_FLAG="--resume"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No checkpoint found. Starting fresh."
        RESUME_FLAG=""
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$PROJECT_DIR" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    python scripts/train_gpu.py \
        --model cnn_lstm \
        --seed 0 \
        --cloud-channel \
        --num-workers "$NUM_WORKERS" \
        --checkpoint-every-batches 500 \
        --experiment-dir "$EXP_DIR" \
        --checkpoint "$EXP_DIR/checkpoints/best_model.pt" \
        $RESUME_FLAG \
        >> "$LOG_DIR/auto_training.log" 2>&1

    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code $EXIT_CODE."

    if [ "$EXIT_CODE" -eq 0 ]; then
        date '+%F %T' > "$DONE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished normally. Stopping (not resuming a finished run)."
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Non-zero exit (crash). Restarting in 30 seconds..."
    sleep 30
done
