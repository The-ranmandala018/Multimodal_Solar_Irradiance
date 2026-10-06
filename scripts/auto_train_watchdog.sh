#!/bin/bash
# Automatic training watchdog for the Solar Irradiance multimodal model.
# If training exits, restart it automatically with --resume.
# The training script itself supports mid-epoch checkpoints.

set -u

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
CONDA_ENV="solar_gpu"
GPU_ID="1"
GPU_IDS="0,1"

LOG_DIR="$PROJECT_DIR/experiments/baseline_lstm_cnn/logs"
CHECKPOINT="$PROJECT_DIR/experiments/baseline_lstm_cnn/checkpoints/last_checkpoint.pt"
MID_CHECKPOINT="$PROJECT_DIR/experiments/baseline_lstm_cnn/checkpoints/mid_epoch_checkpoint.pt"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR"

echo "============================================================"
echo "Automatic Solar Irradiance Training Watchdog"
echo "GPU: $GPU_ID"
echo "============================================================"

while true; do
    # Do not start a second copy if training is already running.
    if pgrep -f "python scripts/train_gpu.py" > /dev/null; then
        sleep 60
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training is not running."

    # If either an epoch or mid-epoch checkpoint exists, resume.
    if [ -f "$MID_CHECKPOINT" ] || [ -f "$CHECKPOINT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint found. Resuming training."
        RESUME_FLAG="--resume"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No checkpoint found. Starting fresh."
        RESUME_FLAG=""
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID"     PYTHONNOUSERSITE=1     PYTHONPATH="$PROJECT_DIR"     python scripts/train_gpu.py         --epochs 30         --batch-size 8         --num-workers 4         --lr 1e-4         --patience 5         --checkpoint-every-batches 100         $RESUME_FLAG         >> "$LOG_DIR/auto_training.log" 2>&1

    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code $EXIT_CODE."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in 30 seconds..."
    sleep 30
done
