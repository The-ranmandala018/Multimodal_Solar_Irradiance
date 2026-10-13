#!/bin/bash
# Train the baseline CNN+LSTM into its OWN experiment folder, with a crash watchdog.
#
# Usage:
#   scripts/train_with_watchdog.sh <experiment_name> <seed> [max_epochs=30] [patience=5]
#
# Examples:
#   scripts/train_with_watchdog.sh baseline_seed1 1                # same recipe as the baseline run
#   scripts/train_with_watchdog.sh long_seed1 1 50 50              # 50 epochs, early stopping effectively off
#
# Recipe is identical to the original baseline run (single GPU, batch 8,
# lr 1e-4, AdamW, MSE) so the results are comparable. Only the seed and the
# output folder differ.
#
# Differences from auto_train_watchdog.sh:
#   * writes to experiments/<experiment_name>/, never to baseline_lstm_cnn/
#   * seeds the run
#   * restarts (with --resume) only when training CRASHES; when training
#     finishes normally it writes a DONE file and stops, instead of resuming
#     a finished run and training an extra epoch
#   * safe to run again after a server restart: it resumes from the last
#     checkpoint, or does nothing if the run already finished

set -u

if [ $# -lt 2 ]; then
    sed -n '2,15p' "$0"
    exit 1
fi

EXP_NAME="$1"
SEED="$2"
EPOCHS="${3:-30}"
PATIENCE="${4:-5}"

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
CONDA_ENV="${CONDA_ENV:-solar_gpu}"
GPU_ID="${GPU_ID:-1}"

if [ "$EXP_NAME" = "baseline_lstm_cnn" ]; then
    echo "Refusing to use baseline_lstm_cnn: that folder holds the original run."
    exit 1
fi

EXP_DIR="experiments/$EXP_NAME"
cd "$PROJECT_DIR" || exit 1
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints"

LOG="$EXP_DIR/logs/auto_training.log"
DONE="$EXP_DIR/DONE"
LAST="$EXP_DIR/checkpoints/last_checkpoint.pt"
MID="$EXP_DIR/checkpoints/mid_epoch_checkpoint.pt"

if [ -f "$DONE" ]; then
    echo "$EXP_NAME already finished on $(cat "$DONE"). Nothing to do."
    exit 0
fi

if pgrep -f "train_gpu.py.*--experiment-dir $EXP_DIR" > /dev/null; then
    echo "Training for $EXP_NAME is already running."
    exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR"

echo "Experiment: $EXP_NAME  seed=$SEED  max_epochs=$EPOCHS  patience=$PATIENCE  GPU=$GPU_ID"
echo "Log: $PROJECT_DIR/$LOG"

while true; do
    if [ -f "$MID" ] || [ -f "$LAST" ]; then
        RESUME="--resume"
        echo "[$(date '+%F %T')] Checkpoint found. Resuming." | tee -a "$LOG"
    else
        RESUME=""
        echo "[$(date '+%F %T')] No checkpoint. Starting fresh." | tee -a "$LOG"
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_gpu.py \
        --epochs "$EPOCHS" \
        --batch-size 8 \
        --num-workers 4 \
        --lr 1e-4 \
        --patience "$PATIENCE" \
        --checkpoint-every-batches 100 \
        --seed "$SEED" \
        --experiment-dir "$EXP_DIR" \
        --checkpoint "$EXP_DIR/checkpoints/best_model.pt" \
        $RESUME >> "$LOG" 2>&1
    CODE=$?

    if [ "$CODE" -eq 0 ]; then
        date '+%F %T' > "$DONE"
        echo "[$(date '+%F %T')] Training finished normally. Stopping." | tee -a "$LOG"
        exit 0
    fi

    echo "[$(date '+%F %T')] Training exited with code $CODE. Restarting in 30 s." | tee -a "$LOG"
    sleep 30
done
