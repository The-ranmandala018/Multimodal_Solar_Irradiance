#!/bin/bash
# Automatic 2-GPU test-evaluation watchdog.
# The evaluator saves per-rank progress checkpoints and resumes automatically.

set -u

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
CONDA_ENV="solar_gpu"
GPU_IDS="0,1"

EVAL_DIR="$PROJECT_DIR/experiments/baseline_lstm_cnn/evaluation"
LOG_DIR="$PROJECT_DIR/experiments/baseline_lstm_cnn/logs"
EVAL_LOG="$LOG_DIR/auto_evaluation.log"

mkdir -p "$EVAL_DIR" "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR"

echo "============================================================"
echo "Automatic Solar Irradiance Test-Evaluation Watchdog"
echo "GPUs: $GPU_IDS"
echo "============================================================"

while true; do
    # Do not start a second evaluation if one is already running.
    if pgrep -f "scripts/evaluate_test.py" > /dev/null; then
        sleep 60
        continue
    fi

    # Final metrics mean evaluation is already complete.
    if [ -f "$EVAL_DIR/test_metrics.csv" ] && [ -f "$EVAL_DIR/test_predictions.npz" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final test evaluation already exists."
        echo "Nothing to restart."
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting/resuming test evaluation."

    CUDA_VISIBLE_DEVICES="$GPU_IDS"     PYTHONNOUSERSITE=1     PYTHONPATH="$PROJECT_DIR"     python -m torch.distributed.run         --standalone         --nproc_per_node=2         scripts/evaluate_test.py         --batch-size 8         --num-workers 4         --checkpoint-every-batches 100         >> "$EVAL_LOG" 2>&1

    EXIT_CODE=$?

    if [ -f "$EVAL_DIR/test_metrics.csv" ] && [ -f "$EVAL_DIR/test_predictions.npz" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test evaluation completed successfully."
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation exited with code $EXIT_CODE."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saved progress will be used on restart."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in 30 seconds..."
    sleep 30
done
