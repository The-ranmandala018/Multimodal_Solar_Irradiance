#!/bin/bash

GPU_ID=1

# GPU must be below these limits
MAX_UTIL=5
MAX_MEMORY_MB=1000

# Require GPU to be free 3 times consecutively
REQUIRED_FREE_CHECKS=3
FREE_CHECKS=0

echo "=========================================="
echo " Solar Irradiance GPU Watcher"
echo " GPU: $GPU_ID"
echo " Check interval: 5 minutes"
echo "=========================================="

while true
do
    GPU_UTIL=$(nvidia-smi -i $GPU_ID \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')

    GPU_MEM=$(nvidia-smi -i $GPU_ID \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')

    echo "$(date '+%Y-%m-%d %H:%M:%S') | GPU $GPU_ID | Util: ${GPU_UTIL}% | Memory: ${GPU_MEM} MB"

    if [ "$GPU_UTIL" -le "$MAX_UTIL" ] && [ "$GPU_MEM" -le "$MAX_MEMORY_MB" ]; then

        FREE_CHECKS=$((FREE_CHECKS + 1))

        echo "GPU looks free: $FREE_CHECKS/$REQUIRED_FREE_CHECKS"

        if [ "$FREE_CHECKS" -ge "$REQUIRED_FREE_CHECKS" ]; then

            echo "=========================================="
            echo "GPU $GPU_ID is confirmed free!"
            echo "Starting GPU batch smoke test..."
            echo "=========================================="

            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export PYTHONNOUSERSITE=1

            cd /storage2/CV_Irradiance/Multimodal_Solar_Irradiance

            PYTHONPATH=. /userhomes/shehan15/miniconda3/envs/solar_gpu/bin/python scripts/gpu_batch_smoke_test.py

            echo "=========================================="
            echo "GPU batch smoke test finished."
            echo "Watcher stopped."
            echo "=========================================="

            exit 0
        fi

    else

        FREE_CHECKS=0
        echo "GPU is busy. Waiting 5 minutes..."

    fi

    sleep 300
done

