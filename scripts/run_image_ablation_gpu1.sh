#!/usr/bin/env bash
# Run the image ablation test on physical GPU 1, one mode at a time, with
# mode-level resume.
#
# image_ablation_test.py itself cannot resume: it keeps predictions in memory
# and writes its CSV only when it finishes. So this wrapper runs each mode
# (real, shuffled, zero, ...) as a separate process with its own output folder and
# skips any mode whose CSV already exists. If the job dies, rerunning this
# script redoes only the unfinished mode(s). The subset and shuffle depend
# only on --seed, so separate per-mode runs score the same samples.
#
# Usage (from anywhere):  bash scripts/run_image_ablation_gpu1.sh [SUBSET] [MODES...]
#   default MODES: real shuffled zero
#   e.g.  bash scripts/run_image_ablation_gpu1.sh 20000 sametime stale10 stale30 stale60
set -euo pipefail

PROJECT_DIR="/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
PYTHON="$HOME/miniconda3/envs/solar_gpu/bin/python"
GPU_ID=1
SUBSET="${1:-20000}"
shift || true
MODES=("$@")
[[ ${#MODES[@]} -eq 0 ]] && MODES=(real shuffled zero)
SEED=0
TAG="subset${SUBSET}"
EVAL_DIR="$PROJECT_DIR/experiments/baseline_lstm_cnn/evaluation"
RUN_DIR="$EVAL_DIR/image_ablation_${TAG}"
LOG="$EVAL_DIR/image_ablation_${TAG}.log"
COMBINED="$EVAL_DIR/image_ablation_metrics_${TAG}.csv"

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR"

# Refuse to start a second copy.
exec 9>"$RUN_DIR/.lock"
if ! flock -n 9; then
    echo "Another ablation run already holds $RUN_DIR/.lock -- not starting." >&2
    exit 1
fi

# Append, never truncate, so earlier attempts stay in the log.
exec >>"$LOG" 2>&1
echo "===== $(date '+%F %T') start: GPU $GPU_ID, subset $SUBSET, seed $SEED, modes: ${MODES[*]} ====="

for MODE in "${MODES[@]}"; do
    OUT="$RUN_DIR/$MODE"
    if [[ -f "$OUT/image_ablation_metrics_${TAG}.csv" ]]; then
        echo "--- $MODE already done, skipping ($OUT)"
        continue
    fi
    echo "--- $(date '+%F %T') running mode: $MODE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT_DIR" \
        "$PYTHON" -u scripts/image_ablation_test.py \
        --mode "$MODE" --subset "$SUBSET" --seed "$SEED" --output-dir "$OUT"
    echo "--- $(date '+%F %T') finished mode: $MODE"
done

# Rebuild the combined table from every finished mode. It is derived only
# from the per-mode CSVs above (which are never rewritten), so regenerating
# it just adds rows for newly finished modes.
"$PYTHON" - "$RUN_DIR" "$TAG" "$COMBINED" <<'EOF'
import csv, sys
from pathlib import Path
run_dir, tag, combined = sys.argv[1:]
order = ["real", "sametime", "stale10", "stale30", "stale60", "shuffled", "zero"]
header, rows = None, []
for mode in order:
    path = Path(run_dir) / mode / f"image_ablation_metrics_{tag}.csv"
    if path.exists():
        with open(path) as f:
            r = list(csv.reader(f))
        header = r[0]
        rows.append(r[1])
with open(combined, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
rmse = {row[0]: float(row[3]) for row in rows}
if "real" in rmse:
    print("\n=== relative to real ===")
    for mode, v in rmse.items():
        if mode != "real":
            d = v - rmse["real"]
            print(f"{mode}:  RMSE {d:+.2f} W/m2  ({100 * d / rmse['real']:+.2f}% vs real)")
print(f"\nSaved: {combined}")
EOF
echo "===== $(date '+%F %T') all modes done ====="
