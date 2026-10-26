#!/usr/bin/env bash
# Step 6 driver (validation only; no test predictions). Resumable: rerun after any interruption.
#   - search for A, B, C in parallel (same 30 settings each), resuming finished trials
#   - stability folds for A, B, C
#   - final runs: Ridge-A, Ridge-B; LGBM A, B, C (own best), A0 (A's best), B_parity (B's best), 5 seeds
# A final run directory counts as finished only when config.json exists (written last);
# unfinished run directories are removed and redone.
set -euo pipefail
cd "$(dirname "$0")"
PY="env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$(cd ../.. && pwd) /storage2/CV_Irradiance/.envs/lgbm_cloud/bin/python"
RUNS=../../experiments/lightgbm_cloud_features/runs
LOGS=../../experiments/lightgbm_cloud_features/logs
mkdir -p "$LOGS"

for s in A B C; do
  if [ ! -f "$RUNS/search_$s/best.json" ]; then
    $PY search_lgbm.py --features $s --threads 24 >> "$LOGS/search_$s.log" 2>&1 &
  fi
done
wait
for s in A B C; do test -f "$RUNS/search_$s/best.json" || { echo "search $s did not finish"; exit 1; }; done

for s in A B C; do
  if [ ! -f "$RUNS/search_$s/stability_summary.json" ]; then
    rm -f "$RUNS/search_$s/stability_folds.csv"
    $PY stability_folds.py --features $s --threads 24 >> "$LOGS/stability_$s.log" 2>&1 &
  fi
done
wait

run() {  # name model features [params-from-set] [seed]
  local name=$1 model=$2 feats=$3 from=${4:-} seed=${5:-0} params="{}"
  local dir="$RUNS/$name"
  if [ -f "$dir/config.json" ]; then return 0; fi
  rm -rf "$dir"
  if [ -n "$from" ]; then
    params=$($PY -c "import json;p=json.load(open('$RUNS/search_$from/best.json'))['params'];p['num_threads']=24;print(json.dumps(p))")
  fi
  $PY train_tabular.py --model $model --features $feats --run-name $name --seed $seed --params "$params" >> "$LOGS/$name.log" 2>&1
}

run final_ridge_A ridge A
run final_ridge_B ridge B
for seed in 0 1 2 3 4; do
  run final_lgbm_A_seed$seed lgbm A A $seed &
  run final_lgbm_B_seed$seed lgbm B B $seed &
  run final_lgbm_C_seed$seed lgbm C C $seed &
  wait
  run final_lgbm_A0_seed$seed lgbm A0 A $seed &
  run final_lgbm_B_parity_seed$seed lgbm B_parity B $seed &
  wait
done
echo "STEP6 DONE"
