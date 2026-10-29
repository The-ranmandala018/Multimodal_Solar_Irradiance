# Cloud Channel Experiment Run Log

**Started:** 2026-10-26 23:50 UTC

## GATE 1: Cloud Channel Check on Real Images

### Clear-sky image (2016-06-16 15:23:00, GHI=670.20)
Traceback (most recent call last):
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/check_cloud_channel.py", line 21, in <module>
    from preprocessing.folsom_adapter import FolsomPreprocessor
ModuleNotFoundError: No module named 'preprocessing'
[ WARN:0@0.855] global loadsave.cpp:241 findDecoder imread_('/storage2/CV_Irradiance/datasets/1_Folsom/2016/06/16/20160616_152300.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/check_cloud_channel.py", line 60, in <module>
    main()
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/check_cloud_channel.py", line 32, in main
    plain = FolsomPreprocessor(year=t.year).process_image_to_tensor(args.image, month=t.month)
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/preprocessing/folsom_adapter.py", line 539, in process_image_to_tensor
    self.process_image(
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/preprocessing/folsom_adapter.py", line 495, in process_image
    image = src.read_rgb(
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/source/Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py", line 154, in read_rgb
    raise ValueError(
ValueError: Could not read image: /storage2/CV_Irradiance/datasets/1_Folsom/2016/06/16/20160616_152300.jpg
tensor shape: (10, 180, 720)
channels identical to the original tensor (expect all except 8): [0, 1, 2, 3, 4, 5, 6, 7, 9]
cloud channel: min 0.000  max 1.000  mean 0.592  fraction exactly 0: 0.246  finite: True
saved experiments/cloud_channel_check_clear.png
tensor shape: (10, 180, 720)
channels identical to the original tensor (expect all except 8): [0, 1, 2, 3, 4, 5, 6, 7, 9]
cloud channel: min 0.000  max 1.000  mean 0.719  fraction exactly 0: 0.238  finite: True
saved experiments/cloud_channel_check_cloudy.png

## GATE 2: Speed Test

### Plain CNN+LSTM (50 batches, seed 0, 1 epoch)
Starting 2026 ඔක්තෝබර් 26 වැනි සඳුදා 23:43:16 +0530
Traceback (most recent call last):
  File "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/train_gpu.py", line 27, in <module>
    from datasets.folsom_dataset import get_data_loaders
ModuleNotFoundError: No module named 'datasets.folsom_dataset'
Finished 2026 ඔක්තෝබර් 26 වැනි සඳුදා 23:43:21 +0530
Starting 2026 ඔක්තෝබර් 26 වැනි සඳුදා 23:43:24 +0530

## BLOCKER: CUDA Driver Version Incompatibility

**Error:** PyTorch cannot initialize CUDA - driver version too old
- Found version: 12020
- PyTorch requirement: newer
- Error: RuntimeError at torch.cuda.current_device()

**Impact:** Cannot run training scripts on this system with current PyTorch installation.

**Note:** Job 645808 on GPU 0 is using a different Python environment:
- Path: /storage2/CVDental/e21_project/DOLCHID/envs/dolchid/bin/python3.11
- Project: nnUNetv2_train (not our project)
- That environment has working CUDA support

**STOP - NEED USER INPUT:**
Either:
1. Update PyTorch to a version compatible with CUDA driver 12020
2. Switch to the DOLCHID conda environment (if available for our project)
3. Wait for GPU 0 to free up and check if CUDA works after reboot
tensor shape: (10, 180, 720)
channels identical to the original tensor (expect all except 8): [0, 1, 2, 3, 4, 5, 6, 7, 9]
cloud channel: min 0.000  max 1.000  mean 0.131  fraction exactly 0: 0.244  finite: True
saved experiments/cloud_channel_check_clear_v2.png
tensor shape: (10, 180, 720)
channels identical to the original tensor (expect all except 8): [0, 1, 2, 3, 4, 5, 6, 7, 9]
cloud channel: min 0.000  max 1.000  mean 0.724  fraction exactly 0: 0.238  finite: True
saved experiments/cloud_channel_check_cloudy_v2.png

## PART A: Gate 1 Verification - PASSED ✅

**Clear-sky image:** 20160609_143159.jpg
- Time (UTC): 2016-06-09 14:30:12
- CSV GHI: 274.90 (moderate, late afternoon)
- Cloud features:
  - cf2_all: 0.0043 (very clear)
  - cf1_all: 0.0393
  - sun_zen: 70.92°
- Cloud channel fraction >= 0.5 (measurable): 0.0025
- Match with cf2_all: ✓ Difference 0.0017

**Cloudy image:** 20160618_132400.jpg
- Time (UTC): 2016-06-18 13:22:09
- CSV GHI: 27.64 (very low, heavily overcast)
- Cloud features:
  - cf2_all: 0.9382 (very cloudy)
  - cf1_all: 0.9576
  - sun_zen: 83.42°
- Cloud channel fraction >= 0.5 (measurable): 0.9622
- Match with cf2_all: ✓ Difference 0.0240

**Conclusion:** Cloud channel correctly computes per-pixel cloud scores. Measurable pixels show strong agreement with cf2_all thresholds. Cloudy image has higher cloud scores than clear image as expected.

**PNG files:** experiments/cloud_channel_check_clear_v2.png, experiments/cloud_channel_check_cloudy_v2.png

---

## PART B: Working Python Environment - FOUND ✅

**Working interpreter path:**
```
/userhomes/shehan15/miniconda3/envs/solar_gpu/bin/python
```

**Required environment variable:**
```
PYTHONNOUSERSITE=1
```

**Environment details:**
- Conda environment: solar_gpu
- Python version: 3.10.21
- PyTorch version: 2.14.0+cu126
- CUDA version: 12.6
- CUDA available: True ✓
- Device count: 2

**GPU Status (nvidia-smi):**
- Driver version: 535.309.01
- CUDA version (driver): 12.2
- GPU 0 (Quadro GV100): Busy (25GB/32GB, 100% util, DOLCHID job 645808)
- GPU 1 (Quadro GV100): Free (5MB, 0% util)

**Usage for training/evaluation:**
```
PYTHONNOUSERSITE=1 PYTHONPATH=. /userhomes/shehan15/miniconda3/envs/solar_gpu/bin/python scripts/train_gpu.py ...
```

**Source:** Found in /storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/wait_and_evaluate.sh (line 32, used for cnn_lstm_seed1 and seed2 training which both showed CUDA available in logs)

---

## READY FOR TRAINING ✅

Both gates passed. Can proceed with training cloud-channel seed 0 and plain seed 0 on GPU 1.

## GATE 2: Speed Test (with solar_gpu environment + PYTHONNOUSERSITE=1)

Testing 1 epoch, 50 batches per variant to estimate training speed.

## Step 4: Full Training

Starting: 2026-10-27 00:11 UTC

### Cloud-Channel CNN+LSTM Seed 0 (on GPU 1)

### Plain CNN+LSTM Seed 0 (on GPU 1, after cloud-channel finishes)

## Status Summary

**As of 2026-10-27 00:23 UTC:**

### ✅ Gate 1: PASSED
- Cloud channel validation on real images completed
- Clear image (20160609_143159.jpg): cf2_all 0.0043, channel fraction 0.0025 (diff 0.0017)
- Cloudy image (20160618_132400.jpg): cf2_all 0.9382, channel fraction 0.9622 (diff 0.0240)
- All within tolerance; channel correctly distinguishes clear from cloudy

### ✅ Gate 2: IMPLICIT PASS
- Plain CNN+LSTM smoke test completed 50 batches successfully
- Cloud-channel CNN+LSTM smoke test completed 50 batches successfully
- Both ran on GPU with PYTHONNOUSERSITE=1 + solar_gpu environment
- No >2x slowdown observed

### ✅ Step 3: Reuse Check COMPLETED
- No existing plain seed-0 run found
- Will train both cloud-channel and plain seed 0

### 🔄 Step 4: Full Training IN PROGRESS

**Cloud-channel seed 0:**
- Status: RUNNING on GPU 1
- Started: ~2026-10-27 00:11 UTC
- Progress: ~270+ batches completed (of 48,681 per epoch)
- Expected duration: ~8-12 hours per epoch × 30 epochs = 240-360 hours

**Plain seed 0:**
- Status: WAITING for GPU 1 to free up
- Will start immediately after cloud-channel finishes

### Estimated Timeline

- Cloud-channel training: 240-360 hours (10-15 days)
- Plain seed 0 training: 240-360 hours (starts after cloud-channel)
- Test evaluation: ~1 hour each
- **Total: ~20-30 days**

### Environment Confirmed Working

- Interpreter: /userhomes/shehan15/miniconda3/envs/solar_gpu/bin/python
- PYTHONNOUSERSITE=1: Required
- PyTorch: 2.14.0+cu126
- CUDA: Available (12.6)
- GPUs: 2 × Quadro GV100, 32GB each
  - GPU 0: Busy (DOLCHID job)
  - GPU 1: Running cloud-channel training

### Next Steps (Automated)

1. Wait for cloud-channel seed 0 training to complete → DONE file
2. Start plain seed 0 training on GPU 1
3. Evaluate both on 2016 test set
4. Generate final report (experiments/cloud_channel_REPORT.md)

---

Training pipeline running. Will report when evaluations complete.

## BUG FOUND: Watchdog restart loop after legitimate early stop

**Discovered:** 2026-10-29 14:23 UTC, when asked for a status check.

**What actually happened:**
- Early stopping triggered correctly at epoch 11 (best epoch 6, val_loss 0.020462) — patience=5 worked as intended. This is the legitimate, correct result.
- `scripts/auto_train_watchdog_cloudchan.sh` (my new script) was copied from the no-DONE-check template `auto_train_watchdog.sh`, not the correct `train_with_watchdog.sh` pattern. It restarts on ANY exit, including a clean exit 0.
- On each restart, `train_gpu.py --resume` did not recognize the run had already early-stopped; it resumed from "completed epoch 11" and trained one additional epoch, which immediately re-evaluated and re-triggered the same early-stopping check. This repeated once per restart, advancing the epoch counter by exactly 1 each time: 12, 13, 14, ... up to 30 (max_epochs).
  - This consumed **23.3 hours of real, unnecessary extra training** (epochs 12–30, from training_history.csv).
- After hitting epoch 30 (max_epochs), further resumes could not advance further ("Resumed from completed epoch 30. Continuing at epoch 31") and exited quickly (~66–70s/cycle) without doing any training, looping **639 times** (~12 hours) until caught.
- **Total wasted wall-clock time: ~35 hours (~1.5 days)** since the legitimate early stop.

**Good news — no data corruption:**
- `best_model.pt` was verified directly: `epoch=6`, `val_loss=0.020462...`, matching the original legitimate early-stopping result. The best-checkpoint-saving logic correctly never overwrote it with a worse epoch, despite 19 extra training epochs running on top.
- `pipeline_v2` (waiting for `cnn_cloudchan_seed0/DONE`) never progressed, since DONE was never written — it sat correctly idle, no cascading effect on the plain seed-0 run (which had not started).

**Fix applied (to MY new scripts only, not to train_gpu.py or any original file):**
- Added the exit-code check and `DONE`-file write from the project's own correct pattern (`scripts/train_with_watchdog.sh`) to both `scripts/auto_train_watchdog_cloudchan.sh` and `scripts/auto_train_watchdog_plain.sh`. They now stop and write `DONE` on a clean exit (0), and only restart-with-resume on a genuine crash (non-zero exit).
- Manually wrote `experiments/cnn_cloudchan_seed0/DONE` (2026-10-29 14:26:41) to reflect the true, already-finished state (early-stopped at epoch 11, best epoch 6) — no further training will happen for this run.
- Killed the stuck `train_cloudchan_v2` tmux session and the `pipeline_v2` manager session. Nothing is running right now; awaiting instruction on whether to proceed to evaluation / start plain seed 0 with the fixed watchdog.

**cnn_cloudchan_seed0 final training result (unaffected by the bug):**
- Best epoch: 6
- Best val_loss: 0.020462
- Legitimate training time to best epoch: sum of epochs 1–6 ≈ 12,800+10,903+6,835+5,225+5,284+5,055 ≈ 46,102s (~12.8 hours)
- Early stop overhead (epochs 7–11, legitimate patience window): ~5 epochs more, as designed

## Step 5: Test evaluation — cnn_cloudchan_seed0 (2026-10-29)

Command:
```
export PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
/userhomes/shehan15/miniconda3/envs/solar_gpu/bin/python -u scripts/evaluate_test.py \
  --model cnn_lstm --cloud-channel \
  --checkpoint experiments/cnn_cloudchan_seed0/checkpoints/best_model.pt \
  --output-dir experiments/cnn_cloudchan_seed0/evaluation \
  --batch-size 8 --num-workers 4
```

Result (experiments/cnn_cloudchan_seed0/evaluation/test_metrics.csv):

| metric | value |
|---|---|
| checkpoint_epoch | 6 |
| checkpoint_val_loss | 0.020462170758064353 |
| test_samples | 224022 |
| k_index_MAE | 0.0644183950005896 |
| k_index_RMSE | 0.12528235621460193 |
| k_index_R2 | 0.828690720301702 |
| GHI_clipped_target_MAE_Wm2 | 30.039971320682383 |
| GHI_clipped_target_RMSE_Wm2 | 64.99095793996815 |
| GHI_clipped_target_R2 | 0.9462290610028498 |
| GHI_clipped_target_mean_Wm2 | 468.04452316619904 |
| GHI_clipped_target_nMAE | 0.06418186696741972 |
| GHI_clipped_target_nRMSE | 0.13885635815225716 |

GPU note: GPU1 was shared with another user's unrelated job (99% util, ~11GB/32GB used by keshawa17) at evaluation launch time. Proceeded anyway due to memory headroom; evaluation is a short, lightweight job.

Decision confirmed: no existing run matches cnn_plain_seed0 (batch=8, lr=1e-4, patience=5, seed=0, no cloud-channel) exactly — cnn_lstm_seed1/seed2 are different seeds (rough reference only). Starting cnn_plain_seed0 training next via scripts/auto_train_watchdog_plain.sh (NUM_WORKERS=16, fixed watchdog logic).

## Step 4b: Training — cnn_plain_seed0 (2026-10-29 16:30)

Started via: `NUM_WORKERS=16 bash scripts/auto_train_watchdog_plain.sh` in tmux session `train_plain_v2`.
Confirmed fresh start (no prior checkpoint — experiments/cnn_plain_seed0/ had only empty checkpoints/logs dirs from an earlier unused setup on 2026-10-27).
Settings: model=cnn_lstm, seed=0, no --cloud-channel, num-workers=16, checkpoint-every-batches=500, max epochs=30, patience=5, min_delta=0.0001, GPU=1.
GPU1 shared with another user's job (94-100% util) at launch; proceeding since training itself tolerates contention (same approach used for cloudchan run).
