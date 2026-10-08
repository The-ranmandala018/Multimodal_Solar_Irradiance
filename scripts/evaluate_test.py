"""Interruption-safe 2-GPU evaluation of the best Folsom CNN+LSTM checkpoint.

The evaluator mirrors the training data configuration and uses the chronological
2016 test split. Each distributed rank owns an exact, non-padded subset of test
samples. Progress is checkpointed independently per rank so a server stop can
be resumed without re-evaluating completed batches.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from datasets.folsom_dataset import get_data_loaders
from models.multimodal_model import MultimodalSolarForecastModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="experiments/baseline_lstm_cnn/checkpoints/best_model.pt",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="experiments/baseline_lstm_cnn/evaluation")
    p.add_argument("--checkpoint-every-batches", type=int, default=100)
    p.add_argument(
        "--reset-evaluation",
        action="store_true",
        help="Ignore existing per-rank evaluation checkpoints and start test evaluation again.",
    )
    return p.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_state(model, state):
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def metric_sums(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return np.array(
        [
            len(y_true),
            np.sum(np.abs(err)),
            np.sum(err ** 2),
            np.sum(y_true),
            np.sum(y_true ** 2),
        ],
        dtype=np.float64,
    )


def metrics_from_sums(s):
    n, abs_err, sq_err, y_sum, y_sq_sum = [float(x) for x in s]
    mae = abs_err / n
    rmse = float(np.sqrt(sq_err / n))
    ss_tot = y_sq_sum - (y_sum * y_sum / n)
    r2 = float(1.0 - sq_err / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def save_rank_checkpoint(path, position, y_k_true, y_k_pred, y_ghi_true, y_ghi_pred):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        position=np.array([position], dtype=np.int64),
        k_index_true=np.asarray(y_k_true, dtype=np.float32),
        k_index_pred=np.asarray(y_k_pred, dtype=np.float32),
        ghi_clipped_true=np.asarray(y_ghi_true, dtype=np.float32),
        ghi_clipped_pred=np.asarray(y_ghi_pred, dtype=np.float32),
    )
    os.replace(tmp, path)


def load_rank_checkpoint(path):
    with np.load(path, allow_pickle=False) as d:
        position = int(d["position"][0])
        return (
            position,
            d["k_index_true"].tolist(),
            d["k_index_pred"].tolist(),
            d["ghi_clipped_true"].tolist(),
            d["ghi_clipped_pred"].tolist(),
        )


def make_config(args, distributed, rank, world_size):
    return {
        "data": {
            "csv_path": "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv",
            "image_root": "/storage2/CV_Irradiance/datasets/1_Folsom",
            "sequence_length": 40,
            "sampling_rate_sec": 60,
            "image_tolerance_sec": 120,
            "latitude": 38.642,
            "longitude": -121.148,
            "altitude": 60.0,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": 2,
            "distributed": distributed,
            "rank": rank,
            "world_size": world_size,
        },
        "model": {"horizons": [10]},
        "training": {"val_split": 0.1},
    }


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("=== INTERRUPTION-SAFE TEST EVALUATION ===")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Checkpoint: {args.checkpoint}")
        print("Test split: chronological 2016")
        print("Progress checkpoints are saved independently for each rank.")

    config = make_config(args, distributed, rank, world_size)
    _, _, test_loader = get_data_loaders(config)

    # DistributedSampler pads by default when N is not divisible by world_size.
    # Reconstruct its exact local index stream and remove padded positions so
    # every real test sample is evaluated exactly once globally.
    sampler_indices = list(iter(test_loader.sampler))
    dataset_size = len(test_loader.dataset)
    total_size = len(sampler_indices) * world_size
    local_real_positions = [
        j for j in range(len(sampler_indices))
        if rank + j * world_size < dataset_size
    ]
    local_indices = [sampler_indices[j] for j in local_real_positions]

    if rank == 0:
        print(f"Global test samples: {dataset_size}")
        print(f"Exact samples assigned to rank 0: {len(local_indices)}")

    rank_ckpt = output_dir / f"rank_{rank}_progress.npz"
    if args.reset_evaluation and rank_ckpt.exists():
        rank_ckpt.unlink()

    position = 0
    y_k_true, y_k_pred = [], []
    y_ghi_true, y_ghi_pred = [], []

    if rank_ckpt.exists():
        position, y_k_true, y_k_pred, y_ghi_true, y_ghi_pred = load_rank_checkpoint(rank_ckpt)
        if rank == 0:
            print(f"Resuming rank 0 from local sample position {position}/{len(local_indices)}")
        else:
            print(f"[rank {rank}] Resuming from local sample position {position}/{len(local_indices)}")

    if position > len(local_indices):
        raise RuntimeError(
            f"Invalid evaluation checkpoint for rank {rank}: "
            f"position={position}, local_samples={len(local_indices)}"
        )

    remaining_indices = local_indices[position:]
    remaining_dataset = Subset(test_loader.dataset, remaining_indices)
    remaining_loader = DataLoader(
        remaining_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = MultimodalSolarForecastModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_state(model, checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", None)
    best_val_loss = checkpoint.get("val_loss", None)

    if rank == 0:
        print(f"Loaded checkpoint epoch: {best_epoch}")
        print(f"Checkpoint validation loss: {best_val_loss}")
        print(f"Remaining batches for rank 0: {len(remaining_loader)}")

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, weather, targets, ghi_cs) in enumerate(remaining_loader, start=1):
            images = images.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                pred = model(images, weather)

            pred = pred.detach().float().cpu().numpy().reshape(-1)
            true = targets.cpu().numpy().reshape(-1)
            cs = ghi_cs.cpu().numpy().reshape(-1)

            y_k_pred.extend(pred.tolist())
            y_k_true.extend(true.tolist())

            # The training target is k_index clipped to [0, 1.2].
            # Therefore these are GHI values reconstructed from the same
            # clipped target used by the trained model.
            y_ghi_pred.extend((pred * cs).tolist())
            y_ghi_true.extend((true * cs).tolist())

            processed_position = position + len(y_k_pred) - len(y_k_true) + 0
            # The two lists always grow equally; calculate progress from
            # the number of samples accumulated in this resumed segment.
            processed_position = position + len(y_k_true)

            if (
                batch_idx % args.checkpoint_every_batches == 0
                or batch_idx == len(remaining_loader)
            ):
                save_rank_checkpoint(
                    rank_ckpt,
                    processed_position,
                    y_k_true,
                    y_k_pred,
                    y_ghi_true,
                    y_ghi_pred,
                )

            if batch_idx == 1 or batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[rank {rank}] {batch_idx}/{len(remaining_loader)} batches, "
                    f"local progress {processed_position}/{len(local_indices)}, "
                    f"elapsed={elapsed/60:.1f} min"
                )

    # Make sure every rank has finished before rank 0 merges shard results.
    if distributed:
        dist.barrier()

    # Rank-local checkpoint now contains the complete local result.
    if rank == 0:
        print("All ranks completed. Merging exact test predictions...")

    all_k_true = []
    all_k_pred = []
    all_ghi_true = []
    all_ghi_pred = []

    for r in range(world_size):
        path = output_dir / f"rank_{r}_progress.npz"
        if not path.exists():
            raise RuntimeError(f"Missing completed evaluation shard: {path}")
        _, kt, kp, gt, gp = load_rank_checkpoint(path)
        all_k_true.extend(kt)
        all_k_pred.extend(kp)
        all_ghi_true.extend(gt)
        all_ghi_pred.extend(gp)

    # Only rank 0 writes the final global artifacts.
    if rank == 0:
        k_mae, k_rmse, k_r2 = regression_metrics(all_k_true, all_k_pred)
        ghi_mae, ghi_rmse, ghi_r2 = regression_metrics(all_ghi_true, all_ghi_pred)

        ghi_mean = float(np.mean(all_ghi_true))
        ghi_nmae = ghi_mae / ghi_mean if ghi_mean != 0 else float("nan")
        ghi_nrmse = ghi_rmse / ghi_mean if ghi_mean != 0 else float("nan")

        results = [
            ("checkpoint", args.checkpoint),
            ("checkpoint_epoch", best_epoch),
            ("checkpoint_val_loss", best_val_loss),
            ("test_samples", len(all_k_true)),
            ("k_index_MAE", k_mae),
            ("k_index_RMSE", k_rmse),
            ("k_index_R2", k_r2),
            ("GHI_clipped_target_MAE_Wm2", ghi_mae),
            ("GHI_clipped_target_RMSE_Wm2", ghi_rmse),
            ("GHI_clipped_target_R2", ghi_r2),
            ("GHI_clipped_target_mean_Wm2", ghi_mean),
            ("GHI_clipped_target_nMAE", ghi_nmae),
            ("GHI_clipped_target_nRMSE", ghi_nrmse),
        ]

        print("\n=== FINAL TEST RESULTS ===")
        for name, value in results:
            if isinstance(value, (float, np.floating)):
                print(f"{name}: {value:.6f}")
            else:
                print(f"{name}: {value}")

        csv_path = output_dir / "test_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(results)

        np.savez_compressed(
            output_dir / "test_predictions.npz",
            k_index_true=np.asarray(all_k_true, dtype=np.float32),
            k_index_pred=np.asarray(all_k_pred, dtype=np.float32),
            ghi_clipped_true=np.asarray(all_ghi_true, dtype=np.float32),
            ghi_clipped_pred=np.asarray(all_ghi_pred, dtype=np.float32),
        )

        print(f"Saved metrics: {csv_path}")
        print(f"Saved predictions: {output_dir / 'test_predictions.npz'}")

        # Rank progress files are deliberately kept after success so that the
        # completed evaluation remains recoverable/auditable.
        print("Per-rank progress checkpoints retained.")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
