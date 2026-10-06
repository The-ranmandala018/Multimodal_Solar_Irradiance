"""GPU training entry point for the multimodal Folsom irradiance model.

Supports resumable training both between epochs and within an epoch. Mid-epoch
checkpoints periodically save the model/optimizer state, accumulated loss,
batch position, and the epoch's DataLoader seed so an interrupted epoch can be
reconstructed and continued.

The Folsom preprocessing/calibration remains CPU-side because the original
pipeline uses NumPy/OpenCV-style operations and must remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.amp import GradScaler, autocast

from datasets.folsom_dataset import get_data_loaders
from models.multimodal_model import MultimodalSolarForecastModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop after this many consecutive epochs without meaningful validation improvement.",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Minimum validation-loss decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--max-train-batches", type=int, default=0,
        help="0 means the full training loader.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="experiments/baseline_lstm_cnn/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--experiment-dir", type=str,
        default="experiments/baseline_lstm_cnn",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint if it exists.",
    )
    parser.add_argument(
        "--ddp", action="store_true",
        help="Use all GPUs launched by torchrun with DistributedDataParallel.",
    )
    parser.add_argument(
        "--checkpoint-every-batches", type=int, default=100,
        help="Save a resumable checkpoint every N training batches.",
    )
    return parser.parse_args()


def _rng_state() -> dict:
    state = {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    if "torch_rng_state" in state:
        # Checkpoints created by different PyTorch versions can deserialize
        # the RNG state as a generic uint8 Tensor. Convert explicitly to the
        # ByteTensor expected by torch.set_rng_state().
        torch_state = torch.as_tensor(
            state["torch_rng_state"], dtype=torch.uint8, device="cpu"
        )
        torch.set_rng_state(torch_state)

    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])

    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        cuda_states = [
            torch.as_tensor(s, dtype=torch.uint8, device="cpu")
            for s in state["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _save_mid_epoch_checkpoint(
    path: str,
    *,
    epoch: int,
    batch_idx: int,
    running_loss: float,
    batches: int,
    epoch_seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    best_val_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "mid_epoch",
            "epoch": epoch,
            "batch_idx": batch_idx,
            "running_loss": running_loss,
            "batches": batches,
            "epoch_seed": epoch_seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "rng_state": _rng_state(),
        },
        path,
    )


def main() -> None:
    args = parse_args()

    ddp = args.ddp
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    is_main_process = rank == 0

    if args.checkpoint_every_batches < 1:
        raise ValueError("--checkpoint-every-batches must be >= 1")

    if not ddp:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            print("WARNING: CUDA is not available; training will run on CPU.")
    elif is_main_process:
        print(f"DDP enabled: {world_size} GPUs, local rank {local_rank}")

    config = {
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
            "distributed": ddp,
            "rank": rank,
            "world_size": world_size,
        },
        "model": {
            "horizons": [10],
        },
        "training": {
            "val_split": 0.1,
        },
    }

    print("Building DataLoaders...")
    train_loader, val_loader, _ = get_data_loaders(config)

    model = MultimodalSolarForecastModel().to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    if is_main_process:
        print(f"Device: {device}")
        print(f"Distributed: {ddp} (world_size={world_size})")
        print(f"AMP: {use_amp}")
        print(f"Train batches per rank: {len(train_loader)}")
        print(f"Validation batches per rank: {len(val_loader)}")
        print(f"Max epochs: {args.epochs}")
        print(f"Early stopping patience: {args.patience}")
        print(f"Early stopping min_delta: {args.min_delta}")
        print(f"Mid-epoch checkpoint interval: {args.checkpoint_every_batches} batches")

    experiment_dir = args.experiment_dir
    logs_dir = os.path.join(experiment_dir, "logs")
    plots_dir = os.path.join(experiment_dir, "plots")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history_path = os.path.join(logs_dir, "training_history.csv")
    last_checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pt")
    mid_checkpoint_path = os.path.join(checkpoints_dir, "mid_epoch_checkpoint.pt")

    start_epoch = 1
    resume_batch = 0
    resume_running_loss = 0.0
    resume_batches = 0
    resume_epoch_seed = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if ddp and args.resume and os.path.exists(mid_checkpoint_path):
        raise RuntimeError(
            "A single-GPU mid-epoch checkpoint exists. Finish Epoch 1 with "
            "the existing single-GPU resume first. DDP cannot safely preserve "
            "that mid-epoch sample position because samples are partitioned "
            "across ranks. After an epoch-complete checkpoint exists, resume "
            "with --ddp."
        )

    if args.resume and os.path.exists(mid_checkpoint_path):
        checkpoint = torch.load(mid_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint["epoch"])
        resume_batch = int(checkpoint["batch_idx"])
        resume_running_loss = float(checkpoint.get("running_loss", 0.0))
        resume_batches = int(checkpoint.get("batches", resume_batch))
        resume_epoch_seed = int(checkpoint["epoch_seed"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        _restore_rng_state(checkpoint.get("rng_state", {}))

        print(
            f"Resuming from middle of epoch {start_epoch}: "
            f"after batch {resume_batch}/{len(train_loader)}."
        )
    elif args.resume and os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        _restore_rng_state(checkpoint.get("rng_state", {}))
        print(
            f"Resumed from completed epoch {checkpoint['epoch']}. "
            f"Continuing at epoch {start_epoch}."
        )
    elif not args.resume:
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "epoch_seconds"]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start = time.perf_counter()

        if epoch == start_epoch and resume_batch > 0:
            epoch_seed = resume_epoch_seed
            running_loss = resume_running_loss
            batches = resume_batches
            skip_batches = resume_batch
        else:
            epoch_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            running_loss = 0.0
            batches = 0
            skip_batches = 0

        # The dataset's training loader is shuffled. Resetting the global torch
        # seed immediately before iterator creation reconstructs the same
        # RandomSampler order for a resumed epoch.
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)

        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if is_main_process:
            print(
                f"Epoch {epoch}: starting at batch "
            f"{skip_batches + 1 if skip_batches else 1}/{len(train_loader)}"
        )

        for batch_idx, (images, weather, targets, _) in enumerate(train_loader, start=1):
            if batch_idx <= skip_batches:
                continue

            if args.max_train_batches and batch_idx > args.max_train_batches:
                break

            images = images.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                predictions = model(images, weather)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1

            if is_main_process and (batch_idx == 1 or batch_idx % 10 == 0):
                if device.type == "cuda":
                    mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    print(
                        f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.6f} GPU_mem={mem:.2f}GB"
                    )
                else:
                    print(
                        f"epoch={epoch} batch={batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.6f}"
                    )

            if is_main_process and batch_idx % args.checkpoint_every_batches == 0:
                _save_mid_epoch_checkpoint(
                    mid_checkpoint_path,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    running_loss=running_loss,
                    batches=batches,
                    epoch_seed=epoch_seed,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                )
                print(
                    f"Saved mid-epoch checkpoint: epoch={epoch}, "
                    f"batch={batch_idx}"
                )

        # Reset resume-only state after completing the resumed epoch.
        resume_batch = 0
        resume_running_loss = 0.0
        resume_batches = 0
        resume_epoch_seed = None

        if ddp:
            train_stats = torch.tensor([running_loss, batches], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            running_loss = train_stats[0].item()
            batches = int(train_stats[1].item())

        train_loss = running_loss / max(batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, weather, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                weather = weather.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    predictions = model(images, weather)
                    loss = criterion(predictions, targets)

                val_loss_sum += loss.item()
                val_batches += 1

        if ddp:
            stats = torch.tensor(
                [val_loss_sum, val_batches],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            val_loss_sum = stats[0].item()
            val_batches = int(stats[1].item())

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.perf_counter() - start

        if is_main_process:
            print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} time={elapsed:.1f}s"
        )

        if is_main_process:
            with open(history_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{elapsed:.3f}"]
                )

        improved = val_loss < best_val_loss - args.min_delta

        if is_main_process and improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint_path = args.checkpoint
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "rng_state": _rng_state(),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        elif is_main_process:
            epochs_without_improvement += 1
            print(
                f"No meaningful validation improvement: "
                f"{epochs_without_improvement}/{args.patience}"
            )

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}"
                )

        should_stop = False
        if ddp:
            stop_tensor = torch.tensor(
                [1 if (is_main_process and epochs_without_improvement >= args.patience) else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        else:
            should_stop = epochs_without_improvement >= args.patience

        if is_main_process:
            torch.save(
                {
                "checkpoint_type": "epoch_complete",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "rng_state": _rng_state(),
            },
            last_checkpoint_path,
        )
            print(f"Saved latest checkpoint: {last_checkpoint_path}")

        if ddp:
            dist.barrier()

        # An epoch is complete, so the mid-epoch checkpoint is no longer needed.
        if is_main_process and os.path.exists(mid_checkpoint_path):
            os.remove(mid_checkpoint_path)

        if should_stop:
            break

    try:
        import matplotlib.pyplot as plt

        epochs, train_losses, val_losses = [], [], []
        with open(history_path, newline="") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))

        if epochs:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_losses, label="Training loss")
            plt.plot(epochs, val_losses, label="Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.title("Baseline CNN + LSTM Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "training_validation_loss.png")
            plt.savefig(plot_path, dpi=160)
            plt.close()
            print(f"Saved loss curve: {plot_path}")
    except Exception as exc:
        print(f"WARNING: Could not create loss plot: {exc}")

    print(f"Training history: {history_path}")
    if is_main_process:
        print(
            f"Training finished. Best epoch: {best_epoch}, "
            f"best val_loss: {best_val_loss:.6f}"
        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
