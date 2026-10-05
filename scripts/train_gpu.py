"""GPU training entry point for the multimodal Folsom irradiance model.

Pipeline:
    Folsom 10-channel projection -> CNN -> 256-D
    40 x 7 weather history -> LSTM -> 128-D
    concatenation -> 384-D -> regression -> k-index forecast

The Folsom preprocessing/calibration remains CPU-side because the original
pipeline uses NumPy/OpenCV-style operations and must remain unchanged.
Model forward/backward, loss, and optimizer work are moved to CUDA when
available.

This script is intended for the real training stage after 2014-2016
calibration is complete and the DataLoader batch smoke test passes.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch
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
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many consecutive epochs without meaningful validation improvement.")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum validation-loss decrease required to reset early-stopping patience.")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0 means the full training loader.")
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/baseline_lstm_cnn/checkpoints/best_model.pt")
    parser.add_argument("--experiment-dir", type=str,
                        default="experiments/baseline_lstm_cnn")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA is not available; training will run on CPU.")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Max epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Early stopping min_delta: {args.min_delta}")

    experiment_dir = args.experiment_dir
    logs_dir = os.path.join(experiment_dir, "logs")
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    history_path = os.path.join(logs_dir, "training_history.csv")
    last_checkpoint_path = os.path.join(
        experiment_dir, "checkpoints", "last_checkpoint.pt"
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if args.resume and os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(
            f"Resumed from epoch {checkpoint['epoch']}. "
            f"Continuing at epoch {start_epoch}."
        )
    elif not args.resume:
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "epoch_seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        start = time.perf_counter()

        for batch_idx, (images, weather, targets, _) in enumerate(train_loader, start=1):
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

            if batch_idx == 1 or batch_idx % 10 == 0:
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

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} time={elapsed:.1f}s"
        )

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{elapsed:.3f}"])

        improved = val_loss < best_val_loss - args.min_delta

        if improved:
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
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
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
                break

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            last_checkpoint_path,
        )
        print(f"Saved latest checkpoint: {last_checkpoint_path}")

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
    print(
        f"Training finished. Best epoch: {best_epoch}, "
        f"best val_loss: {best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()
