"""Parameter count, inference speed, training-step time and peak GPU memory: CNN+LSTM vs Mamba-LSTM.

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_efficiency.py \
        --output experiments/mamba_lstm/comparison/efficiency.csv

Both models get the real input shapes (10 x 180 x 720 image, 40 x 7 weather)
and random data. Inference follows the MERCon 2026 protocol (eval mode, 10
warm-up + 50 timed iterations) at batch 4 and at our evaluation batch 8, with
fp16 autocast as in evaluate_test.py. The training step is forward + backward +
AdamW step with AMP at the training batch of 8 on one GPU. Data loading and
preprocessing are excluded, so these are model-only numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.mamba_lstm_model import build_mamba_lstm  # noqa: E402
from models.multimodal_model import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="experiments/mamba_lstm/config.yaml")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    args = parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    device = torch.device("cuda")
    with open(REPO / args.config) as f:
        cfg = yaml.safe_load(f)

    builders = {
        "cnn_lstm": lambda: build_model("cnn_lstm"),
        f"mamba_lstm_{cfg['model']['scale']}": lambda: build_mamba_lstm(cfg["model"], pretrained=False),
    }
    rows = []
    for name, make in builders.items():
        torch.manual_seed(0)
        model = make().to(device)
        params = sum(p.numel() for p in model.parameters())
        visual = sum(p.numel() for p in model.visual_encoder.parameters())

        model.eval()
        for bs in (4, 8):
            img = torch.randn(bs, 10, 180, 720, device=device)
            wx = torch.randn(bs, 40, 7, device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            def infer():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    model(img, wx)

            sec = timed(infer, args.warmup, args.iters)
            rows.append([name, "inference_fp16", bs, params, visual, f"{sec * 1000:.2f}",
                         f"{bs / sec:.1f}", f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f}"])

        model.train()
        bs = int(cfg["training"]["global_batch_size"])
        img = torch.randn(bs, 10, 180, 720, device=device)
        wx = torch.randn(bs, 40, 7, device=device)
        y = torch.rand(bs, 1, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        def train_step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(img, wx).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        sec = timed(train_step, args.warmup, args.iters)
        rows.append([name, "train_step_amp", bs, params, visual, f"{sec * 1000:.2f}",
                     f"{bs / sec:.1f}", f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f}"])
        del model, opt
        torch.cuda.empty_cache()

    header = ["model", "measurement", "batch_size", "params_total", "params_visual_encoder",
              "ms_per_batch", "samples_per_sec", "peak_mem_alloc_gb"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    gpu = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu}, torch {torch.__version__}")
    for r in rows:
        print(dict(zip(header, r)))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
