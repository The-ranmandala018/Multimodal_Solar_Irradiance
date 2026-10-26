"""Train the Mamba-LSTM model (VMamba visual encoder + the baseline LSTM/fusion/head).

Single GPU:
    python scripts/train_mamba_lstm.py --seed 0 --run-dir experiments/mamba_lstm/seed0
Two GPUs (DistributedDataParallel):
    torchrun --standalone --nproc_per_node 2 scripts/train_mamba_lstm.py --seed 0 \
        --run-dir experiments/mamba_lstm/seed0
Normally launched through scripts/run_mamba_lstm.sh (crash watchdog + log file).

Same data, split, normalisation, target, loss, optimiser and early-stopping rule
as scripts/train_gpu.py; settings come from experiments/mamba_lstm/config.yaml.
scripts/train_gpu.py is not modified or imported.

Batching. Each epoch is a seeded permutation of the training samples cut into
global batches of `global_batch_size` (8). With W GPUs every rank takes its
share of the same global batch, and the loss is scaled so that the averaged
DDP gradient equals the gradient of the global-batch mean: one optimiser step
sees the same 8 samples whether W is 1 or 2. The order depends only on
(seed, epoch), so resuming skips the finished batches without loading them,
and a run can resume with a different number of GPUs.

Checkpoints (in <run-dir>/checkpoints/):
    last.pt       full training state; written every `checkpoint_minutes`, before
                  validation and after each epoch. Written atomically (temp file +
                  rename); the previous one is kept as last_prev.pt.
    best.pt       model weights of the best validation epoch (atomic).
Resume is automatic when last.pt exists (--resume auto). A clean stop:
`touch <run-dir>/STOP` or SIGTERM/SIGINT; the state is saved within ~25 steps.

Exit codes: 0 finished (DONE written), 75 stopped on request (resumable),
76 CUDA out of memory, 77 repeated non-finite loss, 78 configuration or
checkpoint problem. scripts/run_mamba_lstm.sh restarts only on other codes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.folsom_dataset import get_data_loaders  # noqa: E402
from models.mamba_lstm_model import build_mamba_lstm  # noqa: E402

EXIT_DONE = 0
EXIT_STOPPED = 75
EXIT_OOM = 76
EXIT_NONFINITE = 77
EXIT_CONFIG = 78

CHECKPOINT_FORMAT = "mamba_lstm_train_v1"
SYNC_EVERY = 25  # steps between cross-rank stop/save checks

# Keys that may change between resumes without invalidating a checkpoint.
RUNTIME_KEYS = {
    "data": {"num_workers", "prefetch_factor"},
    "training": {"checkpoint_minutes", "log_every_steps", "max_consecutive_nonfinite", "min_grad_scale"},
}

_stop_signal = None
_exit_file = None  # <run-dir>/last_exit, read by scripts/run_mamba_lstm.sh


def record_exit(code: int, reason: str) -> None:
    """Leave the exit reason for the launcher (torchrun hides worker exit codes)."""
    if _exit_file is not None:
        try:
            _exit_file.write_text(f"{code}\n{datetime.now():%F %T} {reason}\n")
        except OSError:
            pass


def _on_signal(signum, _frame):
    global _stop_signal
    _stop_signal = signum


def _die_with_parent():
    """Under torchrun, get SIGTERM if torchrun dies (e.g. SIGKILL) instead of running on as an orphan."""
    if "TORCHELASTIC_RUN_ID" in os.environ:
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except OSError:
            pass


def _worker_init(_worker_id):
    # Ctrl+C reaches the whole process group; let the main process handle it
    # instead of killing the DataLoader workers mid-batch.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="experiments/mamba_lstm/config.yaml")
    p.add_argument("--run-dir", required=True,
                   help="Output folder for this run (checkpoints/, logs/, plots/).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume", choices=["auto", "never", "require"], default="auto",
                   help="auto: resume if a checkpoint exists, else start fresh. "
                        "never: start fresh; refuse if the run dir has a run. "
                        "require: fail unless a valid checkpoint exists.")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    p.add_argument("--patience", type=int, default=None, help="Override training.patience.")
    p.add_argument("--checkpoint-minutes", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--progress-interval", type=float, default=60.0,
                   help="Seconds between progress lines when output is not a terminal.")
    p.add_argument("--allow-config-change", action="store_true",
                   help="Resume even if the protocol settings differ from the checkpoint.")
    # Smoke-test limits. They are recorded in the run config.
    p.add_argument("--max-steps-per-epoch", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- utilities

def config_hash(cfg: dict) -> str:
    protocol = copy.deepcopy(cfg)
    for section, keys in RUNTIME_KEYS.items():
        for key in keys:
            protocol.get(section, {}).pop(key, None)
    blob = json.dumps(protocol, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def atomic_torch_save(obj, path: Path, keep_prev: Path | None = None) -> None:
    """Write to <path>.tmp, fsync, read it back, then rename into place.

    The existing file is only moved (to keep_prev) after the new one has been
    written and verified, so a known-good checkpoint always exists on disk.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    check = torch.load(tmp, map_location="cpu", weights_only=False)
    if check.get("format") != obj.get("format") or check.get("model_state_dict", {}).keys() != obj["model_state_dict"].keys():
        raise RuntimeError(f"verification of {tmp} failed; {path} was left unchanged")
    del check
    if keep_prev is not None and path.exists():
        os.replace(path, keep_prev)
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_csv_atomic(path: Path, header, rows) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def software_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    for name in ("cv2", "timm", "mamba_ssm", "pandas", "pvlib"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            info[name] = None
    return info


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": run("git", "status", "--porcelain", "--untracked-files=no"),
    }


SOURCE_FILES = [
    "scripts/train_mamba_lstm.py", "models/mamba_lstm_model.py", "models/mamba_encoder.py",
    "models/backbones/vmamba/vmamba.py", "models/temporal.py", "models/fusion.py", "models/heads.py",
    "datasets/folsom_dataset.py", "datasets/chrono.py", "preprocessing/folsom_adapter.py",
    "preprocessing/projection.py", "source/Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py",
]


def source_info(config_path: Path) -> dict:
    """Git revision plus sha256 of every source file the run depends on (works for uncommitted code)."""
    files = {}
    for rel in SOURCE_FILES + [str(config_path.relative_to(REPO)) if config_path.is_relative_to(REPO) else str(config_path)]:
        path = REPO / rel if not os.path.isabs(rel) else Path(rel)
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    return {"git": git_info(), "source_hash": combined, "files": files}


class GlobalBatchSampler(Sampler):
    """This rank's share of each global batch of a seeded per-epoch permutation."""

    def __init__(self, n, global_batch, rank, world, seed, epoch, start_step=0, max_steps=0):
        g = torch.Generator()
        g.manual_seed(seed * 100_003 + epoch)
        perm = torch.randperm(n, generator=g).tolist()
        batches = [perm[i:i + global_batch] for i in range(0, n, global_batch)]
        if len(batches[-1]) < world:  # a rank would get no sample
            batches = batches[:-1]
        if max_steps:
            batches = batches[:max_steps]
        self.steps_per_epoch = len(batches)
        self.start_step = start_step
        self.rank, self.world = rank, world
        self._batches = batches

    def global_size(self, step: int) -> int:
        return len(self._batches[step])

    def __iter__(self):
        for batch in self._batches[self.start_step:]:
            yield np.array_split(np.asarray(batch), self.world)[self.rank].tolist()

    def __len__(self):
        return self.steps_per_epoch - self.start_step


class Progress:
    """tqdm bar on a terminal; periodic tqdm-formatted lines in a log file."""

    def __init__(self, total, initial, desc, interval, enabled):
        from tqdm import tqdm
        self.enabled = enabled
        self.tty = sys.stderr.isatty()
        self.total, self.n, self.initial, self.desc = total, initial, initial, desc
        self.interval = interval
        self.start = time.time()
        self.last_print = 0.0
        self.postfix = ""
        self._tqdm_cls = tqdm
        self.bar = tqdm(total=total, initial=initial, desc=desc, dynamic_ncols=True,
                        smoothing=0.05) if (enabled and self.tty) else None

    def update(self, n, postfix: str):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.update(n - self.bar.n)
            self.bar.set_postfix_str(postfix, refresh=False)
            return
        self.n, self.postfix = n, postfix
        now = time.time()
        if now - self.last_print >= self.interval or n == self.total:
            self.last_print = now
            elapsed = now - self.start
            rate = (n - self.initial) / elapsed if elapsed > 0 else None
            line = self._tqdm_cls.format_meter(
                n, self.total, elapsed, prefix=self.desc, rate=rate, postfix=postfix, ncols=0,
            )
            print(f"[{datetime.now():%F %T}] {line}", flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    _die_with_parent()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required (the VMamba selective-scan kernel is CUDA-only).")
        sys.exit(EXIT_CONFIG)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30),
                                device_id=torch.device("cuda", local_rank))
    else:
        rank, local_rank = 0, 0
    device = torch.device("cuda", local_rank)
    main_proc = rank == 0

    def log(msg):
        if main_proc:
            print(f"[{datetime.now():%F %T}] {msg}", flush=True)

    def all_sum(values):
        t = torch.tensor(values, dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()

    def all_max(values):
        t = torch.tensor(values, dtype=torch.int64, device=device)
        if distributed:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.tolist()

    def barrier():
        if distributed:
            dist.barrier()

    def fail(code, msg):
        print(f"[rank {rank}] ERROR: {msg}", flush=True)
        record_exit(code, msg)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(code)

    # ---------------- configuration
    with open(REPO / args.config if not os.path.isabs(args.config) else args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg, dcfg = cfg["training"], cfg["data"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.patience is not None:
        tcfg["patience"] = args.patience
    if args.checkpoint_minutes is not None:
        tcfg["checkpoint_minutes"] = args.checkpoint_minutes
    if args.num_workers is not None:
        dcfg["num_workers"] = args.num_workers
    cfg["seed"] = args.seed
    if args.max_steps_per_epoch or args.max_val_batches:
        cfg["smoke_limits"] = {"max_steps_per_epoch": args.max_steps_per_epoch,
                               "max_val_batches": args.max_val_batches}
    if tcfg.get("loss") != "mse" or tcfg.get("scheduler") != "none":
        fail(EXIT_CONFIG, "only loss=mse and scheduler=none (the baseline protocol) are implemented")
    gbs = int(tcfg["global_batch_size"])
    if gbs % world:
        fail(EXIT_CONFIG, f"global_batch_size {gbs} is not divisible by {world} GPUs")
    chash = config_hash(cfg)
    config_path = REPO / args.config if not os.path.isabs(args.config) else Path(args.config)
    src = source_info(config_path)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    ckpt_dir, logs_dir, plots_dir = run_dir / "checkpoints", run_dir / "logs", run_dir / "plots"
    last_path, prev_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "last_prev.pt", ckpt_dir / "best.pt"
    history_path, steps_path = logs_dir / "training_history.csv", logs_dir / "steps.csv"
    stop_file, done_file = run_dir / "STOP", run_dir / "DONE"

    # One trainer per run dir, taken before anything that can fail, so that only
    # the lock holder ever writes <run-dir>/last_exit. The OS drops the lock
    # however the process ends.
    lock_fd = None
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(run_dir / ".train.lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ERROR: another trainer is already using {run_dir} (lock held). Check "
                  "`pgrep -af train_mamba_lstm`; do not start a second one.", flush=True)
            sys.exit(EXIT_CONFIG)  # deliberately not recorded in last_exit: not our run
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()} {socket.gethostname()}\n".encode())
    barrier()
    global _exit_file
    _exit_file = run_dir / "last_exit"

    # ---------------- locate checkpoint (every rank reads the same files)
    checkpoint, loaded_from = None, None
    candidates = [p for p in (last_path, prev_path) if p.exists()]
    if done_file.exists():
        log(f"{run_dir} is already finished ({done_file.read_text().strip()}). Nothing to do.")
        if main_proc:
            record_exit(EXIT_DONE, "already finished")
        if distributed:
            dist.destroy_process_group()
        sys.exit(EXIT_DONE)
    if args.resume == "never" and (candidates or best_path.exists() or history_path.exists()):
        fail(EXIT_CONFIG, f"{run_dir} already contains a run; use a new --run-dir "
                          "(existing artifacts are never overwritten)")
    corrupt = []
    if args.resume != "never":
        for path in candidates:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if ck.get("format") != CHECKPOINT_FORMAT:
                    raise ValueError(f"unexpected checkpoint format {ck.get('format')!r}")
                checkpoint, loaded_from = ck, path
                break
            except Exception as exc:  # noqa: BLE001
                corrupt.append(path)
                log(f"WARNING: cannot use checkpoint {path}: {exc}")
        if candidates and checkpoint is None:
            fail(EXIT_CONFIG,
                 f"no readable checkpoint among {[str(p) for p in candidates]}. Nothing was "
                 "overwritten. Inspect or move these files, then rerun; to start over use a "
                 "new --run-dir.")
        if checkpoint is None and args.resume == "require":
            fail(EXIT_CONFIG, f"--resume require, but no checkpoint in {ckpt_dir}")
        if checkpoint is None and (best_path.exists() or history_path.exists()):
            fail(EXIT_CONFIG, f"{run_dir} has results but no last.pt/last_prev.pt; refusing to "
                              "start fresh over them. Use a new --run-dir.")
    if checkpoint is not None and checkpoint["config_hash"] != chash:
        msg = (f"protocol settings differ from the checkpoint ({checkpoint['config_hash']} vs "
               f"{chash}); see logs/run_meta.jsonl for the original config")
        if not args.allow_config_change:
            fail(EXIT_CONFIG, msg + ". Pass --allow-config-change to resume anyway.")
        log("WARNING: " + msg)

    barrier()  # every rank has finished reading the checkpoint files
    if main_proc:
        for d in (ckpt_dir, logs_dir, plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        stop_file.unlink(missing_ok=True)
        for path in corrupt:  # keep unreadable files for inspection, out of the rotation
            kept = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d-%H%M%S}")
            os.replace(path, kept)
            log(f"WARNING: unreadable checkpoint preserved as {kept}; resuming from verified {loaded_from}")
    barrier()

    # ---------------- data (same construction as scripts/train_gpu.py)
    log("Building datasets (chronological 2014-2015 / 2016 split)...")
    data_cfg = {
        "data": {
            **{k: dcfg[k] for k in ("csv_path", "image_root", "sequence_length",
                                    "sampling_rate_sec", "image_tolerance_sec",
                                    "latitude", "longitude", "altitude")},
            "batch_size": 1, "num_workers": 0, "distributed": False,
            "rank": 0, "world_size": 1, "load_images": True,
        },
        "model": {"horizons": dcfg["horizons"]},
        "training": {"val_split": dcfg["val_split"]},
    }
    train_loader0, val_loader0, _ = get_data_loaders(data_cfg)
    train_ds, val_ds = train_loader0.dataset, val_loader0.dataset
    n_train, n_val = len(train_ds), len(val_ds)
    local_bs = gbs // world
    nw = int(dcfg["num_workers"])
    loader_kw = dict(num_workers=nw, pin_memory=True, worker_init_fn=_worker_init)
    if nw > 0:
        loader_kw["prefetch_factor"] = int(dcfg["prefetch_factor"])

    val_shards = [s.tolist() for s in np.array_split(np.arange(n_val), world)]
    if args.max_val_batches:
        val_shards = [s[: args.max_val_batches * local_bs] for s in val_shards]
    val_shard = val_shards[rank]
    # Mid-validation stop checks are collective: only run them on batch numbers
    # that every rank reaches (shards can differ by one batch).
    common_val_batches = min(-(-len(s) // local_bs) for s in val_shards)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_shard), batch_size=local_bs,
                            shuffle=False, persistent_workers=nw > 0, **loader_kw)

    # ---------------- model / optimiser
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_mamba_lstm(cfg["model"], pretrained=checkpoint is None and cfg["model"]["pretrained"])
    model.to(device)
    for p in model.parameters():  # some VMamba depthwise weights are non-contiguous; DDP buckets prefer contiguous
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    use_amp = bool(tcfg["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = {
        "epoch": 1, "step_in_epoch": 0, "epoch_complete": False, "global_step": 0,
        "best_val_loss": float("inf"), "best_epoch": 0, "epochs_without_improvement": 0,
        "epoch_sse": 0.0, "epoch_n": 0.0, "epoch_elapsed": 0.0, "history": [],
        "epoch_skipped": 0,  # optimiser steps skipped by GradScaler (inf/NaN gradients) this epoch
    }
    if checkpoint is not None:
        saved_sampler = checkpoint.get("sampler_state", {})
        if saved_sampler and saved_sampler.get("n_train") != n_train:
            fail(EXIT_CONFIG, f"checkpoint was made with {saved_sampler.get('n_train')} training samples, "
                              f"now {n_train}; the data order cannot be continued")
        if checkpoint.get("source", {}).get("source_hash") not in (None, src["source_hash"]):
            log(f"NOTE: source code changed since the checkpoint was written "
                f"({checkpoint['source']['source_hash']} -> {src['source_hash']}); see logs/run_meta.jsonl")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        for key in state:
            state[key] = checkpoint.get(key, state[key])
        if checkpoint["world_size"] == world:
            rs = checkpoint["rng_states"][rank]
            torch.set_rng_state(rs["torch"])
            torch.cuda.set_rng_state(rs["cuda"], device)
            np.random.set_state(rs["numpy"])
            random.setstate(rs["python"])
        else:
            log(f"NOTE: checkpoint was written with {checkpoint['world_size']} GPU(s), now {world}; "
                "data order is unchanged, dropout RNG streams are not restored.")
        if state["epoch_complete"]:
            state["epoch"] += 1
            state["step_in_epoch"] = 0
            state["epoch_complete"] = False
            state["epoch_sse"] = state["epoch_n"] = state["epoch_elapsed"] = 0.0
            state["epoch_skipped"] = 0
        log(f"RESUMED from {loaded_from}: epoch {state['epoch']}, step {state['step_in_epoch']} "
            f"of that epoch, global step {state['global_step']}, best val {state['best_val_loss']:.6f} "
            f"(epoch {state['best_epoch']}).")
    else:
        log("Starting a new run (no checkpoint).")
        torch.manual_seed(args.seed + 1000 * rank)  # per-rank dropout/drop-path stream
    del checkpoint

    ddp_model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True) if distributed else model
    n_params = sum(p.numel() for p in model.parameters())
    n_visual = sum(p.numel() for p in model.visual_encoder.parameters())

    # ---------------- run metadata (one JSON line per launch)
    if main_proc:
        meta = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": world, "per_gpu_batch": local_bs,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "software": software_info(), "source": src,
            "config": cfg, "config_hash": chash,
            "resumed_from": str(loaded_from) if loaded_from else None,
            "n_train": n_train, "n_val": n_val,
            "params_total": n_params, "params_visual_encoder": n_visual,
            "fp32_scan_modules": model.visual_encoder.fp32_scan_modules,
        }
        with open(logs_dir / "run_meta.jsonl", "a") as f:
            f.write(json.dumps(meta, default=str) + "\n")
        log(f"world_size={world} per_gpu_batch={local_bs} global_batch={gbs} amp={use_amp} "
            f"params={n_params/1e6:.2f}M (visual {n_visual/1e6:.2f}M) train={n_train} val={n_val} "
            f"config_hash={chash} fp32_scan_modules={model.visual_encoder.fp32_scan_modules} "
            f"grad_scale={scaler.get_scale():g}")

    def rewrite_history():
        if main_proc:
            cols = ["epoch", "train_loss", "val_loss", "lr", "epoch_seconds", "train_samples_per_sec",
                    "peak_mem_alloc_gb_max", "peak_mem_reserved_gb_max", "world_size", "improved",
                    "skipped_steps", "grad_scale_end"]
            write_csv_atomic(history_path, cols, [[h.get(c, "") for c in cols] for h in state["history"]])

    rewrite_history()

    def gather_rng():
        mine = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                "numpy": np.random.get_state(), "python": random.getstate()}
        if not distributed:
            return [mine]
        out = [None] * world
        dist.all_gather_object(out, mine)
        return out

    def save_last(local_sse, local_n, epoch_elapsed, reason):
        """Collective: every rank must call it at the same point."""
        sse, n = all_sum([local_sse, local_n])
        rng = gather_rng()
        if main_proc:
            ck = {
                "format": CHECKPOINT_FORMAT, "saved_at": datetime.now().isoformat(timespec="seconds"),
                "reason": reason, "config": cfg, "config_hash": chash, "world_size": world,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": None,  # constant learning rate (baseline protocol)
                "rng_states": rng, "seed": args.seed, "source": src,
                "sampler_state": {
                    "type": "GlobalBatchSampler", "seed": args.seed, "epoch": state["epoch"],
                    "next_step": state["step_in_epoch"], "global_batch_size": gbs, "n_train": n_train,
                    "order": "torch.randperm(n_train, Generator().manual_seed(seed * 100003 + epoch))",
                },
                **{k: v for k, v in state.items() if k not in ("epoch_sse", "epoch_n", "epoch_elapsed")},
                "epoch_sse": sse, "epoch_n": n, "epoch_elapsed": epoch_elapsed,
            }
            t0 = time.time()
            atomic_torch_save(ck, last_path, keep_prev=prev_path)
            log(f"Saved {last_path.name} ({reason}; epoch {state['epoch']} step "
                f"{state['step_in_epoch']}, {time.time() - t0:.1f}s)")
        barrier()

    if main_proc and not steps_path.exists():
        with open(steps_path, "w", newline="") as f:
            csv.writer(f).writerow(["time", "epoch", "step_in_epoch", "global_step", "window_loss",
                                    "lr", "samples_per_sec", "gpu_mem_alloc_gb_rank0",
                                    "grad_scale", "skipped_steps_window"])

    epochs, patience, min_delta = int(tcfg["epochs"]), int(tcfg["patience"]), float(tcfg["min_delta"])
    ckpt_seconds = float(tcfg["checkpoint_minutes"]) * 60
    log_every = int(tcfg["log_every_steps"])
    max_nonfinite = int(tcfg["max_consecutive_nonfinite"])
    min_scale = float(tcfg.get("min_grad_scale", 1.0))
    last_val = float(state["history"][-1]["val_loss"]) if state["history"] else None
    finished_reason = None

    def status(loss, run_seconds):
        best = (f"{state['best_val_loss']:.5f}@ep{state['best_epoch']}"
                if state["best_epoch"] else "n/a")
        val_txt = f"{last_val:.5f}" if last_val is not None else "n/a"
        h, rem = divmod(int(run_seconds), 3600)
        return (f"loss={loss:.5f} val={val_txt} best={best} "
                f"patience_left={patience - state['epochs_without_improvement']}/{patience} "
                f"lr={optimizer.param_groups[0]['lr']:.1e} scale={scaler.get_scale():g} "
                f"skipped={state['epoch_skipped']} gstep={state['global_step']} run={h}h{rem // 60:02d}m")

    while state["epoch"] <= epochs:
        epoch = state["epoch"]
        sampler = GlobalBatchSampler(n_train, gbs, rank, world, args.seed, epoch,
                                     start_step=state["step_in_epoch"],
                                     max_steps=args.max_steps_per_epoch)
        steps_per_epoch = sampler.steps_per_epoch
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kw)
        ddp_model.train()
        torch.cuda.reset_peak_memory_stats(device)

        local_sse = state["epoch_sse"] if main_proc else 0.0
        local_n = state["epoch_n"] if main_proc else 0.0
        epoch_t0 = time.time() - state["epoch_elapsed"]
        win_sse = win_n = 0.0
        win_skipped = 0
        win_t0 = time.time()
        last_save = time.time()
        nonfinite = 0
        step = state["step_in_epoch"]
        samples_this_launch = 0
        launch_t0 = time.time()
        progress = Progress(steps_per_epoch, step, f"Seed {args.seed} | Epoch {epoch}/{epochs}",
                            args.progress_interval, main_proc)
        done_seconds = sum(float(h["epoch_seconds"]) for h in state["history"])
        if step:
            log(f"Epoch {epoch}: continuing at step {step + 1}/{steps_per_epoch} "
                f"(skipping {step} finished batches without loading them)")

        for images, weather, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            global_n = sampler.global_size(step)
            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred = ddp_model(images, weather)
                sse_t = ((pred.float() - targets) ** 2).sum()
                loss = sse_t * world / global_n  # DDP averages over ranks -> global mean
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if use_amp and scaler.get_scale() < scale_before:  # step skipped: inf/NaN gradients
                    state["epoch_skipped"] += 1
                    win_skipped += 1
            except torch.cuda.OutOfMemoryError:
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                fail(EXIT_OOM,
                     f"CUDA out of memory at epoch {epoch} step {step + 1} (per-GPU batch {local_bs}, "
                     f"peak {peak:.2f} GB). {last_path.name} from the last save is intact. Check "
                     "nvidia-smi for other processes, then rerun the same command to resume. Do not "
                     "change global_batch_size: that changes the protocol.")
            sse = sse_t.item()
            if np.isfinite(sse):
                nonfinite = 0
                local_sse += sse
                local_n += targets.numel()
                win_sse += sse
                win_n += targets.numel()
            else:
                nonfinite += 1
            step += 1
            state["step_in_epoch"] = step
            state["global_step"] += 1
            samples_this_launch += global_n

            if main_proc:
                progress.update(step, status(local_sse / max(local_n, 1), done_seconds + time.time() - epoch_t0))

            if step % log_every == 0:
                w_sse, w_n = all_sum([win_sse, win_n])
                if main_proc:
                    rate = samples_this_launch / max(time.time() - launch_t0, 1e-9)
                    with open(steps_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            datetime.now().isoformat(timespec="seconds"), epoch, step,
                            state["global_step"], f"{w_sse / max(w_n, 1):.8f}",
                            optimizer.param_groups[0]["lr"], f"{rate:.2f}",
                            f"{torch.cuda.memory_allocated(device) / 2**30:.3f}",
                            f"{scaler.get_scale():g}", win_skipped])
                win_sse = win_n = 0.0
                win_skipped = 0
                win_t0 = time.time()

            if step % SYNC_EVERY == 0 or step == steps_per_epoch:
                stop = int(_stop_signal is not None or (main_proc and stop_file.exists()))
                due = int(main_proc and time.time() - last_save >= ckpt_seconds)
                bad = int(nonfinite > max_nonfinite)
                collapsed = int(use_amp and scaler.get_scale() < min_scale)
                stop, due, bad, collapsed = all_max([stop, due, bad, collapsed])
                if collapsed:
                    fail(EXIT_NONFINITE, f"GradScaler scale collapsed to {scaler.get_scale():g} (< {min_scale:g}) at "
                                         f"epoch {epoch} step {step}: gradients are underflowing and the model has "
                                         f"stopped learning ({state['epoch_skipped']} skipped steps this epoch). Not "
                                         f"saving this state; {last_path.name} is the last saved state.")
                if bad:
                    fail(EXIT_NONFINITE, f"more than {max_nonfinite} consecutive non-finite losses at "
                                         f"epoch {epoch} step {step}; not saving this state. Resume "
                                         f"restarts from {last_path.name}.")
                if (due or stop) and step < steps_per_epoch:
                    save_last(local_sse, local_n, time.time() - epoch_t0,
                              "stop requested" if stop else "periodic")
                    last_save = time.time()
                if stop:
                    progress.close()
                    if step == steps_per_epoch:
                        save_last(local_sse, local_n, time.time() - epoch_t0, "stop requested")
                    log("Stopped on request; resume with the same command.")
                    if main_proc:
                        record_exit(EXIT_STOPPED, f"stopped at epoch {epoch} step {step}")
                    if distributed:
                        dist.destroy_process_group()
                    sys.exit(EXIT_STOPPED)
        progress.close()
        del train_loader

        train_seconds = time.time() - epoch_t0
        # State at the end of the training pass, before validation.
        save_last(local_sse, local_n, train_seconds, "end of training pass")
        sse, n = all_sum([local_sse, local_n])
        train_loss = sse / max(n, 1)

        # ---------------- validation (exact, unpadded, sharded over ranks)
        log(f"Epoch {epoch}: validating on {n_val if not args.max_val_batches else 'a subset of'} samples...")
        ddp_model.eval()
        v_sse = v_n = 0.0
        val_progress = Progress(len(val_loader), 0, f"Seed {args.seed} | Val {epoch}/{epochs}",
                                args.progress_interval, main_proc)
        with torch.no_grad():
            for b, (images, weather, targets, _) in enumerate(val_loader, start=1):
                images = images.to(device, non_blocking=True)
                weather = weather.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred = model(images, weather)
                v_sse += ((pred.float() - targets) ** 2).sum().item()
                v_n += targets.numel()
                if main_proc:
                    val_progress.update(b, f"val_loss(rank0 so far)={v_sse / max(v_n, 1):.5f}")
                if b % SYNC_EVERY == 0 and b <= common_val_batches:
                    stop = int(_stop_signal is not None or (main_proc and stop_file.exists()))
                    if all_max([stop])[0]:
                        log("Stopped on request during validation; resume repeats validation.")
                        if main_proc:
                            record_exit(EXIT_STOPPED, f"stopped during validation of epoch {epoch}")
                        if distributed:
                            dist.destroy_process_group()
                        sys.exit(EXIT_STOPPED)
        val_progress.close()
        # The final all-reduce runs after every rank has finished its shard.
        v_sse, v_n = all_sum([v_sse, v_n])
        val_loss = v_sse / max(v_n, 1)
        last_val = val_loss

        peaks = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                              torch.cuda.max_memory_reserved(device) / 2**30], device=device)
        if distributed:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        epoch_seconds = time.time() - epoch_t0

        improved = val_loss < state["best_val_loss"] - min_delta
        if improved:
            state["best_val_loss"], state["best_epoch"] = val_loss, epoch
            state["epochs_without_improvement"] = 0
            if main_proc:
                atomic_torch_save({
                    "format": "mamba_lstm_best_v1", "model_type": "mamba_lstm",
                    "epoch": epoch, "global_step": state["global_step"], "val_loss": val_loss,
                    "model_state_dict": model.state_dict(), "config": cfg, "config_hash": chash,
                    "seed": args.seed, "source": src,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }, best_path)
        else:
            state["epochs_without_improvement"] += 1
        barrier()

        state["history"].append({
            "epoch": epoch, "train_loss": f"{train_loss:.8f}", "val_loss": f"{val_loss:.8f}",
            "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": f"{epoch_seconds:.1f}",
            "train_samples_per_sec": f"{n / max(train_seconds, 1e-9):.2f}",
            "peak_mem_alloc_gb_max": f"{peaks[0].item():.3f}",
            "peak_mem_reserved_gb_max": f"{peaks[1].item():.3f}",
            "world_size": world, "improved": int(improved),
            "skipped_steps": state["epoch_skipped"], "grad_scale_end": f"{scaler.get_scale():g}",
        })
        # Remaining time: at most until early stopping or max epochs, at this epoch's pace.
        epochs_left = min(epochs - epoch, patience - state["epochs_without_improvement"])
        log(f"Seed {args.seed} | Epoch {epoch}/{epochs} done: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best={state['best_val_loss']:.6f}@ep{state['best_epoch']} "
            f"epoch_time={epoch_seconds / 60:.1f} min peak_mem={peaks[0].item():.2f} GB "
            + ("(new best; saved best.pt)" if improved else
               f"(no improvement {state['epochs_without_improvement']}/{patience})")
            + f" | at most {epochs_left} more epoch(s), <= {epochs_left * epoch_seconds / 3600:.1f} h")

        state["epoch_complete"] = True
        save_last(0.0, 0.0, 0.0, "end of epoch")
        rewrite_history()
        state.update(epoch=epoch + 1, step_in_epoch=0, epoch_complete=False,
                     epoch_sse=0.0, epoch_n=0.0, epoch_elapsed=0.0, epoch_skipped=0)

        if state["epochs_without_improvement"] >= patience:
            finished_reason = f"early stopping at epoch {epoch}"
            break
    else:
        finished_reason = f"reached max epochs ({epochs})"

    if main_proc:
        done_file.write_text(f"{datetime.now():%F %T} {finished_reason}; best epoch "
                             f"{state['best_epoch']} val_loss {state['best_val_loss']:.6f}\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            h = state["history"]
            plt.figure(figsize=(8, 5))
            plt.plot([r["epoch"] for r in h], [float(r["train_loss"]) for r in h], label="Training loss")
            plt.plot([r["epoch"] for r in h], [float(r["val_loss"]) for r in h], label="Validation loss")
            plt.axvline(state["best_epoch"], color="grey", ls=":", label=f"best epoch {state['best_epoch']}")
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss (clear-sky index)")
            plt.title(f"Mamba-LSTM (VMamba-{cfg['model']['scale']}), seed {args.seed}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "training_validation_loss.png", dpi=160)
            plt.close()
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: could not draw the loss plot: {exc}")
        record_exit(EXIT_DONE, finished_reason)
        log(f"Training finished: {finished_reason}. Best epoch {state['best_epoch']}, "
            f"val_loss {state['best_val_loss']:.6f}. Wrote {done_file}.")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(EXIT_DONE)


if __name__ == "__main__":
    main()
