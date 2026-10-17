"""Checkpointed, resumable chunk storage for the cloud-feature extraction.

One chunk = one day folder (motion state resets every day, so chunked results are
identical to a single pass). For each chunk:

  1. the worker writes <parts>/.tmp/<key>.<pid>.parquet and fsyncs it,
  2. reads it back and validates it (row count and paths equal the chunk's input,
     every feature column present, verify_cloud_features.verify() checks pass),
  3. atomically renames it to <parts>/<key>.parquet,
  4. writes the ledger record <parts>/_done/<key>.json (atomic) with the row count,
     the file's sha256 and the frozen-config id. The ledger record is the "complete" marker.

On restart, a chunk is skipped only if its ledger record exists, the config id matches,
the part file exists with the recorded sha256, and the row count equals the chunk's
current input. Anything else (missing, partial, corrupted, stale) is deleted and redone.

The frozen-config id is the sha256 of the hashes of every file that defines the
features. Before any work, those hashes are compared with FROZEN_V1_manifest.json; a
mismatch aborts the run, so checkpoints can only be produced by the frozen v1 configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from common import EXP_DIR, PROJECT_DIR

MANIFEST = EXP_DIR / "FROZEN_V1_manifest.json"
# Files that define the feature values (code and fitted parameters). The extraction
# driver and this module only orchestrate and are deliberately not part of the id.
FEATURE_DEFINING = [
    "scripts/lgbm_cloud/cloud_features.py",
    "scripts/lgbm_cloud/sky_geometry.py",
    "scripts/lgbm_cloud/common.py",
    "experiments/lightgbm_cloud_features/features/image_index.parquet",
    "experiments/lightgbm_cloud_features/features/sky_geometry.json",
    "experiments/lightgbm_cloud_features/features/sky_mask_2014.npy",
    "experiments/lightgbm_cloud_features/features/sky_mask_2015.npy",
    "experiments/lightgbm_cloud_features/features/sky_mask_2016.npy",
    "experiments/lightgbm_cloud_features/features/clear_sky_library.npz",
    "experiments/lightgbm_cloud_features/features/cloud_thresholds.json",
]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def frozen_config_id() -> str:
    """Abort unless every feature-defining file matches the frozen v1 manifest."""
    manifest = json.loads(MANIFEST.read_text())["files"]
    current, bad = {}, []
    for rel in FEATURE_DEFINING:
        cur = sha256_file(PROJECT_DIR / rel)
        current[rel] = cur
        if rel not in manifest or manifest[rel]["sha256"] != cur:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"feature-defining files differ from the frozen v1 manifest: {bad}")
    blob = "\n".join(f"{k} {current[k]}" for k in sorted(current))
    return hashlib.sha256(blob.encode()).hexdigest()


def chunk_key(day_dir: str) -> str:
    return day_dir.replace("/", "-")


class ChunkStore:
    def __init__(self, parts_dir: Path, config_id: str):
        self.dir = Path(parts_dir)
        self.done = self.dir / "_done"
        self.tmp = self.dir / ".tmp"
        for d in (self.dir, self.done, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        self.config_id = config_id

    def part(self, key):
        return self.dir / f"{key}.parquet"

    def record(self, key):
        return self.done / f"{key}.json"

    def is_complete(self, key, n_expected) -> tuple[bool, str]:
        rec, part = self.record(key), self.part(key)
        if not rec.exists():
            return False, "no ledger record"
        try:
            r = json.loads(rec.read_text())
        except Exception:
            return False, "unreadable ledger record"
        if r.get("config_id") != self.config_id:
            return False, "config id differs"
        if r.get("n_rows") != n_expected:
            return False, "row count differs from current input"
        if not part.exists():
            return False, "part file missing"
        if sha256_file(part) != r.get("sha256"):
            return False, "part file sha256 mismatch"
        return True, "ok"

    def discard(self, key):
        for p in (self.record(key), self.part(key)):
            if p.exists():
                p.unlink()

    def clean_tmp(self):
        for p in self.tmp.glob("*"):
            p.unlink()

    def commit(self, key, df: pd.DataFrame, expected_paths: list[str], validate, meta: dict) -> dict:
        """Write, read back, validate, atomically publish, then write the ledger record."""
        tmp = self.tmp / f"{key}.{os.getpid()}.parquet"
        df.to_parquet(tmp, index=False)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        back = pd.read_parquet(tmp)
        problems = []
        if len(back) != len(expected_paths):
            problems.append(f"rows {len(back)} != {len(expected_paths)}")
        if sorted(back["path"]) != sorted(expected_paths):
            problems.append("paths differ from chunk input")
        problems += validate(back)
        if problems:
            tmp.unlink()
            raise RuntimeError(f"chunk {key} failed validation: {problems}")
        digest = sha256_file(tmp)
        os.replace(tmp, self.part(key))
        rec = {"key": key, "n_rows": len(back), "sha256": digest, "config_id": self.config_id,
               "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
        rtmp = self.tmp / f"{key}.{os.getpid()}.json"
        rtmp.write_text(json.dumps(rec, indent=1))
        with open(rtmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(rtmp, self.record(key))
        return rec
