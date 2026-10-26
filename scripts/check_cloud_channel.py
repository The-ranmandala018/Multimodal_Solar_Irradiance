"""Look at the cloud channel for one real image (visual and numeric check).

Run on the server before any training:
    PYTHONPATH=. python scripts/check_cloud_channel.py \
        --image /storage2/CV_Irradiance/datasets/1_Folsom/2016/06/15/20160615_180000.jpg \
        --time "2016-06-15 18:00:00" --out /tmp/cloud_channel_check.png

--time is the sample time (UTC, naive), as in the CSV. Writes a PNG with the
remapped RGB (left) and the cloud channel (right) and prints value statistics.
The PNG is derived from a dataset image: keep it local (CC BY 4.0, see
experiments/lightgbm_cloud_features/documentation/DATA_AVAILABILITY.md).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from preprocessing.folsom_adapter import FolsomPreprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--out", default="cloud_channel_check.png")
    args = ap.parse_args()

    t = pd.Timestamp(args.time)
    plain = FolsomPreprocessor(year=t.year).process_image_to_tensor(args.image, month=t.month)
    clouded = FolsomPreprocessor(year=t.year, cloud_channel=True).process_image_to_tensor(
        args.image, month=t.month, cloud_time=t
    )

    same = [i for i in range(10) if i != 8 and bool((plain[i] == clouded[i]).all())]
    print("tensor shape:", tuple(clouded.shape))
    print("channels identical to the original tensor (expect all except 8):", same)
    c = clouded[8].numpy()
    print(f"cloud channel: min {c.min():.3f}  max {c.max():.3f}  mean {c.mean():.3f}  "
          f"fraction exactly 0: {(c == 0).mean():.3f}  finite: {bool(np.isfinite(c).all())}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(np.clip(plain[5:8].permute(1, 2, 0).numpy(), 0, 1))
    ax[0].set_title("remapped RGB (rows: zenith 0 -> 90 deg, columns: azimuth 0 -> 360 deg)")
    im = ax[1].imshow(c, vmin=0, vmax=1, cmap="viridis")
    ax[1].set_title("cloud channel (0 = clear or not measured, 1 = strong cloud)")
    fig.colorbar(im, ax=ax[1], fraction=0.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
