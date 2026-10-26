# Phase 1 alignment checks (read-only)

Run in this order, each with an output folder argument (`out`):

1. `p1_samples.py`: rebuilds the FolsomDataset sample list, split sizes, and the CNN image offset by file name; checks saved prediction order.
2. `p1_images.py`: file-name time vs file modified time for every image; image intervals by month.
3. `p1_lags.py`: CSV clock vs pvlib clear-sky on clear days; image brightness vs k lag, by file-name and modified-time labels.
4. `p1_true_offsets.py`: CNN image offsets and the "image at or before t" rule on the modified-time clock.

Results (JSON/CSV) are in `experiments/lightgbm_cloud_features/checks/`.
