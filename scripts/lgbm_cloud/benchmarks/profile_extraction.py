"""Single-thread cProfile of the cloud-feature extraction on 300 images of 2014-10-31, saved
verbatim from the inline command run on 2026-10-17 (77 ms/image; shares recorded in
experiments/lightgbm_cloud_features/checks/gpu_benchmark_and_checkpoint_test.json).
Run from scripts/lgbm_cloud with OMP_NUM_THREADS=1 and PYTHONPATH=<repo>:scripts/lgbm_cloud.
"""
import cProfile, pstats, io, cv2, pandas as pd
cv2.setNumThreads(1)
import extract_cloud_features as E
idx=pd.read_parquet(E.FEATURES_DIR/"image_index.parquet",columns=["path","day_dir","t_img","sza_t_img","corrupted"])
day=idx[(idx.day_dir=="2014/10/31")&(idx.sza_t_img<=87)].sort_values("t_img").head(300)
E.process_day(day.head(5))  # warm up caches
pr=cProfile.Profile(); pr.enable(); df=E.process_day(day); pr.disable()
print("ms/img", df.ms.median())
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats("tottime").print_stats(18); print(s.getvalue()[:4000])
