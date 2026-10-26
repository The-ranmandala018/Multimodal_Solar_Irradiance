"""LightGBM CPU vs OpenCL-GPU benchmark on feature set B (300 rounds), saved verbatim from the
inline command run on 2026-10-17 (results in experiments/lightgbm_cloud_features/checks/
gpu_benchmark_and_checkpoint_test.json). Run with CUDA_VISIBLE_DEVICES=0 from scripts/lgbm_cloud
with PYTHONPATH=<repo>:scripts/lgbm_cloud. Benchmark only; not used for any reported result.
"""
import time, numpy as np, lightgbm as lgb
from train_tabular import load, LGBM_DEFAULTS
from feature_sets import get_feature_set, TABLE_OF
f=get_feature_set("B"); tr,va,vs,_=load(f,TABLE_OF["B"])
res={}
for name,extra in (("cpu_32thr",{}),("gpu_opencl",{"device_type":"gpu","gpu_device_id":0}),("gpu_opencl_dp",{"device_type":"gpu","gpu_device_id":0,"gpu_use_dp":True})):
    p=dict(LGBM_DEFAULTS); p.update(extra); p.update(seed=0,bagging_seed=0,feature_fraction_seed=0,data_random_seed=0)
    if "device_type" in extra: p.pop("force_row_wise",None); p["deterministic"]=False
    t0=time.time(); m=lgb.train(p,lgb.Dataset(tr[f],tr.k_true),num_boost_round=300); dt=time.time()-t0
    res[name]=(dt,m.predict(vs[f]))
    print(name, f"{dt:.1f} s for 300 rounds, val k-MSE {np.mean((res[name][1]-vs.k_true.values)**2):.6f}")
base=res["cpu_32thr"][1]
for n,(dt,pr) in res.items(): print(n,"max |pred - cpu|",float(np.abs(pr-base).max()))
t0=time.time(); m=lgb.train(dict(LGBM_DEFAULTS,seed=0,bagging_seed=0,feature_fraction_seed=0,data_random_seed=0),lgb.Dataset(tr[f],tr.k_true),300); pr2=m.predict(vs[f])
print("cpu rerun identical:", bool(np.array_equal(pr2,base)))
