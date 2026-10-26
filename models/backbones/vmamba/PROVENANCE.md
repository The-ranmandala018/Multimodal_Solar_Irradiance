# Vendored VMamba source

`vmamba.py` and `LICENSE` are copied **unchanged** from the official VMamba
repository:

- Repository: https://github.com/MzeroMiko/VMamba (MIT License, see `LICENSE`)
- Commit: `2ed52ead062a51a64521ed3871d52914bf532876` (2025-03-07)
- File: `vmamba.py`, sha256 `572207a2b6553ff85d81b602eebda2041623073f01fc0c6d1280c5a2add040f6`

This is the same file the MERCon 2026 benchmark vendors
(`irradiance_benchmark/models/backbones/vmamba/vmamba.py`, byte-identical).

Do not edit it. All adaptation to the Folsom 10-channel projection lives in
`models/mamba_encoder.py`.

## Runtime requirements

`vmamba.py` imports `timm` and `fvcore` at module level and needs a
selective-scan CUDA kernel for usable speed. It looks for, in order,
`selective_scan_cuda_oflex`, `selective_scan_cuda_core`, and
`selective_scan_cuda` (the kernel shipped with `mamba_ssm`). Without any of
them it falls back to a pure-PyTorch scan that is far too slow for training.
The `mamba_lstm` environment uses `selective_scan_cuda` from `mamba_ssm` 2.2.4.

## ImageNet-1K weights

Official VMamba v2 classification checkpoints, as downloaded by the MERCon
benchmark's `setup.sh`:

| Scale | Release URL | sha256 |
|---|---|---|
| Tiny  | https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth | `d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a` |
| Small | https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth | `c540366e366d81acf69251810e7ea7268e0ca8bcb6843f3b773c1f0105bd30cb` |

Weights are not committed to git.
