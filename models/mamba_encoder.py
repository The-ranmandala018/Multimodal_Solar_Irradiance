"""VMamba visual encoder for the Folsom 10-channel 3D projection.

Input:
    (B, 10, 180, 720)

Output:
    (B, 256) visual feature vector

Drop-in replacement for FolsomCNNEncoder: same input tensor, same 256-D
output, so the existing LSTM, fusion and regression head are unchanged.

    input normalisation (fixed, RGB channels only)
    -> VMamba backbone (official Backbone_VSSM, stage-4 output, 768 x 6 x 23)
    -> global average pooling -> Linear(768, 256)

The pooling + linear projection mirrors the CNN encoder. VMamba presets match
the MERCon 2026 benchmark (VMamba v2, forward_type v05_noz).

ImageNet weights are adapted to the 10 input channels. The Folsom channels are
(x, y, z, theta, phi, R, G, B, intensity, solid-angle weight). The pretrained
3-channel stem kernel is copied onto R, G, B (channels 5-7) and the other
seven input channels start at zero, so the stem initially sees exactly the
ImageNet-normalised RGB image and learns to use the geometry/intensity
channels during training. The ImageNet classifier norm is the stage-4 output
norm (outnorm3) in the backbone; the classifier head is dropped.

fp32 selective scan. VMamba's v05 forward passes the scan inputs in the autocast
dtype (force_fp32=False), relying on a scan kernel that returns fp32. The vendored
vmamba.py always dispatches to mamba_ssm's selective_scan_cuda, which returns the
input dtype, so under fp16 autocast the scan output overflows (observed max 70,140 >
65,504) and training collapses (experiments/mamba_lstm/diagnostics/seed1_nonfinite).
The encoder therefore rebinds every SS2D scan to the same call with force_fp32=True:
the scan runs and returns in fp32; everything else stays fp16. No weights change.

VMamba is imported lazily so that importing this module (and the model
registry) does not require timm, fvcore or the selective-scan kernel.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn

BACKBONE_DIR = Path(__file__).resolve().parent / "backbones" / "vmamba"

DEFAULT_WEIGHTS_DIR = os.environ.get(
    "VMAMBA_WEIGHTS_DIR",
    "/storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark/weights",
)

# Same presets as irradiance_benchmark/models/wrappers.py (MERCon 2026).
VMAMBA_PRESETS = {
    "tiny": {"depths": [2, 2, 8, 2], "dims": 96, "ssm_ratio": 1.0, "drop_path_rate": 0.2},
    "small": {"depths": [2, 2, 15, 2], "dims": 96, "ssm_ratio": 2.0, "drop_path_rate": 0.3},
}
VMAMBA_COMMON = {
    "ssm_d_state": 1,
    "ssm_dt_rank": "auto",
    "ssm_conv": 3,
    "ssm_conv_bias": False,
    "forward_type": "v05_noz",
    "mlp_ratio": 4.0,
    "norm_layer": "ln2d",
    "downsample_version": "v3",
    "patchembed_version": "v2",
}

IN_CHANNELS = 10
RGB_CHANNELS = (5, 6, 7)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _import_vmamba():
    path = str(BACKBONE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import vmamba  # noqa: E402  (vendored upstream file)

    return vmamba


def adapt_imagenet_state_dict(state: dict, in_channels: int = IN_CHANNELS) -> dict:
    """Map an ImageNet VMamba classifier state dict onto the 10-channel backbone."""
    adapted = {}
    for key, value in state.items():
        if key.startswith("classifier.head."):
            continue
        if key.startswith("classifier.norm."):
            key = "outnorm3." + key[len("classifier.norm."):]
        adapted[key] = value

    rgb_weight = adapted["patch_embed.0.weight"]
    if rgb_weight.shape[1] != 3:
        raise ValueError(f"Expected a 3-channel stem, got {tuple(rgb_weight.shape)}")
    weight = rgb_weight.new_zeros(
        (rgb_weight.shape[0], in_channels, *rgb_weight.shape[2:])
    )
    weight[:, list(RGB_CHANNELS)] = rgb_weight
    adapted["patch_embed.0.weight"] = weight
    return adapted


class FolsomVMambaEncoder(nn.Module):
    """VMamba encoder for the Folsom 3D projection representation."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        feature_dim: int = 256,
        scale: str = "tiny",
        pretrained: bool = True,
        weight_path: str | None = None,
    ):
        super().__init__()

        if scale not in VMAMBA_PRESETS:
            raise ValueError(f"Unknown VMamba scale {scale!r}; choose from {sorted(VMAMBA_PRESETS)}")

        vmamba = _import_vmamba()
        if not getattr(vmamba, "WITH_SELECTIVESCAN_MAMBA", False):
            print(
                "WARNING: no selective-scan CUDA kernel found; VMamba will use "
                "the pure-PyTorch scan, which is far too slow for training."
            )

        self.in_channels = in_channels
        self.scale = scale
        self.backbone = vmamba.Backbone_VSSM(
            out_indices=(3,),
            pretrained=None,
            patch_size=4,
            in_chans=in_channels,
            num_classes=1000,
            **VMAMBA_PRESETS[scale],
            **VMAMBA_COMMON,
        )

        self.fp32_scan_modules = self._force_fp32_scan()

        mean = torch.zeros(1, in_channels, 1, 1)
        std = torch.ones(1, in_channels, 1, 1)
        for channel, m, s in zip(RGB_CHANNELS, IMAGENET_MEAN, IMAGENET_STD):
            mean[0, channel] = m
            std[0, channel] = s
        self.register_buffer("input_mean", mean, persistent=False)
        self.register_buffer("input_std", std, persistent=False)

        if pretrained:
            if weight_path is None:
                weight_path = os.path.join(DEFAULT_WEIGHTS_DIR, f"vmamba_{scale}_1k.pth")
            self._load_imagenet(weight_path)

        backbone_dim = self.backbone.dims[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(backbone_dim, feature_dim)

    def _force_fp32_scan(self) -> int:
        """Run every SS2D selective scan in fp32 (see module docstring)."""
        count = 0
        for module in self.backbone.modules():
            if type(module).__name__ != "SS2D":
                continue
            core = module.forward_core
            expected = (getattr(core, "func", None) == module.forward_corev2
                        and core.keywords == {"force_fp32": False, "no_einsum": True})
            if not expected:
                raise RuntimeError(f"unexpected SS2D forward_core {core!r}; refusing to patch")
            module.forward_core = partial(module.forward_corev2, force_fp32=True, no_einsum=True)
            count += 1
        if count == 0:
            raise RuntimeError("no SS2D modules found to patch")
        return count

    def _load_imagenet(self, weight_path: str) -> None:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = adapt_imagenet_state_dict(checkpoint["model"], self.in_channels)
        result = self.backbone.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"ImageNet weights {weight_path} do not match VMamba-{self.scale}: "
                f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
        print(f"Loaded ImageNet VMamba-{self.scale} weights: {weight_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of Folsom projection tensors.

        Args:
            x: Tensor with shape (B, 10, 180, 720).

        Returns:
            Tensor with shape (B, feature_dim).
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor (B, C, H, W), got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} Folsom projection channels, got {x.shape[1]}"
            )

        x = (x - self.input_mean) / self.input_std
        x = self.backbone(x)[-1]
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)


__all__ = ["FolsomVMambaEncoder", "VMAMBA_PRESETS", "adapt_imagenet_state_dict"]
