"""Vision Mamba (Vim-tiny style) visual encoder for the Folsom projection.

Drop-in replacement for FolsomCNNEncoder:
    Input:  (B, 10, 180, 720)
    Output: (B, feature_dim)  (default 256)

Design (Vim-tiny sizes: dim 192, depth 24):
    1. Patch embedding: Conv2d(10 -> 192, kernel = stride = 12x12)
       -> 15 x 60 = 900 tokens.
    2. Learnable position embedding (1, 900, 192).
    3. 24 bidirectional Mamba blocks (forward + backward scan, pre-norm,
       residual connection).
    4. LayerNorm, mean pooling over tokens, Linear(192 -> feature_dim).

This is a simplified Vim-tiny. It does not load the ImageNet Vim weights,
because the Folsom input has 10 channels and a 180x720 grid, so the patch
and position embeddings are trained from scratch.

Needs the `mamba-ssm` package and a CUDA GPU. The import is done lazily, so
the CNN models still work on machines without mamba-ssm.
"""

from __future__ import annotations

import torch
from torch import nn


class _VimBlock(nn.Module):
    """Pre-norm bidirectional Mamba block with a residual connection."""

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba  # lazy import (needs CUDA build)

        self.norm = nn.LayerNorm(dim)
        self.forward_ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        self.backward_ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        fwd = self.forward_ssm(h)
        bwd = self.backward_ssm(h.flip(1)).flip(1)
        return x + fwd + bwd


class FolsomVimTinyEncoder(nn.Module):
    """Vim-tiny style encoder for the (10, 180, 720) Folsom projection."""

    def __init__(
        self,
        in_channels: int = 10,
        feature_dim: int = 256,
        dim: int = 192,
        depth: int = 24,
        patch_size: tuple[int, int] = (12, 12),
        image_size: tuple[int, int] = (180, 720),
    ):
        super().__init__()

        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ValueError(
                f"image_size {image_size} must be divisible by "
                f"patch_size {patch_size}"
            )

        self.in_channels = in_channels
        self.image_size = tuple(image_size)

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        num_tokens = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([_VimBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, feature_dim)

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
                f"Expected {self.in_channels} Folsom projection channels, "
                f"got {x.shape[1]}"
            )

        if tuple(x.shape[2:]) != self.image_size:
            raise ValueError(
                f"Expected spatial size {self.image_size}, "
                f"got {tuple(x.shape[2:])}"
            )

        x = self.patch_embed(x)                 # (B, dim, 15, 60)
        x = x.flatten(2).transpose(1, 2)        # (B, 900, dim)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x).mean(dim=1)            # (B, dim)
        return self.projection(x)


__all__ = ["FolsomVimTinyEncoder"]
