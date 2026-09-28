"""Shape-only conversion for the Folsom 3D feature representation.

The source preprocessing creates (129600, 10) float32 vectors. This module
only converts that representation to the spatial tensor expected by a
PyTorch visual encoder. It does not change the feature values.
"""

import numpy as np

EXPECTED_HEIGHT = 180
EXPECTED_WIDTH = 720
EXPECTED_CHANNELS = 10
EXPECTED_VECTOR_SHAPE = (EXPECTED_HEIGHT * EXPECTED_WIDTH, EXPECTED_CHANNELS)


def vectors_to_projection_grid(vectors: np.ndarray) -> np.ndarray:
    """Convert (129600, 10) vectors to (180, 720, 10)."""
    arr = np.asarray(vectors)
    if arr.shape != EXPECTED_VECTOR_SHAPE:
        raise ValueError(
            f"Expected shape {EXPECTED_VECTOR_SHAPE}, got {arr.shape}"
        )
    return arr.reshape(EXPECTED_HEIGHT, EXPECTED_WIDTH, EXPECTED_CHANNELS)


def vectors_to_projection_tensor(vectors: np.ndarray):
    """Convert (129600, 10) to a PyTorch tensor with shape (10, 180, 720)."""
    import torch

    grid = vectors_to_projection_grid(vectors)
    return torch.from_numpy(np.ascontiguousarray(grid)).permute(2, 0, 1)
