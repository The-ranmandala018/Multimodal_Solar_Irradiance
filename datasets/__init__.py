"""Dataset loaders for the multimodal solar irradiance pipeline."""

from .folsom_dataset import FolsomDataset, get_data_loaders

__all__ = ["FolsomDataset", "get_data_loaders"]
