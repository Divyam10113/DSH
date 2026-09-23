"""
PyTorch Dataset and Collate Pipeline for Multi-Series Knee MRI Studies.
Supports loading from preprocessed .npy/.npz files, joined with multi-label CSVs,
and handles variable slice counts across studies via dynamic batch padding with attention masks.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

from augmentations import MultiPlaneAugmenter
from models.knee_model import ABNORMALITIES


class KneeMRIDataset(Dataset):
    """
    Knee MRI Multi-Series Dataset.
    Loads Sagittal, Coronal, and Axial series for each patient study.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Optional[str] = None,
        planes: List[str] = ("sagittal", "coronal", "axial"),
        target_cols: List[str] = ABNORMALITIES,
        is_training: bool = True,
        img_size: Tuple[int, int] = (128, 128),
        synthetic_mode: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.planes = list(planes)
        self.target_cols = [c for c in target_cols if c in self.df.columns]
        self.is_training = is_training
        self.img_size = img_size
        self.synthetic_mode = synthetic_mode
        self.augmenter = MultiPlaneAugmenter(is_training=is_training)

    def __len__(self) -> int:
        return len(self.df)

    def _load_series(self, study_id: str, plane: str) -> Optional[torch.Tensor]:
        """
        Loads a preprocessed numpy array for a given study and plane.
        Expected file structure: {data_dir}/{study_id}/{plane}.npy
        Array shape: (K, H, W) or (K, C, H, W)
        """
        if self.synthetic_mode or self.data_dir is None:
            # Generate realistic synthetic series if real cached files are not mounted yet
            K = np.random.randint(20, 36)
            H, W = self.img_size
            return torch.randn(K, 1, H, W, dtype=torch.float32)

        file_path = os.path.join(self.data_dir, str(study_id), f"{plane}.npy")
        if not os.path.exists(file_path):
            # Also check .npz
            file_path_npz = os.path.join(self.data_dir, str(study_id), f"{plane}.npz")
            if os.path.exists(file_path_npz):
                with np.load(file_path_npz) as data:
                    arr = data["arr_0"]
            else:
                return None
        else:
            arr = np.load(file_path)

        # Convert to float32 tensor; M1's cache stores uint8 0..255 -> rescale to [0, 1]
        tensor = torch.from_numpy(arr)
        tensor = tensor.float() / 255.0 if tensor.dtype == torch.uint8 else tensor.float()

        # Ensure (K, C, H, W) shape
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # (K, 1, H, W)

        # Resize if spatial resolution differs
        K, C, H, W = tensor.shape
        if (H, W) != self.img_size:
            import torchvision.transforms.functional as TF
            tensor = TF.resize(tensor, self.img_size)

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict[str, torch.Tensor]]]:
        row = self.df.iloc[idx]
        # Kaggle CSVs use StudyInstanceUID; keep study_id for older synthetic frames
        study_id = row.get("StudyInstanceUID", row.get("study_id", f"study_{idx}"))

        # Load each plane
        series_dict = {}
        for plane in self.planes:
            tensor = self._load_series(study_id, plane)
            if tensor is not None:
                series_dict[plane] = tensor

        # Apply volume-consistent augmentations
        series_dict = self.augmenter(series_dict)

        # Extract targets if available. NaN = unlabeled finding; the training engine masks it out of the loss.
        if len(self.target_cols) == len(ABNORMALITIES):
            targets = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        elif self.target_cols:
            missing = sorted(set(ABNORMALITIES) - set(self.target_cols))
            raise KeyError(f"Label columns missing from dataframe: {missing}")
        else:
            targets = torch.full((len(ABNORMALITIES),), float("nan"), dtype=torch.float32)

        return {
            "study_id": str(study_id),
            "series": series_dict,
            "targets": targets
        }


def knee_collate_fn(batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    """
    Collate function that dynamically pads variable-length MRI series into batched tensors.
    Produces boolean masks where True indicates padded dummy slices.

    Returns:
        batched_series: Dict[plane, Tensor(B, K_max, C, H, W)]
        batched_masks:  Dict[plane, Tensor(B, K_max) (bool mask)]
        targets:        Tensor(B, 12)
        study_ids:      List of study ID strings
    """
    B = len(batch)
    study_ids = [item["study_id"] for item in batch]
    targets = torch.stack([item["targets"] for item in batch], dim=0)

    planes = ["sagittal", "coronal", "axial"]
    batched_series = {}
    batched_masks = {}

    for plane in planes:
        # Collect all tensors for this plane in the batch
        plane_tensors = [item["series"].get(plane) for item in batch if plane in item["series"]]
        
        if not plane_tensors:
            continue

        # Find maximum slice count K_max in this batch for this plane
        K_max = max(t.shape[0] for t in plane_tensors)
        C, H, W = plane_tensors[0].shape[1:]

        padded_tensor = torch.zeros((B, K_max, C, H, W), dtype=torch.float32)
        mask_tensor = torch.ones((B, K_max), dtype=torch.bool)  # Default: all masked (ignored)

        for i, item in enumerate(batch):
            t = item["series"].get(plane)
            if t is not None:
                K = t.shape[0]
                padded_tensor[i, :K] = t
                mask_tensor[i, :K] = False  # Real slices are unmasked (attend)

        batched_series[plane] = padded_tensor
        batched_masks[plane] = mask_tensor

    return batched_series, batched_masks, targets, study_ids
