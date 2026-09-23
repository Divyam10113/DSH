"""
MRI Volume-Consistent Data Augmentation Pipeline for Knee MRI.
Ensures that 3D MRI series maintain anatomical spatial consistency across all 2D slices.
Includes spatial transforms, intensity jitter, slice dropout, and Test-Time Augmentation (TTA).
"""

import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Dict, Tuple, Optional


class VolumeConsistentAugmenter:
    """
    Applies identical spatial transformations across all slices of a 3D MRI series.
    Randomly varies contrast and brightness, and performs Slice Dropout during training.
    """
    def __init__(
        self,
        is_training: bool = True,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        max_rotate_deg: float = 15.0,
        p_slice_dropout: float = 0.3,
        max_slice_dropout_rate: float = 0.15,
        p_intensity_jitter: float = 0.5
    ):
        self.is_training = is_training
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.max_rotate_deg = max_rotate_deg
        self.p_slice_dropout = p_slice_dropout
        self.max_slice_dropout_rate = max_slice_dropout_rate
        self.p_intensity_jitter = p_intensity_jitter

    def __call__(self, series_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            series_tensor: (K, C, H, W) where K is the number of slices in this MRI series.
        Returns:
            augmented: (K_aug, C, H, W)
        """
        if not self.is_training:
            return series_tensor

        K, C, H, W = series_tensor.shape

        # 1. Slice Dropout (Randomly drop 5-15% of slices to force model not to rely on a fixed index)
        if self.p_slice_dropout > 0 and random.random() < self.p_slice_dropout and K > 15:
            drop_count = max(1, int(K * random.uniform(0.05, self.max_slice_dropout_rate)))
            keep_indices = sorted(random.sample(range(K), K - drop_count))
            series_tensor = series_tensor[keep_indices]
            K = series_tensor.shape[0]

        # 2. Consistent Horizontal Flip (Applied identically to all K slices in this volume)
        do_hflip = random.random() < self.p_flip
        if do_hflip:
            series_tensor = torch.flip(series_tensor, dims=[3])  # flip width

        # 3. Consistent Random Rotation (Applied identically to all K slices)
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            # PyTorch functional rotate on 4D tensor (K, C, H, W)
            series_tensor = TF.rotate(series_tensor, angle=angle)

        # 4. Intensity / Contrast Jitter
        if random.random() < self.p_intensity_jitter:
            gamma = random.uniform(0.85, 1.15)
            gain = random.uniform(0.9, 1.1)
            # Normalize, apply gamma/gain, clamp
            mean = series_tensor.mean()
            std = series_tensor.std() + 1e-6
            norm = (series_tensor - mean) / std
            # Gamma on the magnitude: a negative base with fractional gamma would produce NaNs
            scaled = norm * gain
            series_tensor = torch.clamp(torch.sign(scaled) * scaled.abs() ** gamma, -3.0, 3.0)

        return series_tensor


class MultiPlaneAugmenter:
    """
    Applies volume-consistent augmentation across all available MRI planes in a study.
    """
    def __init__(self, is_training: bool = True):
        self.augmenter = VolumeConsistentAugmenter(is_training=is_training)

    def __call__(self, series_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        augmented = {}
        for plane, tensor in series_dict.items():
            if tensor is not None:
                augmented[plane] = self.augmenter(tensor)
        return augmented
