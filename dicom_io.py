"""
Test-time input adapter for the offline submission notebook.
Owned by Pranav (M4); delegates all DICOM work to Krish's `knee_dicom` package (M1).

Train/test parity: the hidden test DICOMs go through exactly the same code as the training cache
(`knee_dicom.cache.preprocess_study`: decode all transfer syntaxes -> route one series per plane
-> percentile window to [0, 1] -> letterbox resize -> subsample slices -> uint8), then through the
same `to_model_tensor` conversion KneeMRIDataset uses. The PreprocessConfig must equal the one the
training cache was built with (see the cache's meta.json / summary; defaults are 128px, 32 slices).
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PLANES = ("sagittal", "coronal", "axial")


def to_model_tensor(arr: np.ndarray, img_size: Optional[int] = None) -> torch.Tensor:
    """Cached/preprocessed array -> float32 (K, 1, H, W) in [0, 1]; uint8 caches are divided by 255."""
    t = torch.from_numpy(np.asarray(arr))
    t = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
    if t.dim() == 3:
        t = t.unsqueeze(1)
    if img_size is not None and tuple(t.shape[-2:]) != (img_size, img_size):
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t


def load_study(study_dir: str, series_rows: pd.DataFrame, img_size: int = 128, max_slices: int = 32,
               window: str = "standard") -> Dict[str, torch.Tensor]:
    """Raw DICOM study folder + its *_series.csv rows -> {plane: (K, 1, H, W)} via knee_dicom (M1)."""
    try:
        from knee_dicom.cache import preprocess_study
        from knee_dicom.preprocess import PreprocessConfig
    except ImportError as e:
        raise ImportError("knee_dicom (M1, branch krish-knee-dicom) must be importable for raw-DICOM "
                          "inference; include it in the knee-code dataset") from e

    cfg = PreprocessConfig(img_size=img_size, max_slices=max_slices, window=window)
    arrays, meta = preprocess_study(study_dir, series_rows, cfg)
    for key, info in meta.get("series", {}).items():
        if info.get("error"):
            print(f"[dicom_io] {os.path.basename(study_dir)}/{key}: {info['error']}")
    return {p: to_model_tensor(arrays[p]) for p in PLANES if p in arrays}


def load_cached_study(cache_dir: str, study_id: str, img_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Pre-built cache ({cache_dir}/{study_id}/{plane}.npy, e.g. from `knee_dicom.cache --split test`)."""
    out = {}
    for plane in PLANES:
        path = os.path.join(cache_dir, str(study_id), f"{plane}.npy")
        if os.path.exists(path):
            out[plane] = to_model_tensor(np.load(path), img_size)
    return out
