"""Slice normalisation, windowing, resizing and slice-count handling.

MRI has no Hounsfield units, and intensities vary by scanner, coil and protocol. So "windowing" here
means a per-volume percentile window: clip to the [low, high] percentiles of the foreground voxels
and rescale to [0, 1]. Using the whole volume (not each slice) keeps relative brightness between
slices, which matters for findings like effusion that show up as bright fluid on a few slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Named percentile windows. "standard" suits all findings; "fluid" clips less of the bright tail so
# fluid/oedema stays distinguishable; "bone" compresses the bright tail to emphasise cortex/marrow.
WINDOWS = {
    "standard": (0.5, 99.5),
    "fluid": (0.5, 99.9),
    "bone": (1.0, 98.0),
}


@dataclass(frozen=True)
class PreprocessConfig:
    img_size: int = 128           # output H = W (M3's KneeMRIDataset default)
    max_slices: int = 32          # longer series are uniformly subsampled; shorter ones kept as-is
    window: str = "standard"
    keep_aspect: bool = True      # letterbox to square instead of stretching
    dtype: str = "uint8"          # "uint8" (0..255, compact) or "float16" (0..1)


def window_volume(vol: np.ndarray, window: str = "standard") -> np.ndarray:
    """Clip to the chosen percentile window of the foreground and rescale to float32 [0, 1]."""
    low_p, high_p = WINDOWS[window]
    vol = vol.astype(np.float32, copy=False)
    fg = vol[vol > vol.min()]
    sample = fg if fg.size > 100 else vol.ravel()
    if sample.size > 2_000_000:  # percentiles on a subsample are plenty accurate and much faster
        sample = sample[:: sample.size // 2_000_000 + 1]
    lo, hi = np.percentile(sample, [low_p, high_p])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def resize_slice(img: np.ndarray, size: int, keep_aspect: bool = True,
                 pixel_spacing: tuple[float, float] | None = None) -> np.ndarray:
    """Bilinear resize of one float slice to (size, size).

    With keep_aspect, the physical aspect ratio (using PixelSpacing = (row, col) mm) is kept and the
    image is centred on a zero background. Otherwise the slice is stretched to fill the square.
    """
    h, w = img.shape
    if keep_aspect:
        row_mm, col_mm = pixel_spacing if pixel_spacing else (1.0, 1.0)
        phys_h, phys_w = h * row_mm, w * col_mm
        scale = size / max(phys_h, phys_w)
        new_h = max(1, min(size, round(phys_h * scale)))
        new_w = max(1, min(size, round(phys_w * scale)))
    else:
        new_h = new_w = size
    resized = np.asarray(Image.fromarray(img.astype(np.float32)).resize((new_w, new_h), Image.BILINEAR))
    if (new_h, new_w) == (size, size):
        return resized
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def select_slices(n: int, max_slices: int) -> np.ndarray:
    """Indices of the slices to keep: all if n <= max_slices, else evenly spaced across the series.

    Even spacing keeps the full anatomical extent (medial to lateral, etc.) instead of cropping.
    """
    if max_slices <= 0 or n <= max_slices:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_slices)).astype(int))


def to_storage(vol01: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint8":
        return np.round(vol01 * 255.0).astype(np.uint8)
    if dtype == "float16":
        return vol01.astype(np.float16)
    raise ValueError(f"unsupported dtype {dtype!r}")


def preprocess_volume(vol: np.ndarray, cfg: PreprocessConfig,
                      pixel_spacing: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw (K, H, W) volume -> stored (K', S, S) array plus the kept slice indices."""
    keep = select_slices(vol.shape[0], cfg.max_slices)
    windowed = window_volume(vol, cfg.window)  # window on the full series, then subsample
    out = np.stack([resize_slice(windowed[i], cfg.img_size, cfg.keep_aspect, pixel_spacing) for i in keep])
    return to_storage(np.clip(out, 0.0, 1.0), cfg.dtype), keep
