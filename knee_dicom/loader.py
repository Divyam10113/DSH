"""Reading the cache, plus the 2.5D-slab and MIP data variants (Phase 4).

Teammates never touch raw DICOM: `load_study` returns float32 arrays in [0, 1] straight from the
cache. The cache stores uint8 (0..255) by default, so anyone reading the .npy files directly (e.g.
M3's KneeMRIDataset) should divide uint8 arrays by 255 — `to_unit_float` does exactly that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import PLANES


def to_unit_float(arr: np.ndarray) -> np.ndarray:
    """uint8 0..255 -> float32 0..1; float arrays are returned as float32 unchanged."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def load_plane(cache_dir: str | Path, study_id: str, plane: str, mmap: bool = False) -> np.ndarray | None:
    """(K, H, W) float32 in [0, 1] for one plane of one study, or None if that plane is missing."""
    path = Path(cache_dir) / str(study_id) / f"{plane}.npy"
    if not path.exists():
        return None
    return to_unit_float(np.load(path, mmap_mode="r" if mmap else None))


def load_study(cache_dir: str | Path, study_id: str, planes=PLANES) -> dict[str, np.ndarray]:
    """{plane: (K, H, W) float32} for the planes this study has."""
    out = {}
    for plane in planes:
        arr = load_plane(cache_dir, study_id, plane)
        if arr is not None:
            out[plane] = arr
    return out


def read_manifest(cache_dir: str | Path, require_planes=()) -> pd.DataFrame:
    """manifest.csv, optionally filtered to studies that have all of `require_planes`."""
    manifest = pd.read_csv(Path(cache_dir) / "manifest.csv")
    for plane in require_planes:
        manifest = manifest[manifest[f"has_{plane}"] == 1]
    return manifest.reset_index(drop=True)


# -- data variants ----------------------------------------------------------------------------------

def make_slabs(vol: np.ndarray) -> np.ndarray:
    """2.5D representation: (K, H, W) -> (K, 3, H, W) with each slice's neighbours as channels.

    Edge slices repeat themselves. Lets a 2D backbone (3 input channels, ImageNet weights) see a
    little through-plane context, useful for thin structures like the ACL and meniscal tears.
    """
    padded = np.concatenate([vol[:1], vol, vol[-1:]], axis=0)
    return np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=1)


def sliding_mip(vol: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Maximum intensity projection over a sliding window of `thickness` slices: (K, H, W) -> (K, H, W).

    Bright structures (fluid, oedema, cysts) that span several thin slices become more conspicuous.
    """
    if thickness <= 1:
        return vol.copy()
    half = thickness // 2
    padded = np.concatenate([np.repeat(vol[:1], half, 0), vol, np.repeat(vol[-1:], thickness - 1 - half, 0)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, thickness, axis=0)
    return windows.max(axis=-1)
