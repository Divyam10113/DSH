"""Loading train/test CSVs, label matrices and cross-validation folds for the labeled subset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .constants import ID_COL, LABELS, REPORT_COL


def load_split(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read train.csv (required) and test.csv (optional, only used by the leakage guard)."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    missing = [c for c in [ID_COL, REPORT_COL, *LABELS] if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")
    if train[ID_COL].duplicated().any():
        raise ValueError("train.csv has duplicate StudyInstanceUIDs")
    train[REPORT_COL] = train[REPORT_COL].fillna("").astype(str)
    test_path = data_dir / "test.csv"
    test = pd.read_csv(test_path) if test_path.exists() else None
    return train.reset_index(drop=True), test


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_studies, 12) float matrix: 0/1 where a ground-truth label exists, NaN where it doesn't.

    Unlabeled studies may come as blanks, NaN or sentinel values such as -1; anything that is not
    exactly 0 or 1 is treated as unknown.
    """
    Y = df[LABELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    Y[~np.isin(Y, (0.0, 1.0))] = np.nan
    return Y


def labeled_rows(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of studies with at least one ground-truth label."""
    return ~np.isnan(Y).all(axis=1)


def _rarest_positive(Y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Stratification key: index of the rarest positive label in each row (-1 if none)."""
    prevalence = np.nansum(Y, axis=0)
    order = np.argsort(prevalence)  # rarest first
    key = np.full(len(Y), -1)
    for j in order[::-1]:  # most common first, so rarer labels overwrite
        key[Y[:, j] == 1] = j
    values, counts = np.unique(key, return_counts=True)
    return key, dict(zip(values.tolist(), counts.tolist()))


def make_folds(ids: pd.Series, Y: np.ndarray, n_splits: int = 5, seed: int = 42,
               folds_csv: str | Path | None = None) -> np.ndarray:
    """Fold id per study (labeled subset only).

    If `folds_csv` (columns StudyInstanceUID, fold) is given — e.g. the team's shared GroupKFold —
    it is used as-is so our out-of-fold predictions line up with the image model's CV.
    Otherwise folds are stratified on each study's rarest positive finding so every fold sees
    some positives of the rare labels (Fracture, Synovitis, ...).
    """
    if folds_csv is not None:
        folds = pd.read_csv(folds_csv).set_index(ID_COL)["fold"]
        mapped = ids.map(folds)
        if mapped.isna().any():
            raise ValueError(f"{int(mapped.isna().sum())} labeled studies are missing from {folds_csv}")
        return mapped.to_numpy(dtype=int)

    key, counts = _rarest_positive(Y)
    # StratifiedKFold needs >= n_splits members per stratum; fold tiny strata into "no positive".
    for value, count in counts.items():
        if count < n_splits:
            key[key == value] = -1
    if 0 < (key == -1).sum() < n_splits:  # still too small: merge into the largest stratum
        values, counts = np.unique(key[key != -1], return_counts=True)
        key[key == -1] = values[counts.argmax()]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.empty(len(ids), dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(key)), key)):
        fold[val_idx] = k
    return fold
