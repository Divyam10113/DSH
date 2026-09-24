"""Metrics for comparing label sources against the ground-truth labeled subset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .constants import LABELS


def per_label_metrics(Y_true: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """AUC, average precision and F1 per label, using only cells with a ground-truth label.

    `f1@0.5` is the F1 at a fixed 0.5 threshold; `best_f1` is the best achievable F1 over
    thresholds (optimistic, but useful to compare ranking quality of sources).
    """
    rows = []
    for j, label in enumerate(LABELS):
        known = ~np.isnan(Y_true[:, j])
        y, p = Y_true[known, j], P[known, j]
        row = {"label": label, "n": int(known.sum()), "positives": int(y.sum()),
               "auc": np.nan, "ap": np.nan, "f1@0.5": np.nan, "best_f1": np.nan}
        if known.any():
            row["f1@0.5"] = f1_score(y, p >= 0.5, zero_division=0)
        if len(np.unique(y)) == 2:
            row["auc"] = roc_auc_score(y, p)
            row["ap"] = average_precision_score(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
            row["best_f1"] = float(f1.max())
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def macro(metrics: pd.DataFrame) -> dict[str, float]:
    return {col: float(metrics[col].mean(skipna=True)) for col in ["auc", "ap", "f1@0.5", "best_f1"]}


def compare_sources(Y_true: np.ndarray, sources: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-label AUC/F1 for every source side by side, plus a macro summary table."""
    per_label, summary = {}, {}
    for name, P in sources.items():
        m = per_label_metrics(Y_true, P)
        per_label[name] = m[["auc", "f1@0.5", "best_f1"]]
        summary[name] = macro(m)
    per_label_df = pd.concat(per_label, axis=1)
    return per_label_df, pd.DataFrame(summary).T
