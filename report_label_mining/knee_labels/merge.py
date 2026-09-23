"""Merge ground-truth and pseudo labels into labels.csv, with a leakage guard.

Merge policy (per study, per label):
  * a ground-truth label always wins over a pseudo label (flagged with `<label>__true` = 1);
  * otherwise the value is the text model's probability (a soft label in [0, 1]).

`sample_weight` lets the image model trust pseudo-labeled studies less:
  1.0 for fully ground-truth studies, otherwise the mean over labels of
  1.0 (true) or pseudo_weight * confidence (pseudo), where confidence = |2p - 1|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ID_COL, LABELS, REPORT_COL, TRUE_FLAG_SUFFIX


class LeakageError(AssertionError):
    pass


def build_labels(ids: pd.Series, Y_true: np.ndarray, P: np.ndarray, pseudo_weight: float = 0.5) -> pd.DataFrame:
    """labels.csv frame: StudyInstanceUID, 12 soft labels, 12 `__true` flags and metadata columns."""
    if Y_true.shape != P.shape or len(ids) != len(P):
        raise ValueError("ids, Y_true and P must describe the same studies")
    known = ~np.isnan(Y_true)
    values = np.where(known, Y_true, np.clip(P, 0.0, 1.0))
    confidence = np.abs(2 * values - 1)
    cell_weight = np.where(known, 1.0, pseudo_weight * confidence)

    out = pd.DataFrame(values, columns=LABELS)
    out.insert(0, ID_COL, ids.to_numpy())
    for j, label in enumerate(LABELS):
        out[label + TRUE_FLAG_SUFFIX] = known[:, j].astype(int)
    n_true = known.sum(axis=1)
    out["label_source"] = np.select([n_true == len(LABELS), n_true == 0], ["true", "pseudo"], "mixed")
    n_pseudo = len(LABELS) - n_true
    pseudo_conf_sum = np.where(known, 0.0, confidence).sum(axis=1)
    out["pseudo_confidence"] = np.where(n_pseudo > 0, pseudo_conf_sum / np.maximum(n_pseudo, 1), np.nan)
    out["sample_weight"] = cell_weight.mean(axis=1)
    return out


def check_no_leakage(labels: pd.DataFrame, train: pd.DataFrame, Y_true: np.ndarray,
                     test_ids: pd.Series | None = None) -> None:
    """Raise LeakageError if labels.csv could leak information or corrupt ground truth.

    Checks: one row per training study and nothing else; no test studies; no report text;
    every value is a finite probability; ground-truth cells are unchanged and flagged.
    """
    problems = []
    ids = labels[ID_COL]
    if ids.duplicated().any():
        problems.append("duplicate StudyInstanceUIDs")
    if set(ids) != set(train[ID_COL]):
        problems.append("study set differs from train.csv")
    if test_ids is not None:
        overlap = set(ids) & set(test_ids)
        if overlap:
            problems.append(f"{len(overlap)} test studies present")
    if REPORT_COL in labels.columns or any("report" in c.lower() for c in labels.columns):
        problems.append("report text column present")
    values = labels[LABELS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        problems.append("label values outside [0, 1] or NaN")

    aligned = labels.set_index(ID_COL).loc[train[ID_COL]]
    known = ~np.isnan(Y_true)
    got = aligned[LABELS].to_numpy(dtype=float)
    flags = aligned[[l + TRUE_FLAG_SUFFIX for l in LABELS]].to_numpy()
    if not np.array_equal(got[known], Y_true[known]):
        problems.append("ground-truth labels were overwritten")
    if not np.array_equal(flags.astype(bool), known):
        problems.append("__true flags do not match ground-truth availability")

    if problems:
        raise LeakageError("labels.csv failed leakage guard: " + "; ".join(problems))


def prevalence_report(Y_true: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    """Compare label prevalence on the ground-truth subset with the pseudo-labeled studies.

    Large gaps are a red flag (e.g. a rule firing on every report), though some shift is expected
    since the organisers note prevalence differs between splits.
    """
    rows = []
    for j, label in enumerate(LABELS):
        known = ~np.isnan(Y_true[:, j])
        pseudo = labels[label].to_numpy()[~known]
        rows.append({
            "label": label,
            "true_n": int(known.sum()),
            "true_prevalence": float(np.nanmean(Y_true[known, j])) if known.any() else np.nan,
            "pseudo_n": int((~known).sum()),
            "pseudo_mean_prob": float(pseudo.mean()) if len(pseudo) else np.nan,
            "pseudo_prevalence@0.5": float((pseudo >= 0.5).mean()) if len(pseudo) else np.nan,
        })
    return pd.DataFrame(rows).set_index("label")
