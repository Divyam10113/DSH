"""
Validation Strategy & Official Metric for RSNA Knee Abnormality Detection.
Owned by Pranav (M4 - Integration, Training Engine & Submission).

Provides:
- Leak-free, multi-label stratified GroupKFold at study level (folds.csv shared by all members)
- Official competition metric: macro-averaged AUC-ROC over the 12 targets
- Per-label error diagnostics (prevalence, bootstrap CI, hardest false negatives / positives)

Unlabeled studies (NaN targets, awaiting M2 pseudo-labels) are distributed evenly across
folds but are ignored by the metric, so CV is always scored on ground-truth labels only.

CLI:
    python validation.py make-folds --train-csv train.csv --out folds.csv
    python validation.py report --oof runs/exp/oof.csv --labels train.csv --out report.md
"""

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models.knee_model import ABNORMALITIES

ID_COL = "StudyInstanceUID"
TRUE_FLAG_SUFFIX = "__true"  # M2 labels.csv: `<label>__true` = 1 where the value is ground truth


def gold_labels(df: pd.DataFrame, label_cols: List[str] = ABNORMALITIES) -> pd.DataFrame:
    """
    Ground-truth-only view [id_col + labels]. For M2's labels.csv, soft pseudo-labels are masked to NaN
    using the `<label>__true` flags, so CV is never scored against pseudo-labels.
    Plain Kaggle train.csv (NaN = unlabeled) passes through unchanged.
    """
    out = df[[ID_COL] + label_cols].copy()
    for c in label_cols:
        flag = c + TRUE_FLAG_SUFFIX
        if flag in df.columns:
            out.loc[df[flag] != 1, c] = np.nan
    return out


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------

def make_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    id_col: str = ID_COL,
    group_col: Optional[str] = None,
    label_cols: List[str] = ABNORMALITIES,
) -> pd.DataFrame:
    """
    Assigns each study to a fold using iterative multi-label stratification (Sechidis et al., 2011)
    over groups. All rows sharing a group (default: the study itself) land in the same fold.

    Stratification targets: the 12 labels (gold positives only) + an `is_labeled` indicator,
    so every fold receives a fair share of rare positives (e.g. Fracture) and of gold-labeled studies.

    Returns a DataFrame [id_col, 'fold'].
    """
    group_col = group_col or id_col
    label_cols = [c for c in label_cols if c in df.columns]
    rng = np.random.RandomState(seed)

    work = df[[id_col, group_col] + label_cols].copy() if group_col != id_col else df[[id_col] + label_cols].copy()
    labels = gold_labels(df, label_cols)[label_cols].astype(float)
    work["is_labeled"] = labels.notna().any(axis=1).astype(float)
    work[label_cols] = (labels.fillna(0) >= 0.5).astype(float)

    strat_cols = label_cols + ["is_labeled"]
    # Group-level label matrix: a group is positive for a label if any of its studies is
    grouped = work.groupby(group_col)
    Y = grouped[strat_cols].max().astype(int)
    sizes = grouped.size().reindex(Y.index).values
    groups = Y.index.values
    Y = Y.values

    n_groups, n_labels = Y.shape
    desired_size = np.full(n_folds, sizes.sum() / n_folds)
    desired_label = np.tile(Y.T.sum(axis=1, keepdims=True) / n_folds, (1, n_folds)).T  # (folds, labels)
    desired_label = desired_label.astype(float)

    assignment = np.full(n_groups, -1)
    remaining = np.ones(n_groups, dtype=bool)

    def _pick_fold(label_idx: Optional[int]) -> int:
        if label_idx is not None:
            scores = desired_label[:, label_idx]
            cands = np.flatnonzero(scores == scores.max())
        else:
            cands = np.arange(n_folds)
        size_scores = desired_size[cands]
        cands = cands[size_scores == size_scores.max()]
        return int(rng.choice(cands))

    while True:
        pos_remaining = (Y[remaining] > 0).sum(axis=0)
        active = np.flatnonzero(pos_remaining > 0)
        if active.size == 0:
            break
        # Rarest label first: its positives are the hardest to spread evenly
        label_idx = active[np.argmin(pos_remaining[active])]
        idx = np.flatnonzero(remaining & (Y[:, label_idx] > 0))
        rng.shuffle(idx)
        for g in idx:
            f = _pick_fold(label_idx)
            assignment[g] = f
            remaining[g] = False
            desired_label[f] -= Y[g]
            desired_size[f] -= sizes[g]

    # Groups with no positive stratification target: balance fold sizes
    idx = np.flatnonzero(remaining)
    rng.shuffle(idx)
    for g in idx:
        f = _pick_fold(None)
        assignment[g] = f
        desired_size[f] -= sizes[g]

    fold_of_group = pd.Series(assignment, index=groups, name="fold")
    out = work[[id_col]].copy()
    out["fold"] = work[group_col].map(fold_of_group).astype(int).values
    return out.reset_index(drop=True)


def check_no_leakage(folds: pd.DataFrame, group_col: str = ID_COL) -> None:
    """Raises AssertionError if any group appears in more than one fold."""
    n_folds_per_group = folds.groupby(group_col)["fold"].nunique()
    leaked = n_folds_per_group[n_folds_per_group > 1]
    assert leaked.empty, f"Leakage: {len(leaked)} groups span multiple folds, e.g. {list(leaked.index[:5])}"


def fold_summary(df: pd.DataFrame, folds: pd.DataFrame, id_col: str = ID_COL,
                 label_cols: List[str] = ABNORMALITIES) -> pd.DataFrame:
    """Per-fold counts of studies, gold-labeled studies and positives per label."""
    label_cols = [c for c in label_cols if c in df.columns]
    merged = gold_labels(df, label_cols).merge(folds, on=id_col)
    rows = []
    for f, part in merged.groupby("fold"):
        row = {"fold": f, "studies": len(part), "labeled": int(part[label_cols].notna().any(axis=1).sum())}
        for c in label_cols:
            row[c] = int((part[c] >= 0.5).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Official metric
# ---------------------------------------------------------------------------

def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ABNORMALITIES,
) -> Tuple[float, Dict[str, float]]:
    """
    Macro-averaged AUC-ROC, matching the Kaggle metric (mean of the 12 per-label AUCs).

    NaN targets (unlabeled studies) are masked out per label. A label whose validation
    subset lacks both classes has no defined AUC: it is reported as NaN and excluded
    from the mean (Kaggle's hidden test set is guaranteed to contain both classes).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_label = {}
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        t = (y_true[mask, i] >= 0.5).astype(int)  # soft pseudo-labels are binarised for scoring
        if t.size == 0 or t.min() == t.max():
            per_label[name] = float("nan")
            continue
        per_label[name] = float(roc_auc_score(t, y_pred[mask, i]))

    valid = [v for v in per_label.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_label


def bootstrap_auc_ci(t: np.ndarray, p: np.ndarray, n_boot: int = 200, seed: int = 0,
                     alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for a single-label AUC."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if t[idx].min() == t[idx].max():
            continue
        scores.append(roc_auc_score(t[idx], p[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    study_ids: List[str],
    label_names: List[str] = ABNORMALITIES,
    top_k: int = 5,
    n_boot: int = 200,
) -> pd.DataFrame:
    """
    Per-label diagnostic table: labeled count, positives, prevalence, AUC with 95% bootstrap CI,
    and the hardest false negatives (lowest-scored positives) / false positives (highest-scored negatives)
    so the team can open those studies and inspect them.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ids = np.asarray(study_ids)
    rows = []
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        t = (y_true[mask, i] >= 0.5).astype(int)
        p = y_pred[mask, i]
        sid = ids[mask]
        row = {
            "label": name,
            "n_labeled": int(mask.sum()),
            "n_pos": int(t.sum()),
            "prevalence": float(t.mean()) if t.size else float("nan"),
            "auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "hardest_fn": "", "hardest_fp": "",
        }
        if t.size and t.min() != t.max():
            row["auc"] = float(roc_auc_score(t, p))
            row["ci_low"], row["ci_high"] = bootstrap_auc_ci(t, p, n_boot=n_boot)
            pos, neg = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
            row["hardest_fn"] = ";".join(sid[pos[np.argsort(p[pos])[:top_k]]])
            row["hardest_fp"] = ";".join(sid[neg[np.argsort(-p[neg])[:top_k]]])
        rows.append(row)
    return pd.DataFrame(rows)


def report_from_oof(oof: pd.DataFrame, labels: pd.DataFrame, id_col: str = ID_COL,
                    label_names: List[str] = ABNORMALITIES) -> Tuple[float, pd.DataFrame]:
    """Joins out-of-fold predictions with ground truth and returns (macro_auc, error table)."""
    merged = oof.merge(gold_labels(labels, label_names), on=id_col, suffixes=("_pred", ""))
    y_pred = merged[[f"{c}_pred" for c in label_names]].values
    y_true = merged[label_names].values
    macro, _ = compute_macro_auc(y_true, y_pred, label_names)
    table = error_analysis(y_true, y_pred, merged[id_col].astype(str).tolist(), label_names)
    return macro, table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-folds", help="Create study-level stratified folds")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--out", default="folds.csv")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", default=None, help="Column to group by (default: StudyInstanceUID)")

    r = sub.add_parser("report", help="Macro-AUC + per-label error analysis for OOF predictions")
    r.add_argument("--oof", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--out", default=None, help="Optional markdown output path")

    args = parser.parse_args()

    if args.cmd == "make-folds":
        df = pd.read_csv(args.train_csv)
        folds = make_folds(df, n_folds=args.n_folds, seed=args.seed, group_col=args.group_col)
        check_no_leakage(folds.merge(df[[ID_COL] + ([args.group_col] if args.group_col else [])], on=ID_COL),
                         group_col=args.group_col or ID_COL)
        folds.to_csv(args.out, index=False)
        print(fold_summary(df, folds).to_string(index=False))
        print(f"\nSaved {len(folds)} studies across {args.n_folds} folds -> {args.out}")
    else:
        macro, table = report_from_oof(pd.read_csv(args.oof), pd.read_csv(args.labels))
        text = f"# OOF Report\n\n**Macro-AUC:** {macro:.4f}\n\n" + table.to_markdown(index=False, floatfmt=".4f")
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)


if __name__ == "__main__":
    _main()
