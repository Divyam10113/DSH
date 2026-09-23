"""
Cross-Fold / Cross-Model Ensembling for RSNA Knee Abnormality Detection.
Owned by Pranav (M4 - Integration, Training Engine & Submission).

Macro-AUC only cares about ranking, so predictions are blended as per-label rank averages
(robust to models with different calibration). Weights are chosen by greedy forward selection
with replacement (Caruana et al., 2004) on out-of-fold predictions, scored on gold labels only.

CLI:
    # fit weights on OOF predictions (one oof.csv per model, from train.py --fold all)
    python ensemble.py fit --oof effb0=runs/effb0/oof.csv convnext=runs/convnext/oof.csv \
                           --labels train.csv --out ensemble.json
    # apply weights to test predictions of the same models
    python ensemble.py apply --pred effb0=preds_effb0.csv convnext=preds_convnext.csv \
                             --weights ensemble.json --out submission.csv
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.knee_model import ABNORMALITIES
from validation import ID_COL, compute_macro_auc, gold_labels


def rank_normalize(df: pd.DataFrame, label_cols: List[str] = ABNORMALITIES) -> pd.DataFrame:
    """Per-label ranks scaled to (0, 1]; ties share the average rank."""
    out = df.copy()
    out[label_cols] = df[label_cols].rank(method="average", pct=True)
    return out


def align(frames: Dict[str, pd.DataFrame], id_col: str = ID_COL) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Restricts all prediction frames to their common studies, in a fixed order."""
    common = None
    for df in frames.values():
        ids = set(df[id_col].astype(str))
        common = ids if common is None else common & ids
    order = sorted(common)
    for name, df in frames.items():
        if len(df) != len(order):
            print(f"[Ensemble] {name}: using {len(order)}/{len(df)} studies common to all models")
    arrays = {name: df.assign(**{id_col: df[id_col].astype(str)}).set_index(id_col).loc[order, ABNORMALITIES].values
              for name, df in frames.items()}
    return order, arrays


def blend(frames: Dict[str, pd.DataFrame], weights: Dict[str, float], method: str = "rank",
          id_col: str = ID_COL) -> pd.DataFrame:
    """Weighted blend of prediction frames -> one frame [id_col] + 12 labels in [0, 1]."""
    frames = {n: f for n, f in frames.items() if weights.get(n, 0) > 0}
    if method == "rank":
        frames = {n: rank_normalize(f) for n, f in frames.items()}
    order, arrays = align(frames, id_col)
    total = sum(weights[n] for n in arrays)
    blended = sum(weights[n] * arrays[n] for n in arrays) / total
    out = pd.DataFrame(blended, columns=ABNORMALITIES)
    out.insert(0, id_col, order)
    return out


def fit_weights(oofs: Dict[str, pd.DataFrame], labels: pd.DataFrame, method: str = "rank",
                n_iter: int = 30, id_col: str = ID_COL) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Greedy forward selection with replacement on OOF macro-AUC.
    Returns (normalised weights, table of single-model vs ensemble scores).
    """
    if method == "rank":
        oofs = {n: rank_normalize(f) for n, f in oofs.items()}
    order, arrays = align(oofs, id_col)
    lab = gold_labels(labels)  # score on ground truth only, never on M2 pseudo-labels
    lab = lab.assign(**{id_col: lab[id_col].astype(str)}).set_index(id_col)
    missing = set(order) - set(lab.index)
    if missing:
        raise KeyError(f"{len(missing)} OOF studies missing from labels, e.g. {list(missing)[:3]}")
    y_true = lab.loc[order, ABNORMALITIES].values.astype(float)

    single = {n: compute_macro_auc(y_true, a)[0] for n, a in arrays.items()}
    counts = {n: 0 for n in arrays}
    best_name = max(single, key=single.get)
    counts[best_name] = 1
    current = arrays[best_name].copy()
    best_score = single[best_name]

    for _ in range(n_iter):
        k = sum(counts.values())
        trial = {n: compute_macro_auc(y_true, (current * k + a) / (k + 1))[0] for n, a in arrays.items()}
        name = max(trial, key=trial.get)
        if trial[name] <= best_score + 1e-6:
            break
        counts[name] += 1
        current = (current * k + arrays[name]) / (k + 1)
        best_score = trial[name]

    total = sum(counts.values())
    weights = {n: c / total for n, c in counts.items()}
    rows = [{"model": n, "oof_macro_auc": s, "weight": weights[n]} for n, s in single.items()]
    rows.append({"model": "ENSEMBLE", "oof_macro_auc": best_score, "weight": 1.0})
    return weights, pd.DataFrame(rows)


def _parse_named(items: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = item, item
        out[name] = pd.read_csv(path)
    return out


def _main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--oof", nargs="+", required=True, help="name=path/to/oof.csv")
    f.add_argument("--labels", required=True)
    f.add_argument("--method", choices=["rank", "mean"], default="rank")
    f.add_argument("--n-iter", type=int, default=30)
    f.add_argument("--out", default="ensemble.json")

    a = sub.add_parser("apply")
    a.add_argument("--pred", nargs="+", required=True, help="name=path/to/test_preds.csv")
    a.add_argument("--weights", required=True)
    a.add_argument("--out", default="submission.csv")

    args = parser.parse_args(argv)
    if args.cmd == "fit":
        weights, table = fit_weights(_parse_named(args.oof), pd.read_csv(args.labels), args.method, args.n_iter)
        print(table.to_string(index=False, float_format="%.4f"))
        with open(args.out, "w") as fh:
            json.dump({"method": args.method, "weights": weights}, fh, indent=2)
        print(f"Saved -> {args.out}")
    else:
        with open(args.weights) as fh:
            spec = json.load(fh)
        out = blend(_parse_named(args.pred), spec["weights"], spec.get("method", "rank"))
        out.to_csv(args.out, index=False)
        print(f"Saved {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    _main()
