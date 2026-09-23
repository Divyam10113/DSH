"""
Offline Submission Harness for RSNA Knee Abnormality Detection (Kaggle Code Competition).
Owned by Pranav (M4 - Integration, Training Engine & Submission).

Guarantees for the hidden-test run (<= 9 h, internet OFF):
- Models are rebuilt from checkpoint configs with pretrained=False (no weight downloads)
- DICOM decoding runs in parallel DataLoader workers; a corrupt study never crashes the run
- Wall-clock budget: TTA is dropped if the projected runtime gets tight, and once the budget
  is spent the remaining studies get fallback scores — a valid submission.csv is ALWAYS written
- Output has exactly one row per test study, columns in the official order

Usage (see kaggle/submission_notebook.py):
    python inference.py --data-root /kaggle/input/rsna-knee-abnormality-detection \
        --models effb0="/kaggle/input/knee-weights-effb0/*/best.pth" \
        --ensemble /kaggle/input/knee-weights-effb0/ensemble.json --out submission.csv
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dicom_io import load_cached_study, load_study
from ensemble import blend
from models.knee_model import ABNORMALITIES
from train import build_model, config_from_dict
from validation import ID_COL

SUBMISSION_COLUMNS = [ID_COL] + ABNORMALITIES


class TestStudyDataset(Dataset):
    """Decodes one test study per item (in a worker process) -> (study_id, {plane: (K,1,H,W)})."""

    def __init__(self, study_ids: List[str], series_df: Optional[pd.DataFrame], series_root: str,
                 img_size: int, cache_dir: Optional[str] = None, max_slices: int = 32):
        self.study_ids = study_ids
        self.series_groups = ({str(k): g.reset_index(drop=True) for k, g in series_df.groupby(ID_COL)}
                              if series_df is not None else {})
        self.series_root = series_root
        self.img_size = img_size
        self.cache_dir = cache_dir
        self.max_slices = max_slices

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        try:
            if self.cache_dir:
                series = load_cached_study(self.cache_dir, sid, img_size=self.img_size)
            else:
                rows = self.series_groups.get(sid)
                rows = rows if rows is not None else pd.DataFrame()  # knee_dicom also scans the folder
                series = load_study(os.path.join(self.series_root, sid), rows, self.img_size, self.max_slices)
        except Exception as e:
            print(f"[inference] failed to load {sid}: {type(e).__name__}: {e}")
            series = {}
        return sid, series


def load_models(model_globs: Dict[str, str], device: torch.device):
    """
    name -> list of (model, img_size, max_slices); each glob usually matches the K fold checkpoints of one run.
    """
    groups = {}
    for name, pattern in model_globs.items():
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No checkpoints match {pattern} for model '{name}'")
        loaded = []
        for p in paths:
            state = torch.load(p, map_location="cpu", weights_only=False)
            if state.get("labels", ABNORMALITIES) != ABNORMALITIES:
                raise ValueError(f"{p} was trained with a different label order: {state.get('labels')}")
            cfg = config_from_dict(state["config"])
            model = build_model(cfg, pretrained=False)
            model.load_state_dict(state["model"])
            model.to(device).eval()
            if device.type == "cuda":
                model.half()
            loaded.append((model, (cfg.img_size, cfg.img_size), cfg.max_slices))
            print(f"[inference] loaded {name}: {p} (backbone={cfg.backbone_name}, img={cfg.img_size})")
        groups[name] = loaded
    return groups


@torch.no_grad()
def predict_study(models: List[Tuple[torch.nn.Module, Tuple[int, int], int]], series: Dict[str, torch.Tensor],
                  device: torch.device, tta: bool) -> np.ndarray:
    """Mean sigmoid over fold models (and horizontal-flip TTA) for one study -> (12,)."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    preds = []
    for model, size, _ in models:
        batch = {}
        for plane, t in series.items():
            if tuple(t.shape[-2:]) != size:
                t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
            batch[plane] = t[None].to(device, dtype)  # (1, K, 1, H, W)
        views = [batch] + ([{p: torch.flip(t, dims=[-1]) for p, t in batch.items()}] if tta else [])
        for view in views:
            logits, _, _ = model(view, None)
            preds.append(torch.sigmoid(logits.float())[0].cpu().numpy())
    return np.mean(preds, axis=0)


def run_inference(
    data_root: str,
    model_globs: Dict[str, str],
    out_path: str = "submission.csv",
    ensemble_json: Optional[str] = None,
    split: str = "test",
    cache_dir: Optional[str] = None,
    time_budget_hours: float = 8.0,
    tta: bool = True,
    num_workers: int = 4,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    t_start = time.time()
    budget = time_budget_hours * 3600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device} budget={time_budget_hours}h tta={tta}")

    test_df = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
    study_ids = test_df[ID_COL].astype(str).tolist()[:limit] if limit else test_df[ID_COL].astype(str).tolist()
    series_csv = os.path.join(data_root, f"{split}_series.csv")
    series_df = pd.read_csv(series_csv, dtype={ID_COL: str, "SeriesInstanceUID": str}) if os.path.exists(series_csv) else None

    # Write a valid all-0.5 file first: even a crash later leaves a scoreable submission behind
    fallback = pd.DataFrame({ID_COL: test_df[ID_COL].astype(str)})
    for c in ABNORMALITIES:
        fallback[c] = 0.5
    fallback.to_csv(out_path, index=False)

    groups = load_models(model_globs, device)
    # Decode once at the largest resolution/slice count any model uses; predict_study resizes per model
    img_size = max(s[0] for g in groups.values() for _, s, _ in g)
    max_slices = max(k for g in groups.values() for _, _, k in g)
    ds = TestStudyDataset(study_ids, series_df, os.path.join(data_root, f"{split}_series"), img_size,
                          cache_dir, max_slices)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers)

    preds = {name: {} for name in groups}
    n_failed, n_skipped = 0, 0
    loop_start = time.time()
    for i, (sid, series) in enumerate(loader):
        elapsed = time.time() - t_start
        if elapsed > budget:
            n_skipped = len(study_ids) - i
            print(f"[inference] time budget exhausted; {n_skipped} studies get fallback scores")
            break
        if tta and i >= 10:
            per_study = (time.time() - loop_start) / i
            if elapsed + per_study * (len(study_ids) - i) > 0.9 * budget:
                tta = False
                print(f"[inference] projected runtime too long at study {i}; disabling TTA")
        if not series:
            n_failed += 1
            continue
        try:
            for name, models in groups.items():
                preds[name][sid] = predict_study(models, series, device, tta)
        except Exception as e:
            n_failed += 1
            print(f"[inference] prediction failed for {sid}: {type(e).__name__}: {e}")
            for name in groups:
                preds[name].pop(sid, None)
        if (i + 1) % 100 == 0:
            print(f"[inference] {i + 1}/{len(study_ids)} studies, {(time.time() - t_start) / 60:.1f} min")

    # Blend model groups (rank-average with fitted weights, else equal-weight mean)
    frames = {}
    for name, d in preds.items():
        if d:
            f = pd.DataFrame.from_dict(d, orient="index", columns=ABNORMALITIES)
            frames[name] = f.rename_axis(ID_COL).reset_index()
    if frames:
        if ensemble_json and os.path.exists(ensemble_json):
            with open(ensemble_json) as fh:
                spec = json.load(fh)
            weights = {n: spec["weights"].get(n, 0.0) for n in frames}
            method = spec.get("method", "rank")
            if sum(weights.values()) == 0:
                weights, method = {n: 1.0 for n in frames}, "mean"
        else:
            weights, method = {n: 1.0 for n in frames}, "mean"
        blended = blend(frames, weights, method=method) if len(frames) > 1 or method == "mean" else frames[next(iter(frames))]
    else:
        blended = pd.DataFrame(columns=SUBMISSION_COLUMNS)

    # Failed / skipped studies: column mean of successful predictions (neutral rank)
    sub = fallback[[ID_COL]].merge(blended, on=ID_COL, how="left")
    for c in ABNORMALITIES:
        fill = float(blended[c].mean()) if len(blended) else 0.5
        sub[c] = sub[c].fillna(fill).clip(0.0, 1.0)
    sub = sub[SUBMISSION_COLUMNS]
    validate_submission(sub, fallback[ID_COL].tolist())
    sub.to_csv(out_path, index=False)
    print(f"[inference] wrote {out_path}: {len(sub)} rows, failed={n_failed}, skipped={n_skipped}, "
          f"total {(time.time() - t_start) / 60:.1f} min")
    return sub


def validate_submission(sub: pd.DataFrame, expected_ids: List[str]) -> None:
    assert list(sub.columns) == SUBMISSION_COLUMNS, f"Bad columns: {list(sub.columns)}"
    assert len(sub) == len(expected_ids), f"Expected {len(expected_ids)} rows, got {len(sub)}"
    assert sub[ID_COL].tolist() == list(expected_ids), "Study IDs/order differ from test.csv"
    vals = sub[ABNORMALITIES].values
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all(), "Scores must be finite in [0, 1]"


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Competition data dir containing test.csv, test_series/")
    parser.add_argument("--models", nargs="+", required=True, help='name="glob/to/*/best.pth"')
    parser.add_argument("--ensemble", default=None, help="ensemble.json from ensemble.py fit")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--split", default="test", help="'test' for submission, 'train' for local dry runs")
    parser.add_argument("--cache-dir", default=None, help="Use cached .npy tensors instead of raw DICOM")
    parser.add_argument("--time-budget-hours", type=float, default=8.0)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only predict the first N studies (profiling)")
    args = parser.parse_args()

    model_globs = dict(m.split("=", 1) if "=" in m else (f"model{i}", m) for i, m in enumerate(args.models))
    run_inference(args.data_root, model_globs, args.out, args.ensemble, args.split, args.cache_dir,
                  args.time_budget_hours, not args.no_tta, args.num_workers, args.limit)


if __name__ == "__main__":
    _main()
