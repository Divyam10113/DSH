"""
End-to-End Training and Validation Engine for RSNA Knee Abnormality Detection.
Co-owned by Divyam (M3 - Model Architecture) and Pranav (M4 - Integration).
Features:
- Study-level stratified K-fold CV (folds.csv from validation.py) — no study leakage
- Mixed precision training (AMP) on CUDA, gradient accumulation + clipping
- Asymmetric / Focal BCE loss with NaN-masking for unlabeled findings
- Official macro-AUC + per-label AUC every epoch (validation.py)
- Full checkpoint & resume (model, optimizer, scheduler, scaler, epoch, RNG) so a killed
  Kaggle session continues where it stopped; session time budget stops cleanly before the kill
- Out-of-fold predictions (oof.csv) for ensembling, shared experiment tracking

Examples:
    python train.py                                  # synthetic smoke run (no data needed)
    python train.py --train-csv train.csv --folds-csv folds.csv --data-dir /kaggle/input/knee-cache \
                    --fold 0 --epochs 15 --backbone efficientnet_b0 --run-name effb0_gated
    python train.py ... --fold all                   # all folds sequentially, writes combined oof.csv
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import KneeMultiSeriesModel, ABNORMALITIES
from dataset import KneeMRIDataset, knee_collate_fn
from losses import AsymmetricLoss, MultiLabelFocalLoss
from tracking import ExperimentTracker
from validation import ID_COL, check_no_leakage, compute_macro_auc, gold_labels, make_folds

__all__ = ["TrainConfig", "build_model", "compute_macro_auc", "run_training", "run_fold"]


@dataclass
class TrainConfig:
    # Model (M3)
    backbone_name: str = "resnet34"
    pooling_type: str = "gated"
    fusion_type: str = "concat"
    embed_dim: int = 256
    dropout: float = 0.2
    pretrained: bool = False
    img_size: int = 128
    max_slices: int = 32            # must match the M1 cache build; reused by the offline inference
    # Optimisation
    loss: str = "asl"               # 'asl' or 'focal'
    epochs: int = 5
    batch_size: int = 4
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    use_amp: bool = True
    num_workers: int = 2
    seed: int = 42
    # Data / CV
    fold: int = 0
    n_folds: int = 5
    # Session management
    save_dir: str = "checkpoints"
    run_name: str = ""
    resume: bool = True
    resume_from: str = ""           # e.g. a last.pth attached from a previous Kaggle session's output
    max_hours: float = 0.0          # 0 = unlimited; stop cleanly once the next epoch would exceed this
    leaderboard_csv: str = "experiments.csv"
    use_wandb: bool = False

    def default_run_name(self) -> str:
        return self.run_name or f"{self.backbone_name}_{self.pooling_type}_{self.fusion_type}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: TrainConfig, pretrained: Optional[bool] = None) -> KneeMultiSeriesModel:
    """Single source of truth for model construction; inference.py rebuilds from the checkpoint config."""
    return KneeMultiSeriesModel(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained if pretrained is None else pretrained,
        embed_dim=cfg.embed_dim,
        pooling_type=cfg.pooling_type,
        fusion_type=cfg.fusion_type,
        num_targets=len(ABNORMALITIES),
        dropout=cfg.dropout,
    )


def build_criterion(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.loss == "focal":
        return MultiLabelFocalLoss(reduction="none")
    return AsymmetricLoss(reduction="none")


def masked_loss(criterion: torch.nn.Module, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Element-wise loss averaged over labeled entries only (NaN target = finding not labeled).
    `sample_weights` (B,) down-weights pseudo-labeled studies (M2 labels.csv `sample_weight`).
    """
    mask = (~torch.isnan(targets)).float()
    if sample_weights is not None:
        mask = mask * sample_weights[:, None]
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    loss = criterion(logits.float(), safe_targets)
    return (loss * mask).sum() / mask.sum().clamp(min=1e-6)


def _to_device(series_dict, masks_dict, targets, device):
    series_dict = {p: t.to(device, non_blocking=True) for p, t in series_dict.items()}
    masks_dict = {p: m.to(device, non_blocking=True) for p, m in masks_dict.items()}
    return series_dict, masks_dict, targets.to(device, non_blocking=True)


def config_from_dict(d: Dict) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in known})


def make_synthetic_df(num_samples: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic studies with realistic prevalence and ~30% unlabeled rows, for smoke tests."""
    rng = np.random.RandomState(seed)
    data = {ID_COL: [f"study_{i:04d}" for i in range(num_samples)]}
    for col in ABNORMALITIES:
        data[col] = (rng.rand(num_samples) < 0.25).astype(float)
    df = pd.DataFrame(data)
    unlabeled = rng.rand(num_samples) < 0.3
    df.loc[unlabeled, ABNORMALITIES] = np.nan
    return df


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    sample_weights: Optional[Dict[str, float]] = None,
) -> float:
    model.train()
    total_loss = 0.0
    amp_enabled = use_amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    for step, (series_dict, masks_dict, targets, study_ids) in enumerate(loader, start=1):
        series_dict, masks_dict, targets = _to_device(series_dict, masks_dict, targets, device)
        weights = None
        if sample_weights:
            weights = torch.tensor([sample_weights.get(s, 1.0) for s in study_ids], dtype=torch.float32, device=device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits, _, _ = model(series_dict, masks_dict)
        loss = masked_loss(criterion, logits, targets, weights)

        scaler.scale(loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, Dict[str, float], np.ndarray, List[str]]:
    """Returns (avg_loss, macro_auc, per_label_auc, probs (N, 12), study_ids)."""
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    all_targets, all_probs, all_ids = [], [], []

    for series_dict, masks_dict, targets, study_ids in loader:
        series_dict, masks_dict, targets = _to_device(series_dict, masks_dict, targets, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits, _, _ = model(series_dict, masks_dict)
        total_loss += masked_loss(criterion, logits, targets).item()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_ids.extend(study_ids)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)
    macro_auc, per_label = compute_macro_auc(y_true, y_pred)
    return total_loss / max(1, len(loader)), macro_auc, per_label, y_pred, all_ids


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch: int, best_score: float,
                    cfg: TrainConfig) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "config": asdict(cfg),
        "labels": ABNORMALITIES,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-write never corrupts the previous checkpoint


def save_weights(path: str, model, cfg: TrainConfig, score: float, epoch: int) -> None:
    """Inference-only artefact (small): weights + the config needed to rebuild the model offline."""
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "labels": ABNORMALITIES,
                "score": score, "epoch": epoch}, path)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler) -> Tuple[int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    rng = state.get("rng") or {}
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except (KeyError, TypeError, RuntimeError):
        pass
    return int(state["epoch"]), float(state["best_score"])


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def run_fold(df: pd.DataFrame, folds: pd.DataFrame, cfg: TrainConfig, data_dir: Optional[str] = None) -> Dict:
    """Trains one fold with resume support. Returns a summary dict (best score, paths, OOF frame)."""
    t_start = time.time()
    seed_everything(cfg.seed + cfg.fold)
    device = get_device()
    run_dir = os.path.join(cfg.save_dir, cfg.default_run_name(), f"fold{cfg.fold}")
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pth")
    best_path = os.path.join(run_dir, "best.pth")
    print(f"[Training Engine] {cfg.default_run_name()} fold {cfg.fold} on {device} -> {run_dir}")

    merged = df.merge(folds[[ID_COL, "fold"]], on=ID_COL, how="inner")
    train_df = merged[merged["fold"] != cfg.fold].reset_index(drop=True)
    val_df = merged[merged["fold"] == cfg.fold].reset_index(drop=True)
    assert not set(train_df[ID_COL]) & set(val_df[ID_COL]), "Study leakage between train and val"
    # CV is scored on ground truth only (M2 pseudo-labels are masked out via `<label>__true` flags)
    val_gold = gold_labels(val_df).assign(**{ID_COL: val_df[ID_COL].astype(str)}).set_index(ID_COL)
    weight_map = (dict(zip(train_df[ID_COL].astype(str), train_df["sample_weight"].astype(float)))
                  if "sample_weight" in train_df.columns else None)
    print(f"[Training Engine] train={len(train_df)} val={len(val_df)} "
          f"(val gold-labeled={int(val_gold.notna().any(axis=1).sum())}, sample weights={'on' if weight_map else 'off'})")

    synthetic = data_dir is None or not os.path.exists(data_dir)
    size = (cfg.img_size, cfg.img_size)
    train_ds = KneeMRIDataset(train_df, data_dir=data_dir, is_training=True, img_size=size, synthetic_mode=synthetic)
    val_ds = KneeMRIDataset(val_df, data_dir=data_dir, is_training=False, img_size=size, synthetic_mode=synthetic)
    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(cfg.seed + cfg.fold)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=knee_collate_fn,
                              num_workers=cfg.num_workers, pin_memory=pin, drop_last=len(train_ds) > cfg.batch_size,
                              generator=g, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=knee_collate_fn,
                            num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    start_epoch, best_score = 0, -1.0
    resume_path = cfg.resume_from or (last_path if cfg.resume and os.path.exists(last_path) else "")
    if resume_path:
        start_epoch, best_score = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        print(f"[Training Engine] Resumed from {resume_path} at epoch {start_epoch} (best={best_score:.4f})")

    tracker = ExperimentTracker(run_dir, asdict(cfg), leaderboard_csv=cfg.leaderboard_csv,
                                use_wandb=cfg.use_wandb, run_name=f"{cfg.default_run_name()}/fold{cfg.fold}")

    epoch_times: List[float] = []
    stopped_early = False
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        if cfg.max_hours > 0 and epoch_times:
            projected = (time.time() - t_start) + max(epoch_times)
            if projected > cfg.max_hours * 3600:
                print(f"[Training Engine] Time budget {cfg.max_hours}h reached; stopping before epoch {epoch}. "
                      f"Re-run with the same args (or --resume-from {last_path}) to continue.")
                stopped_early = True
                break

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                     cfg.use_amp, cfg.grad_accum_steps, cfg.max_grad_norm, weight_map)
        val_loss, _, _, probs, ids = evaluate(model, val_loader, criterion, device, cfg.use_amp)
        val_auc, per_label = compute_macro_auc(val_gold.loc[ids, ABNORMALITIES].values, probs)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_times.append(time.time() - t0)

        score = val_auc if not np.isnan(val_auc) else -val_loss  # fallback if val has no gold labels
        is_best = score > best_score
        if is_best:
            best_score = score
            save_weights(best_path, model, cfg, best_score, epoch)
            oof = pd.DataFrame(probs, columns=ABNORMALITIES)
            oof.insert(0, ID_COL, ids)
            oof.to_csv(os.path.join(run_dir, "oof.csv"), index=False)
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_score, cfg)

        tracker.log_epoch({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "val_macro_auc": val_auc, "lr": lr, "epoch_sec": epoch_times[-1],
                           **{f"auc_{k}": v for k, v in per_label.items()}})
        star = "★ BEST" if is_best else ""
        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} [{epoch_times[-1]:.1f}s] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-AUC: {val_auc:.4f} {star}")

    finished = not stopped_early
    oof_path = os.path.join(run_dir, "oof.csv")
    summary = {"fold": cfg.fold, "best_val_macro_auc": best_score, "finished": finished,
               "backbone": cfg.backbone_name, "pooling": cfg.pooling_type, "fusion": cfg.fusion_type,
               "best_path": best_path, "hours": (time.time() - t_start) / 3600}
    if finished and epoch_times:  # skip if this session only re-opened an already finished fold
        tracker.log_summary(summary)
    summary["oof"] = pd.read_csv(oof_path) if os.path.exists(oof_path) else None
    print(f"[Training Engine] Fold {cfg.fold} done={finished} best={best_score:.4f} weights={best_path}")
    return summary


def run_training(
    df: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    folds: Optional[pd.DataFrame] = None,
    fold: str = "0",
    cfg: Optional[TrainConfig] = None,
    **overrides,
) -> float:
    """
    Trains one fold (fold='0'..'k-1') or every fold (fold='all', also writes combined oof.csv).
    Backwards compatible with the M3 ablation runner: run_training(backbone_name=..., epochs=...).
    Returns the best validation macro-AUC (mean across folds for 'all').
    """
    cfg = cfg or TrainConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)

    if df is None:
        print("[Training Engine] Generating synthetic dataset for verification run...")
        df = make_synthetic_df()
    if folds is None:
        folds = make_folds(df, n_folds=cfg.n_folds, seed=cfg.seed)
    check_no_leakage(folds)
    cfg.n_folds = int(folds["fold"].nunique())

    fold_ids = list(range(cfg.n_folds)) if str(fold) == "all" else [int(fold)]
    scores, oofs = [], []
    for f in fold_ids:
        cfg.fold = f
        summary = run_fold(df, folds, cfg, data_dir)
        scores.append(summary["best_val_macro_auc"])
        if summary["oof"] is not None:
            oofs.append(summary["oof"])

    if str(fold) == "all" and oofs:
        oof_all = pd.concat(oofs, ignore_index=True)
        out = os.path.join(cfg.save_dir, cfg.default_run_name(), "oof.csv")
        oof_all.to_csv(out, index=False)
        gold = gold_labels(df).assign(**{ID_COL: df[ID_COL].astype(str)}).set_index(ID_COL)
        cv_auc, _ = compute_macro_auc(gold.loc[oof_all[ID_COL].astype(str), ABNORMALITIES].values,
                                      oof_all[ABNORMALITIES].values)
        print(f"[Training Complete] Pooled OOF Macro-AUC: {cv_auc:.4f} -> {out}")
    return float(np.mean(scores))


def _parse_args() -> Tuple[TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", default=None, help="Labels CSV (StudyInstanceUID + 12 labels); synthetic if omitted")
    parser.add_argument("--folds-csv", default=None, help="folds.csv from validation.py; created on the fly if omitted")
    parser.add_argument("--data-dir", default=None, help="Cached tensors root: {data_dir}/{StudyInstanceUID}/{plane}.npy")
    parser.add_argument("--fold", default="0", help="Fold index or 'all'")
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        if f.name == "fold":
            continue
        val = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(val, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=val)
        else:
            parser.add_argument(flag, type=type(val), default=val)
    args = parser.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig) if f.name != "fold"})
    return cfg, args


if __name__ == "__main__":
    cfg, args = _parse_args()
    df = pd.read_csv(args.train_csv) if args.train_csv else None
    folds = pd.read_csv(args.folds_csv) if args.folds_csv else None
    run_training(df=df, data_dir=args.data_dir, folds=folds, fold=args.fold, cfg=cfg)
