"""
End-to-End Training and Validation Engine for RSNA Knee Abnormality Detection.
Co-owned by Divyam (M3 - Model Architecture) and Pranav (M4 - Integration).
Features:
- Mixed precision training (AMP) on CUDA / MPS / CPU
- Macro-averaged AUC-ROC computation across all 12 findings
- Per-abnormality performance breakdown
- Asymmetric / Focal BCE loss
- Checkpoint saving on best validation Macro-AUC
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional

from models import KneeMultiSeriesModel, ABNORMALITIES
from dataset import KneeMRIDataset, knee_collate_fn
from losses import AsymmetricLoss, MultiLabelFocalLoss


def compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Computes Macro-averaged AUC-ROC and per-class AUC-ROC across all 12 target abnormalities.
    Handles unrepresented classes in validation split gracefully.
    """
    per_class_auc = {}
    valid_aucs = []

    for i, name in enumerate(ABNORMALITIES):
        try:
            # Only compute if at least one positive and one negative sample exist
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                per_class_auc[name] = float(score)
                valid_aucs.append(score)
            else:
                per_class_auc[name] = 0.5
        except Exception:
            per_class_auc[name] = 0.5

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return macro_auc, per_class_auc


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> float:
    model.train()
    total_loss = 0.0

    # Device type for autocast
    device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    # Amp scaler only on cuda/cpu, mps uses autocast directly
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))

    for batch_idx, (series_dict, masks_dict, targets, _) in enumerate(loader):
        # Move tensors to device
        for p in series_dict:
            series_dict[p] = series_dict[p].to(device)
            masks_dict[p] = masks_dict[p].to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if device_type in ("cuda", "cpu"):
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits, _, _ = model(series_dict, masks_dict)
                loss = criterion(logits, targets)
        else:
            logits, _, _ = model(series_dict, masks_dict)
            loss = criterion(logits, targets)

        if device_type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for series_dict, masks_dict, targets, _ in loader:
        for p in series_dict:
            series_dict[p] = series_dict[p].to(device)
            masks_dict[p] = masks_dict[p].to(device)
        targets = targets.to(device)

        logits, probs, _ = model(series_dict, masks_dict)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)

    macro_auc, per_class_auc = compute_macro_auc(y_true, y_pred)
    return avg_loss, macro_auc, per_class_auc


def run_training(
    df: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    backbone_name: str = "resnet34",
    pooling_type: str = "gated",
    fusion_type: str = "concat",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-4,
    save_dir: str = "checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[Training Engine] Running on device: {device}")

    # Generate synthetic dataframe if none provided (allows immediate running & testing)
    if df is None:
        print("[Training Engine] Generating synthetic dataset for verification run...")
        num_samples = 20
        synthetic_data = {"study_id": [f"study_{i:04d}" for i in range(num_samples)]}
        for col in ABNORMALITIES:
            # Random binary labels with realistic medical prevalence (15-30%)
            synthetic_data[col] = (np.random.rand(num_samples) < 0.25).astype(float)
        df = pd.DataFrame(synthetic_data)

    # Split into Train / Val (80 / 20)
    val_size = max(2, int(len(df) * 0.2))
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]

    synthetic_mode = (data_dir is None or not os.path.exists(data_dir))
    train_dataset = KneeMRIDataset(train_df, data_dir=data_dir, is_training=True, synthetic_mode=synthetic_mode)
    val_dataset = KneeMRIDataset(val_df, data_dir=data_dir, is_training=False, synthetic_mode=synthetic_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=knee_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=knee_collate_fn)

    # Instantiate Divyam's Model
    model = KneeMultiSeriesModel(
        backbone_name=backbone_name,
        pretrained=False,
        embed_dim=256,
        pooling_type=pooling_type,
        fusion_type=fusion_type
    ).to(device)

    criterion = AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_macro_auc = 0.0
    best_weights_path = os.path.join(save_dir, f"{backbone_name}_{pooling_type}_best.pth")

    print(f"\n[Training Engine] Starting training for {epochs} epochs...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_macro_auc, per_class = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_macro_auc > best_macro_auc
        if is_best:
            best_macro_auc = val_macro_auc
            torch.save(model.state_dict(), best_weights_path)

        star = "★ BEST" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-AUC: {val_macro_auc:.4f} {star}")

    print("-" * 65)
    print(f"[Training Complete] Best Val Macro-AUC: {best_macro_auc:.4f}")
    print(f"[Checkpoints] Saved to {best_weights_path}")
    return best_macro_auc


if __name__ == "__main__":
    run_training(epochs=2, batch_size=2)
