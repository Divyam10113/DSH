import os

import numpy as np
import pandas as pd
import pytest
import torch

from losses import AsymmetricLoss
from models.knee_model import ABNORMALITIES
from train import TrainConfig, make_synthetic_df, masked_loss, run_training
from validation import ID_COL, make_folds


def _cfg(tmp_path, **kw):
    base = dict(backbone_name="resnet34", embed_dim=64, img_size=32, batch_size=2, num_workers=0,
                save_dir=str(tmp_path / "ck"), leaderboard_csv=str(tmp_path / "exp.csv"), run_name="t")
    base.update(kw)
    return TrainConfig(**base)


def test_masked_loss_ignores_nan_targets():
    crit = AsymmetricLoss(reduction="none")
    logits = torch.zeros(2, 3)
    full = masked_loss(crit, logits, torch.tensor([[1., 0., 1.], [0., 0., 0.]]))
    with_nan = masked_loss(crit, logits, torch.tensor([[1., 0., 1.], [float("nan")] * 3]))
    only_row0 = masked_loss(crit, logits[:1], torch.tensor([[1., 0., 1.]]))
    assert torch.isfinite(with_nan)
    assert with_nan.item() == pytest.approx(only_row0.item())
    assert full.item() != pytest.approx(with_nan.item())


def test_train_writes_artifacts_and_resumes(tmp_path):
    df = make_synthetic_df(12)
    folds = make_folds(df, n_folds=3)
    run_training(df=df, folds=folds, fold="0", cfg=_cfg(tmp_path, epochs=1))

    run_dir = tmp_path / "ck" / "t" / "fold0"
    for name in ("last.pth", "best.pth", "oof.csv", "metrics.csv", "config.json"):
        assert (run_dir / name).exists(), name
    assert torch.load(run_dir / "last.pth", weights_only=False)["epoch"] == 1

    # A new "session" with more epochs continues from epoch 1 instead of restarting
    run_training(df=df, folds=folds, fold="0", cfg=_cfg(tmp_path, epochs=2))
    assert torch.load(run_dir / "last.pth", weights_only=False)["epoch"] == 2
    assert pd.read_csv(run_dir / "metrics.csv")["epoch"].tolist() == [1, 2]

    oof = pd.read_csv(run_dir / "oof.csv")
    assert list(oof.columns) == [ID_COL] + ABNORMALITIES
    val_ids = set(folds.loc[folds.fold == 0, ID_COL])
    assert set(oof[ID_COL]) == val_ids  # OOF rows are exactly the held-out studies


def test_time_budget_stops_cleanly(tmp_path):
    df = make_synthetic_df(8)
    folds = make_folds(df, n_folds=2)
    run_training(df=df, folds=folds, fold="0", cfg=_cfg(tmp_path, epochs=5, max_hours=1e-9))
    state = torch.load(tmp_path / "ck" / "t" / "fold0" / "last.pth", weights_only=False)
    assert state["epoch"] == 1  # one epoch is always run, then it stops before the budget is exceeded


def test_all_folds_writes_pooled_oof(tmp_path):
    df = make_synthetic_df(9)
    folds = make_folds(df, n_folds=3)
    run_training(df=df, folds=folds, fold="all", cfg=_cfg(tmp_path, epochs=1))
    oof = pd.read_csv(tmp_path / "ck" / "t" / "oof.csv")
    assert sorted(oof[ID_COL]) == sorted(df[ID_COL])


def test_sample_weights_scale_loss():
    crit = AsymmetricLoss(reduction="none")
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([[0.0], [0.0]])
    only_first = masked_loss(crit, logits, targets, torch.tensor([1.0, 0.0]))
    assert only_first.item() == pytest.approx(masked_loss(crit, logits[:1], targets[:1]).item())


def test_trains_on_m2_labels_file(tmp_path):
    """M2 labels.csv: soft labels everywhere + __true flags + sample_weight; CV uses gold only."""
    df = make_synthetic_df(12)
    rng = np.random.RandomState(1)
    for c in ABNORMALITIES:
        gold = df[c].notna()
        df[c + "__true"] = gold.astype(int)
        df.loc[~gold, c] = rng.rand((~gold).sum())
    df["sample_weight"] = np.where(df[[c + "__true" for c in ABNORMALITIES]].all(axis=1), 1.0, 0.3)
    folds = make_folds(df, n_folds=3)
    score = run_training(df=df, folds=folds, fold="0", cfg=_cfg(tmp_path, epochs=1))
    assert np.isfinite(score)
