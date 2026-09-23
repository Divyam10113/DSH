import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from models.knee_model import ABNORMALITIES
from validation import ID_COL, check_no_leakage, compute_macro_auc, error_analysis, fold_summary, make_folds


def _labels(n=300, seed=0, unlabeled_frac=0.4):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({ID_COL: [f"1.2.840.{i}" for i in range(n)]})
    prevalence = np.linspace(0.03, 0.4, len(ABNORMALITIES))  # includes a rare, Fracture-like label
    for c, p in zip(ABNORMALITIES, prevalence):
        df[c] = (rng.rand(n) < p).astype(float)
    df.loc[rng.rand(n) < unlabeled_frac, ABNORMALITIES] = np.nan
    return df


def test_folds_cover_every_study_once():
    df = _labels()
    folds = make_folds(df, n_folds=5, seed=1)
    assert len(folds) == len(df)
    assert set(folds[ID_COL]) == set(df[ID_COL])
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    check_no_leakage(folds)


def test_folds_deterministic():
    df = _labels()
    pd.testing.assert_frame_equal(make_folds(df, seed=7), make_folds(df, seed=7))


def test_group_column_keeps_groups_together():
    df = _labels(200)
    df["patient"] = [f"p{i // 3}" for i in range(len(df))]  # 3 studies per patient
    folds = make_folds(df, n_folds=4, group_col="patient").merge(df[[ID_COL, "patient"]], on=ID_COL)
    check_no_leakage(folds, group_col="patient")


def test_leakage_detected():
    bad = pd.DataFrame({ID_COL: ["a", "a", "b"], "fold": [0, 1, 0]})
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def test_rare_positives_and_gold_labels_spread_evenly():
    df = _labels(500)
    summary = fold_summary(df, make_folds(df, n_folds=5))
    for col in [ABNORMALITIES[0], "labeled"]:  # rarest label and gold-label count
        counts = summary[col].values
        assert counts.max() - counts.min() <= max(2, 0.15 * counts.mean()), (col, counts)


def test_macro_auc_matches_sklearn_mean():
    rng = np.random.RandomState(0)
    y = (rng.rand(100, 12) < 0.3).astype(float)
    p = rng.rand(100, 12)
    macro, per = compute_macro_auc(y, p)
    expected = np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(12)])
    assert macro == pytest.approx(expected)
    assert list(per) == ABNORMALITIES


def test_macro_auc_masks_unlabeled_and_single_class():
    y = np.array([[1, 0], [0, 0], [np.nan, np.nan], [1, 0]], dtype=float)
    p = np.array([[0.9, 0.1], [0.1, 0.2], [0.0, 0.9], [0.8, 0.3]])
    macro, per = compute_macro_auc(y, p, ["A", "B"])
    assert per["A"] == 1.0  # the NaN row with score 0.0 is ignored
    assert np.isnan(per["B"])  # only negatives -> undefined, excluded from mean
    assert macro == 1.0


def test_error_analysis_lists_hardest_cases():
    y = np.array([[1], [1], [0], [0]], dtype=float)
    p = np.array([[0.9], [0.2], [0.8], [0.1]])
    table = error_analysis(y, p, ["s1", "s2", "s3", "s4"], ["A"], top_k=1, n_boot=20)
    assert table.loc[0, "hardest_fn"] == "s2"
    assert table.loc[0, "hardest_fp"] == "s3"
    assert table.loc[0, "auc"] == pytest.approx(0.75)


def test_gold_labels_masks_m2_pseudo_labels():
    from validation import TRUE_FLAG_SUFFIX, gold_labels
    df = pd.DataFrame({ID_COL: ["a", "b"]})
    for c in ABNORMALITIES:
        df[c] = [1.0, 0.73]                 # b is a soft pseudo-label
        df[c + TRUE_FLAG_SUFFIX] = [1, 0]
    gold = gold_labels(df)
    assert gold.loc[0, ABNORMALITIES].tolist() == [1.0] * 12
    assert gold.loc[1, ABNORMALITIES].isna().all()
    # folds treat pseudo-labeled studies as unlabeled when stratifying
    summary = fold_summary(df, pd.DataFrame({ID_COL: ["a", "b"], "fold": [0, 1]}))
    assert summary["labeled"].tolist() == [1, 0]
