import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from knee_labels.constants import ID_COL, LABELS, TRUE_FLAG_SUFFIX
from knee_labels.data import label_matrix, labeled_rows, make_folds
from knee_labels.merge import LeakageError, build_labels, check_no_leakage
from knee_labels.pipeline import parse_args, run
from knee_labels.text import detect_language, fold_text, split_sentences

from .synthetic import make_dataset, write_dataset


def test_fold_text_and_sentences():
    assert fold_text("Außenmeniskus  Läsion\n  Épanchement") == "aussenmeniskus lasion\nepanchement"
    assert split_sentences("V.a. Riss. Erguss 3.5 mm; kein Ödem") == ["verdacht auf Riss", "Erguss 3.5 mm", "kein Ödem"]


@pytest.mark.parametrize("text,lang", [
    ("There is a tear of the medial meniscus and no effusion.", "en"),
    ("Kein Gelenkerguss, das vordere Kreuzband ist intakt.", "de"),
    ("Rotura del ligamento cruzado anterior con derrame articular.", "es"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_label_matrix_treats_non_binary_as_unknown():
    df = pd.DataFrame({label: [1, 0, np.nan, -1] for label in LABELS})
    Y = label_matrix(df)
    assert np.isnan(Y[2:]).all() and (Y[0] == 1).all()
    assert labeled_rows(Y).tolist() == [True, True, False, False]


def test_folds_cover_labeled_and_respect_shared_file(tmp_path):
    train, _ = make_dataset(n=200)
    Y = label_matrix(train)
    lab = labeled_rows(Y)
    folds = make_folds(train[ID_COL][lab], Y[lab], n_splits=5)
    assert set(folds) == set(range(5))

    shared = pd.DataFrame({ID_COL: train[ID_COL], "fold": np.arange(len(train)) % 3})
    shared.to_csv(tmp_path / "folds.csv", index=False)
    folds = make_folds(train[ID_COL][lab], Y[lab], folds_csv=tmp_path / "folds.csv")
    assert np.array_equal(folds, (np.flatnonzero(lab) % 3))


def test_merge_keeps_truth_and_guard_catches_problems():
    train, test = make_dataset(n=50)
    Y = label_matrix(train)
    P = np.full(Y.shape, 0.9)
    labels = build_labels(train[ID_COL], Y, P)
    check_no_leakage(labels, train, Y, test[ID_COL])

    known = ~np.isnan(Y)
    assert np.array_equal(labels[LABELS].to_numpy()[known], Y[known])
    assert (labels[LABELS].to_numpy()[~known] == 0.9).all()
    assert labels[[l + TRUE_FLAG_SUFFIX for l in LABELS]].to_numpy().astype(bool).tolist() == known.tolist()
    assert set(labels["label_source"]) <= {"true", "pseudo", "mixed"}

    tampered = labels.copy()
    tampered.loc[known[:, 0], "ACL"] = 1 - tampered.loc[known[:, 0], "ACL"]
    with pytest.raises(LeakageError, match="overwritten"):
        check_no_leakage(tampered, train, Y)

    with pytest.raises(LeakageError, match="test studies"):
        check_no_leakage(labels, train, Y, pd.Series([train[ID_COL].iloc[0]]))

    with pytest.raises(LeakageError, match="report"):
        check_no_leakage(labels.assign(Report="x"), train, Y)


def test_pipeline_end_to_end(tmp_path):
    train = write_dataset(tmp_path / "data", n=300)
    out = tmp_path / "out"
    result = run(parse_args(["--data-dir", str(tmp_path / "data"), "--out-dir", str(out), "--n-splits", "3"]))

    labels = pd.read_csv(out / "labels.csv")
    assert len(labels) == len(train) and labels[ID_COL].tolist() == train[ID_COL].tolist()
    for name in ["teacher_probs.csv", "rule_scores.csv", "metrics_per_label.csv", "metrics_summary.csv",
                 "prevalence.csv", "languages.csv", "summary.json"]:
        assert (out / name).exists(), name
    assert json.loads((out / "summary.json").read_text())["n_studies"] == len(train)

    # Pseudo labels on the unlabeled studies should rank the hidden truth well.
    hidden = train.attrs["hidden_Y"]
    unl = ~labeled_rows(label_matrix(train))
    aucs = [roc_auc_score(hidden[unl, j], labels[label].to_numpy()[unl]) for j, label in enumerate(LABELS)]
    assert np.mean(aucs) > 0.9
    assert result["oof_macro_auc"]["final"] > 0.9
