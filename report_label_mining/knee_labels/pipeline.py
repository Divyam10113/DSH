"""End-to-end M2 pipeline: reports -> validated text models -> pseudo-labels -> labels.csv.

    python -m knee_labels.pipeline --data-dir /kaggle/input/rsna-knee-abnormality-detection \\
        --out-dir /kaggle/working/labels [--models tfidf,xlmr] [--folds folds.csv] [--translate]

Outputs (in --out-dir):
    labels.csv              soft labels for 100% of training studies (true > pseudo) + flags/weights
    teacher_probs.csv       text-model probabilities for every study (out-of-fold on the labeled
                            subset) — the soft targets for text-teacher -> image-student distillation
    rule_scores.csv         rule baseline score per study/label
    metrics_per_label.csv   per-label AUC/F1 of each source vs ground truth (out-of-fold)
    metrics_summary.csv     macro averages of the above
    prevalence.csv          ground-truth vs pseudo-label prevalence sanity check
    languages.csv           detected report language counts
    summary.json            headline numbers, incl. whether the classifier beats the rule baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import DEFAULT_KAGGLE_DATA_DIR, ID_COL, LABELS, REPORT_COL
from .data import label_matrix, labeled_rows, load_split, make_folds
from .evaluate import compare_sources
from .merge import build_labels, check_no_leakage, prevalence_report
from .rules import rule_features, scores_from_features
from .text import detect_language
from .tfidf_model import TfidfLabeler

MODEL_CHOICES = ("tfidf", "xlmr")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_model(name: str, args):
    if name == "tfidf":
        return TfidfLabeler(C=args.tfidf_c, use_rule_features=not args.no_rule_features)
    from .xlmr_model import XLMRConfig, XLMRLabeler

    return XLMRLabeler(XLMRConfig(model_name=args.xlmr_model, epochs=args.xlmr_epochs,
                                  batch_size=args.xlmr_batch_size, max_len=args.xlmr_max_len, seed=args.seed))


def _fit_predict(name: str, args, texts, feats, Y, train_idx, pred_idx):
    model = _make_model(name, args)
    if name == "tfidf":
        model.fit(texts.iloc[train_idx], Y[train_idx], feats.iloc[train_idx])
        return model, model.predict_proba(texts.iloc[pred_idx], feats.iloc[pred_idx])
    model.fit(texts.iloc[train_idx], Y[train_idx], log=log)
    return model, model.predict_proba(texts.iloc[pred_idx])


def _release(model) -> None:
    if getattr(model, "model", None) is not None:  # transformer: free GPU memory between folds
        import torch

        model.model = None
        torch.cuda.empty_cache()


SOURCE_CHOICES = ("model", "blend", "rules")


def candidate_probs(model_probs: list[np.ndarray], rule_P: np.ndarray) -> dict[str, np.ndarray]:
    model = np.mean(model_probs, axis=0)
    return {"model": model, "blend": 0.5 * model + 0.5 * rule_P, "rules": rule_P}


def choose_sources(Y: np.ndarray, candidates: dict[str, np.ndarray], rule_only: np.ndarray) -> list[str]:
    """Per label, the candidate with the best out-of-fold AUC (ties prefer the earlier choice).

    Labels with too few labeled positives to learn from always use the rules. Choosing on the same
    out-of-fold predictions makes the reported "final" score slightly optimistic; with only three
    candidates per label the bias is small.
    """
    chosen = []
    for j in range(Y.shape[1]):
        known = ~np.isnan(Y[:, j])
        if rule_only[j] or len(np.unique(Y[known, j])) < 2:
            chosen.append("rules")
            continue
        aucs = {name: roc_auc_score(Y[known, j], candidates[name][known, j]) for name in SOURCE_CHOICES}
        chosen.append(max(SOURCE_CHOICES, key=lambda name: (aucs[name], -SOURCE_CHOICES.index(name))))
    return chosen


def combine(candidates: dict[str, np.ndarray], chosen: list[str]) -> np.ndarray:
    return np.column_stack([candidates[name][:, j] for j, name in enumerate(chosen)])


def run(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models or any(m not in MODEL_CHOICES for m in models):
        raise SystemExit(f"--models must be a comma-separated subset of {MODEL_CHOICES}")

    train, test = load_split(args.data_dir)
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=args.seed).reset_index(drop=True)
    Y = label_matrix(train)
    lab = labeled_rows(Y)
    log(f"{len(train)} training studies; {lab.sum()} with ground-truth labels, {(~lab).sum()} unlabeled")
    if lab.sum() < args.n_splits:
        raise SystemExit("Not enough labeled studies to cross-validate")

    # 1. Language handling -------------------------------------------------------------------------
    langs = train[REPORT_COL].map(detect_language)
    langs.value_counts().rename_axis("lang").rename("reports").to_csv(out / "languages.csv")
    log("languages: " + ", ".join(f"{k}={v}" for k, v in langs.value_counts().items()))

    texts = train[REPORT_COL]
    if args.translate:
        from .translate import translate_reports

        english = translate_reports(train, langs, REPORT_COL, batch_size=args.translate_batch_size,
                                    cache_path=args.translation_cache or out / "translations.csv", log=log)
        # Keep the original too: the multilingual models and rules can use both.
        texts = (train[REPORT_COL] + "\n\n" + english.where(langs != "en", "")).str.strip()

    # 2. Rule baseline on every report -------------------------------------------------------------
    feats = rule_features(texts)
    rule_P = scores_from_features(feats).to_numpy()
    rule_df = pd.DataFrame(rule_P, columns=LABELS)
    rule_df.insert(0, ID_COL, train[ID_COL].values)
    rule_df.to_csv(out / "rule_scores.csv", index=False)
    log("rule features extracted")

    # 3. Out-of-fold evaluation on the labeled subset ----------------------------------------------
    lab_idx = np.flatnonzero(lab)
    unl_idx = np.flatnonzero(~lab)
    Y_lab = Y[lab_idx]
    folds = make_folds(train[ID_COL].iloc[lab_idx], Y_lab, args.n_splits, args.seed, args.folds)
    positives = np.nansum(Y_lab, axis=0)
    rule_only = positives < args.min_positives
    if rule_only.any():
        log("too few positives, using rules for: " + ", ".join(np.array(LABELS)[rule_only]))

    oof = {m: np.zeros((len(lab_idx), len(LABELS))) for m in models}
    for k in np.unique(folds):
        tr, va = lab_idx[folds != k], lab_idx[folds == k]
        for m in models:
            model, P = _fit_predict(m, args, texts, feats, Y, tr, va)
            oof[m][folds == k] = P
            _release(model)
        log(f"fold {k} done ({len(va)} val studies)")

    oof_candidates = candidate_probs(list(oof.values()), rule_P[lab_idx])
    chosen = choose_sources(Y_lab, oof_candidates, rule_only)
    log("source per label: " + ", ".join(f"{l}={c}" for l, c in zip(LABELS, chosen)))
    oof_final = combine(oof_candidates, chosen)
    sources = {"rules": rule_P[lab_idx], **oof}
    if len(models) > 1:
        sources["model_mean"] = np.mean(list(oof.values()), axis=0)
    sources["final"] = oof_final
    per_label, summary = compare_sources(Y_lab, sources)
    per_label.to_csv(out / "metrics_per_label.csv")
    summary.to_csv(out / "metrics_summary.csv")
    log("out-of-fold macro metrics vs ground truth:\n" + summary.round(4).to_string())

    # 4. Fit on all labeled studies, pseudo-label the rest -----------------------------------------
    final_P = np.zeros((len(train), len(LABELS)))
    final_P[lab_idx] = oof_final
    if len(unl_idx):
        unl_probs = []
        for m in models:
            model, P = _fit_predict(m, args, texts, feats, Y, lab_idx, unl_idx)
            if args.save_models and m == "xlmr":
                model.save(out / "xlmr_final")
            unl_probs.append(P)
            _release(model)
        final_P[unl_idx] = combine(candidate_probs(unl_probs, rule_P[unl_idx]), chosen)
    log(f"pseudo-labeled {len(unl_idx)} studies")

    teacher = pd.DataFrame(final_P, columns=LABELS)
    teacher.insert(0, ID_COL, train[ID_COL].values)
    teacher["oof"] = lab.astype(int)
    teacher.to_csv(out / "teacher_probs.csv", index=False)

    # 5. Merge (true > pseudo), leakage guard, sanity checks ---------------------------------------
    labels = build_labels(train[ID_COL], Y, final_P, pseudo_weight=args.pseudo_weight)
    labels["lang"] = langs.values
    test_ids = test[ID_COL] if test is not None and ID_COL in test.columns else None
    check_no_leakage(labels, train, Y, test_ids)
    labels.to_csv(out / "labels.csv", index=False)
    prevalence = prevalence_report(Y, labels)
    prevalence.to_csv(out / "prevalence.csv")
    log("leakage guard passed; prevalence check:\n" + prevalence.round(3).to_string())

    result = {
        "n_studies": int(len(train)),
        "n_labeled": int(lab.sum()),
        "n_pseudo_labeled": int(len(unl_idx)),
        "models": models,
        "translated": bool(args.translate),
        "rule_only_labels": list(np.array(LABELS)[rule_only]),
        "source_per_label": dict(zip(LABELS, chosen)),
        "oof_macro_auc": {name: round(float(v), 4) for name, v in summary["auc"].items()},
        "oof_macro_f1@0.5": {name: round(float(v), 4) for name, v in summary["f1@0.5"].items()},
    }
    # "Done when" from the plan: the classifier beats the rule baseline on held-out labeled reports.
    result["final_minus_rules_auc"] = round(float(summary.loc["final", "auc"] - summary.loc["rules", "auc"]), 4)
    result["final_beats_rule_baseline"] = bool(result["final_minus_rules_auc"] > 0)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    log(f"done: {out / 'labels.csv'} covers {len(labels)}/{len(train)} studies; "
        f"final beats rule baseline: {result['final_beats_rule_baseline']}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_KAGGLE_DATA_DIR)
    p.add_argument("--out-dir", default="/kaggle/working/labels")
    p.add_argument("--models", default="tfidf", help="comma-separated: tfidf,xlmr")
    p.add_argument("--folds", default=None, help="optional CSV (StudyInstanceUID, fold) shared with the team")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-positives", type=int, default=5,
                   help="labels with fewer labeled positives use the rule score instead of a model")
    p.add_argument("--pseudo-weight", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="debug: use a random subset of studies")
    p.add_argument("--no-rule-features", action="store_true")
    p.add_argument("--tfidf-c", type=float, default=4.0)
    p.add_argument("--xlmr-model", default="xlm-roberta-base", help="Hub id or local path (offline)")
    p.add_argument("--xlmr-epochs", type=int, default=4)
    p.add_argument("--xlmr-batch-size", type=int, default=16)
    p.add_argument("--xlmr-max-len", type=int, default=512)
    p.add_argument("--save-models", action="store_true", help="save the final XLM-R model for distillation")
    p.add_argument("--translate", action="store_true", help="add opus-mt English translations of reports")
    p.add_argument("--translate-batch-size", type=int, default=32)
    p.add_argument("--translation-cache", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
