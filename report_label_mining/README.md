# M2 — Report Label Mining (NLP)

**Owner:** Arya Chauhan — 2024A7PS1441G · Group 4, RSNA Knee Abnormality Detection

**Problem statement:** Convert multilingual radiology reports into reliable labels for all 12
findings so the unlabeled training set becomes usable supervision for the image model.

Only a small subset of `train.csv` has ground-truth labels, but every training study has a report.
Reports are **not** available at test time, so this is a *training-time* component: it produces
`labels.csv`, which the image model (M3/M4) trains on.

## Pipeline

```
train.csv reports ─► language detection ─► (optional) opus-mt translation to English
                 ─► rule/keyword extractor (7 languages, negation + uncertainty aware)   [baseline]
                 ─► text classifiers: char/word TF-IDF + rule features (CPU), XLM-R (GPU)
                 ─► out-of-fold validation on the labeled subset (per-label AUC / F1 vs rules)
                 ─► per-label choice of source (model / blend / rules) by out-of-fold AUC
                 ─► pseudo-label every unlabeled study (soft probability + confidence)
                 ─► merge: ground truth > pseudo  ─► leakage guard ─► labels.csv
```

| Slide 5 responsibility | Code |
|---|---|
| Report ingestion, language detection, translation/normalisation | `knee_labels/text.py`, `knee_labels/translate.py` |
| Rule/keyword extractor (baseline) | `knee_labels/rules.py` |
| Multilingual text classifier, 12 heads | `knee_labels/tfidf_model.py`, `knee_labels/xlmr_model.py` |
| Pseudo-labeling with confidence | `knee_labels/pipeline.py` |
| Merge policy (true > pseudo) + leakage guard | `knee_labels/merge.py` |
| Validation against the labeled subset | `knee_labels/evaluate.py`, `knee_labels/data.py` |
| Distillation targets (later, with M3) | `teacher_probs.csv` output |

## Running on Kaggle

Attach the competition data and run (or use `notebooks/m2_label_mining.ipynb`):

```bash
# Fast CPU run (TF-IDF + rules), ~minutes
python -m knee_labels.pipeline --out-dir /kaggle/working/labels

# Full GPU run: add XLM-R, keep the model for distillation
python -m knee_labels.pipeline --out-dir /kaggle/working/labels --models tfidf,xlmr --save-models

# Use the team's shared folds so our out-of-fold predictions match the image model's CV
python -m knee_labels.pipeline --folds /kaggle/input/<folds-dataset>/folds.csv
```

The `--translate` flag adds English translations (Helsinki-NLP opus-mt, needs internet or attached
weights). It is cached to `translations.csv`. Publish the output directory as a Kaggle Dataset
so teammates can attach `labels.csv`.

## Outputs

| File | Contents |
|---|---|
| `labels.csv` | One row per training study (100% coverage). 12 soft label columns (ground truth where available, otherwise the pseudo-label probability), `<label>__true` flags, `label_source` (true/pseudo/mixed), `pseudo_confidence`, `sample_weight`, `lang` |
| `teacher_probs.csv` | Text-model probability for every study (out-of-fold on the labeled subset). Soft targets for text-teacher → image-student distillation |
| `metrics_per_label.csv`, `metrics_summary.csv` | Out-of-fold AUC / F1 of rules vs each model vs final |
| `prevalence.csv` | Ground-truth prevalence vs pseudo-label prevalence (sanity check) |
| `summary.json` | Headline numbers, including `final_beats_rule_baseline` (the slide's "done when") |

**For the image model:** train with BCE on the 12 label columns and weight each study by
`sample_weight`. Alternatively, use the `__true` flags to down-weight pseudo cells per label.
Evaluate CV on ground-truth labels only (`label_source == "true"`), never on pseudo-labels.

## Leakage guard

`check_no_leakage` fails the run if `labels.csv` has anything other than one row per training
study, contains test studies or report text, has values outside [0, 1], or changes or mis-flags a
ground-truth label.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

The tests cover the rules on English/German/Spanish/French/Italian examples, including negation,
uncertainty and laterality. They also cover folds, the merge policy, the leakage guard, and a full
pipeline run on a synthetic multilingual dataset (`tests/synthetic.py`).
