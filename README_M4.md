# M4 — Integration, Training Engine & Submission (Pranav)

End-to-end run order. Everything runs on Kaggle; the only local step is editing code.

| Step | Command / file | Output |
|---|---|---|
| 1. Cache images (M1) | `python -m knee_dicom.cache --split train --out-dir /kaggle/working/knee_cache` | `knee-cache-v1` dataset |
| 2. Mine labels (M2) | `report_label_mining` pipeline | `labels.csv` (soft labels, `__true` flags, `sample_weight`) |
| 3. Shared folds (**once**) | `python validation.py make-folds --train-csv train.csv --out folds.csv` | `knee-folds` dataset |
| 4. Train | [kaggle/train_notebook.py](kaggle/train_notebook.py) → `train.py` | `checkpoints/<run>/fold*/{best,last}.pth`, `oof.csv`, `experiments.csv` |
| 5. Diagnose | `python validation.py report --oof checkpoints/<run>/oof.csv --labels train.csv` | per-label AUC, 95% CI, hardest FN/FP studies |
| 6. Ensemble | `python ensemble.py fit --oof a=runA/oof.csv b=runB/oof.csv --labels train.csv` | `ensemble.json` |
| 7. Package | `scripts/build_offline_wheels.sh`, `scripts/push_code_dataset.sh` | `knee-wheels`, `knee-code` datasets |
| 8. Submit | [kaggle/submission_notebook.py](kaggle/submission_notebook.py) → `inference.py` | `submission.csv` |

## Guarantees
- **No leakage:** folds are split by study (optionally by patient via `--group-col`), iteratively stratified on all 12 labels and on gold-label availability; `check_no_leakage` asserts it.
- **Metric = Kaggle's:** mean of per-label ROC-AUC. Scored on **ground truth only**: M2 pseudo-labels are masked with the `__true` flags.
- **Killed sessions resume:** `last.pth` holds model/optimizer/scheduler/AMP scaler/epoch/RNG and is written atomically. Re-run with the same args, or pass `--resume-from <prev output>/last.pth`. `--max-hours` stops cleanly before Kaggle's limit.
- **Offline submission always valid:** an all-0.5 file is written first. Corrupt studies get neutral scores, TTA switches off automatically when time gets tight, and once the budget is spent the remaining studies get fallback scores. The format is validated before writing.
- **Train/test parity:** test DICOMs go through M1's `knee_dicom.preprocess_study` with the `img_size`/`max_slices` stored in each checkpoint.

## Tests
```bash
pytest tests -q        # needs knee_dicom importable (merge krish-knee-dicom) for the raw-DICOM test
```
