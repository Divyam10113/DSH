# %% [markdown]
# # RSNA Knee — Offline Submission Notebook (internet OFF, <= 9 h)
# Attach as inputs:
# 1. the competition data
# 2. `knee-code`    : this repo, published with scripts/push_code_dataset.sh
# 3. `knee-wheels`  : offline wheels built with scripts/build_offline_wheels.sh
# 4. one dataset per model run with `fold*/best.pth` (+ optional `ensemble.json`)
#
# Nothing here touches the internet: models are rebuilt with pretrained=False and loaded from disk.

# %%
COMP = "/kaggle/input/rsna-knee-abnormality-detection"
CODE = "/kaggle/input/knee-code"
WHEELS = "/kaggle/input/knee-wheels"
MODELS = {
    "effb0": "/kaggle/input/knee-weights-effb0/*/best.pth",
    # "convnext": "/kaggle/input/knee-weights-convnext/*/best.pth",
}
ENSEMBLE_JSON = "/kaggle/input/knee-weights-effb0/ensemble.json"  # optional; equal-weight mean if missing
TIME_BUDGET_HOURS = 8.0  # safety margin under the 9 h hard limit

# %%
import glob, os, subprocess, sys
wheels = glob.glob(f"{WHEELS}/*.whl")
if wheels:  # DICOM decoders for JPEG Lossless / JPEG 2000 transfer syntaxes
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", WHEELS, "-q",
                    "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"], check=False)
sys.path.insert(0, CODE)

# %%
from inference import run_inference

sub = run_inference(
    data_root=COMP,
    model_globs=MODELS,
    out_path="/kaggle/working/submission.csv",
    ensemble_json=ENSEMBLE_JSON,
    time_budget_hours=TIME_BUDGET_HOURS,
    tta=True,
    num_workers=4,
)
sub.head()
