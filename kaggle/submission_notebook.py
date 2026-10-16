# %% [markdown]
# # RSNA Knee — Offline Submission Notebook (internet OFF, <= 9 h)
# Attach as inputs:
# 1. the competition data
# 2. `knee-code`    : this repo, published with scripts/push_code_dataset.sh
# 3. `knee-wheels`  : offline wheels built with scripts/build_offline_wheels.sh
# 4. the per-fold weight datasets `knee-weights-effb0-fold0` .. `fold4` (each holds `fold<k>/best.pth`)
#
# Nothing here touches the internet: models are rebuilt with pretrained=False and loaded from disk.

# %%
import glob, os, subprocess, sys

def find_input(name: str, fallback: str = "", is_file: bool = False) -> str:
    legacy = f"/kaggle/input/{name}"
    if is_file and os.path.isfile(legacy):
        return legacy
    if not is_file and os.path.isdir(legacy):
        return legacy
    comp = f"/kaggle/input/competitions/{name}"
    if is_file and os.path.isfile(comp):
        return comp
    if not is_file and os.path.isdir(comp):
        return comp
    matches = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    for m in matches:
        if is_file and os.path.isfile(m):
            return m
        if not is_file and os.path.isdir(m):
            return m
    return fallback or legacy

COMP = find_input("rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection")
CODE = find_input("knee-code", "/kaggle/input/knee-code")
WHEELS = find_input("knee-wheels", "/kaggle/input/knee-wheels")

MODELS = {
    "effb0": "/kaggle/input/**/knee-weights-effb0-fold*/**/best.pth",  # averages every fold found
    # "resnet34": "/kaggle/input/**/knee-weights-resnet34-fold*/**/best.pth",
}
ENSEMBLE_JSON = None  # with 2+ models: path to ensemble.json from `ensemble.py fit`; None = equal-weight mean
TIME_BUDGET_HOURS = 8.0  # safety margin under the 9 h hard limit

# %%
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
