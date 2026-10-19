# %% [markdown]
# # RSNA Knee — Thin Training Launcher (Kaggle, GPU, internet ON)
# All real code lives in GitHub. This notebook only clones the repo, points it at the data
# and runs one fold. Checkpoints go to /kaggle/working, which becomes this notebook's output.
#
# **Resuming a killed session:** attach this notebook's previous output as an input dataset and
# set `RESUME_FROM` to its `last.pth`. With `MAX_HOURS` set, the engine stops cleanly before
# Kaggle's session limit, so the next session only loses the epoch in progress, never the run.
#
# Paste each `# %%` block into its own notebook cell (or import this file as a notebook via jupytext).

# %%
REPO = "https://github.com/Divyam10113/DSH.git"
BRANCH = "feat/m3-modeling"
# Custom dataset paths (leave empty string to auto-discover under /kaggle/input)
COMP_DIR = ""                                   # auto-finds rsna-knee-abnormality-detection
CACHE_DIR = ""                                  # auto-finds knee-cache-v1
LABELS_PATH = ""                                # auto-finds knee-labels-v1/labels.csv
FOLDS_PATH = ""                                 # auto-finds knee-folds/folds.csv
FOLD = "1"                                      # "0".."4" or "all"
RUN_NAME = "effb0_gated_concat"
RESUME_FROM = ""                                # e.g. /kaggle/input/<prev-output>/checkpoints/<run>/fold0/last.pth
MAX_HOURS = 8.5                                 # below the session limit, leaves time to save outputs
SMOKE = False                                   # True: 200 studies, 1 epoch, to check the pipeline before a full run

# %%
import glob, os, subprocess, sys

def find_comp(explicit: str = "") -> str:
    if explicit and os.path.isdir(explicit):
        return explicit
    for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.isdir(p):
            return p
    for m in glob.glob("/kaggle/input/**/rsna-knee-abnormality-detection", recursive=True):
        if os.path.isdir(m):
            return m
    return "/kaggle/input/competitions/rsna-knee-abnormality-detection"

def find_cache(explicit: str = "") -> str:
    if explicit and os.path.isdir(explicit):
        return explicit
    for p in ("/kaggle/input/notebooks/aryachauhan5/m1-knee-cache-build/knee_cache",
              "/kaggle/input/m1-knee-cache-build/knee_cache",
              "/kaggle/input/knee_cache",
              "/kaggle/input/knee-cache-v1"):
        if os.path.isdir(p):
            return p
    for m in sorted(glob.glob("/kaggle/input/**/manifest.csv", recursive=True)):
        return os.path.dirname(m)
    for m in sorted(glob.glob("/kaggle/input/**/knee_cache", recursive=True)):
        if os.path.isdir(m):
            return m
    return "/kaggle/input/notebooks/aryachauhan5/m1-knee-cache-build/knee_cache"

def find_labels(explicit: str = "") -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in ("/kaggle/input/datasets/aryachauhan5/knee-labels-v1/labels.csv",
              "/kaggle/input/knee-labels-v1/labels.csv"):
        if os.path.isfile(p):
            return p
    for m in sorted(glob.glob("/kaggle/input/**/labels.csv", recursive=True)):
        if "rsna-knee-abnormality-detection" not in m:
            return m
    return "/kaggle/input/datasets/aryachauhan5/knee-labels-v1/labels.csv"

def find_folds(explicit: str = "") -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in ("/kaggle/input/datasets/pranavlorekar4096/knee-folds/folds.csv",
              "/kaggle/input/knee-folds/folds.csv"):
        if os.path.isfile(p):
            return p
    for m in sorted(glob.glob("/kaggle/input/**/folds.csv", recursive=True)):
        return m
    return "/kaggle/input/datasets/pranavlorekar4096/knee-folds/folds.csv"

if not os.path.exists("/kaggle/working/DSH"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "/kaggle/working/DSH"], check=True)
sys.path.insert(0, "/kaggle/working/DSH")
os.chdir("/kaggle/working")

# %%
import pandas as pd
from train import TrainConfig, run_training

COMP = find_comp(COMP_DIR)
CACHE = find_cache(CACHE_DIR)
LABELS = find_labels(LABELS_PATH)
FOLDS = find_folds(FOLDS_PATH)

print(f"Resolved input paths:")
print(f"  COMP:   {COMP} (exists: {os.path.exists(COMP)})")
print(f"  CACHE:  {CACHE} (exists: {os.path.exists(CACHE)})")
print(f"  LABELS: {LABELS} (exists: {os.path.exists(LABELS)})")
print(f"  FOLDS:  {FOLDS} (exists: {os.path.exists(FOLDS)})")

assert os.path.isdir(CACHE), f"Image cache not found at {CACHE}. Attach notebook 'm1-knee-cache-build' via Add Input -> Notebooks. Available inputs: {glob.glob('/kaggle/input/*')}"
labels_csv = LABELS if os.path.exists(LABELS) else f"{COMP}/train.csv"  # before M2 delivers: gold labels only
df = pd.read_csv(labels_csv)
folds = pd.read_csv(FOLDS) if os.path.exists(FOLDS) else None

# Keep only studies with at least one cached plane (manifest.csv is written by knee_dicom.cache)
manifest_csv = f"{CACHE}/manifest.csv"
if os.path.exists(manifest_csv):
    man = pd.read_csv(manifest_csv, dtype={"StudyInstanceUID": str})
    has_any = man[["has_sagittal", "has_coronal", "has_axial"]].any(axis=1)
    cached = set(man.loc[has_any, "StudyInstanceUID"])
    before = len(df)
    df = df[df["StudyInstanceUID"].astype(str).isin(cached)].reset_index(drop=True)
    print(f"cache filter: kept {len(df)}/{before} studies with at least one cached plane")

if SMOKE:
    df = df.sample(min(200, len(df)), random_state=0).reset_index(drop=True)
print(f"labels={labels_csv} studies={len(df)} folds={'shared' if folds is not None else 'generated'} smoke={SMOKE}")

cfg = TrainConfig(
    backbone_name="efficientnet_b0", pooling_type="gated", fusion_type="concat",
    pretrained=True, embed_dim=256,
    img_size=128, max_slices=32,  # must equal the knee_dicom cache build (see the cache's summary/meta.json)
    epochs=1 if SMOKE else 15, batch_size=4, grad_accum_steps=4, lr=3e-4, num_workers=4,
    save_dir="/kaggle/working/checkpoints", run_name=f"{RUN_NAME}_smoke" if SMOKE else RUN_NAME,
    resume_from=RESUME_FROM, max_hours=MAX_HOURS,
    leaderboard_csv="/kaggle/working/experiments.csv", use_wandb=False,
)
score = run_training(df=df, data_dir=CACHE, folds=folds, fold=FOLD, cfg=cfg)
print("best val macro-AUC:", score)

# %% [markdown]
# Next: send Pranav the new row of `experiments.csv`, then publish
# `/kaggle/working/checkpoints/<RUN_NAME>` as the Kaggle Dataset `knee-weights-effb0-fold<FOLD>`.
