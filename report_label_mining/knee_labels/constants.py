"""Shared names for the RSNA Knee Abnormality Detection data."""

ID_COL = "StudyInstanceUID"
REPORT_COL = "Report"

# Order matches the competition's submission header.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Suffix for the per-label "this value is a ground-truth label" flag in labels.csv.
TRUE_FLAG_SUFFIX = "__true"

DEFAULT_KAGGLE_DATA_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
