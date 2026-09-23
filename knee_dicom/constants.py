"""Shared names for the RSNA Knee Abnormality Detection imaging data."""

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
PLANE_COL = "Anatomical_Plane"
FLUID_COL = "Fluid_Sensitive"
FATSAT_COL = "Fat_Suppression"

# Plane names as written by the cache. They match M3's KneeMRIDataset ({study}/{plane}.npy).
PLANES = ("sagittal", "coronal", "axial")

DEFAULT_KAGGLE_DATA_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
