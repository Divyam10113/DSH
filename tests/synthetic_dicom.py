"""Synthetic knee-MRI-like DICOM studies in the competition layout, for tests and dry runs.

    <root>/train.csv, train_series.csv
    <root>/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

Series are written in the competition's transfer syntaxes: Explicit VR LE, Implicit VR LE,
JPEG 2000 (pylibjpeg-openjpeg encoder) and JPEG Lossless (transcoded with GDCM when available).
"""

from __future__ annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless, JPEGLosslessSV1,
                         MRImageStorage, generate_uid)

ORIENTATIONS = {  # ImageOrientationPatient (row cosines, column cosines)
    "Sagittal": [0, 1, 0, 0, 0, -1],
    "Coronal": [1, 0, 0, 0, 0, -1],
    "Axial": [1, 0, 0, 0, 1, 0],
}
NORMAL_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def _phantom(k: int, n: int, rows: int, cols: int, seed: int) -> np.ndarray:
    """A bright ellipse ("joint") whose size varies with slice index, plus noise; uint16."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    r = 0.25 + 0.15 * np.sin(np.pi * (k + 1) / (n + 1))
    blob = (((y - rows / 2) / (rows * r)) ** 2 + ((x - cols / 2) / (cols * r)) ** 2) < 1
    img = 200 + 1500 * blob + 40 * (k % 7) + rng.normal(0, 30, (rows, cols))
    return np.clip(img, 0, 4095).astype(np.uint16)


def _gdcm_transcode(path: Path, uid: str) -> bool:
    try:
        import gdcm
    except ImportError:
        return False
    reader = gdcm.ImageReader()
    reader.SetFileName(str(path))
    if not reader.Read():
        return False
    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax(gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLosslessProcess14_1))
    change.SetInput(reader.GetImage())
    if not change.Change():
        return False
    writer = gdcm.ImageWriter()
    writer.SetFileName(str(path))
    writer.SetFile(reader.GetFile())
    writer.SetImage(change.GetOutput())
    return bool(writer.Write())


def write_slice(path: Path, pixels: np.ndarray, plane: str, index: int, study_uid: str, series_uid: str,
                syntax: str, spacing=(0.4, 0.4), thickness: float = 3.0, instance_number: int | None = None) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.Modality = "MR"
    ds.InstanceNumber = instance_number if instance_number is not None else index + 1
    pos = [0.0, 0.0, 0.0]
    pos[NORMAL_AXIS[plane]] = index * thickness
    ds.ImagePositionPatient = pos
    ds.ImageOrientationPatient = ORIENTATIONS[plane]
    ds.PixelSpacing = list(spacing)
    ds.SliceThickness = thickness
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = pixels.tobytes()

    if syntax == "implicit":
        meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds.save_as(path, implicit_vr=True, little_endian=True, enforce_file_format=True)
        return
    if syntax == "j2k":
        ds.compress(JPEG2000Lossless, pixels)
    ds.save_as(path, enforce_file_format=True)
    if syntax == "jpeg_lossless" and not _gdcm_transcode(path, JPEGLosslessSV1):
        raise RuntimeError("GDCM is needed to write JPEG Lossless test files")


def make_dataset(root: Path, n_studies: int = 4, syntaxes=("explicit", "implicit", "j2k", "jpeg_lossless"),
                 seed: int = 0) -> pd.DataFrame:
    """Write studies with a sagittal PD-FS + sagittal T1 + coronal + axial series each.

    Returns the series table. Slice files are shuffled on disk and InstanceNumbers are scrambled on
    one series, so ordering must come from the patient geometry.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    series_rows, study_ids = [], []
    layout = [  # plane, fluid, fatsat, n slices, rows, cols
        ("Sagittal", 1, 1, 34, 96, 96),
        ("Sagittal", 0, 0, 24, 96, 96),
        ("Coronal", 1, 1, 20, 80, 96),
        ("Axial", 1, 0, 40, 96, 96),
    ]
    for s in range(n_studies):
        study_uid = generate_uid()
        study_ids.append(study_uid)
        for j, (plane, fluid, fatsat, n, rows, cols) in enumerate(layout):
            series_uid = generate_uid()
            syntax = syntaxes[(s + j) % len(syntaxes)]
            folder = root / "train_series" / study_uid / series_uid
            folder.mkdir(parents=True)
            scramble = j == 3
            for k in rng.permutation(n):
                pixels = _phantom(k, n, rows, cols, seed=s * 1000 + j * 100 + int(k))
                write_slice(folder / f"{generate_uid()}.dcm", pixels, plane, int(k), study_uid, series_uid, syntax,
                            instance_number=int(rng.integers(1, 999)) if scramble else None)
            series_rows.append({"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
                                "Fluid_Sensitive": fluid, "Fat_Suppression": fatsat, "Anatomical_Plane": plane,
                                "syntax": syntax, "n_slices": n})
    series = pd.DataFrame(series_rows)
    series.drop(columns=["syntax", "n_slices"]).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(root / "train.csv", index=False)
    return series


if __name__ == "__main__":
    out = Path(tempfile.mkdtemp())
    print(make_dataset(out))
    print(out)
    shutil.rmtree(out)
