import json
import os

import numpy as np
import pandas as pd
import pytest

from ensemble import blend, fit_weights
from inference import SUBMISSION_COLUMNS, run_inference
from models.knee_model import ABNORMALITIES
from train import TrainConfig, build_model, save_weights
from validation import ID_COL

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("knee_dicom", reason="M1 package (branch krish-knee-dicom) needed for raw-DICOM inference")


def _write_dicom(path, pixels, z, instance):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelSpacing = [0.5, 0.5]
    ds.InstanceNumber = instance
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


@pytest.fixture
def fake_competition(tmp_path):
    """3 test studies: one complete, one with a single plane, one with no readable series."""
    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    series_rows = []
    studies = ["1.2.3.1", "1.2.3.2", "1.2.3.3"]
    layout = {"1.2.3.1": ["Sagittal", "Coronal", "Axial"], "1.2.3.2": ["Sagittal"], "1.2.3.3": ["Axial"]}
    for sid in studies:
        for j, plane in enumerate(layout[sid]):
            ser = f"{sid}.{j}"
            series_rows.append({ID_COL: sid, "SeriesInstanceUID": ser, "Fluid_Sensitive": 1,
                                "Fat_Suppression": 0, "Anatomical_Plane": plane})
            d = root / "test_series" / sid / ser
            d.mkdir(parents=True)
            if sid == "1.2.3.3":
                (d / "broken.dcm").write_bytes(b"not a dicom")
                continue
            for k in range(6):  # written in shuffled order; loader must sort spatially
                _write_dicom(str(d / f"{rng.randint(1e9)}.dcm"), rng.randint(0, 1000, (40, 48)), z=float(5 - k), instance=k)
    pd.DataFrame({ID_COL: studies}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "test_series.csv", index=False)
    return root, studies


def _weights(tmp_path, name, seed):
    import torch
    torch.manual_seed(seed)
    cfg = TrainConfig(backbone_name="resnet34", embed_dim=64, img_size=32)
    paths = []
    for fold in range(2):
        d = tmp_path / name / f"fold{fold}"
        d.mkdir(parents=True)
        save_weights(str(d / "best.pth"), build_model(cfg), cfg, 0.5, 1)
        paths.append(d)
    return str(tmp_path / name / "*" / "best.pth")


def test_offline_submission_end_to_end(tmp_path, fake_competition):
    root, studies = fake_competition
    out = tmp_path / "submission.csv"
    models = {"a": _weights(tmp_path, "a", 0), "b": _weights(tmp_path, "b", 1)}
    ens = tmp_path / "ensemble.json"
    ens.write_text(json.dumps({"method": "rank", "weights": {"a": 0.7, "b": 0.3}}))

    sub = run_inference(str(root), models, str(out), ensemble_json=str(ens), num_workers=0, tta=True)

    written = pd.read_csv(out, dtype={ID_COL: str})
    assert list(written.columns) == SUBMISSION_COLUMNS
    assert written[ID_COL].tolist() == studies
    vals = written[ABNORMALITIES].values
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
    # the broken study got the neutral fallback (column mean), the others real predictions
    assert np.allclose(vals[2], vals[:2].mean(axis=0))


def test_ensemble_fit_prefers_better_model():
    rng = np.random.RandomState(0)
    n = 400
    ids = [f"s{i}" for i in range(n)]
    y = (rng.rand(n, 12) < 0.3).astype(float)
    labels = pd.DataFrame(y, columns=ABNORMALITIES).assign(**{ID_COL: ids})
    good = pd.DataFrame(y * 0.6 + rng.rand(n, 12) * 0.7, columns=ABNORMALITIES).assign(**{ID_COL: ids})
    noise = pd.DataFrame(rng.rand(n, 12), columns=ABNORMALITIES).assign(**{ID_COL: ids})

    weights, table = fit_weights({"good": good, "noise": noise}, labels)
    assert weights["good"] > weights["noise"]
    ens_score = table.set_index("model").loc["ENSEMBLE", "oof_macro_auc"]
    assert ens_score >= table.set_index("model").loc["good", "oof_macro_auc"] - 1e-9

    blended = blend({"good": good, "noise": noise}, weights)
    assert list(blended.columns) == [ID_COL] + ABNORMALITIES and len(blended) == n
