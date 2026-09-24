import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee_dicom.cache import build_cache, parse_args, preprocess_study
from knee_dicom.dicom_io import DecodeError, read_frames, read_header
from knee_dicom.loader import load_study, make_slabs, read_manifest, sliding_mip, to_unit_float
from knee_dicom.preprocess import PreprocessConfig, preprocess_volume, resize_slice, select_slices, window_volume
from knee_dicom.routing import attach_planes, route_study
from knee_dicom.series import load_series, plane_from_orientation
from knee_dicom.verify import cache_check, decode_check, montage

from .synthetic_dicom import _phantom, make_dataset

SYNTAXES = {
    "explicit": "1.2.840.10008.1.2.1",
    "implicit": "1.2.840.10008.1.2",
    "j2k": "1.2.840.10008.1.2.4.90",
    "jpeg_lossless": "1.2.840.10008.1.2.4.70",
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    series = make_dataset(root, n_studies=4)
    return root, series


@pytest.fixture(scope="module")
def cache(dataset, tmp_path_factory):
    root, _ = dataset
    out = tmp_path_factory.mktemp("cache")
    summary = build_cache(parse_args(["--data-dir", str(root), "--out-dir", str(out), "--workers", "2",
                                      "--extra-t1"]))
    return out, summary


# -- decoding -------------------------------------------------------------------------------------

@pytest.mark.parametrize("syntax", list(SYNTAXES))
def test_every_transfer_syntax_decodes_losslessly(dataset, syntax):
    root, series = dataset
    row = series[series["syntax"] == syntax].iloc[0]
    folder = root / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
    files = sorted(folder.glob("*.dcm"))
    header = read_header(files[0])
    assert header.transfer_syntax == SYNTAXES[syntax]
    vol = load_series(folder)
    assert vol.pixels.shape[0] == row["n_slices"]
    # Lossless codecs: decoded slice k must equal the phantom written for slice k.
    study_idx = list(series["StudyInstanceUID"].unique()).index(row["StudyInstanceUID"])
    j = list(series[series["StudyInstanceUID"] == row["StudyInstanceUID"]]["SeriesInstanceUID"]).index(
        row["SeriesInstanceUID"])
    n, (rows, cols) = int(row["n_slices"]), vol.pixels.shape[1:]
    for k in (0, n // 2, n - 1):
        expected = _phantom(k, n, rows, cols, seed=study_idx * 1000 + j * 100 + k)
        assert np.array_equal(vol.pixels[k], expected.astype(np.float32))


def test_decode_error_on_garbage(tmp_path):
    bad = tmp_path / "bad.dcm"
    bad.write_bytes(b"not a dicom at all")
    with pytest.raises(Exception):
        read_frames(bad)
    empty = tmp_path / "series"
    empty.mkdir()
    with pytest.raises(DecodeError):
        load_series(empty)


def test_decode_check_reports_all_files(dataset, tmp_path):
    root, series = dataset
    assert decode_check(root, "train", sample=0, seed=0, workers=2, out=str(tmp_path / "r.csv"))
    report = pd.read_csv(tmp_path / "r.csv")
    assert len(report) == series["n_slices"].sum() and report["ok"].all()
    assert set(report["transfer_syntax"]) == set(SYNTAXES.values())


# -- geometry, routing, preprocessing -------------------------------------------------------------

def test_slices_ordered_by_geometry_despite_shuffled_files_and_instance_numbers(dataset):
    root, series = dataset
    row = series[series["Anatomical_Plane"] == "Axial"].iloc[0]  # InstanceNumbers scrambled here
    vol = load_series(root / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"])
    z = [h.position[2] for h in vol.headers]
    assert z == sorted(z) and len(set(z)) == len(z)
    assert vol.inferred_plane == "axial"


@pytest.mark.parametrize("orientation,plane", [
    ([0, 1, 0, 0, 0, -1], "sagittal"), ([1, 0, 0, 0, 0, -1], "coronal"), ([1, 0, 0, 0, 1, 0], "axial"),
    ([0.99, 0.1, 0, 0, 0.1, -0.99], "coronal"),
])
def test_plane_from_orientation(orientation, plane):
    assert plane_from_orientation(np.array(orientation, float)) == plane


def test_routing_prefers_fluid_fatsat_and_keeps_t1(dataset):
    _, series = dataset
    study = attach_planes(series[series["StudyInstanceUID"] == series["StudyInstanceUID"].iloc[0]])
    routes = route_study(study, extra_t1=True)
    sag = study[study["plane"] == "sagittal"].set_index("SeriesInstanceUID")
    assert sag.loc[routes["sagittal"], "Fluid_Sensitive"] == 1
    assert sag.loc[routes["sagittal_t1"], "Fluid_Sensitive"] == 0
    assert set(routes) == {"sagittal", "coronal", "axial", "sagittal_t1"}


def test_preprocess_shapes_range_and_slice_selection():
    vol = np.random.default_rng(0).gamma(2, 300, (120, 200, 150)).astype(np.float32)
    cfg = PreprocessConfig(img_size=64, max_slices=32)
    out, keep = preprocess_volume(vol, cfg, pixel_spacing=(0.5, 0.5))
    assert out.shape == (32, 64, 64) and out.dtype == np.uint8
    assert keep[0] == 0 and keep[-1] == 119  # full anatomical extent kept
    assert list(select_slices(20, 32)) == list(range(20))
    w = window_volume(vol)
    assert w.min() >= 0 and w.max() <= 1
    out16, _ = preprocess_volume(vol, PreprocessConfig(img_size=64, dtype="float16"))
    assert out16.dtype == np.float16 and float(out16.max()) <= 1.0


def test_letterbox_keeps_physical_aspect():
    img = np.ones((100, 200), np.float32)
    out = resize_slice(img, 64, keep_aspect=True, pixel_spacing=(1.0, 1.0))
    filled_rows = np.flatnonzero(out.max(axis=1) > 0)
    assert out.shape == (64, 64) and 30 <= len(filled_rows) <= 34  # 2:1 -> half the height
    stretched = resize_slice(img, 64, keep_aspect=False)
    assert (stretched > 0).all()


def test_blank_volume_does_not_crash():
    out, _ = preprocess_volume(np.zeros((5, 32, 32), np.float32), PreprocessConfig(img_size=16))
    assert out.shape == (5, 16, 16) and out.max() == 0


# -- cache ------------------------------------------------------------------------------------------

def test_cache_layout_matches_m3_dataset(cache, dataset):
    out, summary = cache
    root, series = dataset
    assert summary["n_studies"] == 4 and summary["n_series_failed"] == 0 and summary["studies_all_planes"] == 4
    manifest = read_manifest(out, require_planes=("sagittal", "coronal", "axial"))
    assert len(manifest) == 4 and "study_id" in manifest
    for uid in manifest["study_id"]:
        # M3's KneeMRIDataset reads {data_dir}/{study_id}/{plane}.npy as (K, H, W).
        for plane in ("sagittal", "coronal", "axial", "sagittal_t1"):
            arr = np.load(out / uid / f"{plane}.npy")
            assert arr.ndim == 3 and arr.shape[1:] == (128, 128) and arr.dtype == np.uint8
        study = load_study(out, uid)
        assert set(study) == {"sagittal", "coronal", "axial"}
        assert all(v.dtype == np.float32 and 0 <= v.min() and v.max() <= 1 for v in study.values())
        assert study["sagittal"].shape[0] == 32   # 34 slices subsampled to max 32
        assert study["coronal"].shape[0] == 20    # shorter series kept as-is


def test_cache_resumes_and_checks(cache, dataset, tmp_path):
    out, _ = cache
    root, _ = dataset
    before = {p: p.stat().st_mtime_ns for p in out.glob("*/meta.json")}
    build_cache(parse_args(["--data-dir", str(root), "--out-dir", str(out), "--workers", "1"]))
    assert {p: p.stat().st_mtime_ns for p in out.glob("*/meta.json")} == before  # nothing redone

    assert cache_check(out, root, "train")
    assert len(montage(out, 2, tmp_path / "qc")) == 2

    victim = next(out.glob("*/coronal.npy"))
    saved = victim.read_bytes()
    victim.unlink()
    try:
        assert not cache_check(out, root, "train")
    finally:
        victim.write_bytes(saved)


def test_study_with_broken_series_still_cached(dataset, tmp_path):
    root, series = dataset
    uid = series["StudyInstanceUID"].iloc[0]
    study_dir = tmp_path / uid
    import shutil
    shutil.copytree(root / "train_series" / uid, study_dir)
    rows = series[series["StudyInstanceUID"] == uid].drop(columns=["syntax", "n_slices"])
    coronal = rows.loc[rows["Anatomical_Plane"] == "Coronal", "SeriesInstanceUID"].iloc[0]
    for f in (study_dir / coronal).glob("*.dcm"):
        f.write_bytes(b"corrupt")
    arrays, meta = preprocess_study(study_dir, rows, PreprocessConfig(img_size=32))
    assert "coronal" not in arrays and {"sagittal", "axial"} <= set(arrays)
    assert meta["series"]["coronal"]["error"]
    json.dumps(meta, default=str)


def test_missing_plane_inferred_from_orientation(dataset, tmp_path):
    root, series = dataset
    uid = series["StudyInstanceUID"].iloc[1]
    rows = series[series["StudyInstanceUID"] == uid].drop(columns=["syntax", "n_slices"]).copy()
    rows["Anatomical_Plane"] = np.nan  # e.g. an unlabeled test series
    arrays, meta = preprocess_study(root / "train_series" / uid, rows, PreprocessConfig(img_size=32))
    assert set(arrays) == {"sagittal", "coronal", "axial"}


# -- variants ---------------------------------------------------------------------------------------

def test_slabs_and_mip():
    vol = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)
    slabs = make_slabs(vol)
    assert slabs.shape == (5, 3, 2, 2)
    assert np.array_equal(slabs[2, 0], vol[1]) and np.array_equal(slabs[2, 2], vol[3])
    assert np.array_equal(slabs[0, 0], vol[0]) and np.array_equal(slabs[-1, 2], vol[-1])
    mip = sliding_mip(vol, 3)
    assert mip.shape == vol.shape and np.array_equal(mip[1], vol[2]) and np.array_equal(mip[-1], vol[-1])
    assert to_unit_float(np.array([0, 255], np.uint8)).tolist() == [0.0, 1.0]
