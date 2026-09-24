"""Robust single-slice DICOM decoding across every transfer syntax in the competition data.

The data mixes Explicit VR Little Endian, Implicit VR Little Endian, JPEG Lossless and JPEG 2000.
pydicom decodes the uncompressed ones natively. The compressed ones need a plugin:

    pylibjpeg + pylibjpeg-libjpeg   JPEG Lossless / JPEG-LS / baseline JPEG
    pylibjpeg + pylibjpeg-openjpeg  JPEG 2000
    python-gdcm                     fallback for anything the above cannot handle

Without these, pydicom raises on a chunk of studies (or silently skips them if errors are
swallowed), which is exactly the "~15% of studies fail" trap. `read_frames` tries every available
decoder in turn and raises `DecodeError` only if all of them fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

# Header tags we keep (all are in the competition's allowlist of 86 tags or standard geometry).
_HEADER_TAGS = [
    "SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns",
    "PhotometricInterpretation", "SeriesDescription", "NumberOfFrames",
]

_PYDICOM_MAJOR = int(pydicom.__version__.split(".")[0])


class DecodeError(RuntimeError):
    pass


@dataclass
class SliceHeader:
    path: str
    instance_number: int | None
    position: np.ndarray | None      # ImagePositionPatient (3,)
    orientation: np.ndarray | None   # ImageOrientationPatient (6,)
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    rows: int
    columns: int
    transfer_syntax: str
    series_description: str


def _floats(value, n: int) -> np.ndarray | None:
    try:
        arr = np.asarray([float(v) for v in value], dtype=np.float64)
        return arr if arr.shape == (n,) and np.isfinite(arr).all() else None
    except (TypeError, ValueError):
        return None


def read_header(path: str | Path) -> SliceHeader:
    """Read geometry/metadata only (no pixel data), which is fast even for compressed files."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                         specific_tags=_HEADER_TAGS + ["TransferSyntaxUID"])
    spacing = _floats(ds.get("PixelSpacing", None) or [], 2)
    try:
        instance = int(ds.get("InstanceNumber")) if ds.get("InstanceNumber") is not None else None
    except (TypeError, ValueError):
        instance = None
    try:
        thickness = float(ds.get("SliceThickness")) if ds.get("SliceThickness") is not None else None
    except (TypeError, ValueError):
        thickness = None
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    return SliceHeader(
        path=str(path),
        instance_number=instance,
        position=_floats(ds.get("ImagePositionPatient", None) or [], 3),
        orientation=_floats(ds.get("ImageOrientationPatient", None) or [], 6),
        pixel_spacing=tuple(spacing) if spacing is not None else None,
        slice_thickness=thickness,
        rows=int(ds.get("Rows", 0) or 0),
        columns=int(ds.get("Columns", 0) or 0),
        transfer_syntax=str(ts) if ts else "unknown",
        series_description=str(ds.get("SeriesDescription", "") or ""),
    )


def _decoders():
    """Yield (name, fn(ds) -> ndarray) in preference order, for whatever is installed."""
    yield "default", lambda ds: ds.pixel_array
    if _PYDICOM_MAJOR >= 3:
        from pydicom.pixels import pixel_array as _pixel_array

        for plugin in ("pylibjpeg", "gdcm", "pillow"):
            yield plugin, lambda ds, p=plugin: _pixel_array(ds, decoding_plugin=p)
    else:  # pydicom 2.x: force one handler at a time
        from pydicom import config

        handlers = {h.__name__.split(".")[-1]: h for h in config.pixel_data_handlers}
        for name in ("pylibjpeg_handler", "gdcm_handler", "pillow_handler"):
            if name in handlers:
                def decode(ds, h=handlers[name]):
                    saved = config.pixel_data_handlers
                    config.pixel_data_handlers = [h]
                    try:
                        ds._pixel_array = None
                        return ds.pixel_array
                    finally:
                        config.pixel_data_handlers = saved
                yield name, decode


def read_frames(path: str | Path) -> np.ndarray:
    """Decode one DICOM file to float32 (F, H, W) with rescale and MONOCHROME1 applied.

    F == 1 for ordinary single-slice files (the competition layout is one slice per file).
    """
    ds = pydicom.dcmread(str(path), force=True)
    if "PixelData" not in ds:
        raise DecodeError(f"{path}: no pixel data")
    if not hasattr(ds, "file_meta") or "TransferSyntaxUID" not in ds.file_meta:
        # Some stripped files lack the meta header; assume the common uncompressed default.
        ds.file_meta = getattr(ds, "file_meta", pydicom.dataset.FileMetaDataset())
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    errors = []
    arr = None
    for name, decode in _decoders():
        try:
            arr = decode(ds)
            break
        except Exception as exc:  # try the next decoder
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
    if arr is None:
        raise DecodeError(f"{path} ({ds.file_meta.TransferSyntaxUID}): " + " | ".join(errors))

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and int(ds.get("SamplesPerPixel", 1)) > 1:
        arr = arr[..., :3].mean(axis=-1)  # RGB-encoded greyscale; rare for MRI
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]

    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    if slope != 1 or intercept != 0:
        arr = arr * slope + intercept
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def available_decoders() -> list[str]:
    """Names of the decoding back-ends importable in this environment (for the setup check)."""
    names = []
    for module in ("pylibjpeg", "libjpeg", "openjpeg", "gdcm", "PIL"):
        try:
            __import__(module)
            names.append(module)
        except ImportError:
            pass
    return names
