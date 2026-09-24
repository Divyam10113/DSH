"""Load one MRI series (a folder of single-slice DICOMs) into a correctly ordered 3D volume."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dicom_io import DecodeError, SliceHeader, read_frames, read_header


@dataclass
class Volume:
    pixels: np.ndarray                    # float32 (K, H, W), slices in anatomical order
    headers: list[SliceHeader]            # one per kept slice, same order
    inferred_plane: str | None            # from ImageOrientationPatient, if available
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count of skipped files

    @property
    def spacing(self) -> tuple[float, float] | None:
        return self.headers[0].pixel_spacing if self.headers else None


def plane_from_orientation(orientation: np.ndarray | None) -> str | None:
    """Sagittal / coronal / axial from the slice normal (row x column direction cosines).

    Patient axes: x = right->left, y = anterior->posterior, z = feet->head. The dominant component
    of the normal tells which plane the slices lie in.
    """
    if orientation is None:
        return None
    normal = np.cross(orientation[:3], orientation[3:])
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-6:
        return None
    return ("sagittal", "coronal", "axial")[int(np.argmax(np.abs(normal)))]


def _sort_key(headers: list[SliceHeader]) -> list[float]:
    """Order slices by position along the slice normal; fall back to InstanceNumber, then name.

    The normal's sign depends on how the scanner wrote the orientation, so it is flipped to point
    along the positive patient axis. Every study then runs the same way (sagittal right->left,
    coronal anterior->posterior, axial feet->head), whatever the scanner.
    """
    ref = next((h.orientation for h in headers if h.orientation is not None), None)
    if ref is not None and all(h.position is not None for h in headers):
        normal = np.cross(ref[:3], ref[3:])
        if normal[int(np.argmax(np.abs(normal)))] < 0:
            normal = -normal
        return [float(np.dot(h.position, normal)) for h in headers]
    if all(h.instance_number is not None for h in headers):
        return [float(h.instance_number) for h in headers]
    return list(range(len(headers)))  # headers are already sorted by filename


def load_series(series_dir: str | Path, decode: bool = True) -> Volume:
    """Read every .dcm in `series_dir` and return them as an ordered volume.

    Robustness rules (each logged in `Volume.dropped`):
      * files whose header can't be read are skipped ("bad_header");
      * slices whose size differs from the series' majority size are skipped ("off_size"),
        e.g. a localizer that slipped into the folder;
      * duplicate slice positions keep the first file ("duplicate_position");
      * files that fail to decode with every back-end are skipped ("decode_failed").
    Raises DecodeError if no slice survives.
    """
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        raise DecodeError(f"{series_dir}: no .dcm files")
    dropped: Counter = Counter()

    headers = []
    for f in files:
        try:
            headers.append(read_header(f))
        except Exception:
            dropped["bad_header"] += 1
    if not headers:
        raise DecodeError(f"{series_dir}: no readable headers")

    size = Counter((h.rows, h.columns) for h in headers).most_common(1)[0][0]
    kept = [h for h in headers if (h.rows, h.columns) == size]
    dropped["off_size"] += len(headers) - len(kept)

    keys = _sort_key(kept)
    order = np.argsort(keys, kind="stable")
    ordered, seen = [], set()
    for i in order:
        key = round(keys[i], 3)
        if key in seen and kept[i].position is not None:
            dropped["duplicate_position"] += 1
            continue
        seen.add(key)
        ordered.append(kept[i])

    plane = plane_from_orientation(next((h.orientation for h in ordered if h.orientation is not None), None))
    if not decode:
        return Volume(np.zeros((0, 0, 0), np.float32), ordered, plane, dict(dropped))

    slices, good = [], []
    for h in ordered:
        try:
            frames = read_frames(h.path)
        except Exception:
            dropped["decode_failed"] += 1
            continue
        for frame in frames:
            if frame.shape != size and size != (0, 0):
                dropped["off_size"] += 1
                continue
            slices.append(frame)
            good.append(h)
    if not slices:
        raise DecodeError(f"{series_dir}: no slice could be decoded ({dict(dropped)})")
    return Volume(np.stack(slices).astype(np.float32), good, plane, {k: v for k, v in dropped.items() if v})
