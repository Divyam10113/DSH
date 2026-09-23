"""Sequence routing: choose which series of a study represents each anatomical plane.

A study typically has several series per plane (e.g. sagittal PD fat-sat + sagittal T1). The image
model gets one series per plane, so we route each plane to the most informative series for knee
findings. Fluid-sensitive fat-suppressed sequences (PD-FS / T2-FS / STIR) are the workhorse for tears,
bone bruises, effusion and cysts, so they are preferred:

    fluid + fat-sat  >  fluid  >  fat-sat  >  other (T1 etc.)

Ties go to the series whose slice count is closest to a typical full-coverage series. Optionally, the
best *non*-fluid series (usually T1, good for fracture lines and anatomy) is routed to "<plane>_t1".
"""

from __future__ import annotations

import pandas as pd

from .constants import FATSAT_COL, FLUID_COL, PLANE_COL, PLANES, SERIES_COL

TYPICAL_SLICES = 30  # the competition's median series length


def _flag(value) -> int:
    try:
        return int(float(value) == 1)
    except (TypeError, ValueError):
        return 0


def normalise_plane(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in PLANES else None


def route_study(series: pd.DataFrame, n_slices: dict[str, int] | None = None,
                extra_t1: bool = False) -> dict[str, str]:
    """Map output key ("sagittal", ..., optionally "sagittal_t1") -> SeriesInstanceUID.

    `series` holds one study's rows of *_series.csv, with a "plane" column already normalised
    (the CSV plane, or the plane inferred from DICOM orientation when the CSV value is missing).
    `n_slices` (series uid -> file count) is used for tie-breaking when available.
    """
    n_slices = n_slices or {}
    routes = {}
    for plane in PLANES:
        rows = series[series["plane"] == plane]
        if rows.empty:
            continue

        def rank(row):
            fluid, fatsat = _flag(row[FLUID_COL]), _flag(row[FATSAT_COL])
            k = n_slices.get(row[SERIES_COL], TYPICAL_SLICES)
            return (fluid * 2 + fatsat, -abs(k - TYPICAL_SLICES), k)

        ranked = sorted((rank(r), r[SERIES_COL]) for _, r in rows.iterrows())
        routes[plane] = ranked[-1][1]
        if extra_t1:
            non_fluid = [(score, uid) for score, uid in ranked if score[0] < 2]
            if non_fluid:
                routes[f"{plane}_t1"] = non_fluid[-1][1]
    return routes


def attach_planes(series: pd.DataFrame) -> pd.DataFrame:
    """Add a normalised "plane" column from Anatomical_Plane (None where missing/unknown)."""
    series = series.copy()
    series["plane"] = series[PLANE_COL].map(normalise_plane) if PLANE_COL in series else None
    return series
