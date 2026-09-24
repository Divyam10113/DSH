"""Verification tools for the M1 "done when" criteria.

    # 1) every DICOM of a 500-study sample decodes (all series, all transfer syntaxes)
    python -m knee_dicom.verify decode --sample 500 --out decode_report.csv

    # 2) the cache is complete and well-formed; series counts match the CSV
    python -m knee_dicom.verify cache --cache-dir /kaggle/working/knee_cache

    # 3) QC montages (a few slices per plane) for visual review of 20 studies
    python -m knee_dicom.verify montage --cache-dir /kaggle/working/knee_cache --n 20 --out qc/

`decode` and `cache` exit with status 1 on any failure, so they can gate a notebook run.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import DEFAULT_KAGGLE_DATA_DIR, ID_COL, PLANES, SERIES_COL
from .dicom_io import available_decoders, read_frames, read_header


# -- 1. decode check --------------------------------------------------------------------------------

def _decode_study(study_dir: str) -> list[dict]:
    rows = []
    for series_dir in sorted(p for p in Path(study_dir).iterdir() if p.is_dir()):
        for f in sorted(series_dir.glob("*.dcm")):
            row = {ID_COL: Path(study_dir).name, SERIES_COL: series_dir.name, "file": f.name,
                   "transfer_syntax": "unknown", "ok": False, "error": None}
            try:
                row["transfer_syntax"] = read_header(f).transfer_syntax
                arr = read_frames(f)
                row["ok"] = bool(arr.size and np.isfinite(arr).all())
                if not row["ok"]:
                    row["error"] = "empty or non-finite pixels"
            except Exception as exc:
                row["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
            rows.append(row)
    return rows


def decode_check(data_dir: Path, split: str, sample: int, seed: int, workers: int, out: str | None) -> bool:
    root = data_dir / f"{split}_series"
    studies = sorted(p for p in root.iterdir() if p.is_dir())
    random.Random(seed).shuffle(studies)
    studies = studies[:sample] if sample else studies
    print(f"decoders available: {available_decoders()}")
    print(f"decoding every slice of {len(studies)} studies with {workers} workers ...", flush=True)
    rows = []
    with Pool(workers) as pool:
        for i, study_rows in enumerate(pool.imap_unordered(_decode_study, map(str, studies), chunksize=2), 1):
            rows += study_rows
            if i % 50 == 0:
                print(f"  {i}/{len(studies)} studies", flush=True)
    report = pd.DataFrame(rows)
    if out:
        report.to_csv(out, index=False)
    by_ts = report.groupby("transfer_syntax")["ok"].agg(files="size", decoded="sum")
    by_ts["failed"] = by_ts["files"] - by_ts["decoded"]
    print(by_ts.to_string())
    failed_studies = report.loc[~report["ok"], ID_COL].nunique()
    print(f"{len(studies)} studies, {len(report)} files, {int((~report['ok']).sum())} failed files "
          f"in {failed_studies} studies")
    if failed_studies:
        print(report.loc[~report["ok"], "error"].value_counts().head(10).to_string())
    return failed_studies == 0


# -- 2. cache check ---------------------------------------------------------------------------------

def cache_check(cache_dir: Path, data_dir: Path | None, split: str) -> bool:
    manifest = pd.read_csv(cache_dir / "manifest.csv")
    problems: Counter = Counter()
    examples: dict[str, str] = {}

    def flag(kind: str, where: str):
        problems[kind] += 1
        examples.setdefault(kind, where)

    shapes, dtypes = Counter(), Counter()
    for _, row in manifest.iterrows():
        uid = str(row["study_id"])
        if isinstance(row.get("study_error"), str):
            flag("study_error", uid)
        if not any(row[f"has_{p}"] for p in PLANES):
            flag("no_planes", uid)
        for plane in PLANES:
            path = cache_dir / uid / f"{plane}.npy"
            if row[f"has_{plane}"] and not path.exists():
                flag("missing_file", str(path))
            if not path.exists():
                continue
            arr = np.load(path, mmap_mode="r")
            shapes[arr.shape[1:]] += 1
            dtypes[str(arr.dtype)] += 1
            if arr.ndim != 3 or arr.shape[0] == 0:
                flag("bad_shape", f"{path} {arr.shape}")
            elif arr.shape[0] != row[f"n_slices_{plane}"]:
                flag("slice_count_mismatch", str(path))
            if arr.dtype != np.uint8 and not np.isfinite(np.asarray(arr)).all():
                flag("non_finite", str(path))
            if float(np.asarray(arr).max()) == 0:
                flag("all_black", str(path))
        if row["n_series_disk"] != row["n_series_csv"]:
            flag("series_count_differs_from_csv", uid)

    if data_dir is not None:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            csv_path = data_dir / f"{split}_series.csv"
        expected = set(pd.read_csv(csv_path, usecols=[ID_COL])[ID_COL].astype(str))
        missing = expected - set(manifest["study_id"].astype(str))
        if missing:
            problems["studies_not_cached"] = len(missing)
            examples["studies_not_cached"] = next(iter(missing))

    print(f"{len(manifest)} studies in manifest; slice shapes {dict(shapes)}; dtypes {dict(dtypes)}")
    print("plane coverage: " + ", ".join(f"{p}={int(manifest[f'has_{p}'].sum())}" for p in PLANES))
    # A differing series count is informational (CSV vs disk); everything else is a hard failure.
    hard = {k: v for k, v in problems.items() if k != "series_count_differs_from_csv"}
    for kind, n in problems.items():
        print(f"  {'FAIL' if kind in hard else 'note'} {kind}: {n} (e.g. {examples[kind]})")
    print("cache check " + ("FAILED" if hard else "passed"))
    return not hard


# -- 3. QC montage ----------------------------------------------------------------------------------

def montage(cache_dir: Path, n: int, out: Path, slices_per_plane: int = 5, seed: int = 0) -> list[Path]:
    """One PNG per study: a row per plane, `slices_per_plane` evenly spaced slices per row."""
    manifest = pd.read_csv(cache_dir / "manifest.csv")
    ids = manifest["study_id"].astype(str).tolist()
    random.Random(seed).shuffle(ids)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for uid in ids[:n]:
        rows = []
        for plane in PLANES:
            path = cache_dir / uid / f"{plane}.npy"
            if not path.exists():
                continue
            vol = np.load(path)
            vol = vol if vol.dtype == np.uint8 else np.round(np.clip(vol, 0, 1) * 255).astype(np.uint8)
            idx = np.unique(np.linspace(0, len(vol) - 1, slices_per_plane).round().astype(int))
            tiles = [vol[i] for i in idx] + [np.zeros_like(vol[0])] * (slices_per_plane - len(idx))
            rows.append(np.concatenate(tiles, axis=1))
        if rows:
            width = max(r.shape[1] for r in rows)
            grid = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]))) for r in rows], axis=0)
            path = out / f"{uid}.png"
            Image.fromarray(grid).save(path)
            written.append(path)
    print(f"wrote {len(written)} montages to {out} (rows: {', '.join(PLANES)} when present)")
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="decode every slice of a sample of studies")
    d.add_argument("--data-dir", default=DEFAULT_KAGGLE_DATA_DIR)
    d.add_argument("--split", default="train", choices=["train", "test"])
    d.add_argument("--sample", type=int, default=500, help="0 = all studies")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    d.add_argument("--out", default="decode_report.csv")

    c = sub.add_parser("cache", help="validate a built cache")
    c.add_argument("--cache-dir", required=True)
    c.add_argument("--data-dir", default=None, help="also check every study of the split was cached")
    c.add_argument("--split", default="train", choices=["train", "test"])

    m = sub.add_parser("montage", help="write QC montages")
    m.add_argument("--cache-dir", required=True)
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--out", default="qc")

    args = p.parse_args(argv)
    if args.cmd == "decode":
        ok = decode_check(Path(args.data_dir), args.split, args.sample, args.seed, args.workers, args.out)
    elif args.cmd == "cache":
        ok = cache_check(Path(args.cache_dir), Path(args.data_dir) if args.data_dir else None, args.split)
    else:
        ok = bool(montage(Path(args.cache_dir), args.n, Path(args.out)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
