"""Build the preprocessed cache: raw DICOM studies -> {out}/{StudyInstanceUID}/{plane}.npy.

    python -m knee_dicom.cache --split train --out-dir /kaggle/working/knee_cache \\
        [--img-size 128] [--max-slices 32] [--workers 4] [--shard 0 --num-shards 4] [--limit 500]

The layout is exactly what M3's `KneeMRIDataset` reads: one (K, H, W) array per plane per study.
Each study also gets a `meta.json`. It is written last, so a study with a meta.json is complete,
and a killed session resumes where it stopped. Run with different --shard values in parallel
notebooks/accounts, then `--aggregate-only` over the merged folders to rebuild the manifest.

Outputs besides the arrays:
    manifest.csv    one row per study: study_id, planes present, slice counts, routed series, errors
    series_log.csv  one row per cached series: routing reason, raw/saved slices, dropped files
    summary.json    counts of studies/series/failures, missing planes, cache size
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_KAGGLE_DATA_DIR, FATSAT_COL, FLUID_COL, ID_COL, PLANE_COL, PLANES, SERIES_COL
from .dicom_io import read_header
from .preprocess import WINDOWS, PreprocessConfig, preprocess_volume
from .routing import attach_planes, route_study
from .series import load_series, plane_from_orientation


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------------------------------------------
# One study (also usable in memory by the offline inference notebook)
# -------------------------------------------------------------------------------------------------

def study_series_table(study_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """The study's series (from CSV and/or disk) with plane, flags and file counts.

    Series present on disk but missing from the CSV are included with flags 0. Missing planes are
    inferred from the DICOM orientation of the series' first file.
    """
    on_disk = {p.name: p for p in study_dir.iterdir() if p.is_dir()} if study_dir.exists() else {}
    table = attach_planes(rows) if len(rows) else pd.DataFrame(columns=[SERIES_COL, PLANE_COL, FLUID_COL, FATSAT_COL, "plane"])
    extra = [uid for uid in on_disk if uid not in set(table[SERIES_COL])]
    if extra:
        table = pd.concat([table, pd.DataFrame({SERIES_COL: extra, FLUID_COL: 0, FATSAT_COL: 0, "plane": None})],
                          ignore_index=True)
    table["on_disk"] = table[SERIES_COL].isin(on_disk)
    table["n_files"] = [len(list(on_disk[u].glob("*.dcm"))) if u in on_disk else 0 for u in table[SERIES_COL]]
    for i, row in table.iterrows():
        if pd.isna(row["plane"]) and row["on_disk"] and row["n_files"]:
            try:
                first = next(iter(sorted(on_disk[row[SERIES_COL]].glob("*.dcm"))))
                table.at[i, "plane"] = plane_from_orientation(read_header(first).orientation)
            except Exception:
                pass
    return table


def preprocess_study(study_dir: str | Path, rows: pd.DataFrame, cfg: PreprocessConfig,
                     extra_t1: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Decode, route and preprocess one study. Returns ({key: (K, S, S) array}, meta).

    Never raises for a bad series: failures are recorded in meta["series"][key]["error"] so one
    corrupt series can't kill a 9-hour run.
    """
    study_dir = Path(study_dir)
    table = study_series_table(study_dir, rows)
    usable = table[table["on_disk"] & (table["n_files"] > 0)]
    routes = route_study(usable, dict(zip(usable[SERIES_COL], usable["n_files"])), extra_t1=extra_t1)

    arrays, series_meta = {}, {}
    for key, series_uid in routes.items():
        row = usable[usable[SERIES_COL] == series_uid].iloc[0]
        info = {"series_uid": series_uid, "fluid": int(row[FLUID_COL] == 1), "fatsat": int(row[FATSAT_COL] == 1),
                "n_files": int(row["n_files"]), "error": None}
        try:
            vol = load_series(study_dir / series_uid)
            arr, keep = preprocess_volume(vol.pixels, cfg, vol.spacing)
            arrays[key] = arr
            info.update(n_raw_slices=int(vol.pixels.shape[0]), n_saved_slices=int(arr.shape[0]),
                        raw_shape=list(vol.pixels.shape[1:]), pixel_spacing=vol.spacing,
                        inferred_plane=vol.inferred_plane, dropped=vol.dropped,
                        transfer_syntaxes=sorted({h.transfer_syntax for h in vol.headers}))
        except Exception as exc:
            info["error"] = f"{exc.__class__.__name__}: {exc}"[:500]
        series_meta[key] = info

    meta = {
        ID_COL: study_dir.name,
        "n_series_csv": int(len(rows)),
        "n_series_disk": int(table["on_disk"].sum()),
        "series_missing_on_disk": table.loc[~table["on_disk"], SERIES_COL].tolist(),
        "series_not_in_csv": int(len(table) - len(rows)),
        "routes": routes,
        "series": series_meta,
        "config": asdict(cfg),
    }
    return arrays, meta


def _save_study(task) -> dict:
    study_uid, study_dir, rows, out_dir, cfg, extra_t1, overwrite = task
    target = Path(out_dir) / study_uid
    meta_path = target / "meta.json"
    if meta_path.exists() and not overwrite:
        return {"skipped": True}
    t0 = time.time()
    try:
        arrays, meta = preprocess_study(study_dir, pd.DataFrame(rows), cfg, extra_t1)
    except Exception as exc:  # e.g. study folder missing entirely
        arrays, meta = {}, {ID_COL: study_uid, "routes": {}, "series": {},
                            "study_error": f"{exc.__class__.__name__}: {exc}"[:500]}
    target.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        tmp = target / f".{key}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, target / f"{key}.npy")
    meta["seconds"] = round(time.time() - t0, 2)
    tmp = target / ".meta.tmp.json"
    tmp.write_text(json.dumps(meta, default=str))
    os.replace(tmp, meta_path)  # written last: its presence marks the study as complete
    return meta


# -------------------------------------------------------------------------------------------------
# Whole split
# -------------------------------------------------------------------------------------------------

def list_studies(data_dir: Path, split: str) -> tuple[list[str], pd.DataFrame]:
    series = pd.read_csv(data_dir / f"{split}_series.csv")
    ids = set(series[ID_COL].astype(str))
    study_csv = data_dir / f"{split}.csv"
    if study_csv.exists():
        ids |= set(pd.read_csv(study_csv, usecols=[ID_COL])[ID_COL].astype(str))
    return sorted(ids), series


def aggregate(out_dir: str | Path) -> dict:
    """Scan every {study}/meta.json and write manifest.csv, series_log.csv and summary.json."""
    out_dir = Path(out_dir)
    studies, series_rows = [], []
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        uid = meta[ID_COL]
        row = {"study_id": uid, ID_COL: uid, "n_series_csv": meta.get("n_series_csv"),
               "n_series_disk": meta.get("n_series_disk"), "study_error": meta.get("study_error")}
        n_errors = 0
        for key, info in meta.get("series", {}).items():
            ok = info.get("error") is None
            n_errors += not ok
            series_rows.append({ID_COL: uid, "key": key, **{k: v for k, v in info.items() if k != "dropped"},
                                "dropped": json.dumps(info.get("dropped") or {})})
            if key in PLANES:
                row[f"has_{key}"] = int(ok)
                row[f"n_slices_{key}"] = info.get("n_saved_slices") if ok else 0
                row[f"{key}_series"] = info.get("series_uid")
        for plane in PLANES:
            row.setdefault(f"has_{plane}", 0)
            row.setdefault(f"n_slices_{plane}", 0)
        row["n_series_errors"] = n_errors
        studies.append(row)

    manifest = pd.DataFrame(studies)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    series_log = pd.DataFrame(series_rows)
    series_log.to_csv(out_dir / "series_log.csv", index=False)

    total_bytes = sum(p.stat().st_size for p in out_dir.glob("*/*.npy"))
    summary = {
        "n_studies": int(len(manifest)),
        "n_series_cached": int((series_log["error"].isna()).sum()) if len(series_log) else 0,
        "n_series_failed": int(series_log["error"].notna().sum()) if len(series_log) else 0,
        "n_study_errors": int(manifest["study_error"].notna().sum()) if len(manifest) else 0,
        "studies_missing_plane": {p: int((manifest[f"has_{p}"] == 0).sum()) for p in PLANES} if len(manifest) else {},
        "studies_all_planes": int(manifest[[f"has_{p}" for p in PLANES]].all(axis=1).sum()) if len(manifest) else 0,
        "cache_gb": round(total_bytes / 1e9, 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_cache(args) -> dict:
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(img_size=args.img_size, max_slices=args.max_slices, window=args.window,
                           keep_aspect=not args.stretch, dtype=args.dtype)

    if not args.aggregate_only:
        ids, series = list_studies(data_dir, args.split)
        ids = ids[args.shard::args.num_shards]
        if args.limit:
            ids = ids[:args.limit]
        by_study = {uid: g.to_dict("records") for uid, g in series.assign(**{ID_COL: series[ID_COL].astype(str)}).groupby(ID_COL)}
        root = data_dir / f"{args.split}_series"
        tasks = [(uid, str(root / uid), by_study.get(uid, []), str(out_dir), cfg, args.extra_t1, args.overwrite)
                 for uid in ids]
        log(f"{len(tasks)} studies to process (shard {args.shard}/{args.num_shards}) with {args.workers} workers; "
            f"config {asdict(cfg)}")

        done = skipped = failed = 0
        t0 = time.time()
        with Pool(args.workers) as pool:
            for meta in pool.imap_unordered(_save_study, tasks, chunksize=1):
                if meta.get("skipped"):
                    skipped += 1
                    continue
                done += 1
                failed += bool(meta.get("study_error")) or any(s.get("error") for s in meta.get("series", {}).values())
                if done % args.log_every == 0:
                    rate = done / (time.time() - t0)
                    left = (len(tasks) - done - skipped) / max(rate, 1e-9)
                    log(f"{done + skipped}/{len(tasks)} studies ({skipped} resumed), {failed} with errors, "
                        f"{rate:.2f} studies/s, ~{left / 60:.0f} min left")
        log(f"processed {done}, resumed {skipped}, with errors {failed}")

    summary = aggregate(out_dir)
    log("summary: " + json.dumps(summary))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_KAGGLE_DATA_DIR)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out-dir", default="/kaggle/working/knee_cache")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--max-slices", type=int, default=32, help="0 keeps every slice")
    p.add_argument("--window", default="standard", choices=sorted(WINDOWS))
    p.add_argument("--dtype", default="uint8", choices=["uint8", "float16"])
    p.add_argument("--stretch", action="store_true", help="stretch to square instead of letterboxing")
    p.add_argument("--extra-t1", action="store_true", help="also cache the best non-fluid series as <plane>_t1")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N studies of this shard")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--aggregate-only", action="store_true", help="just rebuild manifest/summary from meta.json")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args(argv)


if __name__ == "__main__":
    build_cache(parse_args())
