# GridSat-B1 global infrared imagery for tasks T1 and T2

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.ibtracs import NIO_BOX

BASE = ("https://www.ncei.noaa.gov/data/"
        "geostationary-ir-channel-brightness-temperature-gridsat-b1/access")
VARS = ["irwin_cdr", "irwvp"]
GRIDSAT_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

# A margin beyond the basin so a storm near the edge still has full context in
# any crop taken around it.
MARGIN_DEG = 6.0
# measured, not guessed: one cropped, compressed timestep on disk.
MB_PER_CROP = 6.1
_north, _west, _south, _east = NIO_BOX
LAT_RANGE = (_south - MARGIN_DEG, _north + MARGIN_DEG)
LON_RANGE = (_west - MARGIN_DEG, _east + MARGIN_DEG)


def url_for(when: pd.Timestamp) -> str:
    return (f"{BASE}/{when.year}/GRIDSAT-B1."
            f"{when.year}.{when.month:02d}.{when.day:02d}.{when.hour:02d}"
            f".v02r01.nc")


def out_path(root: Path, when: pd.Timestamp) -> Path:
    return (root / f"{when.year}" /
            f"nio_{when.year}{when.month:02d}{when.day:02d}_{when.hour:02d}.nc")


def snap_to_grid(times: pd.Series) -> pd.DatetimeIndex:
    # round observation times to the nearest 3-hourly GridSat slot
    t = pd.DatetimeIndex(pd.to_datetime(times))
    return pd.DatetimeIndex(t.round("3h")).unique().sort_values()


def schedule(adt_csv: Path | None, track_df: pd.DataFrame | None,
             every: int = 1) -> pd.DatetimeIndex:
    # timestamps worth downloading, from ADT scenes and/or best track
    stamps = []
    if adt_csv and Path(adt_csv).exists():
        adt = pd.read_csv(adt_csv, parse_dates=["time"])
        stamps.append(snap_to_grid(adt["time"]))
    if track_df is not None and len(track_df):
        stamps.append(snap_to_grid(track_df["ISO_TIME"]))
    if not stamps:
        return pd.DatetimeIndex([])

    all_t = pd.DatetimeIndex(np.unique(np.concatenate([s.values for s in stamps])))
    all_t = all_t[np.isin(all_t.hour, GRIDSAT_HOURS)].sort_values()
    return all_t[::every]


def fetch_one(when: pd.Timestamp, root: Path,
              session: requests.Session | None = None) -> str:
    # download one global file, crop to the basin, drop the original
    target = out_path(root, when)
    if target.exists() and target.stat().st_size > 0:
        return "cached"

    sess = session or requests
    tmp = None
    try:
        with sess.get(url_for(when), timeout=600, stream=True) as r:
            if r.status_code != 200:
                return f"http {r.status_code}"
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
                tmp = Path(fh.name)
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

        with xr.open_dataset(tmp) as ds:
            keep = [v for v in VARS if v in ds.variables]
            sub = ds[keep].sel(
                lat=slice(*LAT_RANGE), lon=slice(*LON_RANGE)).load()
            sub.attrs["source"] = "GridSat-B1 v02r01 (NOAA NCEI)"
            target.parent.mkdir(parents=True, exist_ok=True)
            # compress: brightness temperature fields shrink a long way.
            enc = {v: {"zlib": True, "complevel": 4} for v in keep}
            sub.to_netcdf(target, encoding=enc)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        if target.exists():
            target.unlink(missing_ok=True)
        return f"fail {str(exc)[:70]}"
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def download(times: pd.DatetimeIndex, root: Path, workers: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    todo = [t for t in times if not out_path(root, t).exists()]
    print(f"{len(times):,} timesteps requested, {len(todo):,} still to fetch")
    if not todo:
        return
    print(f"~{len(todo) * 44 / 1024:.1f} GB to transfer, "
          f"~{len(todo) * MB_PER_CROP / 1024:.1f} GB kept on disk\n")

    done = failed = 0
    sess = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, t, root, sess): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if status in ("ok", "cached"):
                done += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  {futures[fut]:%Y-%m-%d %H}Z  {status}")
            if i % 20 == 0:
                print(f"  [{i:,}/{len(todo):,}] {done:,} ok, {failed:,} failed",
                      end="\r", flush=True)
    print(f"\nfinished: {done:,} on disk, {failed:,} failed")


if __name__ == "__main__":
    import argparse

    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Download GridSat-B1 over the NIO")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--every", type=int, default=2,
                    help="keep every Nth 3-hourly slot (2 = 6-hourly)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from ingest.ibtracs import build as build_track

    track = build_track(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    track = track[(track["SEASON"] >= args.from_year)
                  & (track["SEASON"] <= args.to_year)]
    times = schedule(ROOT / "data" / "adt" / "adt_nio.csv", track, args.every)
    times = times[(times.year >= args.from_year) & (times.year <= args.to_year)]

    print(f"GridSat-B1 - North Indian Ocean {args.from_year}-{args.to_year}")
    print(f"  box lat {LAT_RANGE[0]:.0f}..{LAT_RANGE[1]:.0f}  "
          f"lon {LON_RANGE[0]:.0f}..{LON_RANGE[1]:.0f}")
    print(f"  {len(times):,} timesteps at "
          f"{'3-hourly' if args.every == 1 else f'{3 * args.every}-hourly'}\n")

    if args.dry_run:
        print(f"~{len(times) * 44 / 1024:.1f} GB transfer, "
              f"~{len(times) * MB_PER_CROP / 1024:.1f} GB on disk")
        raise SystemExit(0)

    download(times, ROOT / "data" / "gridsat", args.workers)
