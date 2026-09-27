# The split window for every INSAT patch we already have.
#
#     python src/build_insat_split_window.py
#
# TIR1 minus TIR2. Both channels see the same cloud, but the 12 micron window is
# slightly more transparent to ice, so thin cirrus lets it read warmer while
# opaque deep convection looks the same to both. The difference is therefore a
# direct measure of how optically thick a cloud top is, which is exactly the
# judgement that separates an embedded centre from an irregular central dense
# overcast - the two scene classes that collapse when the classifier reads INSAT
# imagery instead of GridSat (0.69 to 0.47 and 0.58 to 0.39).
#
# Writes data/processed/scenes_insat/split_window.npy, aligned row for row with
# patches.csv, plus a few summary statistics per patch for the physics half of
# the hybrid. Nothing existing is modified.

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "processed" / "scenes_insat"
GRID = ROOT / "data" / "insat" / "grid"
AUX = ROOT / "data" / "insat" / "aux"
PATCH_PX = 224
OUT = CACHE / "split_window.npy"
STATS = CACHE / "split_window_stats.csv"

# a split window difference runs from about -2 K in thick convection to +6 K or
# more through thin cirrus; this is the range we scale into the byte image
SW_LO, SW_HI = -2.0, 8.0


def patch_at(ds_ir: xr.Dataset, ds_aux: xr.Dataset, lat: float, lon: float):
    lats, lons = ds_ir["lat"].values, ds_ir["lon"].values
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon).argmin())
    half = PATCH_PX // 2
    if i - half < 0 or i + half > len(lats) or j - half < 0 or j + half > len(lons):
        return None
    sl = (slice(i - half, i + half), slice(j - half, j + half))
    t1 = ds_ir["irwin_cdr"].isel(time=0).values[sl]
    t2 = ds_aux["tir2"].isel(time=0).values[sl]
    if not np.isfinite(t1).any() or not np.isfinite(t2).any():
        return None
    return t1 - t2


def main() -> None:
    from vision.detect import gridsat_path

    meta = pd.read_csv(CACHE / "patches.csv")
    meta["time"] = pd.to_datetime(meta["time"])
    print(f"{len(meta)} INSAT patches to match")

    arr = np.zeros((len(meta), PATCH_PX, PATCH_PX), dtype=np.uint8)
    rows, missing, t0 = [], 0, time.time()
    for n, r in enumerate(meta.itertuples(), 1):
        # patches.csv keeps the ADT analysis time; the scene files are named by
        # the 3-hourly slot the builder rounded it to, so round the same way
        slot = pd.Timestamp(r.time).round("3h").to_pydatetime()
        f_ir, f_aux = gridsat_path(GRID, slot), gridsat_path(AUX, slot)
        sw = None
        if f_ir.exists() and f_aux.exists():
            with xr.open_dataset(f_ir) as a, xr.open_dataset(f_aux) as b:
                sw = patch_at(a, b, float(r.lat), float(r.lon))
        if sw is None:
            missing += 1
            rows.append({"sw_mean": np.nan, "sw_core": np.nan, "sw_ring": np.nan,
                         "sw_thin_frac": np.nan, "sw_std": np.nan})
            continue

        scaled = np.clip((sw - SW_LO) / (SW_HI - SW_LO), 0, 1)
        arr[n - 1] = (np.nan_to_num(scaled, nan=0.5) * 255).astype(np.uint8)

        # storm-relative summaries, the same frame the cold-cloud stats use
        m = PATCH_PX
        yy, xx = np.mgrid[0:m, 0:m]
        rad = np.hypot(yy - m / 2, xx - m / 2) / (m / 2)
        core, ring = sw[rad <= .15], sw[(rad > .15) & (rad <= .4)]
        rows.append({
            "sw_mean": float(np.nanmean(sw)),
            "sw_core": float(np.nanmean(core)) if core.size else np.nan,
            "sw_ring": float(np.nanmean(ring)) if ring.size else np.nan,
            # how much of the patch is semi-transparent cloud rather than
            # opaque tops: the cirrus fraction, in one number
            "sw_thin_frac": float(np.nanmean(sw > 2.0)),
            "sw_std": float(np.nanstd(sw)),
        })
        if n % 100 == 0 or n == len(meta):
            print(f"  {n}/{len(meta)}  {time.time() - t0:.0f}s  missing {missing}",
                  flush=True)

    np.save(OUT, arr)
    stats = pd.concat([meta[["storm", "time", "scene"]].reset_index(drop=True),
                       pd.DataFrame(rows)], axis=1)
    stats.to_csv(STATS, index=False)
    print(f"\nwrote {OUT.name} and {STATS.name}; {missing} patches had no aux file")

    ok = stats.dropna(subset=["sw_mean"])
    print(f"\nsplit window by scene class ({len(ok)} patches):")
    print(f"  {'scene':<10}{'n':>5}{'mean K':>9}{'core':>8}{'ring':>8}{'thin frac':>11}")
    for scene, g in ok.groupby("scene"):
        print(f"  {scene:<10}{len(g):>5}{g.sw_mean.mean():>9.2f}{g.sw_core.mean():>8.2f}"
              f"{g.sw_ring.mean():>8.2f}{g.sw_thin_frac.mean():>11.2f}")
    print("\nif EMBC and IRRCDO separate here, the channel carries what the")
    print("classifier was missing. if they do not, it does not.")


if __name__ == "__main__":
    main()
