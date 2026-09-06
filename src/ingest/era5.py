# download ERA5 environmental fields over the North Indian Ocean

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# this module is both imported and run directly as a script. When run directly,
# `src` is not yet on the path, so the sibling import below would fail before
# any __main__ block could fix it.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.ibtracs import NIO_BOX

# the download box is the basin box plus a margin, because every predictor is
# an area statistic in a storm-relative frame and the widest of them reaches
# 800 km (~7.2 degrees) from the centre. Without the margin, a storm near the
# northern edge of the basin gets its annulus clipped and its "environmental"
# shear is measured only to the south - a directional bias falling hardest on
# northern Bay of Bengal systems, which are the landfalling ones that matter
# most. Widening costs about 0.9 GB and no extra time, since CDS wall time is
# dominated by queue latency rather than bytes.
ANNULUS_MARGIN_DEG = 8.0

_north, _west, _south, _east = NIO_BOX
# [north, west, south, east], as CDS expects.
AREA = [
    _north + ANNULUS_MARGIN_DEG,
    _west - ANNULUS_MARGIN_DEG,
    _south - ANNULUS_MARGIN_DEG,
    _east + ANNULUS_MARGIN_DEG,
]
GRID = [1.0, 1.0]
SYNOPTIC_TIMES = ["00:00", "06:00", "12:00", "18:00"]

PRESSURE_LEVELS = ["200", "500", "700", "850"]

# CDS enforces a per-request cost limit. Measured empirically against the live
# API: one variable across four levels for a full season (2,240 fields) is
# accepted, while three variables together (6,720 fields) is refused as
# "cost limits exceeded". So requests are issued one variable at a time, with a
# per-month fallback for unusually busy seasons, and merged locally afterwards.
#
# humidity is only read at 500 and 700 hPa, so it is not requested at four.
PRESSURE_REQUESTS = [
    ("u_component_of_wind", PRESSURE_LEVELS),
    ("v_component_of_wind", PRESSURE_LEVELS),
    ("relative_humidity", ["500", "700"]),
]
SHORT = {"u_component_of_wind": "u", "v_component_of_wind": "v",
         "relative_humidity": "r"}

SINGLE_VARIABLES = [
    "sea_surface_temperature",
    "total_column_water_vapour",
    "mean_sea_level_pressure",
]


def storm_calendar(track: pd.DataFrame) -> dict[int, dict[str, list[str]]]:
    # map each season to the months and days on which any storm was active
    calendar: dict[int, dict[str, list[str]]] = {}
    for year, group in track.groupby(track["ISO_TIME"].dt.year):
        calendar[int(year)] = {
            "months": sorted({f"{m:02d}" for m in group["ISO_TIME"].dt.month.unique()}),
            "days": sorted({f"{d:02d}" for d in group["ISO_TIME"].dt.day.unique()}),
        }
    return calendar


def _request(client, dataset: str, payload: dict, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 0:
        print(f"  skip  {target.name} (already downloaded)")
        return True
    print(f"  fetch {target.name} ...", flush=True)
    try:
        client.retrieve(dataset, payload, str(target))
        print(f"  done  {target.name}  ({target.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue to next year
        print(f"  FAIL  {target.name}: {str(exc)[:300]}")
        if target.exists():
            target.unlink()
        return False


def _merge_to(parts: list[Path], final: Path) -> bool:
    # combine part files into one dataset, then remove the parts
    import xarray as xr

    try:
        loaded = []
        for p in parts:
            with xr.open_dataset(p) as ds:
                loaded.append(ds.load())
        merged = xr.merge(loaded, compat="override")
        merged.to_netcdf(final)
        merged.close()
        for ds in loaded:
            ds.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  MERGE FAIL {final.name}: {str(exc)[:200]}")
        return False

    for p in parts:
        p.unlink(missing_ok=True)
    print(f"  merged {final.name}  ({final.stat().st_size / 1e6:.1f} MB)")
    return True


def _pressure_year(client, year: int, spec: dict, out_dir: Path) -> bool:
    final = out_dir / f"era5_pl_{year}.nc"
    if final.exists() and final.stat().st_size > 0:
        print(f"  skip  {final.name} (already downloaded)")
        return True

    base = {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "time": SYNOPTIC_TIMES,
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    parts: list[Path] = []

    for variable, levels in PRESSURE_REQUESTS:
        tag = SHORT[variable]
        part = out_dir / f"_pl_{year}_{tag}.nc"
        payload = {**base, "variable": [variable], "pressure_level": levels,
                   "month": spec["months"], "day": spec["days"]}

        if _request(client, "reanalysis-era5-pressure-levels", payload, part):
            parts.append(part)
            continue

        # busy season - retry this variable one month at a time.
        print(f"  retrying {tag} {year} month by month")
        monthly: list[Path] = []
        for month in spec["months"]:
            mp = out_dir / f"_pl_{year}_{tag}_{month}.nc"
            if _request(client, "reanalysis-era5-pressure-levels",
                        {**payload, "month": [month]}, mp):
                monthly.append(mp)
        if not monthly or not _merge_to(monthly, part):
            return False
        parts.append(part)

    return _merge_to(parts, final)


def _base_payload(year: int, spec: dict) -> dict:
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": spec["months"],
        "day": spec["days"],
        "time": SYNOPTIC_TIMES,
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _plan(calendar: dict, targets: list[int], out_dir: Path) -> list[dict]:
    # One task per (season, variable)
    tasks = []
    for year in sorted(targets, reverse=True):
        spec = calendar[year]
        if not (out_dir / f"era5_pl_{year}.nc").exists():
            for variable, levels in PRESSURE_REQUESTS:
                part = out_dir / f"_pl_{year}_{SHORT[variable]}.nc"
                if not part.exists():
                    tasks.append({
                        "dataset": "reanalysis-era5-pressure-levels",
                        "payload": {**_base_payload(year, spec),
                                    "variable": [variable], "pressure_level": levels},
                        "target": part, "year": year, "spec": spec,
                        "label": f"{year} pl:{SHORT[variable]}",
                    })
        sl = out_dir / f"era5_sl_{year}.nc"
        if not sl.exists():
            tasks.append({
                "dataset": "reanalysis-era5-single-levels",
                "payload": {**_base_payload(year, spec), "variable": SINGLE_VARIABLES},
                "target": sl, "year": year, "spec": spec, "label": f"{year} sl",
            })
    return tasks


def _run_task(task: dict) -> bool:
    # each worker builds its own client - cdsapi.Client is not thread-safe
    import cdsapi

    client = cdsapi.Client(quiet=True, progress=False)
    if _request(client, task["dataset"], task["payload"], task["target"]):
        return True

    # busy season: retry this variable one month at a time.
    if "pressure_level" not in task["payload"]:
        return False
    monthly = []
    for month in task["spec"]["months"]:
        mp = task["target"].with_name(f"{task['target'].stem}_{month}.nc")
        if _request(client, task["dataset"], {**task["payload"], "month": [month]}, mp):
            monthly.append(mp)
    return bool(monthly) and _merge_to(monthly, task["target"])


def download(track: pd.DataFrame, out_dir: Path, years: list[int] | None = None,
             workers: int = 4) -> None:
    # fetch every season, several CDS jobs in flight at once
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    calendar = storm_calendar(track)
    targets = sorted(calendar) if years is None else [y for y in sorted(calendar) if y in years]
    tasks = _plan(calendar, targets, out_dir)

    print(f"ERA5 download - {len(targets)} seasons, box {AREA}, grid {GRID[0]} deg")
    print(f"{len(tasks)} requests outstanding, {workers} concurrent\n")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            label = futures[fut]["label"]
            try:
                got = fut.result()
            except Exception as exc:  # noqa: BLE001
                got, exc_msg = False, str(exc)[:150]
                print(f"  ERROR {label}: {exc_msg}")
            done, failed = (done + 1, failed) if got else (done, failed + 1)
            print(f"  [{done + failed}/{len(tasks)}] {label} "
                  f"{'ok' if got else 'FAILED'}", flush=True)

    # merge the per-variable parts once everything for a season has landed.
    print("\nmerging seasons ...")
    merged = 0
    for year in sorted(targets, reverse=True):
        final = out_dir / f"era5_pl_{year}.nc"
        if final.exists():
            continue
        parts = [out_dir / f"_pl_{year}_{SHORT[v]}.nc" for v, _ in PRESSURE_REQUESTS]
        if all(p.exists() for p in parts) and _merge_to(parts, final):
            merged += 1

    print(f"\nfinished: {done} requests ok, {failed} failed, {merged} seasons merged")


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / "src"))
    from ingest.ibtracs import build

    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print("usage: python src/ingest/era5.py [YEAR ...]   (no years = all seasons)")
        raise SystemExit(0)
    years = [int(a) for a in args] or None
    track = build(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    download(track, ROOT / "data" / "era5", years)
