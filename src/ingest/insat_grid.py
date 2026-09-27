"""put the downloaded INSAT archive on the model grid, one file per GridSat slot.

    python src/ingest/insat_grid.py [--workers 4]
"""

# reads the manifests insat_archive.py wrote, so each output file carries the
# GridSat slot it was matched to rather than the scan's own minute. output is
# data/insat/grid/<year>/nio_YYYYMMDD_HH.nc, the same naming as GridSat, which
# is what lets every training script and endpoint read it as source="insat".

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "insat" / "raw"
GRID = ROOT / "data" / "insat" / "grid"
SATELLITE = {"3RIMG": "INSAT-3DR", "3DIMG": "INSAT-3D", "3SIMG": "INSAT-3DS"}


def _one(raw: str, slot_iso: str) -> tuple[str, str]:
    from ingest.insat_regrid import to_gridsat
    from vision.detect import gridsat_path

    slot = datetime.fromisoformat(slot_iso)
    out = gridsat_path(GRID, slot)
    if out.exists():
        return slot_iso, "exists"
    ds = to_gridsat(raw)
    ds = ds.assign_coords(time=[np.datetime64(slot, "s")])
    ds.attrs["satellite"] = SATELLITE.get(Path(raw).name[:5], "INSAT")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("irwin_cdr", "irwvp")}
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(out)
    return slot_iso, "written"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--satellites", default="3R,3D",
                    help="which archive manifests to read, e.g. 3R or 3R,3D")
    args = ap.parse_args()

    jobs = []
    for sat in args.satellites.split(","):
        manifest = ROOT / "data" / "insat" / f"manifest_{sat.strip()}.json"
        if not manifest.exists():
            continue
        for slot, entry in json.loads(manifest.read_text()).items():
            if entry and (RAW / entry["identifier"]).exists():
                jobs.append((str(RAW / entry["identifier"]), slot))
    print(f"{len(jobs)} granules with a matched slot on disk", flush=True)

    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, raw, slot): slot for raw, slot in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  failed {futures[fut]}: {str(exc)[:120]}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {time.time() - t0:.0f}s  failed {failed}", flush=True)


if __name__ == "__main__":
    main()
