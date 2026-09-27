"""put the INSAT channels the first pass discarded onto the model grid.

    python src/ingest/insat_aux_grid.py [--workers 4]

Every L1C granule carries six channels and the first regrid kept two: the 10.8
micron window and the 6.8 micron water vapour. The other four have been sitting
on disk unused ever since. This writes two of them - 12.0 micron TIR2 and 3.9
micron MIR - to a parallel tree, data/insat/aux/, one file per GridSat slot with
the same naming as everything else.

Why TIR2 in particular: TIR1 minus TIR2 is the split window difference, which
separates optically thick convective cloud from thin cirrus. Thin cirrus over an
eyewall is the mechanism behind two failures we have measured and not fixed - the
severe storm under-read, and the collapse of the EMBC and IRRCDO scene classes on
INSAT imagery. A single window channel cannot see through it and two can.

Nothing here touches data/insat/grid/. The existing files, and every model and
endpoint that reads them, are left exactly as they are; this is additive, and if
the extra channels turn out not to help, the directory is deleted and nothing
else changes.
"""

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
AUX = ROOT / "data" / "insat" / "aux"
SATELLITE = {"3RIMG": "INSAT-3DR", "3DIMG": "INSAT-3D", "3SIMG": "INSAT-3DS"}


def _one(raw: str, slot_iso: str) -> tuple[str, str]:
    from ingest.insat_regrid import AUX_CHANNELS, to_gridsat_aux
    from vision.detect import gridsat_path

    slot = datetime.fromisoformat(slot_iso)
    out = gridsat_path(AUX, slot)
    if out.exists():
        return slot_iso, "exists"
    ds = to_gridsat_aux(raw)
    ds = ds.assign_coords(time=[np.datetime64(slot, "s")])
    ds.attrs["satellite"] = SATELLITE.get(Path(raw).name[:5], "INSAT")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in AUX_CHANNELS}
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(out)
    return slot_iso, "written"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--satellites", default="3R,3D")
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

    t0, done, failed, written = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, raw, slot): slot for raw, slot in jobs}
        for fut in as_completed(futures):
            try:
                _, how = fut.result()
                written += how == "written"
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  failed {futures[fut]}: {str(exc)[:120]}", flush=True)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {time.time() - t0:.0f}s  "
                      f"written {written}  failed {failed}", flush=True)
    print(f"\ndone: {written} written, {failed} failed, into {AUX}", flush=True)


if __name__ == "__main__":
    main()
