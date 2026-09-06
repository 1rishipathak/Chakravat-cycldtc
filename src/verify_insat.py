# run the trained models on INSAT-3DR imagery they have never seen

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dataset import load_dataset                      # noqa: E402
from ingest.ibtracs import imd_category               # noqa: E402
from ingest.insat_regrid import to_gridsat, valid_bounds   # noqa: E402
from vision.scenes import extract_patch               # noqa: E402

INSAT_RAW = ROOT / "data" / "insat" / "raw"
GRIDSAT = ROOT / "data" / "gridsat"

# Amphan, 18 May 2020 - a 125 kt system, the hardest end of the range.
STORM_SID = "2020136N10088"
CASES = [
    ("GridSat  8 km", "2020-05-18 09:00", GRIDSAT / "2020/nio_20200518_09.nc"),
    ("INSAT-3DR 4 km", "2020-05-18 09:00",
     INSAT_RAW / "3RIMG_18MAY2020_0902_L1C_ASIA_MER_V01R00.h5"),
    ("INSAT-3DR 4 km", "2020-05-18 12:00",
     INSAT_RAW / "3RIMG_18MAY2020_1202_L1C_ASIA_MER_V01R00.h5"),
    ("GridSat  8 km", "2020-05-18 15:00", GRIDSAT / "2020/nio_20200518_15.nc"),
    ("INSAT-3DR 4 km", "2020-05-18 16:00",
     INSAT_RAW / "3RIMG_18MAY2020_1602_L1C_ASIA_MER_V01R00.h5"),
]


def main() -> None:
    from api.vision import MODELS, _as_tensor

    scene_model, classes = MODELS.scene()
    intensity_model = MODELS.intensity()
    if scene_model is None or intensity_model is None:
        raise SystemExit("vision models not trained - nothing to verify")

    track = (load_dataset()
             .assign(ISO_TIME=lambda d: pd.to_datetime(d["ISO_TIME"]))
             .query("SID == @STORM_SID")
             .set_index("ISO_TIME").sort_index())

    def truth_at(ts: pd.Timestamp):
        cols = ["LAT", "LON", "vmax_kt"]
        s = (track[cols].reindex(track.index.union([ts]))
             .interpolate("index").loc[ts])
        return float(s.LAT), float(s.LON), float(s.vmax_kt)

    def infer(ds, lat, lon):
        patch = extract_patch(ds, lat, lon)
        if patch is None:
            return None
        x = _as_tensor(patch).to(MODELS.device)
        with torch.no_grad():
            raw = float(intensity_model(x).squeeze().cpu())
            probs = torch.softmax(scene_model(x), dim=1)[0].cpu().numpy()
        kt = raw * MODELS.intensity_scale[1] + MODELS.intensity_scale[0]
        slope, intercept = MODELS.intensity_cal
        kt = max((kt - intercept) / max(slope, 1e-6), 0.0)
        k = int(np.argmax(probs))
        return kt, classes[k], float(probs[k])

    print("=" * 78)
    print("INSAT-3DR through models trained only on GridSat - no retraining")
    print("=" * 78)
    print(f"\nAMPHAN ({STORM_SID}), 18 May 2020 - peak 130 kt, Super Cyclonic Storm\n")
    print(f"{'sensor':<15} {'time':<6} {'truth':>6} {'T3 kt':>7} {'err':>7}  "
          f"{'T2 scene':<9} {'conf':>5}")
    print("-" * 66)

    rows = []
    for label, when, path in CASES:
        ts = pd.Timestamp(when)
        lat, lon, truth = truth_at(ts)
        if not path.exists():
            print(f"{label:<15} {ts:%H:%M}  file not on disk - "
                  f"run src/ingest/insat.py")
            continue
        ds = to_gridsat(path) if path.suffix == ".h5" else xr.open_dataset(path)
        out = infer(ds, lat, lon)
        if out is None:
            bounds = valid_bounds(path) if path.suffix == ".h5" else None
            why = (f"storm at {lat:.1f}N outside strip {bounds}"
                   if bounds else "no usable patch")
            print(f"{label:<15} {ts:%H:%M}  {why}")
            continue
        kt, scene, conf = out
        print(f"{label:<15} {ts:%H:%M} {truth:>6.0f} {kt:>7.1f} {kt - truth:>+7.1f}  "
              f"{scene:<9} {100 * conf:>4.0f}%")
        rows.append((label, kt - truth))

    ins = [e for lab, e in rows if lab.startswith("INSAT")]
    grs = [e for lab, e in rows if lab.startswith("GridSat")]
    if ins and grs:
        print("-" * 66)
        print(f"mean |error|   INSAT {np.mean(np.abs(ins)):.1f} kt "
              f"(n={len(ins)})    GridSat {np.mean(np.abs(grs)):.1f} kt (n={len(grs)})")

    print(f"\n{'READ THIS BEFORE QUOTING THE NUMBERS':^78}")
    print("The result to take away is that INSAT imagery runs through the")
    print("existing models at all, unretrained, and returns plausible values on")
    print("a 125 kt storm. It is NOT evidence that INSAT beats GridSat: this is")
    print("one storm, and it sits in the vision training split, so the GridSat")
    print("column had seen it and the INSAT column had not.")


if __name__ == "__main__":
    main()
