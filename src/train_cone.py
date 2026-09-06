# train the ensemble and verify its cone is honestly calibrated

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_dataset                       # noqa: E402
from features.build import HORIZONS                    # noqa: E402
from eval.splits import storm_split, describe          # noqa: E402
from eval.metrics import great_circle_km               # noqa: E402
from models.ensemble import EnsembleForecaster, DEFAULT_LEVELS   # noqa: E402

OUT = ROOT / "reports"
ART = ROOT / "artifacts"
N_MEMBERS = 8

TRACK_TARGETS = [f"y_dlat_{h}" for h in HORIZONS] + [f"y_dlon_{h}" for h in HORIZONS]
INTENSITY_TARGETS = [f"y_dv_{h}" for h in HORIZONS]


def main() -> None:
    print("=" * 78)
    print(f"CHAKRAVAT - ensemble forecast cone ({N_MEMBERS} members)")
    print("=" * 78)

    data = load_dataset()
    splits = storm_split(data)
    print(describe(splits))
    train, val, test = splits["train"], splits["val"], splits["test"]

    print(f"\ntraining {N_MEMBERS} bootstrap members ...")
    ens = EnsembleForecaster(n_members=N_MEMBERS).fit(
        train, TRACK_TARGETS + INTENSITY_TARGETS)

    print("calibrating cone multipliers on validation seasons ...")
    ens.calibrate(val, HORIZONS)

    print("\n" + "=" * 78)
    print("CONE CALIBRATION - multipliers learned on 2016-2019")
    print("=" * 78)
    print(f"  {'horizon':>8} " + " ".join(f"{int(lv*100):>7}%" for lv in DEFAULT_LEVELS))
    for h in HORIZONS:
        vals = [ens.multipliers.get(("track", h, lv)) for lv in DEFAULT_LEVELS]
        if all(v is not None for v in vals):
            print(f"  {h:>6} h " + " ".join(f"{v:>8.2f}" for v in vals))

    print("\n" + "=" * 78)
    print("COVERAGE ON TEST - does the cone mean what it says?")
    print("=" * 78)
    print(f"  {'horizon':>8} {'n':>5} " +
          " ".join(f"{int(lv*100):>3}% cone  radius" for lv in DEFAULT_LEVELS))

    rows = []
    for h in HORIZONS:
        sub = test[test[f"y_dlat_{h}"].notna() & test[f"y_dlon_{h}"].notna()]
        if len(sub) < 30:
            continue
        line = f"  {h:>6} h {len(sub):>5} "
        entry = {"task": "track", "horizon_h": h, "n": int(len(sub))}
        for lv in DEFAULT_LEVELS:
            cone = ens.track_cone(sub, h, lv)
            err = great_circle_km(sub[f"true_lat_{h}"], sub[f"true_lon_{h}"],
                                  cone["lat"], cone["lon"])
            covered = float(np.mean(err <= cone["radius_km"]))
            median_r = float(np.median(cone["radius_km"]))
            line += f"   {100*covered:>5.1f}% {median_r:>6.0f}km"
            entry[f"coverage_{int(lv*100)}"] = covered
            entry[f"median_radius_km_{int(lv*100)}"] = median_r
        print(line)
        rows.append(entry)

    print("\n  (nominal levels are " +
          ", ".join(f"{int(lv*100)}%" for lv in DEFAULT_LEVELS) +
          " - coverage should land near those)")

    print("\n" + "=" * 78)
    print("INTENSITY BAND COVERAGE ON TEST")
    print("=" * 78)
    print(f"  {'horizon':>8} {'n':>5} " +
          " ".join(f"{int(lv*100):>3}% band   width" for lv in DEFAULT_LEVELS))
    for h in HORIZONS:
        sub = test[test[f"y_dv_{h}"].notna()]
        if len(sub) < 30:
            continue
        truth = sub["vmax_kt"].to_numpy() + sub[f"y_dv_{h}"].to_numpy()
        line = f"  {h:>6} h {len(sub):>5} "
        entry = {"task": "intensity", "horizon_h": h, "n": int(len(sub))}
        for lv in DEFAULT_LEVELS:
            band = ens.intensity_band(sub, h, lv)
            covered = float(np.mean(np.abs(truth - band["vmax"]) <= band["half_width_kt"]))
            width = float(np.median(2 * band["half_width_kt"]))
            line += f"   {100*covered:>5.1f}% {width:>6.1f}kt"
            entry[f"coverage_{int(lv*100)}"] = covered
            entry[f"median_width_kt_{int(lv*100)}"] = width
        print(line)
        rows.append(entry)

    ART.mkdir(exist_ok=True)
    joblib.dump(ens, ART / "ensemble_cone.joblib")
    (OUT / "cone_calibration.json").write_text(json.dumps(
        {"n_members": N_MEMBERS,
         "multipliers": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in ens.multipliers.items()},
         "coverage": rows}, indent=2))
    print(f"\nSaved {ART / 'ensemble_cone.joblib'}")
    print(f"Wrote {OUT / 'cone_calibration.json'}")


if __name__ == "__main__":
    main()
