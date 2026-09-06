# verify the end-to-end pipeline against best track, on a named storm

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ingest.ibtracs import build, imd_category      # noqa: E402
from eval.metrics import great_circle_km            # noqa: E402


def verify(name: str, when: pd.Timestamp, hours: int = 48) -> None:
    from pipeline import Chakravat

    track = build(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    storm = track[track["NAME"].str.upper().str.startswith(name)].sort_values("ISO_TIME")
    if storm.empty:
        print(f"{name}: not in best track")
        return

    res = Chakravat(verbose=False).run(when, hours=hours)
    if not res.get("available") or not res["storms"]:
        print(f"{name}: nothing detected")
        return

    # the detection closest to where the storm actually was.
    at = storm[storm["ISO_TIME"] == when]
    if at.empty:
        at = storm.iloc[[-1]]
    tlat, tlon = float(at["LAT"].iloc[0]), float(at["LON"].iloc[0])
    best = min(res["storms"],
               key=lambda s: float(great_circle_km(tlat, tlon, s["lat"], s["lon"])))

    if not best.get("prediction"):
        print(f"{name}: detected, but track too short to forecast")
        return
    # use the issue time the pipeline reports, not the last detection: they
    # differ whenever the GridSat slot is not synoptic.
    issued = pd.Timestamp(best["prediction"]["issued_at"])
    print("=" * 74)
    print(f"{name}  -  window ends {when}, forecast issued {issued}")
    print("=" * 74)

    truth_now = storm[storm["ISO_TIME"] == issued]
    if not truth_now.empty:
        alat = float(truth_now["LAT"].iloc[0])
        alon = float(truth_now["LON"].iloc[0])
        akt = float(truth_now["vmax_kt"].iloc[0])
        d = float(great_circle_km(alat, alon, best["lat"], best["lon"]))
        print(f"  detection   {best['lat']:.1f}N {best['lon']:.1f}E   "
              f"actual {alat:.1f}N {alon:.1f}E   off by {d:.0f} km")
        print(f"  intensity   {best['vmax_kt']:.0f} kt   actual {akt:.0f} kt   "
              f"({best['vmax_kt'] - akt:+.0f} kt)")
        print(f"  scene       {best['scene']}  "
              f"(confidence {best['scene_confidence']:.2f})"
              if best["scene_confidence"] else f"  scene       {best['scene']}")

    if not best.get("prediction"):
        print("\n  no forecast - detected track too short")
        return

    print(f"\n  {'lead':>5}  {'forecast':<22} {'actual':<22} {'error':>8}  cone")
    inside = total = 0
    for p in best["prediction"]["forecast"]:
        vt = issued + pd.Timedelta(hours=p["horizon_h"])
        row = storm[storm["ISO_TIME"] == vt]
        if row.empty:
            continue
        alat, alon = float(row["LAT"].iloc[0]), float(row["LON"].iloc[0])
        akt = float(row["vmax_kt"].iloc[0])
        err = float(great_circle_km(alat, alon, p["lat"], p["lon"]))
        ok = err <= p["radius_km"]
        inside += ok
        total += 1
        print(f"  {p['horizon_h']:>3} h  "
              f"{p['lat']:.1f}N {p['lon']:.1f}E {p['vmax_kt']:>4.0f} kt   "
              f"{alat:.1f}N {alon:.1f}E {akt:>4.0f} kt   "
              f"{err:>6.0f} km  {'in' if ok else 'OUT':>3} (r={p['radius_km']:.0f})")
    if total:
        print(f"\n  inside the {100*best['prediction'].get('level', 0.67):.0f}% cone: "
              f"{inside}/{total}")


if __name__ == "__main__":
    cases = [("AMPHAN", "2020-05-20 00:00"),
             ("TAUKTAE", "2021-05-17 12:00"),
             ("BIPARJOY", "2023-06-13 00:00")]
    if len(sys.argv) > 2:
        cases = [(sys.argv[1].upper(), sys.argv[2])]
    for nm, t in cases:
        verify(nm, pd.Timestamp(t))
        print()
