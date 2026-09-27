# How big is the damaging wind field, as a function of how strong the storm is?
#
#     python src/wind_radii.py
#
# The cone says where the centre may go. Nobody is hurt by a centre. To say
# whether a place gets gale-force wind you need the size of the wind field as
# well as the position, and that is what this measures: the radius of 34, 50 and
# 64 knot winds against intensity, from the JTWC wind radii carried in IBTrACS
# for this basin.
#
# It writes reports/wind_radii.json, which the probability swaths read. The 34 kt
# table already existed inside models/impact.py as a hard-coded list; this
# replaces the guesswork for the other two thresholds and puts all three
# somewhere they can be checked.
#
# Quadrants are averaged. A real wind field is not circular - the right-forward
# quadrant is usually the largest because the storm's own motion adds to the
# circulation - and the asymmetry is in the report for anyone who wants it, but
# the swath code uses the mean radius and says so.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPORT = ROOT / "reports" / "wind_radii.json"
NM_TO_KM = 1.852
BANDS = [(34, 48), (48, 64), (64, 90), (90, 999)]
THRESHOLDS = (34, 50, 64)
QUADRANTS = ("NE", "SE", "SW", "NW")


def main() -> None:
    from ingest.ibtracs import build

    track = build(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    raw = pd.read_csv(ROOT / "data" / "raw" / "ibtracs.NI.csv",
                      low_memory=False, skiprows=[1])
    raw["ISO_TIME"] = pd.to_datetime(raw["ISO_TIME"], errors="coerce")
    df = raw.merge(track[["SID", "ISO_TIME", "vmax_kt"]], on=["SID", "ISO_TIME"],
                   how="inner")

    print("=" * 76)
    print("WIND RADII - how far the damaging wind reaches, by intensity")
    print("=" * 76)

    out = {"source": "JTWC wind radii in IBTrACS, North Indian Ocean",
           "units": "km, mean of the four quadrants",
           "note": "quadrant means; the real field is asymmetric and the "
                   "asymmetry is reported separately",
           "thresholds": {}}

    for thr in THRESHOLDS:
        cols = [f"USA_R{thr}_{q}" for q in QUADRANTS]
        if not all(c in df.columns for c in cols):
            print(f"R{thr}: columns missing, skipped")
            continue
        q = df[cols].apply(pd.to_numeric, errors="coerce")
        # IBTrACS uses 0 both for "no winds this strong" and for "not reported";
        # a fix where every quadrant is zero or blank carries no information.
        has = q.notna().any(axis=1) & (q.fillna(0) > 0).any(axis=1)
        sub = df[has].copy()
        sub["radius_km"] = q[has].mean(axis=1, skipna=True) * NM_TO_KM

        print(f"\nR{thr}: {len(sub)} fixes with a reported radius, "
              f"{sub['SID'].nunique()} storms, "
              f"{sub['ISO_TIME'].dt.year.min():.0f}-{sub['ISO_TIME'].dt.year.max():.0f}")
        print(f"  {'intensity':<12}{'n':>6}{'median km':>11}{'mean':>8}{'p25':>7}{'p75':>7}")
        table = []
        for lo, hi in BANDS:
            m = (sub["vmax_kt"] >= lo) & (sub["vmax_kt"] < hi)
            if m.sum() < 10:
                table.append({"from_kt": lo, "to_kt": hi, "n": int(m.sum()),
                              "median_km": None})
                print(f"  {f'{lo}-{hi}':<12}{m.sum():>6}      too few")
                continue
            r = sub.loc[m, "radius_km"]
            table.append({"from_kt": lo, "to_kt": hi, "n": int(m.sum()),
                          "median_km": float(r.median()), "mean_km": float(r.mean()),
                          "p25_km": float(r.quantile(.25)), "p75_km": float(r.quantile(.75))})
            print(f"  {f'{lo}-{hi}':<12}{m.sum():>6}{r.median():>11.0f}{r.mean():>8.0f}"
                  f"{r.quantile(.25):>7.0f}{r.quantile(.75):>7.0f}")

        # asymmetry, for the record: which quadrant is biggest
        qm = (q[has].mean() * NM_TO_KM).to_dict()
        out["thresholds"][str(thr)] = {
            "bands": table,
            "quadrant_mean_km": {k.split("_")[-1]: float(v) for k, v in qm.items()},
            "fixes": int(len(sub)), "storms": int(sub["SID"].nunique()),
        }
        print("  quadrant means: " + "  ".join(
            f"{k.split('_')[-1]} {v:.0f}" for k, v in qm.items()))

    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORT}")
    print("\nbelow each threshold the radius is zero: a 40 kt storm has no 64 kt")
    print("winds to draw. that is why the tables start where they do.")


if __name__ == "__main__":
    main()
