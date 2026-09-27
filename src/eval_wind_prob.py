# Does "40% chance of gales" mean it happened 40% of the time?
#
#     python src/eval_wind_prob.py [--threshold 34] [--max-storms 0]
#
# A probability that is never checked is decoration. The check is a reliability
# diagram: bin every forecast by the probability it gave, and compare that to
# how often the thing actually happened in each bin. A perfectly reliable
# forecast lies on the diagonal. This is the same discipline the cone already
# gets - fit the spread on held-out data, then verify the coverage claimed.
#
# Truth is built from the observed best track over the same window: where the
# storm actually went, how strong it actually was, and the wind field size from
# the measured radius table. That last part is shared with the forecast side, so
# be clear about what this verifies. Position and intensity are genuinely
# independent - they are what the models predicted and what actually happened -
# but the radius model is common to both, so this diagram tests track and
# intensity skill, not whether the wind field size is right. Checking that needs
# per-quadrant observed radii, which IBTrACS reports for too few fixes here.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports" / "wind_prob_reliability.json"
BINS = np.array([0.0, 0.0001, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0001])
HELD_OUT = range(2020, 2026)
WINDOW_H = 72


def truth_field(storm: pd.DataFrame, t0: pd.Timestamp, lat, lon, threshold: int):
    """Where the storm actually put winds of this strength, over the window."""
    from models.impact import gale_radius_km
    from models.wind_prob import KM_PER_DEG, densify

    later = storm[(storm["ISO_TIME"] > t0)
                  & (storm["ISO_TIME"] <= t0 + pd.Timedelta(hours=WINDOW_H))]
    if later.empty:
        return None
    pts = [((r.ISO_TIME - t0).total_seconds() / 3600.0,
            float(r.LAT), float(r.LON), float(r.vmax_kt)) for r in later.itertuples()]
    pts = [(0.0, float(storm[storm.ISO_TIME == t0]["LAT"].iloc[0]),
            float(storm[storm.ISO_TIME == t0]["LON"].iloc[0]),
            float(storm[storm.ISO_TIME == t0]["vmax_kt"].iloc[0]))] + pts

    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cos_lat = np.cos(np.radians(LA))
    hit = np.zeros(LA.shape, dtype=bool)
    for _h, la, lo, v in densify(pts, 1.0):
        r = gale_radius_km(v, threshold)
        if r <= 0:
            continue
        dy = (LA - la) * KM_PER_DEG
        dx = (LO - lo) * KM_PER_DEG * cos_lat
        hit |= (dy * dy + dx * dx) <= r * r
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=34, choices=[34, 50, 64])
    ap.add_argument("--deg", type=float, default=0.25)
    ap.add_argument("--max-storms", type=int, default=0)
    ap.add_argument("--min-kt", type=float, default=34.0,
                    help="only issue from fixes at least this strong")
    ap.add_argument("--fit-seasons", default="2020,2021,2022",
                    help="seasons used to fit the calibration; the rest verify it")
    args = ap.parse_args()
    fit_seasons = {int(x) for x in args.fit_seasons.split(",") if x.strip()}

    import joblib

    from dataset import load_dataset
    from features.build import HORIZONS
    from models.landfall import add_landfall_targets
    from models import wind_prob as wp

    data = add_landfall_targets(load_dataset(verbose=False))
    ens = joblib.load(ROOT / "artifacts" / "ensemble_cone.joblib")
    lat, lon = wp.basin_grid(args.deg)

    test = data[data["SEASON"].isin(HELD_OUT)]
    sids = list(dict.fromkeys(test["SID"]))
    if args.max_storms:
        sids = sids[:args.max_storms]

    print("=" * 78)
    print(f"WIND PROBABILITY RELIABILITY - {args.threshold} kt within {WINDOW_H} h")
    print("=" * 78)
    print(f"{len(sids)} held-out storms, grid {len(lat)}x{len(lon)} at {args.deg} deg\n")

    hits = np.zeros(len(BINS) - 1)       # observed occurrences per forecast bin
    total = np.zeros(len(BINS) - 1)      # forecasts per bin
    psum = np.zeros(len(BINS) - 1)       # forecast probability sum, for the x axis
    cases, t0 = 0, time.time()
    brier_num, brier_n, climo_hits = 0.0, 0, 0.0
    # eight members give nine possible raw values, so the calibration is a
    # lookup over those nine rather than a fitted curve
    grid_hits = {}                       # raw value -> observed count, per split
    grid_n = {}

    for n, sid in enumerate(sids, 1):
        storm = data[data["SID"] == sid].sort_values("ISO_TIME")
        for _, row in storm.iterrows():
            if row["vmax_kt"] < args.min_kt:
                continue
            issue = pd.Timestamp(row["ISO_TIME"])
            truth = truth_field(storm, issue, lat, lon, args.threshold)
            if truth is None or not truth.any():
                continue
            one = storm[storm["ISO_TIME"] == issue]
            try:
                tracks = wp.member_tracks(ens, one, HORIZONS)
            except Exception:                      # noqa: BLE001 - short track
                continue
            # raw, deliberately: this script is what fits the calibration, so
            # scoring an already-calibrated field would be circular
            field = wp.probability_field(tracks, lat, lon,
                                         thresholds=(args.threshold,),
                                         calibrated=False)
            p = field[args.threshold].ravel()
            y = truth.ravel().astype(float)

            split = "fit" if int(row["SEASON"]) in fit_seasons else "verify"
            vals, inv = np.unique(np.round(p, 6), return_inverse=True)
            cnt = np.bincount(inv, minlength=len(vals))
            pos = np.bincount(inv, weights=y, minlength=len(vals))
            for v, c_, h_ in zip(vals, cnt, pos):
                grid_n.setdefault(split, {}).setdefault(float(v), 0)
                grid_hits.setdefault(split, {}).setdefault(float(v), 0.0)
                grid_n[split][float(v)] += int(c_)
                grid_hits[split][float(v)] += float(h_)

            idx = np.digitize(p, BINS) - 1
            np.add.at(total, idx, 1.0)
            np.add.at(hits, idx, y)
            np.add.at(psum, idx, p)
            brier_num += float(np.sum((p - y) ** 2))
            brier_n += p.size
            climo_hits += float(y.sum())
            cases += 1

        if n % 5 == 0 or n == len(sids):
            print(f"  {n}/{len(sids)} storms, {cases} forecasts, "
                  f"{time.time() - t0:.0f}s", flush=True)

    if not cases:
        raise SystemExit("no verifiable forecasts")

    brier = brier_num / brier_n
    base = climo_hits / brier_n
    brier_climo = base * (1 - base)
    bss = 1 - brier / brier_climo if brier_climo > 0 else float("nan")

    print(f"\nreliability, {cases} forecasts over {brier_n:,} grid points")
    print(f"  {'forecast':>12}{'observed':>11}{'n':>12}")
    rows = []
    for i in range(len(BINS) - 1):
        if total[i] == 0:
            continue
        fx, ob = psum[i] / total[i], hits[i] / total[i]
        rows.append({"bin_lo": float(BINS[i]), "bin_hi": float(BINS[i + 1]),
                     "forecast": float(fx), "observed": float(ob),
                     "n": int(total[i])})
        print(f"  {fx:>11.3f}{ob:>11.3f}{int(total[i]):>12,}")

    print(f"\n  Brier score      {brier:.5f}")
    print(f"  base rate        {base:.5f}")
    print(f"  Brier skill      {bss:+.3f}  (against always forecasting the base rate)")

    # the calibration: what a raw member fraction has actually meant, fitted on
    # the fit seasons only, forced upward-monotone so a larger fraction can
    # never map to a smaller probability
    fit_n, fit_h = grid_n.get("fit", {}), grid_hits.get("fit", {})
    raw_vals = sorted(v for v in fit_n if fit_n[v] >= 500)
    mapping, running = [], 0.0
    for v in raw_vals:
        obs = fit_h[v] / fit_n[v]
        running = max(running, obs)
        mapping.append({"raw": v, "calibrated": running, "n": int(fit_n[v])})

    def apply_map(v: float) -> float:
        if not mapping:
            return v
        if v <= mapping[0]["raw"]:
            return mapping[0]["calibrated"]
        for a_, b_ in zip(mapping, mapping[1:]):
            if v <= b_["raw"]:
                span = b_["raw"] - a_["raw"]
                f = 0.0 if span <= 0 else (v - a_["raw"]) / span
                return a_["calibrated"] + f * (b_["calibrated"] - a_["calibrated"])
        return mapping[-1]["calibrated"]

    ver_n, ver_h = grid_n.get("verify", {}), grid_hits.get("verify", {})
    ver_rows, ver_brier, ver_pts, ver_pos = [], 0.0, 0, 0.0
    for v in sorted(ver_n):
        n_ = ver_n[v]
        if n_ < 500:
            continue
        obs = ver_h[v] / n_
        cal = apply_map(v)
        ver_rows.append({"raw": v, "calibrated": cal, "observed": obs, "n": int(n_)})
    for v in sorted(ver_n):
        n_, o_ = ver_n[v], ver_h[v]
        cal = apply_map(v)
        ver_brier += o_ * (cal - 1.0) ** 2 + (n_ - o_) * cal ** 2
        ver_pts += n_
        ver_pos += o_
    ver_brier = ver_brier / ver_pts if ver_pts else float("nan")
    ver_base = ver_pos / ver_pts if ver_pts else float("nan")
    ver_bss = (1 - ver_brier / (ver_base * (1 - ver_base))
               if ver_pts and 0 < ver_base < 1 else float("nan"))

    print(f"\ncalibration fitted on {sorted(fit_seasons)}, verified on the rest")
    print(f"  {'members':>9}{'raw':>8}{'calibrated':>12}{'observed':>10}{'n':>12}")
    for r in ver_rows:
        print(f"  {r['raw']*8:>8.0f}/8{r['raw']:>8.3f}{r['calibrated']:>12.3f}"
              f"{r['observed']:>10.3f}{r['n']:>12,}")
    print(f"  calibrated Brier skill on the verify seasons: {ver_bss:+.3f}")

    out = {"calibration": {"fit_seasons": sorted(fit_seasons), "mapping": mapping,
                           "verification": ver_rows,
                           "calibrated_brier": ver_brier,
                           "calibrated_brier_skill": ver_bss},
           "threshold_kt": args.threshold, "window_h": WINDOW_H,
           "grid_deg": args.deg, "storms": len(sids), "forecasts": cases,
           "grid_points": int(brier_n), "reliability": rows,
           "brier": brier, "base_rate": base, "brier_skill_score": bss,
           "verifies": "track and intensity; the wind field radius model is "
                       "shared with the truth side and is not tested here"}
    path = REPORT.with_name(f"wind_prob_reliability_{args.threshold}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
