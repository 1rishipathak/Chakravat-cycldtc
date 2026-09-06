# train and score the landfall module against hand-computable baselines

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_dataset                       # noqa: E402
from eval.splits import storm_split                    # noqa: E402
from eval.metrics import great_circle_km, contingency  # noqa: E402
from models.landfall import (                          # noqa: E402
    LandfallForecaster, add_landfall_targets, add_coast_features,
    baseline_hours, baseline_position, baseline_intensity, snap_to_coast,
)
from models import coastline as cl                      # noqa: E402

OUT = ROOT / "reports"
ART = ROOT / "artifacts"


def _track_crossings(sub):
    # where each row's forecast track first meets the coast
    from features.build import HORIZONS

    n = len(sub)
    out = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    path = ART / "ensemble_cone.joblib"
    if not (cl.available() and path.exists()):
        print("  (no ensemble or coastline - skipping track crossing)")
        return out

    ens = joblib.load(path)
    lat_h, lon_h = {}, {}
    for h in HORIZONS:
        c = ens.track_cone(sub, h)
        lat_h[h], lon_h[h] = c["lat"], c["lon"]

    lat0 = sub["LAT"].to_numpy()
    lon0 = sub["LON"].to_numpy()
    for i in range(n):
        pts = [(lat0[i], lon0[i])]
        pts += [(lat_h[h][i], lon_h[h][i]) for h in HORIZONS]
        if not all(np.isfinite(a) and np.isfinite(b) for a, b in pts):
            continue
        hit = cl.first_crossing(pts)
        if hit is None:
            continue
        out[0][i], out[1][i], frac = hit
        out[2][i] = cl.crossing_time_h(frac, list(HORIZONS))
    return out


def main() -> None:
    print("=" * 78)
    print("CHAKRAVAT - landfall module")
    print("=" * 78)

    data = add_landfall_targets(load_dataset())
    if not cl.available():
        print("\nNO COASTLINE on disk - run src/ingest/coastline.py")
    else:
        print("\nattaching coastline features ...")
    data = add_coast_features(data)
    splits = storm_split(data)
    train, test = splits["train"], splits["test"]

    n_land = int(data["lf_hours"].notna().sum())
    print(f"\npre-landfall forecast points (<=72 h out): {n_land:,}")
    print(f"  train {int(train['lf_hours'].notna().sum()):,}   "
          f"test {int(test['lf_hours'].notna().sum()):,}")
    print(f"  storms making landfall: "
          f"{data[data['lf_hours'].notna()]['SID'].nunique()}")

    print("\ntraining ...")
    model = LandfallForecaster().fit(train)

    sub = test[test["lf_hours"].notna()].copy()
    results = {}

    print("\n" + "=" * 78)
    print("LANDFALL FORECASTS - test seasons 2020-2025")
    print("=" * 78)

    # --- timing ---
    truth_h = sub["lf_hours"].to_numpy()
    pred_h = model.predict(sub, "lf_hours")
    base_h = baseline_hours(sub)
    mae_model = float(np.mean(np.abs(pred_h - truth_h)))
    mae_base = float(np.mean(np.abs(np.clip(base_h, 0, 200) - truth_h)))
    print(f"\nTiming - hours until landfall  (n={len(sub):,})")
    print(f"  distance / speed baseline : {mae_base:6.2f} h MAE")
    print(f"  chakravat                 : {mae_model:6.2f} h MAE   "
          f"({100*(1-mae_model/mae_base):+.1f}%)")
    results["timing"] = {"n": len(sub), "baseline_mae_h": mae_base,
                         "model_mae_h": mae_model}

    # --- position ---
    pred_lat = model.predict(sub, "lf_lat")
    pred_lon = model.predict(sub, "lf_lon")
    b_lat, b_lon = baseline_position(sub, np.clip(base_h, 0, 200))
    err_model = great_circle_km(sub["lf_lat"], sub["lf_lon"], pred_lat, pred_lon)
    err_base = great_circle_km(sub["lf_lat"], sub["lf_lon"], b_lat, b_lon)
    print(f"\nPosition - where it crosses the coast")
    print(f"  motion extrapolation      : {np.mean(err_base):6.1f} km mean, "
          f"{np.median(err_base):6.1f} km median")
    print(f"  chakravat                 : {np.mean(err_model):6.1f} km mean, "
          f"{np.median(err_model):6.1f} km median   "
          f"({100*(1-np.mean(err_model)/np.mean(err_base)):+.1f}%)")
    # guardrail: a landfall coordinate that is not on the coast is wrong by
    # construction. Snapping cannot aim at a different stretch of coast, but it
    # removes the component of the error pointing out to sea or inland.
    snap_lat, snap_lon = snap_to_coast(pred_lat, pred_lon)
    err_snap = great_circle_km(sub["lf_lat"], sub["lf_lon"], snap_lat, snap_lon)
    print(f"  + snapped to coastline    : {np.mean(err_snap):6.1f} km mean, "
          f"{np.median(err_snap):6.1f} km median   "
          f"({100*(1-np.mean(err_snap)/np.mean(err_model)):+.1f}% vs raw)")

    # how often is the answer even on a coast? Previously unmeasured.
    if cl.available():
        _, _, d_raw, _ = cl.nearest_coast_bulk(pred_lat, pred_lon)
        on_coast = float(np.mean(d_raw <= 25.0))
        print(f"  raw predictions within 25 km of any coast: {100*on_coast:.0f}%")
    else:
        on_coast = float("nan")

    # the real fix: landfall position *is* where the forecast track crosses the
    # coast. Taking the intersection makes it consistent with the track
    # forecast (122 km at 24 h) instead of accumulating a separate error, and
    # removes a model rather than adding one.
    cross_lat, cross_lon, cross_h = _track_crossings(sub)
    got = np.isfinite(cross_lat)
    if got.any():
        err_cross = great_circle_km(sub["lf_lat"], sub["lf_lon"],
                                    cross_lat, cross_lon)
        # like-for-like. The rows where a track finds land are not a random
        # subset - they are storms already aimed at the coast, which may
        # simply be the easier ones. Comparing the crossing on those rows
        # against the regressor on *all* rows would flatter it.
        print(f"\n  - on the {got.sum()} fixes whose forecast track reaches land "
              f"({100*got.mean():.0f}% of test) -")
        print(f"  regression, same rows     : {np.mean(err_model[got]):6.1f} km mean, "
              f"{np.median(err_model[got]):6.1f} km median")
        print(f"  snapped, same rows        : {np.mean(err_snap[got]):6.1f} km mean, "
              f"{np.median(err_snap[got]):6.1f} km median")
        print(f"  track x coastline         : {np.mean(err_cross[got]):6.1f} km mean, "
              f"{np.median(err_cross[got]):6.1f} km median   "
              f"({100*(1-np.mean(err_cross[got])/np.mean(err_model[got])):+.1f}% vs regression)")

        # hybrid: use the crossing where the track finds one, snapped
        # regression where it does not. This is what would ship.
        hy_lat = np.where(got, cross_lat, snap_lat)
        hy_lon = np.where(got, cross_lon, snap_lon)
        err_hy = great_circle_km(sub["lf_lat"], sub["lf_lon"], hy_lat, hy_lon)
        print(f"  HYBRID (crossing + snap)  : {np.mean(err_hy):6.1f} km mean, "
              f"{np.median(err_hy):6.1f} km median   "
              f"({100*(1-np.mean(err_hy)/np.mean(err_model)):+.1f}% vs raw)")

        # timing implied by the crossing, against the timing model. Reported
        # for information: the timing model is our best landfall number and is
        # not replaced unless the crossing beats it outright.
        okh = got & np.isfinite(cross_h)
        if okh.sum() >= 15:
            mae_cross_h = float(np.mean(np.abs(cross_h[okh] - truth_h[okh])))
            mae_model_h = float(np.mean(np.abs(pred_h[okh] - truth_h[okh])))
            print(f"  timing from crossing      : {mae_cross_h:6.2f} h MAE  "
                  f"(model on same rows {mae_model_h:.2f} h) - "
                  f"{'crossing wins' if mae_cross_h < mae_model_h else 'model kept'}")
            results["timing_from_crossing"] = {
                "n": int(okh.sum()), "crossing_mae_h": mae_cross_h,
                "model_mae_h_same_rows": mae_model_h}
    else:
        err_cross = np.full(len(sub), np.nan)
        err_hy = err_snap
        print("\n  track x coastline         : no crossings found")

    results["position"] = {"baseline_mean_km": float(np.mean(err_base)),
                           "model_mean_km": float(np.mean(err_model)),
                           "model_median_km": float(np.median(err_model)),
                           "snapped_mean_km": float(np.mean(err_snap)),
                           "snapped_median_km": float(np.median(err_snap)),
                           "crossing_n": int(got.sum()),
                           "crossing_mean_km": float(np.mean(err_cross[got]))
                           if got.any() else None,
                           "hybrid_mean_km": float(np.mean(err_hy)),
                           "hybrid_median_km": float(np.median(err_hy)),
                           "raw_frac_within_25km_of_coast": on_coast}

    # --- intensity at landfall ---
    truth_v = sub["lf_vmax"].to_numpy()
    pred_v = model.predict(sub, "lf_vmax")
    base_v = baseline_intensity(sub)
    mae_v = float(np.mean(np.abs(pred_v - truth_v)))
    mae_bv = float(np.mean(np.abs(base_v - truth_v)))
    print(f"\nIntensity at landfall")
    print(f"  persistence baseline      : {mae_bv:6.2f} kt MAE")
    print(f"  chakravat                 : {mae_v:6.2f} kt MAE   "
          f"({100*(1-mae_v/mae_bv):+.1f}%)")
    results["intensity"] = {"baseline_mae_kt": mae_bv, "model_mae_kt": mae_v}

    # --- will it land at all ---
    allsub = test[test["lf_will_land"].notna()]
    prob = model.predict_will_land(allsub)
    y = allsub["lf_will_land"].to_numpy()
    c = contingency(y, prob, 0.5)
    print(f"\nWill it make landfall within 72 h?  (base rate "
          f"{100*np.mean(y):.1f}%, n={len(allsub):,})")
    print(f"  POD {c['pod']:.2f}   FAR {c['far']:.2f}   CSI {c['csi']:.2f}")
    results["will_land"] = {"base_rate": float(np.mean(y)), **c}

    # --- by lead time, which is how it would be used ---
    print("\nAccuracy by lead time")
    print(f"  {'lead':>10} {'n':>5} {'timing':>9} {'position':>11} {'intensity':>11}")
    bands = [(0, 12), (12, 24), (24, 48), (48, 72)]
    by_lead = []
    for lo, hi in bands:
        m = (truth_h > lo) & (truth_h <= hi)
        if m.sum() < 15:
            continue
        row = {"lead_from_h": lo, "lead_to_h": hi, "n": int(m.sum()),
               "timing_mae_h": float(np.mean(np.abs(pred_h[m] - truth_h[m]))),
               "position_mean_km": float(np.mean(err_model[m])),
               "intensity_mae_kt": float(np.mean(np.abs(pred_v[m] - truth_v[m])))}
        print(f"  {lo:>3}-{hi:>3} h {row['n']:>5} {row['timing_mae_h']:>7.2f} h "
              f"{row['position_mean_km']:>8.1f} km {row['intensity_mae_kt']:>8.2f} kt")
        by_lead.append(row)
    results["by_lead"] = by_lead

    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "landfall_model.joblib")
    (OUT / "landfall_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {ART / 'landfall_model.joblib'}")
    print(f"Wrote {OUT / 'landfall_results.json'}")


if __name__ == "__main__":
    main()
