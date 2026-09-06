# train and evaluate the prediction stack on real IBTrACS North Indian

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build                                    # noqa: E402
from features.build import build_dataset, HORIZONS                  # noqa: E402
from dataset import load_dataset                              # noqa: E402
from eval.splits import storm_split, frame_split, describe          # noqa: E402
from eval import metrics                                            # noqa: E402
from models.baselines import Persistence, DecayPersistence, Cliper  # noqa: E402
from models.gbm import GbmForecaster                                # noqa: E402
from models.residual import ResidualForecaster                      # noqa: E402
from models.ri import RiClassifier                                  # noqa: E402

RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

TRACK_TARGETS = [f"y_dlat_{h}" for h in HORIZONS] + [f"y_dlon_{h}" for h in HORIZONS]
INTENSITY_TARGETS = [f"y_dv_{h}" for h in HORIZONS]


def rule(char: str = "=", n: int = 78) -> None:
    print(char * n)


def evaluate_track(test: pd.DataFrame, cliper: Cliper, gbm: GbmForecaster,
                   residual: ResidualForecaster) -> list[dict]:
    rows = []
    for h in HORIZONS:
        mask = test[f"y_dlat_{h}"].notna() & test[f"y_dlon_{h}"].notna()
        sub = test[mask]
        if len(sub) < 20:
            continue
        true_lat = sub[f"true_lat_{h}"].to_numpy()
        true_lon = sub[f"true_lon_{h}"].to_numpy()
        lat0, lon0 = sub["LAT"].to_numpy(), sub["LON"].to_numpy()

        p_lat, p_lon = Persistence.predict_track(sub, h)
        candidates = {
            "persistence": (p_lat, p_lon),
            "cliper": (lat0 + cliper.predict(sub, f"y_dlat_{h}"),
                       lon0 + cliper.predict(sub, f"y_dlon_{h}")),
            "gbm_plain": (lat0 + gbm.predict(sub, f"y_dlat_{h}"),
                          lon0 + gbm.predict(sub, f"y_dlon_{h}")),
            "chakravat": (lat0 + residual.predict(sub, f"y_dlat_{h}"),
                          lon0 + residual.predict(sub, f"y_dlon_{h}")),
        }
        for name, (la, lo) in candidates.items():
            r = metrics.track_error(true_lat, true_lon, la, lo)
            r.update(horizon_h=h, model=name)
            rows.append(r)
    return rows


def evaluate_intensity(test: pd.DataFrame, cliper: Cliper, gbm: GbmForecaster,
                       residual: ResidualForecaster) -> list[dict]:
    decay = DecayPersistence()
    rows = []
    for h in HORIZONS:
        sub = test[test[f"y_dv_{h}"].notna()]
        if len(sub) < 20:
            continue
        truth = sub[f"y_dv_{h}"].to_numpy()
        preds = {
            "persistence": Persistence.predict_intensity(sub, h),
            "decay_persistence": decay.predict_intensity(sub, h),
            "cliper": cliper.predict(sub, f"y_dv_{h}"),
            "gbm_plain": gbm.predict(sub, f"y_dv_{h}"),
            "chakravat": residual.predict(sub, f"y_dv_{h}"),
        }
        for name, pred in preds.items():
            r = metrics.intensity_error(truth, pred)
            r.update(horizon_h=h, model=name)
            rows.append(r)
    return rows


def evaluate_ri(test: pd.DataFrame, clf: RiClassifier, threshold: float) -> dict:
    sub = test[test["y_ri"].notna()]
    y_true = sub["y_ri"].to_numpy()
    prob = clf.predict_proba(sub)

    out = metrics.brier_skill_score(y_true, prob, climatology=clf.base_rate)
    out["test_base_rate"] = float(np.mean(y_true))
    out["operating_threshold"] = float(threshold)
    out["contingency_at_operating"] = metrics.contingency(y_true, prob, threshold)
    for thr in (0.1, 0.2, 0.3, 0.5):
        out[f"contingency@{thr}"] = metrics.contingency(y_true, prob, thr)
    return out


def leakage_demo(data: pd.DataFrame) -> dict:
    # same model, same data, two splits
    results = {}
    for label, splits in (("storm_split_honest", storm_split(data)),
                          ("frame_split_leaky", frame_split(data))):
        gbm = GbmForecaster().fit(splits["train"], ["y_dv_24"])
        test = splits["test"]
        sub = test[test["y_dv_24"].notna()]
        pred = gbm.predict(sub, "y_dv_24")
        results[label] = metrics.intensity_error(sub["y_dv_24"].to_numpy(), pred)
    return results


def main() -> None:
    rule()
    print("CHAKRAVAT - prediction stack, North Indian Ocean")
    rule()

    data = load_dataset()
    print(f"\nforecast-ready rows: {len(data):,}   storms: {data['SID'].nunique()}")

    splits = storm_split(data)
    print("\nSplit (chronological, by season - no storm spans two sets):")
    print(describe(splits))
    train, val, test = splits["train"], splits["val"], splits["test"]
    all_targets = TRACK_TARGETS + INTENSITY_TARGETS

    print("\nFitting baselines and models ...")
    cliper = Cliper().fit(train, all_targets)
    gbm = GbmForecaster().fit(train, all_targets)
    residual = ResidualForecaster().fit(train, all_targets)

    ri_clf = RiClassifier().fit(train, val)
    ri_threshold = ri_clf.pick_threshold(val, min_pod=0.5)

    track_rows = evaluate_track(test, cliper, gbm, residual)
    intensity_rows = evaluate_intensity(test, cliper, gbm, residual)
    ri = evaluate_ri(test, ri_clf, ri_threshold)
    leak = leakage_demo(data)

    print("\n")
    rule()
    print("TRACK FORECAST - mean position error, km (test seasons 2020-2025)")
    rule()
    tdf = pd.DataFrame(track_rows).pivot(index="horizon_h", columns="model", values="mean_km")
    tdf = tdf[["persistence", "cliper", "gbm_plain", "chakravat"]]
    tdf["skill_vs_cliper_%"] = 100 * (1 - tdf["chakravat"] / tdf["cliper"])
    n_by_h = pd.DataFrame(track_rows).groupby("horizon_h")["n"].max()
    tdf.insert(0, "n", n_by_h)
    print(tdf.round(1).to_string())

    print("\n")
    rule()
    print("INTENSITY FORECAST - MAE, knots (test seasons 2020-2025)")
    rule()
    idf = pd.DataFrame(intensity_rows).pivot(index="horizon_h", columns="model", values="mae_kt")
    idf = idf[["persistence", "decay_persistence", "cliper", "gbm_plain", "chakravat"]]
    idf["skill_vs_cliper_%"] = 100 * (1 - idf["chakravat"] / idf["cliper"])
    idf.insert(0, "n", pd.DataFrame(intensity_rows).groupby("horizon_h")["n"].max())
    print(idf.round(2).to_string())

    print("\n")
    rule()
    print("RAPID INTENSIFICATION - +30 kt in 24 h")
    rule()
    print(f"  train base rate      {ri['base_rate'] * 100:.2f}%")
    print(f"  test base rate       {ri['test_base_rate'] * 100:.2f}%")
    print(f"  Brier score          {ri['brier']:.4f}")
    print(f"  Brier (climatology)  {ri['brier_climatology']:.4f}")
    print(f"  Brier skill score    {ri['bss']:+.3f}   <- positive means real skill")
    print(f"\n  operating threshold  {ri['operating_threshold']:.2f} "
          f"(chosen on validation for POD >= 0.5)")
    for thr in (0.1, 0.2, 0.3, 0.5):
        c = ri[f"contingency@{thr}"]
        print(f"  @{thr}: POD {c['pod']:.2f}  FAR {c['far']:.2f}  CSI {c['csi']:.2f}  "
              f"(hits {int(c['hits'])}, misses {int(c['misses'])}, FA {int(c['false_alarms'])})")

    print("\n")
    rule()
    print("LEAKAGE DEMONSTRATION - identical model, 24 h intensity MAE")
    rule()
    honest = leak["storm_split_honest"]["mae_kt"]
    leaky = leak["frame_split_leaky"]["mae_kt"]
    print(f"  random frame split (leaky)   {leaky:.2f} kt   <- what many repos report")
    print(f"  held-out seasons (honest)    {honest:.2f} kt   <- what it actually does")
    print(f"  optimism from leakage        {honest - leaky:+.2f} kt "
          f"({100 * (honest - leaky) / leaky:+.0f}%)")

    ART = ROOT / "artifacts"
    ART.mkdir(exist_ok=True)
    joblib.dump(
        {"cliper": cliper, "residual": residual, "ri": ri_clf, "ri_threshold": ri_threshold},
        ART / "prediction_models.joblib",
    )
    print(f"\nSaved models to {ART / 'prediction_models.joblib'}")

    payload = {
        "dataset": {
            "rows": int(len(data)),
            "storms": int(data["SID"].nunique()),
            "splits": {k: {"rows": int(len(v)), "storms": int(v["SID"].nunique())}
                       for k, v in splits.items()},
        },
        "track": track_rows,
        "intensity": intensity_rows,
        "rapid_intensification": ri,
        "leakage_demo": leak,
    }
    (OUT / "prediction_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT / 'prediction_results.json'}")


if __name__ == "__main__":
    main()
