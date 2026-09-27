# confidence intervals for the served forecast's skill over CLIPER
#
#     python src/eval_forecast_ci.py
#
# "11% better than CLIPER" on 51 storms says nothing about whether a different
# 51 storms would say 0% or 20%. this scores the forecast the API actually
# serves (the ensemble mean) and CLIPER on identical held-out rows, then
# resamples whole storms 2,000 times and re-computes skill each time. the pair
# is resampled together, so the interval is for the difference, which is the
# claim, rather than for each error separately.

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_dataset          # noqa: E402
from eval import metrics                  # noqa: E402
from eval.splits import storm_split       # noqa: E402
from features.build import HORIZONS       # noqa: E402

REPS = 2000


def skill_ci(storms: np.ndarray, err_model: np.ndarray, err_base: np.ndarray, seed: int = 0):
    uniq = np.unique(storms)
    groups = [np.flatnonzero(storms == s) for s in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(REPS):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vals.append(100 * (1 - err_model[idx].mean() / err_base[idx].mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.asarray(vals) > 0))


def main() -> None:
    d = load_dataset(verbose=False)
    test = storm_split(d)["test"]
    ens = joblib.load(ROOT / "artifacts" / "ensemble_cone.joblib")
    cliper = joblib.load(ROOT / "artifacts" / "prediction_models.joblib")["cliper"]

    out = []
    print(f"held-out storms {test['SID'].nunique()}   rows {len(test)}")
    print(f"{'task':<10}{'lead':>5}{'n':>6}{'storms':>7}{'CLIPER':>9}{'served':>9}"
          f"{'skill':>8}   95% interval     share of resamples > 0")
    for task in ("track", "intensity"):
        for h in HORIZONS:
            if task == "track":
                sub = test[test[f"y_dlat_{h}"].notna() & test[f"y_dlon_{h}"].notna()]
                tlat = sub["LAT"].to_numpy() + sub[f"y_dlat_{h}"].to_numpy()
                tlon = sub["LON"].to_numpy() + sub[f"y_dlon_{h}"].to_numpy()
                c = ens.track_cone(sub, h, 0.67)
                em = metrics.great_circle_km(tlat, tlon, np.asarray(c["lat"]), np.asarray(c["lon"]))
                eb = metrics.great_circle_km(tlat, tlon,
                                             sub["LAT"].to_numpy() + cliper.predict(sub, f"y_dlat_{h}"),
                                             sub["LON"].to_numpy() + cliper.predict(sub, f"y_dlon_{h}"))
                unit = "km"
            else:
                sub = test[test[f"y_dv_{h}"].notna()]
                truth = sub["vmax_kt"].to_numpy() + sub[f"y_dv_{h}"].to_numpy()
                em = np.abs(np.asarray(ens.intensity_band(sub, h, 0.67)["vmax"]) - truth)
                eb = np.abs(sub["vmax_kt"].to_numpy() + cliper.predict(sub, f"y_dv_{h}") - truth)
                unit = "kt"
            em, eb = np.asarray(em, float), np.asarray(eb, float)
            skill = 100 * (1 - em.mean() / eb.mean())
            lo, hi, pos = skill_ci(sub["SID"].to_numpy(), em, eb)
            out.append({"task": task, "horizon_h": h, "n": int(len(sub)),
                        "storms": int(sub["SID"].nunique()), "unit": unit,
                        "cliper": float(eb.mean()), "served": float(em.mean()),
                        "skill_pct": float(skill), "skill_ci95": [lo, hi],
                        "share_of_resamples_positive": pos})
            print(f"{task:<10}{h:>5}{len(sub):>6}{sub['SID'].nunique():>7}{eb.mean():>9.2f}"
                  f"{em.mean():>9.2f}{skill:>+7.1f}%   [{lo:+5.1f}, {hi:+5.1f}]      {pos:.3f}")

    (ROOT / "reports" / "forecast_skill_ci.json").write_text(json.dumps(
        {"model": "served ensemble mean", "split": "held-out seasons 2020-2025",
         "resamples": REPS, "results": out}, indent=2))
    print("wrote reports/forecast_skill_ci.json")


if __name__ == "__main__":
    main()
