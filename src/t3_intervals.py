# An intensity estimate that says how wrong it usually is.
#
#     python src/t3_intervals.py
#
# T3 prints one number. Its error is not the same size everywhere - the model
# reads weak systems high and severe ones low, by about 6 kt and 13 kt
# respectively - so a single RMSE quoted next to it understates the risk at the
# top of the scale, which is the end anyone cares about.
#
# The fix needs no retraining. We already hold out-of-fold predictions for every
# patch, each made by a model that never saw that storm, so the residuals are an
# honest sample of the error. Bin them by what the model *said* - the only thing
# available at inference - take empirical quantiles inside each bin, and you have
# an interval. This is conformal prediction, and it is the same move the forecast
# cone already makes: fit the spread on held-out data, then verify the coverage
# you claimed.
#
# The bins are chosen on the prediction, not the truth. Binning by truth would
# leak: at inference nobody knows the truth, and an interval fitted that way
# looks far better than it is.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

OOF = ROOT / "reports" / "cv" / "t3_best_track_rotate_joint_oof.csv"
OUT = ROOT / "reports" / "t3_intervals.json"

# edges on the PREDICTED value, in knots
EDGES = [0, 34, 48, 64, 90, 1e9]
NAMES = ["<34", "34-48", "48-64", "64-90", "90+"]
LEVELS = (0.50, 0.67, 0.90)


def band_of(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(pred, EDGES[1:-1]), 0, len(NAMES) - 1)


def quantiles(resid: np.ndarray, level: float) -> tuple[float, float]:
    lo = (1 - level) / 2
    return float(np.quantile(resid, lo)), float(np.quantile(resid, 1 - lo))


def main() -> None:
    d = pd.read_csv(OOF)
    d = d[np.isfinite(d["truth"]) & np.isfinite(d["cnn_raw"])].reset_index(drop=True)
    pred, truth = d["cnn_raw"].to_numpy(), d["truth"].to_numpy()
    resid = truth - pred                      # what to add to the prediction
    band = band_of(pred)
    folds = d["fold"].to_numpy()

    print("=" * 74)
    print("T3 PREDICTION INTERVALS - fitted on out-of-fold residuals")
    print("=" * 74)
    print(f"{len(d)} patches with best-track truth, {d['storm'].nunique()} storms\n")

    print("residuals by predicted band (truth minus prediction):")
    print(f"  {'predicted':<10}{'n':>6}{'median':>9}{'mean':>8}{'p16':>8}{'p84':>8}")
    for b, name in enumerate(NAMES):
        m = band == b
        if not m.sum():
            continue
        r = resid[m]
        print(f"  {name:<10}{m.sum():>6}{np.median(r):>9.1f}{r.mean():>8.1f}"
              f"{np.quantile(r, .16):>8.1f}{np.quantile(r, .84):>8.1f}")

    # Honest coverage: the quantiles for a fold come from the other folds only.
    out = {"source": OOF.name, "n": int(len(d)), "storms": int(d["storm"].nunique()),
           "edges_kt": EDGES[:-1] + ["inf"], "bands": NAMES, "levels": {}}
    print("\ncoverage, each fold's interval fitted on the other four:")
    print(f"  {'level':>6}{'target':>9}{'measured':>10}{'mean width':>12}")
    for level in LEVELS:
        lo_hi = np.zeros((len(d), 2))
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            for b in range(len(NAMES)):
                fit = tr & (band == b)
                use = te & (band == b)
                if not use.sum():
                    continue
                if fit.sum() < 20:            # too thin to fit a band: pool
                    fit = tr
                lo, hi = quantiles(resid[fit], level)
                lo_hi[use] = (lo, hi)
        lower, upper = pred + lo_hi[:, 0], pred + lo_hi[:, 1]
        inside = (truth >= lower) & (truth <= upper)
        width = float(np.mean(upper - lower))
        print(f"  {level:>6.2f}{level:>9.2f}{inside.mean():>10.3f}{width:>10.1f} kt")

        # the table that will actually be served, fitted on everything
        table = {}
        for b, name in enumerate(NAMES):
            m = band == b
            if m.sum() >= 20:
                lo, hi = quantiles(resid[m], level)
            else:
                lo, hi = quantiles(resid, level)
            table[name] = {"lower_offset_kt": lo, "upper_offset_kt": hi, "n": int(m.sum())}
        out["levels"][f"{level:.2f}"] = {
            "coverage": float(inside.mean()),
            "mean_width_kt": width,
            "by_band": table,
            "coverage_by_band": {
                NAMES[b]: float(inside[band == b].mean()) if (band == b).sum() else None
                for b in range(len(NAMES))},
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    print("\nnote the bands here are on the PREDICTION, and the -13 kt bias in")
    print("limitations is on the TRUTH. they are not the same number and both are")
    print("right: when truth is 100 kt the model says about 87, but when the model")
    print("says 100 the truth is usually near it. that is regression to the mean")
    print("seen from its two ends, and only the first is measurable at inference,")
    print("which is why the interval is conditioned this way. it still leans")
    print("upward at the top of the scale, which is the under-read showing through.")


if __name__ == "__main__":
    main()
