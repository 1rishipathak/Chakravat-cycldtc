# Does seeing through the cirrus fix the storms we read low?
#
#     python src/eval_microwave.py
#
# T3 reads the strongest storms low - 8 kt at 64-90 and 13 kt at 90+ - and four
# separate attempts to correct that failed, because none of them addressed the
# cause. An infrared window channel sees the top of the cirrus canopy, and above
# a mature eyewall the canopy is uniformly cold whatever is happening beneath it.
# The information is not in the image to be recovered.
#
# At 89 GHz the canopy is transparent and precipitation-sized ice scatters the
# signal, so the eyewall prints as a cold ring around a warm eye. That is a
# direct measure of the thing the infrared cannot see.
#
# The test: on the 456 patches that have an overpass, does adding microwave
# structure to the existing T3 estimate reduce its error? The CNN's own
# out-of-fold predictions are reused, so this is a correction fitted on top of a
# model that never saw these storms, evaluated on the same folds it was.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MW = ROOT / "data" / "processed" / "microwave"
OOF = ROOT / "reports" / "cv" / "t3_best_track_rotate_joint_oof.csv"
REPORT = ROOT / "reports" / "microwave_intensity.json"
GRID_DEG = 0.05


def features(grid: np.ndarray) -> dict:
    """Storm-relative structure of the 89 GHz scattering signal.

    Cold means ice aloft means deep convection. The eye/eyewall contrast is the
    part infrared cannot supply: a mature storm has a warm centre ringed by very
    cold convection, and how deep that contrast runs is what the Dvorak analyst
    is really estimating when they look at a microwave pass.
    """
    n = grid.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r_px = np.hypot(yy - n / 2, xx - n / 2)
    r_km = r_px * GRID_DEG * 111.32

    def band(lo, hi):
        m = (r_km >= lo) & (r_km < hi) & np.isfinite(grid)
        return grid[m] if m.sum() >= 20 else np.array([np.nan])

    core = band(0, 25)          # the eye, if there is one
    wall = band(25, 75)         # where an eyewall lives
    outer = band(75, 200)       # the rainbands

    f = {
        "mw_min": float(np.nanmin(grid)) if np.isfinite(grid).any() else np.nan,
        "mw_core_mean": float(np.nanmean(core)),
        "mw_wall_mean": float(np.nanmean(wall)),
        "mw_wall_min": float(np.nanmin(wall)) if np.isfinite(wall).any() else np.nan,
        "mw_outer_mean": float(np.nanmean(outer)),
        # a warm eye inside a cold wall: positive when the eye is clear
        "mw_eye_contrast": float(np.nanmean(core) - np.nanmean(wall)),
        # how much of the storm is scattering hard, at three depths
        "mw_frac_250": float(np.nanmean(grid < 250)),
        "mw_frac_200": float(np.nanmean(grid < 200)),
        "mw_frac_150": float(np.nanmean(grid < 150)),
        "mw_std": float(np.nanstd(grid)),
    }
    return f


def main() -> None:
    from sklearn.linear_model import RidgeCV  # noqa: F401 - used in the gate block too

    idx = pd.read_csv(MW / "index.csv")
    oof = pd.read_csv(OOF)
    oof["time"] = oof["time"].astype(str)
    idx["time"] = idx["time"].astype(str)

    # the joint run concatenated GridSat patches then INSAT ones, and the two
    # share (storm, time) - the same storm at the same slot through two
    # instruments. joining on that key alone matched every microwave patch
    # twice. the microwave grids were cut against the GridSat table, so that is
    # the half to keep.
    n_gridsat = int(pd.read_csv(ROOT / "data" / "processed" / "scenes"
                                / "patches.csv")["bt_vmax_kt"].notna().sum())
    if len(oof) > n_gridsat:
        oof = oof.iloc[:n_gridsat].copy()
    print(f"using the {len(oof)} GridSat rows of the joint out-of-fold table")

    rows = []
    for r in idx.itertuples():
        path = MW / r.file
        if not path.exists():
            continue
        g = np.load(path)
        if not np.isfinite(g).any():
            continue
        rows.append({"storm": r.storm, "time": r.time, "gap_min": r.gap_min,
                     "sensor": r.sensor, **features(g)})
    mw = pd.DataFrame(rows)
    print(f"{len(mw)} patches with usable 89 GHz imagery")

    df = oof.merge(mw, on=["storm", "time"], how="inner")
    df = df[np.isfinite(df["truth"]) & np.isfinite(df["cnn_raw"])].reset_index(drop=True)
    feat_cols = [c for c in mw.columns if c.startswith("mw_")]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"{len(df)} of those join to a T3 out-of-fold prediction, "
          f"{df['storm'].nunique()} storms\n")
    if len(df) < 120:
        raise SystemExit("too few joined patches to say anything")

    y = df["truth"].to_numpy()
    base = df["cnn_raw"].to_numpy()
    folds = df["fold"].to_numpy()
    storms = df["storm"].to_numpy()

    def rmse(p):
        return float(np.sqrt(np.mean((p - y) ** 2)))

    # A correction fitted per fold on the other folds, exactly like the CNN.
    #
    # Ridge rather than boosted trees, and this is not a detail: with a few
    # hundred training rows a tree ensemble fits the residual noise, and the
    # control below proves it - a refit carrying no new information at all made
    # the error worse. A penalised linear model on standardised features can
    # still express "colder eyewall means stronger storm" without inventing
    # structure that is not there.
    def corrected(cols, label):
        pred = np.full(len(df), np.nan)
        X = df[cols].to_numpy(dtype=float)
        for f in sorted(set(folds)):
            te, tr = folds == f, folds != f
            if tr.sum() < 60 or te.sum() == 0:
                pred[te] = base[te]
                continue
            mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
            sd = np.where(sd > 1e-9, sd, 1.0)
            m = RidgeCV(alphas=np.logspace(-1, 4, 24))
            m.fit((X[tr] - mu) / sd, y[tr] - base[tr])
            pred[te] = base[te] + m.predict((X[te] - mu) / sd)
        return label, pred

    from cv_vision import storm_bootstrap

    results = {}
    print(f"{'model':<34}{'RMSE':>8}{'95% interval':>18}{'bias':>8}{'MAE':>8}")
    for label, pred in [("T3 alone (the served model)", base),
                        corrected(["cnn_raw"], "T3 + a refit on itself"),
                        corrected(["cnn_raw"] + feat_cols, "T3 + 89 GHz structure"),
                        corrected(feat_cols, "89 GHz structure alone")]:
        r = rmse(pred)
        ci = storm_bootstrap(storms, lambda i: float(np.sqrt(np.mean(
            (pred[i] - y[i]) ** 2))))
        results[label] = {"rmse": r, "ci95": list(ci),
                          "bias": float(np.mean(pred - y)),
                          "mae": float(np.mean(np.abs(pred - y)))}
        print(f"{label:<34}{r:>8.2f}   [{ci[0]:.2f},{ci[1]:.2f}]"
              f"{np.mean(pred - y):>8.2f}{np.mean(np.abs(pred - y)):>8.2f}")

    # the band that motivated all of this
    _, with_mw = corrected(["cnn_raw"] + feat_cols, "x")
    print(f"\nbias by truth band, which is where the problem was:")
    print(f"  {'band':<10}{'n':>5}{'T3 alone':>11}{'with 89 GHz':>13}")
    bands = [("<34", 0, 34), ("34-48", 34, 48), ("48-64", 48, 64),
             ("64-90", 64, 90), ("90+", 90, 999)]
    band_rows = []
    for name, lo, hi in bands:
        m = (y >= lo) & (y < hi)
        if m.sum() < 8:
            continue
        a, b = float(np.mean(base[m] - y[m])), float(np.mean(with_mw[m] - y[m]))
        band_rows.append({"band": name, "n": int(m.sum()), "t3": a, "with_mw": b})
        print(f"  {name:<10}{int(m.sum()):>5}{a:>11.1f}{b:>13.1f}")

    # The subset that looks like a discovery, and the control that undoes it.
    #
    # Restricted to storms whose TRUTH is 64 kt or more, adding 89 GHz appears
    # to cut the error by about 3 kt. It is not real. Selecting rows by truth
    # while the model is known to read that band low guarantees the selected
    # residuals are biased positive, and any model with an intercept banks that
    # bias whether or not it is given microwave at all - which is why the
    # control, refitted on the T3 estimate alone, scores better still.
    #
    # The honest version gates on the prediction, because that is what exists at
    # runtime. Gated that way the gain disappears and reverses.
    print(f"\n{'gate':<34}{'n':>5}{'T3':>8}{'+89GHz':>9}{'control':>9}")
    gates = []
    for label, sub in (("predicted 64 kt or more", df[df["cnn_raw"] >= 64]),
                       ("predicted 50 kt or more", df[df["cnn_raw"] >= 50]),
                       ("truth 64 kt or more (not usable)", df[df["truth"] >= 64])):
        if len(sub) < 60:
            continue
        sub = sub.reset_index(drop=True)
        yy, bb, ff = (sub["truth"].to_numpy(), sub["cnn_raw"].to_numpy(),
                      sub["fold"].to_numpy())

        def fit(cols):
            X = sub[cols].to_numpy(dtype=float)
            pr = bb.copy()
            for k in sorted(set(ff)):
                te, tr = ff == k, ff != k
                if tr.sum() < 40 or te.sum() == 0:
                    continue
                mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
                sd = np.where(sd > 1e-9, sd, 1.0)
                mm = RidgeCV(alphas=np.logspace(-1, 4, 24))
                mm.fit((X[tr] - mu) / sd, yy[tr] - bb[tr])
                pr[te] = bb[te] + mm.predict((X[te] - mu) / sd)
            return float(np.sqrt(np.mean((pr - yy) ** 2)))

        row = {"gate": label, "n": int(len(sub)),
               "t3": float(np.sqrt(np.mean((bb - yy) ** 2))),
               "with_mw": fit(["cnn_raw"] + feat_cols),
               "control": fit(["cnn_raw"])}
        gates.append(row)
        print(f"{label:<34}{row['n']:>5}{row['t3']:>8.2f}"
              f"{row['with_mw']:>9.2f}{row['control']:>9.2f}")
    print("\nwhere the control beats the microwave column, the apparent gain was")
    print("the selection, not the instrument.")

    out = {"gates": gates,
           "patches": int(len(df)), "storms": int(df["storm"].nunique()),
           "median_gap_min": float(df["gap_min"].median()),
           "features": feat_cols, "results": results, "bands": band_rows,
           "note": "a correction fitted per fold on the other folds, on top of "
                   "T3 out-of-fold predictions, so nothing here has seen its "
                   "own storm"}
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORT}")

    a = results["T3 alone (the served model)"]
    c = results["T3 + 89 GHz structure"]
    d = results["T3 + a refit on itself"]
    print(f"\n89 GHz moves RMSE {a['rmse']:.2f} -> {c['rmse']:.2f} kt")
    print(f"a refit with no new information moves it to {d['rmse']:.2f}, which is")
    print("the control: any gain below that is the refit, not the instrument.")
    inside = a["ci95"][0] <= c["rmse"] <= a["ci95"][1]
    print(f"the new value sits {'inside' if inside else 'OUTSIDE'} the interval "
          f"of the old one, [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]")


if __name__ == "__main__":
    main()
