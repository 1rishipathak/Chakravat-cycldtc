# Does the split window recover the two scene classes INSAT loses?
#
#     python src/eval_t2_split_window.py
#
# Scene typing reads 0.72 on GridSat and 0.60 on INSAT, and the paired test in
# src/eval_t2_sensors.py showed the gap is not scene composition - it is the
# sensor, and it lives almost entirely in EMBC and IRRCDO, the two classes
# defined by how cold and how uniform the central overcast is rather than by a
# geometric feature. EYE barely moves at all.
#
# TIR1 minus TIR2 measures exactly that property: how optically thick the cloud
# top is. GridSat has no second window channel, so this can only ever be an
# INSAT-side model, which is fine - detection already keeps one checkpoint per
# sensor for the same reason.
#
# The CNN half is expensive and unchanged, so its saved out-of-fold predictions
# are reused and only the statistics half is refitted, fold for fold. That makes
# this directly comparable with the numbers in limitations section 7 and costs
# seconds rather than a GPU afternoon.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_scene as t2                       # noqa: E402
from cv_vision import storm_bootstrap          # noqa: E402

GRIDSAT = ROOT / "data" / "processed" / "scenes"
INSAT = ROOT / "data" / "processed" / "scenes_insat"
OOF = ROOT / "reports" / "cv" / "sensors_t2_gridsat_oof.csv"
SW = INSAT / "split_window_stats.csv"
REPORT = ROOT / "reports" / "t2_split_window.json"
SW_COLS = ["sw_mean", "sw_core", "sw_ring", "sw_thin_frac", "sw_std"]


def lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def main() -> None:
    pg, mg, classes, yg, _ = t2.load_scenes(rebuild_check=False, cache=GRIDSAT)
    pi, mi, _, yi, _ = t2.load_scenes(rebuild_check=False, cache=INSAT, classes=classes)
    patches = np.concatenate([pg, pi])
    meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")],
                     ignore_index=True)
    y = np.concatenate([yg, yi])
    sensor = meta["sensor"].to_numpy()
    storms = meta["storm"].to_numpy()
    n_cls = len(classes)

    oof = pd.read_csv(OOF)
    assert len(oof) == len(meta), (len(oof), len(meta))
    folds = oof["fold"].to_numpy()
    cnn_p = oof[[f"cnn_{c}" for c in classes]].to_numpy()

    feats = t2.stats_features(patches, np.arange(len(y)))

    sw = pd.read_csv(SW)
    sw["time"] = sw["time"].astype(str)
    key = mi.assign(time=mi["time"].astype(str))[["storm", "time"]]
    joined = key.merge(sw[["storm", "time"] + SW_COLS].assign(
        time=lambda d: d["time"].astype(str)), on=["storm", "time"], how="left")
    sw_i = joined[SW_COLS].to_numpy(dtype=np.float32)
    have = np.isfinite(sw_i).all(axis=1)
    print(f"{len(meta)} patches: {len(mg)} gridsat, {len(mi)} insat")
    print(f"split window present for {have.sum()} of {len(mi)} insat patches "
          f"({have.mean():.0%})")

    # gaps filled with the column median so a missing scene cannot drop a patch
    med = np.nanmedian(np.where(np.isfinite(sw_i), sw_i, np.nan), axis=0)
    sw_i = np.where(np.isfinite(sw_i), sw_i, med)

    insat_rows = np.flatnonzero(sensor == "insat")
    sw_all = np.zeros((len(meta), len(SW_COLS)), dtype=np.float32)
    sw_all[insat_rows] = sw_i

    def run(use_sw: bool) -> np.ndarray:
        """Per-sensor statistics models, optionally with the split window."""
        p = np.full((len(y), n_cls), np.nan)
        for f in sorted(set(folds)):
            test, train = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
            for s in ("gridsat", "insat"):
                fit = train[sensor[train] == s]
                want = test[sensor[test] == s]
                if not len(want):
                    continue
                X = feats
                if use_sw and s == "insat":
                    X = np.hstack([feats, sw_all])
                p[want] = lr().fit(X[fit], y[fit]).predict_proba(X[want])
        return p

    insat_storms = set(mi["storm"])
    in_era = meta["storm"].isin(insat_storms).to_numpy()

    def score(probs, s):
        rows = np.flatnonzero((sensor == s) & in_era)
        pr, t, st = probs[rows].argmax(1), y[rows], storms[rows]
        f1 = t2.macro_f1(t, pr, n_cls)
        ci = storm_bootstrap(st, lambda i: t2.macro_f1(t[i], pr[i], n_cls))
        per = dict(zip(classes, [float(v) for v in t2.per_class_f1(t, pr, n_cls)]))
        return f1, ci, per, len(rows)

    out = {"classes": classes, "sw_columns": SW_COLS,
           "insat_patches_with_split_window": int(have.sum()),
           "configurations": {}}

    print(f"\n{'stat half':<28}{'GridSat hybrid':>22}{'INSAT hybrid':>22}"
          f"{'INSAT stats alone':>20}")
    for label, use_sw in (("per sensor, no split window", False),
                          ("per sensor + split window", True)):
        stat_p = run(use_sw)
        hyb = (1 - t2.BLEND) * cnn_p + t2.BLEND * stat_p
        gf, gci, gper, gn = score(hyb, "gridsat")
        inf, ici, iper, inn = score(hyb, "insat")
        sf, _, sper, _ = score(stat_p, "insat")
        print(f"{label:<28}{gf:>8.3f} [{gci[0]:.3f},{gci[1]:.3f}]"
              f"{inf:>8.3f} [{ici[0]:.3f},{ici[1]:.3f}]{sf:>20.3f}")
        out["configurations"][label] = {
            "gridsat_hybrid": gf, "gridsat_ci95": list(gci),
            "insat_hybrid": inf, "insat_ci95": list(ici),
            "insat_stats_alone": sf,
            "insat_per_class": iper, "insat_stats_per_class": sper,
        }

    a = out["configurations"]["per sensor, no split window"]
    b = out["configurations"]["per sensor + split window"]
    print(f"\nthe two classes this was built for, hybrid F1 on INSAT:")
    print(f"  {'class':<10}{'without':>10}{'with':>10}{'change':>10}")
    for c in classes:
        x, z = a["insat_per_class"][c], b["insat_per_class"][c]
        print(f"  {c:<10}{x:>10.2f}{z:>10.2f}{z - x:>+10.2f}")

    # "better" has to mean better than the noise. a gain that sits inside the
    # interval of the thing it beats is a direction, not a result.
    gain = b["insat_hybrid"] - a["insat_hybrid"]
    inside = a["insat_ci95"][0] <= b["insat_hybrid"] <= a["insat_ci95"][1]
    out["gain_insat"] = gain
    out["gain_inside_the_interval"] = bool(inside)
    out["verdict"] = ("measured direction, inside the interval" if inside
                      else "improvement beyond the interval")
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORT}")
    print(f"\nINSAT hybrid {a['insat_hybrid']:.3f} -> {b['insat_hybrid']:.3f} "
          f"({gain:+.3f}); the new value sits "
          f"{'inside' if inside else 'outside'} the old interval "
          f"[{a['insat_ci95'][0]:.3f},{a['insat_ci95'][1]:.3f}]")
    print(f"verdict: {out['verdict']}")
    print("GridSat is untouched either way - it has no second window channel, so")
    print("its statistics model is the same model in both rows.")


if __name__ == "__main__":
    main()
