# does making the cold-cloud half sensor-aware close the T2 gap on INSAT?
#
#     python src/eval_t2_stat_variants.py
#
# answer: no. it lifts that half a long way on INSAT and the blend barely
# moves, because the CNN is the half that is weak. kept so the numbers
# quoted in limitations.md section 7 can be regenerated.
#
# the CNN half is expensive and unchanged, so we reuse its saved out-of-fold
# predictions and refit only the statistics half, fold for fold, exactly as
# cv_sensors.run_t2 did. that makes this comparable to sensors_t2_*.json and
# costs seconds rather than a GPU afternoon.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import train_scene as t2                       # noqa: E402
from cv_vision import storm_bootstrap          # noqa: E402

GRIDSAT = ROOT / "data" / "processed" / "scenes"
INSAT = ROOT / "data" / "processed" / "scenes_insat"

pg, mg, classes, yg, _ = t2.load_scenes(rebuild_check=False, cache=GRIDSAT)
pi, mi, _, yi, _ = t2.load_scenes(rebuild_check=False, cache=INSAT, classes=classes)
patches = np.concatenate([pg, pi])
meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")], ignore_index=True)
y = np.concatenate([yg, yi])
sensor = meta["sensor"].to_numpy()
storms = meta["storm"].to_numpy()
n = len(classes)

oof = pd.read_csv(ROOT / "reports" / "cv" / "sensors_t2_gridsat_oof.csv")
assert len(oof) == len(meta), (len(oof), len(meta))
assert (oof["sensor"].to_numpy() == sensor).all(), "row order differs from the OOF file"
assert (oof["storm"].astype(str).to_numpy() == meta["storm"].astype(str).to_numpy()).all()
folds = oof["fold"].to_numpy()
cnn_p = oof[[f"cnn_{c}" for c in classes]].to_numpy()
saved_stat = oof[[f"stat_{c}" for c in classes]].to_numpy()

feats = t2.stats_features(patches, np.arange(len(y)))
is_insat = (sensor == "insat").astype(np.float32)[:, None]
print(f"patches {len(y)}  gridsat {(sensor=='gridsat').sum()}  insat {(sensor=='insat').sum()}"
      f"  features {feats.shape[1]}  classes {classes}")


def lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def variant(name, fit_fn):
    """fit_fn(train_rows, test_rows) -> probabilities for test_rows"""
    p = np.full((len(y), n), np.nan)
    for f in sorted(set(folds)):
        test = np.flatnonzero(folds == f)
        train = np.flatnonzero(folds != f)
        p[test] = fit_fn(train, test)
    assert not np.isnan(p).any()
    return name, p


def gridsat_only(train, test):
    rows = train[sensor[train] == "gridsat"]
    m = lr().fit(feats[rows], y[rows])
    return m.predict_proba(feats[test])


def pooled(train, test):
    m = lr().fit(feats[train], y[train])
    return m.predict_proba(feats[test])


def with_indicator(train, test):
    x = np.hstack([feats, is_insat])
    m = lr().fit(x[train], y[train])
    return m.predict_proba(x[test])


def per_sensor_scaled(train, test):
    # standardise inside each sensor, then one shared logistic model: the
    # shapes are shared, the offsets are not
    x = np.empty_like(feats)
    for s in ("gridsat", "insat"):
        fit_rows = train[sensor[train] == s]
        sc = StandardScaler().fit(feats[fit_rows])
        rows = np.flatnonzero(sensor == s)
        x[rows] = sc.transform(feats[rows])
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x[train], y[train])
    return m.predict_proba(x[test])


def separate_models(train, test):
    out = np.zeros((len(test), n))
    for s in ("gridsat", "insat"):
        fit_rows = train[sensor[train] == s]
        m = lr().fit(feats[fit_rows], y[fit_rows])
        want = np.flatnonzero(sensor[test] == s)
        if len(want):
            out[want] = m.predict_proba(feats[test][want])
    return out


VARIANTS = [("gridsat only (served)", gridsat_only),
            ("pooled, no sensor term", pooled),
            ("pooled + sensor flag", with_indicator),
            ("per-sensor scaling", per_sensor_scaled),
            ("separate per sensor", separate_models)]

insat_storms = set(mi["storm"])
in_era = meta["storm"].isin(insat_storms).to_numpy()


def score(probs, s):
    rows = np.flatnonzero((sensor == s) & in_era)
    p, t, st = probs[rows].argmax(1), y[rows], storms[rows]
    f1 = t2.macro_f1(t, p, n)
    ci = storm_bootstrap(st, lambda i: t2.macro_f1(t[i], p[i], n))
    return f1, ci, len(rows)


# sanity: the saved stat probabilities should match the served variant
_, reproduced = variant("check", gridsat_only)
print(f"\nreproduction check vs the saved stat column: "
      f"max abs diff {np.abs(reproduced - saved_stat).max():.4f}")

print(f"\n{'stat half':<24}{'GridSat hybrid':>22}{'INSAT hybrid':>22}{'INSAT stat alone':>20}")
for name, fn in VARIANTS:
    _, stat_p = variant(name, fn)
    hyb = (1 - t2.BLEND) * cnn_p + t2.BLEND * stat_p
    gf, gci, gn = score(hyb, "gridsat")
    inf, ici, inn = score(hyb, "insat")
    sf, _, _ = score(stat_p, "insat")
    print(f"{name:<24}{gf:>8.3f} [{gci[0]:.3f},{gci[1]:.3f}]{inf:>8.3f} [{ici[0]:.3f},{ici[1]:.3f}]"
          f"{sf:>20.3f}")
print(f"\n(n = {gn} GridSat and {inn} INSAT patches, INSAT-era storms only,"
      f" blend {t2.BLEND}, CNN half unchanged and GridSat-trained)")
