# does INSAT help? GridSat-only, joint and INSAT-only training, scored per sensor
#
#     python src/cv_sensors.py t3 --mode gridsat
#     python src/cv_sensors.py t3 --mode joint
#     python src/cv_sensors.py t2 --mode insat
#
# the INSAT cache holds the same ADT records at the same slots as the GridSat
# one, so folds are hashed on storm id across both: a storm's GridSat and INSAT
# patches always sit in the same fold and neither sensor can leak a test storm
# into training. every mode predicts every patch of both sensors once, and the
# summary scores each sensor only on storms that have INSAT imagery, so the
# three modes are compared on identical rows.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_vision import fold_of, imd_index, storm_bootstrap   # noqa: E402

GRIDSAT_CACHE = ROOT / "data" / "processed" / "scenes"
INSAT_CACHE = ROOT / "data" / "processed" / "scenes_insat"
OUT = ROOT / "reports" / "cv"
MODES = ("gridsat", "joint", "insat")


def train_rows(sensor: np.ndarray, rows: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gridsat":
        return rows[sensor[rows] == "gridsat"]
    if mode == "insat":
        return rows[sensor[rows] == "insat"]
    return rows


def run_t3(args, device, log):
    import train_intensity_gridsat as t3

    pg, mg, yg = t3.load_patches("best_track", GRIDSAT_CACHE)
    pi, mi, yi = t3.load_patches("best_track", INSAT_CACHE)
    patches = np.concatenate([pg, pi])
    meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")], ignore_index=True)
    y = np.concatenate([yg, yi])
    sensor = meta["sensor"].to_numpy()
    folds = fold_of(meta["storm"], args.folds)
    insat_storms = set(mi["storm"])
    log(f"T3 sensors  mode={args.mode}  gridsat {len(yg)}  insat {len(yi)}  "
        f"insat-era storms {len(insat_storms)}")

    pred = np.full(len(y), np.nan)
    for f in range(args.folds):
        t0 = time.time()
        test = np.flatnonzero(folds == f)
        val = train_rows(sensor, np.flatnonzero(folds == (f + 1) % args.folds), args.mode)
        train = train_rows(sensor, np.flatnonzero((folds != f) & (folds != (f + 1) % args.folds)),
                           args.mode)
        log(f"\nfold {f}  train {len(train)}  val {len(val)}  test {len(test)}")
        model, info = t3.fit(patches, y, train, val, device, aug="rotate", seed=args.seed + f,
                             log=log, epochs=args.epochs or t3.EPOCHS)
        pred[test] = t3.apply_calibration(
            t3.predict_idx(model, patches, y, test, (info["y_mean"], info["y_std"]), device), None)
        log(f"  fold {f} done ({time.time() - t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    storms = meta["storm"].to_numpy()
    result = {"task": "t3", "mode": args.mode, "insat_era_storms": len(insat_storms)}
    for s in ("gridsat", "insat"):
        rows = np.flatnonzero((sensor == s) & meta["storm"].isin(insat_storms).to_numpy())
        p, t, st = pred[rows], y[rows], storms[rows]
        summ = t3.summarise(p, t)
        summ["rmse_ci95"] = storm_bootstrap(st, lambda i: float(np.sqrt(np.mean((p[i] - t[i]) ** 2))))
        summ["imd_category_exact"] = float((imd_index(p) == imd_index(t)).mean())
        result[f"on_{s}"] = summ
        log(f"  on {s:<8} n={len(rows)}  RMSE {summ['rmse']:.2f} "
            f"[{summ['rmse_ci95'][0]:.2f}, {summ['rmse_ci95'][1]:.2f}]  bias {summ['bias']:+.2f}")
    oof = meta[["storm", "time", "sensor"]].copy()
    oof["truth"], oof["pred"], oof["fold"] = y, pred, folds
    return result, oof


def run_t2(args, device, log):
    import train_scene as t2

    pg, mg, classes, yg, _ = t2.load_scenes(rebuild_check=False, cache=GRIDSAT_CACHE)
    pi, mi, _, yi, _ = t2.load_scenes(rebuild_check=False, cache=INSAT_CACHE, classes=classes)
    patches = np.concatenate([pg, pi])
    meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")], ignore_index=True)
    y = np.concatenate([yg, yi])
    sensor = meta["sensor"].to_numpy()
    folds = fold_of(meta["storm"], args.folds)
    insat_storms = set(mi["storm"])
    n = len(classes)
    log(f"T2 sensors  mode={args.mode}  gridsat {len(yg)}  insat {len(yi)}  classes {classes}")

    feats = t2.stats_features(patches, np.arange(len(y)))
    cnn_p = np.full((len(y), n), np.nan)
    stat_p = np.full((len(y), n), np.nan)
    for f in range(args.folds):
        t0 = time.time()
        test = np.flatnonzero(folds == f)
        val = train_rows(sensor, np.flatnonzero(folds == (f + 1) % args.folds), args.mode)
        train = train_rows(sensor, np.flatnonzero((folds != f) & (folds != (f + 1) % args.folds)),
                           args.mode)
        log(f"\nfold {f}  train {len(train)}  val {len(val)}  test {len(test)}")
        model, _ = t2.fit(patches, meta, y, classes, train, val, device, aug=args.aug,
                          seed=args.seed + f, log=log, epochs=args.epochs or t2.EPOCHS,
                          norm=args.norm)
        _, cnn_p[test] = t2.predict_idx(model, patches, meta, classes, test, device,
                                        norm=args.norm)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
        fit_rows = np.concatenate([train, val])
        lr.fit(feats[fit_rows], y[fit_rows])
        stat_p[test] = lr.predict_proba(feats[test])
        log(f"  fold {f} done ({time.time() - t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    storms = meta["storm"].to_numpy()
    hyb = (1 - t2.BLEND) * cnn_p + t2.BLEND * stat_p
    result = {"task": "t2", "mode": args.mode, "augmentation": args.aug,
              "normalisation": args.norm, "classes": classes,
              "insat_era_storms": len(insat_storms)}
    for s in ("gridsat", "insat"):
        rows = np.flatnonzero((sensor == s) & meta["storm"].isin(insat_storms).to_numpy())
        out = {"n": int(len(rows))}
        for name, probs in (("cnn", cnn_p), ("stats", stat_p), ("hybrid", hyb)):
            p = probs[rows].argmax(1)
            t, st = y[rows], storms[rows]
            out[name] = {"macro_f1": t2.macro_f1(t, p, n),
                         "macro_f1_ci95": storm_bootstrap(st, lambda i: t2.macro_f1(t[i], p[i], n)),
                         "accuracy": float((p == t).mean())}
        result[f"on_{s}"] = out
        log(f"  on {s:<8} n={len(rows)}  hybrid macro-F1 {out['hybrid']['macro_f1']:.3f} "
            f"{[round(x, 3) for x in out['hybrid']['macro_f1_ci95']]}  cnn {out['cnn']['macro_f1']:.3f}")
    oof = meta[["storm", "time", "sensor", "scene"]].copy()
    oof["fold"] = folds
    for j, c in enumerate(classes):
        oof[f"cnn_{c}"], oof[f"stat_{c}"] = cnn_p[:, j], stat_p[:, j]
    return result, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["t2", "t3"])
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--aug", default="rotate")
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "patch"],
                    help="patch standardises each channel by its own spread, which "
                         "removes any constant offset between the two sensors")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    name = (f"sensors_{args.task}_{args.mode}"
            + ("" if args.norm == "imagenet" else f"_{args.norm}norm")
            + (f"_smoke{args.epochs}" if args.epochs else ""))
    logf = open(OUT / f"{name}.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    result, oof = (run_t3 if args.task == "t3" else run_t2)(args, device, log)
    result["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    oof.to_csv(OUT / f"{name}_oof.csv", index=False)
    log(f"\nwrote {OUT / (name + '.json')}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()
