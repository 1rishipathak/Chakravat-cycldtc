# storm-grouped cross-validation for T2 and T3, with bootstrap intervals
#
#     python src/cv_vision.py t3 --target best_track --aug rotate
#     python src/cv_vision.py t2 --aug rotate
#
# a single held-out split of 14 storms moves a lot between draws - the same T3
# checkpoint scored 13.9 kt on its validation storms and 10.95 on its test
# storms. k-fold by storm gives every patch exactly one prediction from a model
# that never saw its storm, so the score uses all ~950 patches instead of ~100,
# and resampling whole storms gives an honest 95% interval.
#
# each fold holds one bucket out for testing, uses the next bucket for early
# stopping and calibration, and trains on the rest. nothing from the test
# bucket touches training, stopping or calibration.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import IMD_CATEGORIES               # noqa: E402
from vision.scenes import AUGMENTATIONS                 # noqa: E402

OUT = ROOT / "reports" / "cv"
BOOT_REPS = 2000


def fold_of(storms: pd.Series, k: int) -> np.ndarray:
    # hashed on storm id, so adding storms never reshuffles existing ones
    return storms.map(lambda s: int(hashlib.md5(f"cv:{s}".encode()).hexdigest()[:8], 16) % k
                      ).to_numpy()


def storm_bootstrap(storms: np.ndarray, stat, reps: int = BOOT_REPS, seed: int = 0):
    # 95% interval for stat(idx), resampling whole storms with replacement
    uniq = np.unique(storms)
    groups = [np.flatnonzero(storms == s) for s in uniq]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def imd_index(kt: np.ndarray) -> np.ndarray:
    edges = [lo for _, lo, _ in IMD_CATEGORIES[1:]]
    return np.digitize(np.asarray(kt, float), edges)


# ---------------------------------------------------------------------- T3
def run_t3(args, device, log):
    import train_intensity_gridsat as t3

    patches, meta, y = (t3.load_joint(args.target) if args.sources == "joint"
                        else t3.load_patches(args.target))
    env = t3.env_matrix(meta) if args.env else None
    if args.env and env is None:
        raise SystemExit("no era5.csv sidecar - run src/add_era5_to_patches.py first")
    if "sensor" not in meta.columns:
        meta = meta.assign(sensor="gridsat")
    folds = fold_of(meta["storm"], args.folds)
    storms = meta["storm"].to_numpy()
    log(f"T3 CV  target={args.target}  aug={args.aug}  patches={len(meta)}  "
        f"storms={meta['storm'].nunique()}  folds={args.folds}")
    for f in range(args.folds):
        log(f"  fold {f}: {int((folds == f).sum())} patches, "
            f"{meta.loc[folds == f, 'storm'].nunique()} storms")

    raw = np.full(len(y), np.nan)
    cal = np.full(len(y), np.nan)
    mean_b = np.full(len(y), np.nan)
    cold_b = np.full(len(y), np.nan)
    fold_info = []
    for f in range(args.folds):
        t0 = time.time()
        test = np.flatnonzero(folds == f)
        val = np.flatnonzero(folds == (f + 1) % args.folds)
        train = np.flatnonzero((folds != f) & (folds != (f + 1) % args.folds))
        log(f"\nfold {f}  train {len(train)}  val {len(val)}  test {len(test)}")

        model, info = t3.fit(patches, y, train, val, device, aug=args.aug, env=env,
                             seed=args.seed + f, log=log, epochs=args.epochs or t3.EPOCHS)
        scale = (info["y_mean"], info["y_std"])
        es = info.get("env_scaled")
        c = t3.fit_calibration(
            t3.predict_idx(model, patches, y, val, scale, device, env=es), y[val])
        p = t3.predict_idx(model, patches, y, test, scale, device, env=es)
        raw[test] = t3.apply_calibration(p, None)
        cal[test] = t3.apply_calibration(p, c)

        base = t3.fit_baselines(patches, y, np.concatenate([train, val]), test)
        mean_b[test], cold_b[test] = base["predict_the_mean"], base["cold_cloud"]
        # env_scaled is the fold's standardised environment matrix, needed for
        # predicting and far too large to write into a report
        info = {k: v for k, v in info.items() if k != "env_scaled"}
        fold_info.append({"fold": f, **info, "calibration": c,
                          "test_rmse_raw": t3.rmse(raw[test], y[test]),
                          "test_rmse_cal": t3.rmse(cal[test], y[test]),
                          "seconds": round(time.time() - t0)})
        log(f"  fold {f} test RMSE raw {fold_info[-1]['test_rmse_raw']:.2f}  "
            f"calibrated {fold_info[-1]['test_rmse_cal']:.2f}  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    def summary(pred, rows=None):
        rows = np.arange(len(y)) if rows is None else rows
        p, t, st = pred[rows], y[rows], storms[rows]
        s = t3.summarise(p, t)
        s["rmse_ci95"] = storm_bootstrap(st, lambda i: float(np.sqrt(np.mean((p[i] - t[i]) ** 2))))
        s["mae_ci95"] = storm_bootstrap(st, lambda i: float(np.mean(np.abs(p[i] - t[i]))))
        s["bias_ci95"] = storm_bootstrap(st, lambda i: float(np.mean(p[i] - t[i])))
        cat_t, cat_p = imd_index(t), imd_index(p)
        s["imd_category_exact"] = float((cat_t == cat_p).mean())
        s["imd_category_within_one"] = float((np.abs(cat_t - cat_p) <= 1).mean())
        return s

    # the headline stays GridSat rows over every storm, so a joint run is
    # comparable with the GridSat-only one it replaces; INSAT rows are reported
    # beside it rather than mixed into the same number
    sensor = meta["sensor"].to_numpy()
    grid_rows = np.flatnonzero(sensor == "gridsat")
    result = {
        "task": "t3", "target": args.target, "augmentation": args.aug,
        "sources": args.sources, "folds": args.folds,
        "patches": int(len(grid_rows)), "storms": int(meta["storm"].nunique()),
        "scored_on": "gridsat patches",
        "cnn_raw": summary(raw, grid_rows), "cnn_recalibrated": summary(cal, grid_rows),
        "predict_the_mean": summary(mean_b, grid_rows), "cold_cloud": summary(cold_b, grid_rows),
        "fold_detail": fold_info,
    }
    if (sensor == "insat").any():
        result["on_insat"] = summary(raw, np.flatnonzero(sensor == "insat"))

    # the operational objective technique on the same scenes, when the truth
    # is best track. ADT's wind exists for most but not all patches.
    if args.target == "best_track":
        has = np.flatnonzero((meta["vmax_kt"] > 0).to_numpy() & (sensor == "gridsat"))
        adt = meta["vmax_kt"].to_numpy(dtype=float)
        bench = {"n": int(len(has)), "storms": int(meta.iloc[has]["storm"].nunique()),
                 "adt": summary(adt, has)}
        t_h, a_h, s_h = y[has], adt[has], storms[has]
        for name, pred in (("cnn_raw", raw), ("cnn_recalibrated", cal)):
            bench[name] = summary(pred, has)
            p_h = pred[has]
            # paired: both errors come from the same resampled storms
            bench[f"{name}_minus_adt_rmse_ci95"] = storm_bootstrap(
                s_h, lambda i, p_h=p_h: float(np.sqrt(np.mean((p_h[i] - t_h[i]) ** 2))
                                              - np.sqrt(np.mean((a_h[i] - t_h[i]) ** 2))))
        result["adt_benchmark"] = bench

    cols = ["storm", "season", "time", "scene", "lat", "lon", "vmax_kt", "bt_vmax_kt"]
    oof = meta[[c for c in cols if c in meta.columns]].copy()
    oof["truth"], oof["fold"] = y, folds
    oof["cnn_raw"], oof["cnn_recalibrated"] = raw, cal
    oof["predict_the_mean"], oof["cold_cloud"] = mean_b, cold_b
    return result, oof


# ---------------------------------------------------------------------- T2
def run_t2(args, device, log):
    import train_scene as t2

    patches, meta, classes, y, _ = t2.load_scenes(rebuild_check=False)
    folds = fold_of(meta["storm"], args.folds)
    storms = meta["storm"].to_numpy()
    n = len(classes)
    log(f"T2 CV  aug={args.aug}  patches={len(meta)}  storms={meta['storm'].nunique()}  "
        f"classes={classes}")

    pred = np.full(len(y), -1)
    probs = np.full((len(y), n), np.nan)
    major = np.full(len(y), -1)
    cold = np.full(len(y), -1)
    fold_info = []
    for f in range(args.folds):
        t0 = time.time()
        test = np.flatnonzero(folds == f)
        val = np.flatnonzero(folds == (f + 1) % args.folds)
        train = np.flatnonzero((folds != f) & (folds != (f + 1) % args.folds))
        log(f"\nfold {f}  train {len(train)}  val {len(val)}  test {len(test)}")
        model, info = t2.fit(patches, meta, y, classes, train, val, device,
                             aug=args.aug, seed=args.seed + f, log=log,
                             epochs=args.epochs or t2.EPOCHS)
        pred[test], probs[test] = t2.predict_idx(model, patches, meta, classes, test, device)
        base = t2.fit_baselines(patches, y, np.concatenate([train, val]), test, n)
        major[test], cold[test] = base["majority"], base["cold_cloud_stats"]
        fold_info.append({"fold": f, **info,
                          "test_macro_f1": t2.macro_f1(y[test], pred[test], n),
                          "seconds": round(time.time() - t0)})
        log(f"  fold {f} test macro-F1 {fold_info[-1]['test_macro_f1']:.3f}  "
            f"({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    def summary(p):
        return {
            "macro_f1": t2.macro_f1(y, p, n),
            "macro_f1_ci95": storm_bootstrap(storms, lambda i: t2.macro_f1(y[i], p[i], n)),
            "accuracy": float((y == p).mean()),
            "accuracy_ci95": storm_bootstrap(storms, lambda i: float((y[i] == p[i]).mean())),
            "per_class_f1": dict(zip(classes, t2.per_class_f1(y, p, n))),
            "support": dict(zip(classes, np.bincount(y, minlength=n).tolist())),
            "confusion": [[int(np.sum((y == i) & (p == j))) for j in range(n)]
                          for i in range(n)],
        }

    result = {"task": "t2", "augmentation": args.aug, "folds": args.folds,
              "classes": classes, "patches": int(len(y)),
              "storms": int(meta["storm"].nunique()),
              "cnn": summary(pred), "majority": summary(major),
              "cold_cloud_stats": summary(cold), "fold_detail": fold_info}
    oof = meta[["storm", "season", "time", "scene", "lat", "lon"]].copy()
    oof["fold"], oof["pred"] = folds, [classes[i] for i in pred]
    for j, c in enumerate(classes):
        oof[f"p_{c}"] = probs[:, j]
    return result, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["t2", "t3"])
    ap.add_argument("--target", choices=["best_track", "adt"], default="best_track")
    ap.add_argument("--sources", choices=["gridsat", "joint"], default="gridsat")
    ap.add_argument("--env", action="store_true",
                    help="give T3 the ERA5 environment alongside the image")
    ap.add_argument("--aug", choices=AUGMENTATIONS, default="rotate")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0,
                    help="cap epochs per fold; only for smoke-testing the runner")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    name = (f"t3_{args.target}_{args.aug}"
            + ("_joint" if args.sources == "joint" else "")
            + ("_env" if args.env else "")
            if args.task == "t3" else f"t2_{args.aug}")
    if args.epochs:
        name += f"_smoke{args.epochs}"
    log_path = OUT / f"{name}.log"
    logf = open(log_path, "w", encoding="utf-8")

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
