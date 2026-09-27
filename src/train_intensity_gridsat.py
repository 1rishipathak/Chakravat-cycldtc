# T3 in the operational domain: intensity from GridSat imagery
#
#     python src/train_intensity_gridsat.py                    best-track target
#     python src/train_intensity_gridsat.py --target adt       the original target
#     python src/train_intensity_gridsat.py --aug rotate --no-calibrate

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision.scenes import (AUGMENTATIONS, IMAGENET_MEAN, IMAGENET_STD,  # noqa: E402
                           augment, storm_split)

CACHE = ROOT / "data" / "processed" / "scenes"
INSAT_CACHE = ROOT / "data" / "processed" / "scenes_insat"
OUT, ART = ROOT / "reports", ROOT / "artifacts"
PRETRAINED = ART / "intensity_vision.pt"

BACKBONE = "convnext_tiny"
EPOCHS, PATIENCE, BATCH, LR, SEED = 60, 12, 32, 5e-5, 0

# Huber at delta = 1 sigma turns linear about 22 kt from the truth, so the
# gradient on a badly-missed severe cyclone is no larger than on a marginal
# one. Measured consequence: +8 kt bias below 48 kt and -18 kt above 90.
# widening the quadratic region restores the pull on the extremes.
HUBER_DELTA = 2.5
# intensity is not uniformly sampled - the training set holds 152 patches
# between 34 and 47 kt and 59 above 90. Weighting by inverse bin frequency
# stops the common middle from dominating the gradient.
WEIGHT_BINS = [0, 34, 48, 64, 90, 999]
BANDS = [("<34", 0, 34), ("34-48", 34, 48), ("48-64", 48, 64),
         ("64-90", 64, 90), ("90+", 90, 999)]

# what T3 learns to read. "best_track" is IMD's 3-minute wind from IBTrACS,
# interpolated to the scene time by vision/best_track.py - the scale the IMD
# categories are defined on and the one T4 was trained on. "adt" is ADT's own
# estimate, which T3 was originally trained and scored against by mistake. ADT
# uses 1-minute winds and reads about 7.6 kt above IMD across the archive, so
# the old 10.95 kt RMSE measured agreement with ADT, not accuracy.
TARGETS = {"best_track": "bt_vmax_kt", "adt": "vmax_kt"}


def load_env(cache: Path, meta: pd.DataFrame) -> np.ndarray | None:
    """The ERA5 columns for these patches, in the order the patches come in.

    Written by src/add_era5_to_patches.py to a sidecar, so a cache rebuild
    cannot take them with it. Returns None if the sidecar is not there, which
    is what keeps the image-only model the default.
    """
    path = cache / "era5.csv"
    if not path.exists():
        return None
    from features.environment import ENV_FEATURES

    side = pd.read_csv(path)
    side["time"] = side["time"].astype(str)
    key = meta.assign(time=meta["time"].astype(str))[["storm", "time"]]
    joined = key.merge(side, on=["storm", "time"], how="left")
    if len(joined) != len(meta):                       # duplicate keys in the sidecar
        joined = joined.drop_duplicates(subset=["storm", "time"]).reset_index(drop=True)
    return joined[ENV_FEATURES].to_numpy(dtype=np.float32)


class EnvNet(nn.Module):
    """The same trunk, with the environment concatenated at the head.

    Late fusion rather than early: the convolutional trunk stays a picture
    model and keeps its Digital Typhoon weights, and the environment joins as
    numbers where the decision is made. Early fusion would have meant
    broadcasting ten scalars into image planes and retraining the trunk from
    scratch, which throws away the transfer that is worth the most here.
    """

    def __init__(self, backbone: str, n_env: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.trunk = timm.create_model(backbone, pretrained=pretrained,
                                       num_classes=0, in_chans=3)
        self.n_env = int(n_env)
        d = self.trunk.num_features + self.n_env
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x, env=None):
        f = self.trunk(x)
        if self.n_env:
            f = torch.cat([f, env], dim=1)
        return self.head(f)


class IntensityPatches(Dataset):
    # GridSat patch -> intensity in knots, on whatever scale `y` carries

    def __init__(self, patches, y, idx, train=False, weights=None, aug="reflect",
                 env=None):
        self.patches, self.y, self.idx, self.train = patches, y, idx, train
        self.weights, self.aug, self.env = weights, aug, env

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        a = self.patches[i].astype(np.float32) / 255.0
        if self.train:
            a = augment(a, self.aug)
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        w = 1.0 if self.weights is None else float(self.weights[i])
        e = (np.zeros(0, dtype=np.float32) if self.env is None
             else self.env[i].astype(np.float32))
        return (torch.from_numpy(a), torch.from_numpy(e),
                torch.tensor(float(self.y[i]), dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32))


def load_patches(target: str = "best_track", cache: Path = CACHE):
    # patches, their metadata and the target, restricted to rows that have one
    col = TARGETS[target]
    patches = np.load(cache / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(cache / "patches.csv")
    if col not in meta.columns:
        raise SystemExit(f"patches.csv has no {col} column - "
                         f"run src/vision/best_track.py first")
    keep = meta[col].notna() & (meta[col] > 0)
    rows = np.flatnonzero(keep.to_numpy())
    meta = meta[keep].reset_index(drop=True)
    return np.asarray(patches[rows]), meta, meta[col].to_numpy(dtype=np.float64)


def load_joint(target: str = "best_track"):
    # GridSat and INSAT patches in one table, with a sensor column.
    #
    # the INSAT cache holds the same ADT records at the same slots, so a storm's
    # patches from both sensors hash to the same split and neither can leak the
    # other's test storms. cross-validation said this costs nothing on GridSat
    # (13.50 -> 13.41 kt) and repairs INSAT (15.11 -> 13.77, bias -4.7 -> -0.7),
    # which is what lets one model read the live INSAT feed.
    pg, mg, yg = load_patches(target, CACHE)
    pi, mi, yi = load_patches(target, INSAT_CACHE)
    patches = np.concatenate([pg, pi])
    meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")],
                     ignore_index=True)
    return patches, meta, np.concatenate([yg, yi])


def env_matrix(meta: pd.DataFrame) -> np.ndarray | None:
    """ERA5 columns for a patch table, whether one sensor or both.

    load_joint concatenates gridsat rows then insat rows, so the environment is
    assembled in the same order and stays aligned with the patches.
    """
    if "sensor" not in meta.columns:
        return load_env(CACHE, meta)
    blocks = []
    for sensor, cache in (("gridsat", CACHE), ("insat", INSAT_CACHE)):
        sub = meta[meta["sensor"] == sensor].reset_index(drop=True)
        if sub.empty:
            continue
        e = load_env(cache, sub)
        if e is None:
            return None
        blocks.append(e)
    return np.concatenate(blocks) if blocks else None


def stats_features(patches, idx):
    out, n = [], patches.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    for i in idx:
        ir = patches[i, :, :, 0].astype(np.float32) / 255.0
        wv = patches[i, :, :, 1].astype(np.float32) / 255.0
        core, ring = ir[r <= .15], ir[(r > .15) & (r <= .4)]
        out.append([ir.min(), ir.mean(), ir.std(),
                    float((ir < .25).mean()), float((ir < .35).mean()),
                    float((ir < .5).mean()),
                    core.mean(), core.std(), ring.mean(),
                    core.mean() - ring.mean(), wv.mean(), float((ir - wv).mean())])
    return np.asarray(out, dtype=np.float32)


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def summarise(pred, truth) -> dict:
    # overall error plus bands - one figure hides opposite-signed band errors
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    e = pred - truth
    bands = {}
    for lab, lo, hi in BANDS:
        m = (truth >= lo) & (truth < hi)
        bands[lab] = ({"n": int(m.sum()), "bias": float(e[m].mean()),
                       "rmse": float(np.sqrt(np.mean(e[m] ** 2)))}
                      if m.sum() >= 3 else {"n": int(m.sum()), "bias": None, "rmse": None})
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()), "bands": bands}


def fit_baselines(patches, y, fit_idx, eval_idx) -> dict[str, np.ndarray]:
    # predict-the-mean and a ridge on cold-cloud statistics
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    sm.fit(stats_features(patches, fit_idx), y[fit_idx])
    return {"predict_the_mean": np.full(len(eval_idx), float(y[fit_idx].mean())),
            "cold_cloud": sm.predict(stats_features(patches, eval_idx))}


def fit(patches, y, train_idx, val_idx, device, aug="reflect", seed=SEED,
        log=print, epochs=EPOCHS, env=None):
    # fine-tune from the Digital Typhoon weights, early-stopped on validation RMSE
    import timm
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_env = 0 if env is None else env.shape[1]
    if n_env:
        model = EnvNet(BACKBONE, n_env).to(device)
    else:
        model = timm.create_model(BACKBONE, pretrained=True, num_classes=1,
                                  in_chans=3).to(device)
    transferred = False
    if PRETRAINED.exists():
        ckpt = torch.load(PRETRAINED, map_location=device, weights_only=False)
        try:
            if n_env:
                # the checkpoint is a plain timm model; its trunk weights fit
                # ours, its one-output head does not and is dropped.
                trunk_only = {k: v for k, v in ckpt["state_dict"].items()
                              if not k.startswith("head.")}
                missing, _ = model.trunk.load_state_dict(trunk_only, strict=False)
                transferred = len(missing) < 5
            else:
                model.load_state_dict(ckpt["state_dict"])
                transferred = True
        except Exception as exc:  # noqa: BLE001
            log(f"  could not transfer weights ({str(exc)[:60]}); starting from ImageNet")

    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())

    env_scaled, env_scale = None, None
    if n_env:
        mu = np.nanmean(env[train_idx], axis=0)
        sd = np.nanstd(env[train_idx], axis=0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        env_scale = (mu.tolist(), sd.tolist())
        env_scaled = (np.where(np.isfinite(env), env, mu) - mu) / sd
        env_scaled = env_scaled.astype(np.float32)
        gaps = float(np.mean(~np.isfinite(env)))
        log(f"  environment: {n_env} features, {gaps:.1%} of values imputed")

    # inverse-frequency weights, computed on the training rows only.
    bins = np.clip(np.digitize(y, WEIGHT_BINS) - 1, 0, len(WEIGHT_BINS) - 2)
    counts = np.bincount(bins[train_idx], minlength=len(WEIGHT_BINS) - 1)
    inv = np.where(counts > 0, counts.max() / np.maximum(counts, 1), 1.0)
    weights = inv[bins]

    train_loader = DataLoader(IntensityPatches(patches, y, train_idx, train=True,
                                               weights=weights, aug=aug,
                                               env=env_scaled),
                              batch_size=BATCH, shuffle=True,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(IntensityPatches(patches, y, val_idx, env=env_scaled),
                            batch_size=BATCH, pin_memory=(device == "cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")
    scale = (y_mean, y_std)

    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for x, ev, yy_, w in train_loader:
            x, ev, yy_, w = (x.to(device), ev.to(device),
                             yy_.to(device), w.to(device))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                out = model(x, ev) if n_env else model(x)
                per = loss_fn(out.squeeze(-1), (yy_ - y_mean) / y_std)
                loss = (per * w).sum() / w.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        v = rmse(predict(model, val_loader, scale, device), y[val_idx])
        flag = ""
        if v < best - 1e-4:
            best, best_epoch, bad, flag = v, epoch, 0, "  *"
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 5 == 0 or flag:
            log(f"  epoch {epoch:>2}  val RMSE {v:6.2f} kt  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            log(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, {"y_mean": y_mean, "y_std": y_std, "transferred": transferred,
                   "best_epoch": best_epoch, "best_val_rmse": float(best),
                   "n_env": n_env, "env_scale": env_scale,
                   "env_scaled": env_scaled}


def predict(model, loader, scale, device) -> np.ndarray:
    # raw model output in knots, before calibration
    model.eval()
    uses_env = getattr(model, "n_env", 0) > 0
    out = []
    with torch.no_grad():
        for x, ev, _, _ in loader:
            x = x.to(device)
            pred = model(x, ev.to(device)) if uses_env else model(x)
            out.append(pred.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) * scale[1] + scale[0]


def predict_idx(model, patches, y, idx, scale, device, env=None) -> np.ndarray:
    loader = DataLoader(IntensityPatches(patches, y, idx, env=env), batch_size=BATCH)
    return predict(model, loader, scale, device)


def fit_calibration(pred, truth) -> dict:
    # predicted = slope x truth + intercept, fitted where it will not be scored
    slope, intercept = np.polyfit(np.asarray(truth, float), np.asarray(pred, float), 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def apply_calibration(pred, cal: dict | None) -> np.ndarray:
    # invert the fitted line to undo shrinkage toward the mean; never below 0 kt
    pred = np.asarray(pred, float)
    if cal:
        pred = (pred - cal["intercept"]) / max(cal["slope"], 1e-6)
    return np.clip(pred, 0.0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), default="best_track")
    ap.add_argument("--sources", choices=["gridsat", "joint"], default="gridsat",
                    help="train on GridSat alone or on GridSat and INSAT together")
    ap.add_argument("--aug", choices=AUGMENTATIONS, default="reflect")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                    help="serve with the validation-fitted linear recalibration")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="",
                    help="write a candidate (artifacts/intensity_gridsat_<tag>.pt, "
                         "reports/candidates/) instead of replacing the served model")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    report_dir = OUT / "candidates" if args.tag else OUT
    report_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"CHAKRAVAT T3-GridSat - intensity in the operational domain  [{device}]")
    print(f"target {args.target}   sources {args.sources}   augmentation {args.aug}   "
          f"calibrate {args.calibrate}")
    print("=" * 78)

    patches, meta, y = (load_joint(args.target) if args.sources == "joint"
                        else load_patches(args.target))
    if "sensor" not in meta.columns:
        meta = meta.assign(sensor="gridsat")
    print(f"\npatches {len(meta):,}   storms {meta['storm'].nunique()}   "
          f"intensity {y.min():.0f}-{y.max():.0f} kt")

    splits = storm_split(meta)
    for k, v in splits.items():
        print(f"  {k:<6} {len(v):>4} patches  {meta.loc[v, 'storm'].nunique():>3} storms")
    y_test = y[splits["test"]]

    # the headline test set stays GridSat, so this run is comparable with every
    # earlier one; INSAT test rows are scored separately below
    sensor = meta["sensor"].to_numpy()
    test_by_sensor = {s: splits["test"][sensor[splits["test"]] == s]
                      for s in sorted(set(sensor))}
    base = fit_baselines(patches, y, splits["train"], splits["test"])
    print("\nbaselines on test ...")
    for name, p in base.items():
        print(f"  {name:<18} RMSE {rmse(p, y_test):.2f} kt")

    print("\ntraining ...")
    model, info = fit(patches, y, splits["train"], splits["val"], device,
                      aug=args.aug, seed=args.seed)
    scale = (info["y_mean"], info["y_std"])
    val_raw = predict_idx(model, patches, y, splits["val"], scale, device)
    test_raw = predict_idx(model, patches, y, splits["test"], scale, device)
    cal = fit_calibration(val_raw, y[splits["val"]])

    raw = summarise(apply_calibration(test_raw, None), y_test)
    calibrated = summarise(apply_calibration(test_raw, cal), y_test)
    served = calibrated if args.calibrate else raw

    print("\n" + "=" * 78)
    print(f"T3-GridSat RESULTS - held-out storms, truth = {args.target}")
    print("=" * 78)
    for name, s in (("raw", raw), ("calibrated", calibrated)):
        print(f"  {BACKBONE} {name:<11} RMSE {s['rmse']:6.2f} kt   MAE {s['mae']:.2f}   "
              f"bias {s['bias']:+.2f}")
        print("    " + " | ".join(f"{b}: n={v['n']} bias {v['bias']:+.1f}"
                                  for b, v in s["bands"].items() if v["bias"] is not None))
    print(f"  validation fit: predicted = {cal['slope']:.3f} x truth + {cal['intercept']:.2f}")

    # the operational objective method on exactly the same scenes
    benchmark = None
    if args.target == "best_track":
        test_meta = meta.iloc[splits["test"]]
        has_adt = (test_meta["vmax_kt"] > 0).to_numpy()
        if has_adt.sum():
            served_pred = apply_calibration(test_raw, cal if args.calibrate else None)
            benchmark = {
                "n": int(has_adt.sum()),
                "chakravat": summarise(served_pred[has_adt], y_test[has_adt]),
                "adt": summarise(test_meta["vmax_kt"].to_numpy()[has_adt], y_test[has_adt]),
            }
            print(f"\n  same {benchmark['n']} test scenes against IMD best track: "
                  f"T3 RMSE {benchmark['chakravat']['rmse']:.2f} kt, "
                  f"ADT {benchmark['adt']['rmse']:.2f} kt")

    ART.mkdir(exist_ok=True)
    by_sensor = {}
    for s, rows in test_by_sensor.items():
        if len(rows) < 10:
            continue
        pred = apply_calibration(predict_idx(model, patches, y, rows, scale, device),
                                 cal if args.calibrate else None)
        by_sensor[s] = summarise(pred, y[rows])
        print(f"  on {s:<8} n={len(rows):>4}  RMSE {by_sensor[s]['rmse']:6.2f} kt  "
              f"bias {by_sensor[s]['bias']:+.2f}")

    ckpt = {"state_dict": model.state_dict(), "backbone": BACKBONE,
            "y_mean": info["y_mean"], "y_std": info["y_std"], "domain": "gridsat",
            "target": args.target, "aug": args.aug, "sources": args.sources}
    if args.calibrate:
        ckpt["calibration"] = cal
    torch.save(ckpt, ART / f"intensity_gridsat{suffix}.pt")

    (report_dir / f"t3_intensity_gridsat{suffix}.json").write_text(json.dumps({
        "backbone": BACKBONE, "target": args.target, "augmentation": args.aug,
        "sources": args.sources, "by_sensor": by_sensor,
        "calibrated": args.calibrate, "patches": int(len(meta)),
        "storms": int(meta["storm"].nunique()), "transferred": info["transferred"],
        "splits": {k: {"patches": int(len(v)), "storms": int(meta.loc[v, "storm"].nunique())}
                   for k, v in splits.items()},
        "predict_the_mean_rmse_kt": rmse(base["predict_the_mean"], y_test),
        "cold_cloud_rmse_kt": rmse(base["cold_cloud"], y_test),
        "cnn_rmse_kt": served["rmse"], "cnn_mae_kt": served["mae"],
        "cnn_bias_kt": served["bias"],
        "raw": raw, "recalibrated": calibrated, "adt_benchmark": benchmark,
    }, indent=2))
    (report_dir / f"t3_calibration{suffix}.json").write_text(json.dumps(
        {"target": args.target, "applied": args.calibrate, **cal,
         "before": raw, "after": calibrated}, indent=2))
    print(f"\nSaved {ART / f'intensity_gridsat{suffix}.pt'}")


if __name__ == "__main__":
    main()
