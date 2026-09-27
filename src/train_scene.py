# task T2: classify the Dvorak cloud pattern from satellite imagery
#
#     python src/train_scene.py [--aug reflect|rotate] [--tag NAME]

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.adt import DVORAK_CLASSES                       # noqa: E402
from vision.scenes import (AUGMENTATIONS, ScenePatches, build_cache,  # noqa: E402
                           storm_split)

ADT = ROOT / "data" / "adt" / "adt_nio.csv"
GRIDSAT = ROOT / "data" / "gridsat"
CACHE = ROOT / "data" / "processed" / "scenes"
INSAT_CACHE = ROOT / "data" / "processed" / "scenes_insat"
OUT, ART = ROOT / "reports", ROOT / "artifacts"

BACKBONE = "convnext_tiny"
# ConvNeXt will not train at 3e-4. Verified directly: on a fixed 100-sample
# subset it plateaus at a Huber loss of 0.72 - the value for predicting the
# mean - and never descends, while the same model at 5e-5 reaches 0.015. The
# first T3 run looked like a data problem for exactly this reason.
EPOCHS, PATIENCE, BATCH, LR, SEED = 40, 8, 48, 5e-5, 0
MIN_PER_CLASS = 40

# served T2 is a hybrid: the CNN's probabilities averaged with a logistic model
# on cold-cloud statistics. storm-grouped 5-fold CV over all 974 patches:
# CNN 0.638 macro-F1, logistic 0.650, average 0.699, with the gain over either
# alone clear of zero at 95%. ADT decides scene type from cloud-top temperature
# rules, which the statistics capture directly; the CNN adds spatial structure
# (it is far better on EYE). equal weights, so nothing is tuned on held-out data.
BLEND = 0.5


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> list[float]:
    out = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0)
    return out


def stats_features(patches: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # brightness-temperature summaries in a storm-relative frame
    out = []
    n = patches.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    for i in idx:
        ir = patches[i, :, :, 0].astype(np.float32) / 255.0
        wv = patches[i, :, :, 1].astype(np.float32) / 255.0
        core, ring, outer = ir[r <= .15], ir[(r > .15) & (r <= .4)], ir[r > .4]
        out.append([
            ir.min(), ir.mean(), ir.std(),
            float((ir < .25).mean()), float((ir < .35).mean()), float((ir < .5).mean()),
            core.mean(), core.std(), ring.mean(), ring.std(), outer.mean(),
            core.mean() - ring.mean(),          # eye vs eyewall contrast
            core.min() - ring.min(),
            wv.mean(), wv.std(), float((ir - wv).mean()),
        ])
    return np.asarray(out, dtype=np.float32)


def _available_scene_count() -> int:
    # labelled scenes whose GridSat timestep is currently on disk
    from vision.scenes import gridsat_path

    adt = pd.read_csv(ADT, parse_dates=["time"])
    adt = adt[adt["scene"].isin(DVORAK_CLASSES)].copy()
    adt["slot"] = pd.DatetimeIndex(adt["time"]).round("3h")
    adt = adt.drop_duplicates(subset=["storm", "slot"])
    return int(adt["slot"].map(lambda s: gridsat_path(GRIDSAT, s).exists()).sum())


def load_scenes(rebuild_check: bool = True, cache: Path = CACHE, classes: list[str] | None = None):
    # patches, metadata restricted to learnable classes, class list, labels
    if rebuild_check:
        # rebuild when more imagery has arrived, not just when the cache is
        # absent. GridSat downloads incrementally, so a cache built earlier
        # silently pins the model to however many scenes existed at the time -
        # 768 of the 979 now available, in the first run.
        #
        # compared against what the last build saw, not against the patch
        # count: five scenes always sit too near the crop edge, so 974 patches
        # from 979 scenes looked like missing imagery and every run rebuilt the
        # cache - twelve minutes, and it wiped the best-track columns T3 needs.
        import json
        available = _available_scene_count()
        info = CACHE / "build.json"
        seen = (json.loads(info.read_text())["with_imagery"] if info.exists()
                else len(pd.read_csv(CACHE / "patches.csv")) if (CACHE / "patches.csv").exists()
                else 0)
        if not (CACHE / "patches.npy").exists() or available > seen:
            print(f"\nbuilding patch cache from ADT + GridSat "
                  f"(last build saw {seen:,}, available {available:,}) ...")
            build_cache(ADT, GRIDSAT, CACHE, DVORAK_CLASSES)
            from vision.best_track import add_columns, load_tracks
            add_columns(CACHE / "patches.csv", load_tracks())

    patches = np.load(cache / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(cache / "patches.csv", parse_dates=["time"])
    counts = meta["scene"].value_counts()
    # a second cache (INSAT) must use the GridSat class list, not its own counts
    if classes is None:
        classes = [c for c in DVORAK_CLASSES if counts.get(c, 0) >= MIN_PER_CLASS]
    keep = meta["scene"].isin(classes).to_numpy()
    rows = np.flatnonzero(keep)
    meta = meta[keep].reset_index(drop=True)
    y = meta["scene"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return np.asarray(patches[rows]), meta, classes, y, counts


def load_joint(rebuild_check: bool = True):
    # GridSat and INSAT patches in one table, with a sensor column; the INSAT
    # cache uses the GridSat class list rather than its own counts
    pg, mg, classes, yg, counts = load_scenes(rebuild_check=rebuild_check, cache=CACHE)
    pi, mi, _, yi, _ = load_scenes(rebuild_check=False, cache=INSAT_CACHE, classes=classes)
    patches = np.concatenate([pg, pi])
    meta = pd.concat([mg.assign(sensor="gridsat"), mi.assign(sensor="insat")],
                     ignore_index=True)
    return patches, meta, classes, np.concatenate([yg, yi]), counts


def fit_stat_model(patches, y, fit_idx):
    # the cold-cloud logistic model, the non-CNN half of the hybrid
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(stats_features(patches, fit_idx), y[fit_idx])
    return model


def fit_baselines(patches, y, fit_idx, eval_idx, n_classes) -> dict[str, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    major = int(pd.Series(y[fit_idx]).mode()[0])
    stat = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    stat.fit(stats_features(patches, fit_idx), y[fit_idx])
    return {"majority": np.full(len(eval_idx), major),
            "cold_cloud_stats": stat.predict(stats_features(patches, eval_idx))}


def fit(patches, meta, y, classes, train_idx, val_idx, device, aug="reflect",
        seed=SEED, log=print, epochs=EPOCHS, norm="imagenet"):
    # ImageNet ConvNeXt, class-weighted, early-stopped on validation macro-F1
    import timm
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = timm.create_model(BACKBONE, pretrained=True,
                              num_classes=len(classes), in_chans=3).to(device)
    train_loader = DataLoader(ScenePatches(patches, meta, train_idx, classes,
                                           train=True, aug=aug, norm=norm),
                              batch_size=BATCH, shuffle=True, num_workers=0,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(ScenePatches(patches, meta, val_idx, classes, norm=norm),
                            batch_size=BATCH, num_workers=0,
                            pin_memory=(device == "cuda"))

    # weight by inverse frequency: curved band outnumbers irregular CDO five to
    # one, and unweighted training simply predicts the common class.
    freq = np.bincount(y[train_idx], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor((freq.sum() / np.maximum(freq, 1)) / len(classes),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, t in train_loader:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = loss_fn(model(x), t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss) * len(t)
            seen += len(t)
        sched.step()

        vp, _ = predict(model, val_loader, device)
        vf1 = macro_f1(y[val_idx], vp, len(classes))
        flag = ""
        if vf1 > best + 1e-4:
            best, best_epoch, bad, flag = vf1, epoch, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 5 == 0 or flag:
            log(f"  epoch {epoch:>2}  train {total/max(seen,1):5.3f}  "
                f"val macro-F1 {vf1:.3f}  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            log(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_macro_f1": float(best)}


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    # predicted class and softmax probabilities
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.softmax(model(x.to(device, non_blocking=True)).float(), 1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
    return np.concatenate(preds), np.concatenate(probs)


def predict_idx(model, patches, meta, classes, idx, device, norm="imagenet"):
    return predict(model, DataLoader(ScenePatches(patches, meta, idx, classes, norm=norm),
                                     batch_size=BATCH, num_workers=0), device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=AUGMENTATIONS, default="reflect")
    ap.add_argument("--sources", choices=["gridsat", "joint"], default="gridsat",
                    help="train on GridSat alone or on GridSat and INSAT together")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="",
                    help="write a candidate (artifacts/scene_classifier_<tag>.pt, "
                         "reports/candidates/) instead of replacing the served model")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    report_dir = OUT / "candidates" if args.tag else OUT
    report_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"CHAKRAVAT T2 - Dvorak scene classification  [{BACKBONE}, {device}]"
          f"   sources {args.sources}   augmentation {args.aug}")
    print("=" * 78)

    patches, meta, classes, y, counts = (load_joint() if args.sources == "joint"
                                         else load_scenes())
    if "sensor" not in meta.columns:
        meta = meta.assign(sensor="gridsat")
    print(f"\npatches: {len(meta):,}   storms: {meta['storm'].nunique()}")
    print(f"classes kept (>= {MIN_PER_CLASS} samples): {classes}")
    for c in classes:
        print(f"  {c:<8} {counts[c]:>5,}")

    splits = storm_split(meta)
    for name, idx in splits.items():
        print(f"  {name:<6} {len(idx):>5,} patches  "
              f"{meta.loc[idx, 'storm'].nunique():>3} storms")
    y_test = y[splits["test"]]

    print("\nbaselines ...")
    base = fit_baselines(patches, y, splits["train"], splits["test"], len(classes))
    base_major = macro_f1(y_test, base["majority"], len(classes))
    acc_major = float((y_test == base["majority"]).mean())
    base_stat = macro_f1(y_test, base["cold_cloud_stats"], len(classes))
    acc_stat = float((y_test == base["cold_cloud_stats"]).mean())
    print(f"  majority class         macro-F1 {base_major:.3f}  acc {acc_major:.3f}")
    print(f"  cold-cloud statistics  macro-F1 {base_stat:.3f}  acc {acc_stat:.3f}")

    print("\ntraining ...")
    model, _ = fit(patches, meta, y, classes, splits["train"], splits["val"], device,
                   aug=args.aug, seed=args.seed)
    tp, tprob = predict_idx(model, patches, meta, classes, splits["test"], device)
    tt = y_test
    cnn_f1, cnn_acc = macro_f1(tt, tp, len(classes)), float((tt == tp).mean())

    # the statistics half sees train and validation; neither half sees test
    stat_model = fit_stat_model(patches, y, np.concatenate([splits["train"], splits["val"]]))
    sprob = stat_model.predict_proba(stats_features(patches, splits["test"]))
    hp = ((1 - BLEND) * tprob + BLEND * sprob).argmax(1)
    hyb_f1, hyb_acc = macro_f1(tt, hp, len(classes)), float((tt == hp).mean())
    stat_f1 = macro_f1(tt, sprob.argmax(1), len(classes))

    print("\n" + "=" * 78)
    print("T2 RESULTS - held-out storms")
    print("=" * 78)
    print(f"  majority class         macro-F1 {base_major:.3f}   acc {acc_major:.3f}")
    print(f"  cold-cloud statistics  macro-F1 {base_stat:.3f}   acc {acc_stat:.3f}")
    print(f"  {BACKBONE:<21}  macro-F1 {cnn_f1:.3f}   acc {cnn_acc:.3f}")
    print(f"  stats logistic (tr+va) macro-F1 {stat_f1:.3f}")
    print(f"  hybrid, served         macro-F1 {hyb_f1:.3f}   acc {hyb_acc:.3f}")

    print("\n  hybrid per-class F1 and confusion (rows = truth)")
    print("           " + "".join(f"{c:>9}" for c in classes) + "     F1")
    f1s = per_class_f1(tt, hp, len(classes))
    for i, c in enumerate(classes):
        row = [int(np.sum((tt == i) & (hp == j))) for j in range(len(classes))]
        print(f"  {c:<8} " + "".join(f"{v:>9,}" for v in row) + f"  {f1s[i]:>6.3f}")

    sensor = meta["sensor"].to_numpy()
    by_sensor = {}
    for s in sorted(set(sensor)):
        rows = splits["test"][sensor[splits["test"]] == s]
        if len(rows) < 10:
            continue
        _, cnn_p = predict_idx(model, patches, meta, classes, rows, device)
        sp = stat_model.predict_proba(stats_features(patches, rows))
        hp_s = ((1 - BLEND) * cnn_p + BLEND * sp).argmax(1)
        by_sensor[s] = {"n": int(len(rows)), "hybrid_macro_f1": macro_f1(y[rows], hp_s, len(classes)),
                        "accuracy": float((y[rows] == hp_s).mean())}
        print(f"  on {s:<8} n={len(rows):>4}  hybrid macro-F1 "
              f"{by_sensor[s]['hybrid_macro_f1']:.3f}")

    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "backbone": BACKBONE, "aug": args.aug, "sources": args.sources,
                "stat_model": stat_model, "blend": BLEND},
               ART / f"scene_classifier{suffix}.pt")
    (report_dir / f"t2_scene{suffix}.json").write_text(json.dumps(
        {"backbone": BACKBONE, "classes": classes, "augmentation": args.aug,
         "sources": args.sources, "by_sensor": by_sensor,
         "patches": int(len(meta)), "storms": int(meta["storm"].nunique()),
         "majority": {"macro_f1": base_major, "accuracy": acc_major},
         "cold_cloud_stats": {"macro_f1": base_stat, "accuracy": acc_stat},
         "cnn": {"macro_f1": cnn_f1, "accuracy": cnn_acc},
         "stats_logistic": {"macro_f1": stat_f1},
         "hybrid": {"macro_f1": hyb_f1, "accuracy": hyb_acc, "blend": BLEND,
                    "per_class_f1": dict(zip(classes, f1s))},
         "served": "hybrid"}, indent=2))
    print(f"\nSaved {ART / f'scene_classifier{suffix}.pt'}")


if __name__ == "__main__":
    main()
