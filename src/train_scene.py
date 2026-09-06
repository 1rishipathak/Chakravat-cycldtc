# task T2: classify the Dvorak cloud pattern from satellite imagery

from __future__ import annotations

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
from vision.scenes import ScenePatches, build_cache, storm_split   # noqa: E402

ADT = ROOT / "data" / "adt" / "adt_nio.csv"
GRIDSAT = ROOT / "data" / "gridsat"
CACHE = ROOT / "data" / "processed" / "scenes"
OUT, ART = ROOT / "reports", ROOT / "artifacts"

BACKBONE = "convnext_tiny"
# ConvNeXt will not train at 3e-4. Verified directly: on a fixed 100-sample
# subset it plateaus at a Huber loss of 0.72 - the value for predicting the
# mean - and never descends, while the same model at 5e-5 reaches 0.015. The
# first T3 run looked like a data problem for exactly this reason.
EPOCHS, PATIENCE, BATCH, LR, SEED = 40, 8, 48, 5e-5, 0
MIN_PER_CLASS = 40


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


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"CHAKRAVAT T2 - Dvorak scene classification  [{BACKBONE}, {device}]")
    print("=" * 78)

    # rebuild when more imagery has arrived, not just when the cache is absent.
    # GridSat downloads incrementally, so a cache built earlier silently pins
    # the model to however many scenes existed at the time - 768 of the 979
    # now available, in the first run.
    available = _available_scene_count()
    cached = 0
    if (CACHE / "patches.csv").exists():
        cached = len(pd.read_csv(CACHE / "patches.csv"))
    if not (CACHE / "patches.npy").exists() or cached < available:
        print(f"\nbuilding patch cache from ADT + GridSat "
              f"(cached {cached:,}, available {available:,}) ...")
        build_cache(ADT, GRIDSAT, CACHE, DVORAK_CLASSES)

    patches = np.load(CACHE / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(CACHE / "patches.csv", parse_dates=["time"])
    print(f"\npatches: {len(meta):,}   storms: {meta['storm'].nunique()}")

    counts = meta["scene"].value_counts()
    classes = [c for c in DVORAK_CLASSES if counts.get(c, 0) >= MIN_PER_CLASS]
    meta = meta[meta["scene"].isin(classes)].reset_index(drop=True)
    print(f"classes kept (>= {MIN_PER_CLASS} samples): {classes}")
    for c in classes:
        print(f"  {c:<8} {counts[c]:>5,}")

    splits = storm_split(meta)
    for name, idx in splits.items():
        print(f"  {name:<6} {len(idx):>5,} patches  "
              f"{meta.loc[idx, 'storm'].nunique():>3} storms")

    y = meta["scene"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_test = y[splits["test"]]

    # ---- baselines -------------------------------------------------------
    print("\nbaselines ...")
    major = int(pd.Series(y[splits["train"]]).mode()[0])
    base_major = macro_f1(y_test, np.full_like(y_test, major), len(classes))
    acc_major = float((y_test == major).mean())
    print(f"  majority class         macro-F1 {base_major:.3f}  acc {acc_major:.3f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    xs_tr = stats_features(patches, splits["train"])
    xs_te = stats_features(patches, splits["test"])
    stat = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    stat.fit(xs_tr, y[splits["train"]])
    p_stat = stat.predict(xs_te)
    base_stat = macro_f1(y_test, p_stat, len(classes))
    acc_stat = float((y_test == p_stat).mean())
    print(f"  cold-cloud statistics  macro-F1 {base_stat:.3f}  acc {acc_stat:.3f}"
          f"  ({time.time()-t0:.0f}s)")

    # ---- CNN -------------------------------------------------------------
    import timm
    model = timm.create_model(BACKBONE, pretrained=True,
                              num_classes=len(classes), in_chans=3).to(device)

    loaders = {
        name: DataLoader(ScenePatches(patches, meta, idx, classes,
                                      train=(name == "train")),
                         batch_size=BATCH, shuffle=(name == "train"),
                         num_workers=0, pin_memory=(device == "cuda"))
        for name, idx in splits.items()
    }

    # weight by inverse frequency: curved band outnumbers irregular CDO five to
    # one, and unweighted training simply predicts the common class.
    freq = np.bincount(y[splits["train"]], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor((freq.sum() / np.maximum(freq, 1)) / len(classes),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, t in loader:
                out = model(x.to(device, non_blocking=True))
                preds.append(out.argmax(1).cpu().numpy())
                truths.append(t.numpy())
        return np.concatenate(preds), np.concatenate(truths)

    print(f"\nparameters: {sum(p.numel() for p in model.parameters()):,}")
    print("training ...")
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, t in loaders["train"]:
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

        vp, vt = evaluate(loaders["val"])
        vf1 = macro_f1(vt, vp, len(classes))
        flag = ""
        if vf1 > best + 1e-4:
            best, bad, flag = vf1, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  epoch {epoch:>2}  train {total/max(seen,1):5.3f}  "
              f"val macro-F1 {vf1:.3f}  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    tp, tt = evaluate(loaders["test"])
    cnn_f1, cnn_acc = macro_f1(tt, tp, len(classes)), float((tt == tp).mean())

    print("\n" + "=" * 78)
    print("T2 RESULTS - held-out storms")
    print("=" * 78)
    print(f"  majority class         macro-F1 {base_major:.3f}   acc {acc_major:.3f}")
    print(f"  cold-cloud statistics  macro-F1 {base_stat:.3f}   acc {acc_stat:.3f}")
    print(f"  {BACKBONE:<21}  macro-F1 {cnn_f1:.3f}   acc {cnn_acc:.3f}")

    print("\n  per-class F1 and confusion (rows = truth)")
    print("           " + "".join(f"{c:>9}" for c in classes) + "     F1")
    for i, c in enumerate(classes):
        row = [int(np.sum((tt == i) & (tp == j))) for j in range(len(classes))]
        tp_i, fp_i = row[i], int(np.sum((tp == i) & (tt != i)))
        fn_i = sum(row) - row[i]
        prec = tp_i / (tp_i + fp_i) if tp_i + fp_i else 0.0
        rec = tp_i / (tp_i + fn_i) if tp_i + fn_i else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {c:<8} " + "".join(f"{v:>9,}" for v in row) + f"  {f1:>6.3f}")

    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "backbone": BACKBONE}, ART / "scene_classifier.pt")
    (OUT / "t2_scene.json").write_text(json.dumps(
        {"backbone": BACKBONE, "classes": classes,
         "patches": int(len(meta)), "storms": int(meta["storm"].nunique()),
         "majority": {"macro_f1": base_major, "accuracy": acc_major},
         "cold_cloud_stats": {"macro_f1": base_stat, "accuracy": acc_stat},
         "cnn": {"macro_f1": cnn_f1, "accuracy": cnn_acc}}, indent=2))
    print(f"\nSaved {ART / 'scene_classifier.pt'}")
    print(f"Wrote {OUT / 't2_scene.json'}")


if __name__ == "__main__":
    main()
