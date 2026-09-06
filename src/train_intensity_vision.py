# task T3: estimate cyclone intensity directly from a satellite image

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

from vision.dataset import TyphoonFrames, load_labels, storm_split  # noqa: E402

DATA = ROOT / "data" / "digital_typhoon" / "AU"
OUT = ROOT / "reports"
ART = ROOT / "artifacts"

BACKBONE = "convnext_tiny"
EPOCHS = 40
PATIENCE = 8
BATCH = 64
# ConvNeXt will not train at 3e-4. Verified directly: on a fixed 100-sample
# subset it plateaus at a Huber loss of 0.72 - the value for predicting the
# mean - and never descends, while the same model at 5e-5 reaches 0.015. The
# first T3 run looked like a data problem for exactly this reason.
LR = 5e-5
SEED = 0


def build_model(name: str = BACKBONE) -> nn.Module:
    import timm
    # A single scalar head on an ImageNet-pretrained trunk. With a few thousand
    # frames the trunk matters far more than head capacity.
    return timm.create_model(name, pretrained=True, num_classes=1, in_chans=3)


def image_statistics(frame: pd.DataFrame, size: int = 64) -> np.ndarray:
    # dvorak-flavoured summary features: how cold, how much, how symmetric
    from PIL import Image

    feats = []
    for path in frame["path"]:
        img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        # Digital Typhoon PNGs are brightness temperature: darker = colder tops.
        cy = cx = size // 2
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.hypot(yy - cy, xx - cx)
        core = a[r <= size * 0.15]
        ring = a[(r > size * 0.15) & (r <= size * 0.35)]
        rings = [a[(r > lo * size) & (r <= hi * size)].mean()
                 for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45))]
        feats.append([
            a.min(), a.mean(), a.std(),
            float((a < 0.25).mean()), float((a < 0.35).mean()), float((a < 0.5).mean()),
            core.mean(), core.std(), ring.mean(),
            core.mean() - ring.mean(),           # eye-versus-eyewall contrast
            *rings,
        ])
    return np.asarray(feats, dtype=np.float32)


def evaluate(model, loader, device, y_mean: float = 0.0,
             y_std: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    # predictions returned in knots, whatever space the model trained in
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device, non_blocking=True)).squeeze(-1)
            preds.append(out.cpu().numpy() * y_std + y_mean)
            truths.append(y.numpy())
    return np.concatenate(preds), np.concatenate(truths)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = pred - truth
    return {"n": int(len(err)),
            "rmse_kt": float(np.sqrt(np.mean(err ** 2))),
            "mae_kt": float(np.mean(np.abs(err))),
            "bias_kt": float(np.mean(err))}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"CHAKRAVAT T3 - intensity from imagery  [{BACKBONE}, {device}]")
    print("=" * 78)

    if not (DATA / "labels.csv").exists():
        raise SystemExit(f"no labels at {DATA}. Run src/ingest/digital_typhoon.py first.")

    df = load_labels(DATA)
    print(f"\nframes on disk: {len(df):,}   storms: {df['storm_id'].nunique()}")
    print(f"seasons {df['season'].min()}-{df['season'].max()}   "
          f"intensity {df['wind_kt'].min():.0f}-{df['wind_kt'].max():.0f} kt")

    splits = storm_split(df)
    for name, part in splits.items():
        print(f"  {name:<6} {len(part):>6,} frames  {part['storm_id'].nunique():>4} storms")

    # ---- baselines -------------------------------------------------------
    print("\nbaselines ...")
    train_mean = float(splits["train"]["wind_kt"].mean())
    test_truth = splits["test"]["wind_kt"].to_numpy()
    base_mean = scores(np.full(len(test_truth), train_mean), test_truth)
    print(f"  predict-the-mean       RMSE {base_mean['rmse_kt']:.2f} kt")

    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    x_tr = image_statistics(splits["train"])
    x_te = image_statistics(splits["test"])
    stat_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    stat_model.fit(x_tr, splits["train"]["wind_kt"].to_numpy())
    base_stat = scores(stat_model.predict(x_te), test_truth)
    print(f"  cold-cloud statistics  RMSE {base_stat['rmse_kt']:.2f} kt "
          f"({time.time() - t0:.0f}s)")

    # ---- CNN -------------------------------------------------------------
    # standardise the target. The first attempt regressed raw knots, and the
    # network flatly refused to learn: validation RMSE sat at 17.08 kt for
    # every epoch against a val standard deviation of 17.1 and a
    # predict-the-mean score of 17.87. It had collapsed to the mean and stayed
    # there. With a head initialised near zero and a Huber loss that is linear
    # beyond 10 kt, the gradient pulling the output up to a ~39 kt bias is
    # constant and small, so optimisation spends itself on the offset and never
    # reaches the image-to-intensity mapping. Centring and scaling removes that
    # offset from the problem entirely.
    y_mean = float(splits["train"]["wind_kt"].mean())
    y_std = float(splits["train"]["wind_kt"].std())
    print(f"\ntarget standardised: mean {y_mean:.1f} kt, std {y_std:.1f} kt")

    loaders = {
        name: DataLoader(TyphoonFrames(part, train=(name == "train")),
                         batch_size=BATCH, shuffle=(name == "train"),
                         num_workers=0, pin_memory=(device == "cuda"))
        for name, part in splits.items()
    }

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    # Huber rather than MSE: best-track intensity is quantised to 5 kt steps and
    # carries real outliers, which a squared loss would chase.
    # delta is now in standard deviations, not knots.
    loss_fn = nn.HuberLoss(delta=1.0)

    print(f"\nparameters: {sum(p.numel() for p in model.parameters()):,}")
    print("training ...")
    best, best_state, bad = np.inf, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = loss_fn(model(x).squeeze(-1), (y - y_mean) / y_std)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss) * len(y)
            seen += len(y)
        sched.step()

        vp, vt = evaluate(model, loaders["val"], device, y_mean, y_std)
        vrmse = scores(vp, vt)["rmse_kt"]
        flag = ""
        if vrmse < best - 1e-4:
            best, bad, flag = vrmse, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  epoch {epoch:>2}  train {total/max(seen,1):6.2f}  "
              f"val RMSE {vrmse:6.2f} kt  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    tp, tt = evaluate(model, loaders["test"], device, y_mean, y_std)
    cnn = scores(tp, tt)

    print("\n" + "=" * 78)
    print("T3 RESULTS - held-out storms")
    print("=" * 78)
    print(f"  predict-the-mean       RMSE {base_mean['rmse_kt']:6.2f} kt")
    print(f"  cold-cloud statistics  RMSE {base_stat['rmse_kt']:6.2f} kt")
    print(f"  {BACKBONE:<21}  RMSE {cnn['rmse_kt']:6.2f} kt   "
          f"MAE {cnn['mae_kt']:.2f}   bias {cnn['bias_kt']:+.2f}")
    print(f"\n  reference: Deepti 13.24 kt RMSE on infrared imagery")

    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": BACKBONE,
                "y_mean": y_mean, "y_std": y_std},
               ART / "intensity_vision.pt")
    (OUT / "t3_intensity_vision.json").write_text(json.dumps(
        {"backbone": BACKBONE,
         "frames": int(len(df)), "storms": int(df["storm_id"].nunique()),
         "splits": {k: {"frames": int(len(v)), "storms": int(v["storm_id"].nunique())}
                    for k, v in splits.items()},
         "predict_the_mean": base_mean, "cold_cloud_stats": base_stat,
         "cnn": cnn}, indent=2))
    print(f"\nSaved {ART / 'intensity_vision.pt'}")
    print(f"Wrote {OUT / 't3_intensity_vision.json'}")


if __name__ == "__main__":
    main()
