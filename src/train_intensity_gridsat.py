# T3 in the operational domain: intensity from GridSat imagery

from __future__ import annotations

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

from vision.scenes import IMAGENET_MEAN, IMAGENET_STD, storm_split  # noqa: E402

CACHE = ROOT / "data" / "processed" / "scenes"
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


class IntensityPatches(Dataset):
    # GridSat patch -> best-track intensity in knots

    def __init__(self, patches, meta, idx, train=False, weights=None):
        self.patches, self.meta, self.idx, self.train = patches, meta, idx, train
        self.weights = weights

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        a = self.patches[i].astype(np.float32) / 255.0
        if self.train:
            r = np.random.randint(4)
            if r:
                a = np.rot90(a, r, axes=(0, 1))
            if np.random.rand() < 0.5:
                a = np.rot90(a[:, ::-1], 2, axes=(0, 1))
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        y = float(self.meta.iloc[i]["vmax_kt"])
        w = 1.0 if self.weights is None else float(self.weights[i])
        return (torch.from_numpy(a), torch.tensor(y, dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32))


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


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f"CHAKRAVAT T3-GridSat - intensity in the operational domain  [{device}]")
    print("=" * 78)

    patches = np.load(CACHE / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(CACHE / "patches.csv")
    keep = meta["vmax_kt"].notna() & (meta["vmax_kt"] > 0)
    meta = meta[keep].reset_index(drop=True)
    patches = patches[np.flatnonzero(keep.to_numpy())]
    print(f"\npatches {len(meta):,}   storms {meta['storm'].nunique()}   "
          f"intensity {meta['vmax_kt'].min():.0f}-{meta['vmax_kt'].max():.0f} kt")

    splits = storm_split(meta)
    for k, v in splits.items():
        print(f"  {k:<6} {len(v):>4} patches  {meta.loc[v,'storm'].nunique():>3} storms")

    y = meta["vmax_kt"].to_numpy()
    y_test = y[splits["test"]]
    y_mean, y_std = float(y[splits["train"]].mean()), float(y[splits["train"]].std())

    print("\nbaselines ...")
    print(f"  predict-the-mean       RMSE {rmse(np.full_like(y_test, y_mean), y_test):.2f} kt")
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    sm.fit(stats_features(patches, splits["train"]), y[splits["train"]])
    base_stat = rmse(sm.predict(stats_features(patches, splits["test"])), y_test)
    print(f"  cold-cloud statistics  RMSE {base_stat:.2f} kt")

    import timm
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=1,
                              in_chans=3).to(device)
    transferred = False
    if PRETRAINED.exists():
        ckpt = torch.load(PRETRAINED, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["state_dict"])
            transferred = True
            print(f"\ninitialised from Digital Typhoon weights "
                  f"({PRETRAINED.name}) - pretrain, then fine-tune")
        except Exception as exc:  # noqa: BLE001
            print(f"\ncould not transfer weights ({str(exc)[:60]}); "
                  f"starting from ImageNet")

    # inverse-frequency weights, computed on the training split only.
    bins = np.digitize(y, WEIGHT_BINS) - 1
    counts = np.bincount(bins[splits["train"]], minlength=len(WEIGHT_BINS) - 1)
    inv = np.where(counts > 0, counts.max() / np.maximum(counts, 1), 1.0)
    weights = inv[bins]
    print("  sample weights by band: " +
          ", ".join(f"{WEIGHT_BINS[i]}-{WEIGHT_BINS[i+1]}: x{inv[i]:.1f}"
                    for i in range(len(inv))))

    loaders = {k: DataLoader(IntensityPatches(patches, meta, v, train=(k == "train"),
                                              weights=weights),
                             batch_size=BATCH, shuffle=(k == "train"),
                             pin_memory=(device == "cuda"))
               for k, v in splits.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")

    def evaluate(loader):
        model.eval()
        p, t = [], []
        with torch.no_grad():
            for x, yy_, _ in loader:
                out = model(x.to(device)).squeeze(-1)
                p.append(out.cpu().numpy() * y_std + y_mean)
                t.append(yy_.numpy())
        return np.concatenate(p), np.concatenate(t)

    print("training ...")
    best, best_state, bad = np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); t0 = time.time()
        for x, yy_, w in loaders["train"]:
            x, yy_, w = x.to(device), yy_.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                per = loss_fn(model(x).squeeze(-1), (yy_ - y_mean) / y_std)
                loss = (per * w).sum() / w.sum()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        vp, vt = evaluate(loaders["val"])
        v = rmse(vp, vt)
        flag = ""
        if v < best - 1e-4:
            best, bad, flag = v, 0, "  *"
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 2 == 0 or flag:
            print(f"  epoch {epoch:>2}  val RMSE {v:6.2f} kt  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    tp, tt = evaluate(loaders["test"])
    cnn = rmse(tp, tt)
    mae = float(np.mean(np.abs(tp - tt)))

    print("\n" + "=" * 78)
    print("T3-GridSat RESULTS - held-out storms, operational domain")
    print("=" * 78)
    print(f"  predict-the-mean       RMSE {rmse(np.full_like(y_test, y_mean), y_test):6.2f} kt")
    print(f"  cold-cloud statistics  RMSE {base_stat:6.2f} kt")
    print(f"  {BACKBONE:<21}  RMSE {cnn:6.2f} kt   MAE {mae:.2f}   "
          f"bias {np.mean(tp - tt):+.2f}")
    print("\n  bias by intensity band - a single figure hides opposite errors")
    for lo, hi in zip(WEIGHT_BINS[:-1], WEIGHT_BINS[1:]):
        m_ = (tt >= lo) & (tt < hi)
        if m_.sum() >= 3:
            print(f"    {lo:>3}-{hi if hi < 999 else '+':<4} n={int(m_.sum()):>3}  "
                  f"bias {np.mean(tp[m_] - tt[m_]):>+6.1f} kt  "
                  f"RMSE {np.sqrt(np.mean((tp[m_] - tt[m_]) ** 2)):>5.1f} kt")
    print(f"\n  transferred from Digital Typhoon: {transferred}")
    print(f"  reference: Deepti 13.24 kt on INSAT infrared (same basin, finer sensor)")

    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": BACKBONE,
                "y_mean": y_mean, "y_std": y_std, "domain": "gridsat"},
               ART / "intensity_gridsat.pt")
    (OUT / "t3_intensity_gridsat.json").write_text(json.dumps(
        {"backbone": BACKBONE, "patches": int(len(meta)),
         "storms": int(meta["storm"].nunique()), "transferred": transferred,
         "predict_the_mean_rmse_kt": rmse(np.full_like(y_test, y_mean), y_test),
         "cold_cloud_rmse_kt": base_stat,
         "cnn_rmse_kt": cnn, "cnn_mae_kt": mae}, indent=2))
    print(f"\nSaved {ART / 'intensity_gridsat.pt'}")


if __name__ == "__main__":
    main()
