# task T1: find cyclones in a basin-wide scene and fix their centres

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build as build_track            # noqa: E402
from eval.metrics import great_circle_km                   # noqa: E402
from vision.detect import (                                # noqa: E402
    BasinScenes, build_index, season_split, find_peaks,
    SCENE_H, SCENE_W, HEAT_H, HEAT_W, STRIDE, gridsat_path,
)

GRIDSAT = ROOT / "data" / "gridsat"
RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
OUT, ART = ROOT / "reports", ROOT / "artifacts"

EPOCHS, PATIENCE, BATCH, LR, SEED = 40, 10, 8, 1.5e-4, 0
WEIGHT_DECAY = 5e-4   # raised with the learning rate lowered: the first run
                      # peaked at epoch 3 and then memorised the training set
MATCH_KM = 150.0          # a peak counts as a hit within this distance


class HeatmapNet(nn.Module):
    # timm encoder with a light decoder to a single-channel heatmap

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            backbone, pretrained=True, features_only=True, in_chans=3)
        chs = self.encoder.feature_info.channels()
        # encoder strides are 2,4,8,16,32. Decoding must reach stride 4 to match
        # the target heatmap; stopping one level short halves the resolution and
        # doubles the quantisation floor on every centre fix.
        self.lat3 = nn.Conv2d(chs[-1], 128, 1)
        self.lat2 = nn.Conv2d(chs[-2], 128, 1)
        self.lat1 = nn.Conv2d(chs[-3], 128, 1)
        self.lat0 = nn.Conv2d(chs[-4], 128, 1)
        self.smooth = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 1, 1)
        # start with low probability everywhere: the heatmap is almost all
        # background, and without this the model spends its first epochs
        # unlearning a confident positive prior.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        feats = self.encoder(x)
        up = lambda a, b: nn.functional.interpolate(  # noqa: E731
            a, size=b.shape[-2:], mode="nearest")
        p = self.lat3(feats[-1])
        p = up(p, feats[-2]) + self.lat2(feats[-2])
        p = up(p, feats[-3]) + self.lat1(feats[-3])
        p = up(p, feats[-4]) + self.lat0(feats[-4])
        return self.head(self.smooth(p))


def focal_loss(pred_logits, target, weight=None,
               alpha: float = 2.0, beta: float = 4.0):
    # CenterNet penalty-reduced focal loss
    # always in float32. Under autocast the sigmoid lands in fp16, where the
    # clamp floor and log together underflow and the loss goes NaN - which is
    # what killed the first run at epoch 11.
    pred = torch.sigmoid(pred_logits.float()).clamp(1e-4, 1 - 1e-4)
    target = target.float()
    pos = target.ge(0.99).float()
    neg = 1.0 - pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    if weight is not None:
        w = weight.float()
        pos_loss, neg_loss = pos_loss * w, neg_loss * w
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def coldest_pixel_baseline(when: pd.Timestamp, n_expected: int):
    # coldest local minima of brightness temperature, as centre estimates
    import scipy.ndimage as ndi

    with xr.open_dataset(gridsat_path(GRIDSAT, when)) as ds:
        ir = ds["irwin_cdr"].isel(time=0).values.astype(np.float32)
        lats, lons = ds["lat"].values, ds["lon"].values
    ir = np.nan_to_num(ir, nan=300.0)
    smooth = ndi.uniform_filter(ir, size=25)
    mins = ndi.minimum_filter(smooth, size=61, mode="nearest")
    mask = (smooth == mins)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    order = np.argsort(smooth[ys, xs])[:max(n_expected, 1)]
    return [(float(lats[ys[k]]), float(lons[xs[k]])) for k in order]


def match(pred_pts, true_pts, radius_km: float = MATCH_KM,
          return_matched: bool = False):
    # greedy nearest matching; returns (hits, errors_km, n_pred, n_true)
    used, errors = set(), []
    for plat, plon in pred_pts:
        best, best_d = None, np.inf
        for j, (tlat, tlon) in enumerate(true_pts):
            if j in used:
                continue
            d = float(great_circle_km(tlat, tlon, plat, plon))
            if d < best_d:
                best, best_d = j, d
        if best is not None and best_d <= radius_km:
            used.add(best)
            errors.append(best_d)
    if return_matched:
        return len(used), errors, len(pred_pts), len(true_pts), used
    return len(used), errors, len(pred_pts), len(true_pts)


def _prf(hits, n_pred, n_true):
    prec = hits / n_pred if n_pred else 0.0
    rec = hits / n_true if n_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# IMD bands. Detection difficulty is not uniform across these: a 20 kt
# depression often has no organised convective signature in infrared and can be
# indistinguishable from ordinary monsoon convection, while a cyclonic storm is
# unmistakable. A single pooled recall hides which of those the model is
# failing at, and in this test set 79% of fixes are below Cyclonic Storm.
INTENSITY_BANDS = [("D <28", 0, 28), ("DD 28-33", 28, 34),
                   ("CS 34-47", 34, 48), ("SCS+ 48+", 48, 999)]


def predict_scene(model, ds, i, index, device):
    # model peaks for one scene, in geographic coordinates
    when = pd.Timestamp(index.iloc[i]["time"])
    x = ds[i][0]
    with torch.no_grad():
        heat = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    # reuse the dataset's memoised grid rather than reopening the file. The
    # lat/lon axes are identical in every GridSat crop, and opening the netCDF
    # again per scene cost ~680 ms each - about eight minutes across the test
    # set, purely to re-read constants.
    lats, lons = ds._latlon(when)
    return when, heat, lats, lons


def evaluate(model, index, track, device, threshold=0.30,
             cache=None, cache_offset=0, with_baseline=True,
             min_kt: float = 0.0):
    # min_kt restricts scoring to systems at or above that intensity
    model.eval()
    ds = BasinScenes(index, track, GRIDSAT, cache=cache, cache_offset=cache_offset)
    hits = n_pred = n_true = 0
    errors, base_errors = [], []
    b_hits = b_pred = 0
    band_hits = {b[0]: 0 for b in INTENSITY_BANDS}
    band_total = {b[0]: 0 for b in INTENSITY_BANDS}

    for i in range(len(ds)):
        when, heat, lats, lons = predict_scene(model, ds, i, index, device)
        here = track[(track["ISO_TIME"] == when) & (track["vmax_kt"] >= min_kt)]
        true_pts = [(r.LAT, r.LON) for r in here.itertuples()]
        true_kt = [float(r.vmax_kt) for r in here.itertuples()]
        if not true_pts:
            continue

        pred_pts = [BasinScenes.from_pixel(y * STRIDE, x_ * STRIDE, lats, lons)
                    for y, x_, _ in find_peaks(heat, threshold)]
        h, errs, np_, nt, matched = match(pred_pts, true_pts, return_matched=True)
        hits += h; errors += errs; n_pred += np_; n_true += nt

        for j, kt in enumerate(true_kt):
            for name, lo, hi in INTENSITY_BANDS:
                if lo <= kt < hi:
                    band_total[name] += 1
                    if j in matched:
                        band_hits[name] += 1
                    break

        if with_baseline:
            bh, berrs, bp, _ = match(coldest_pixel_baseline(when, len(true_pts)),
                                     true_pts)
            b_hits += bh; b_pred += bp; base_errors += berrs

    prec, rec, f1 = _prf(hits, n_pred, n_true)
    _, _, bf1 = _prf(b_hits, b_pred, n_true)
    return {
        "n_scenes": int(len(index)), "n_storms": int(n_true),
        "min_kt": float(min_kt), "threshold": float(threshold),
        "precision": prec, "recall": rec, "f1": f1,
        "median_km": float(np.median(errors)) if errors else float("nan"),
        "mean_km": float(np.mean(errors)) if errors else float("nan"),
        "baseline_f1": bf1,
        "baseline_median_km": (float(np.median(base_errors))
                               if base_errors else float("nan")),
        "recall_by_intensity": {
            name: {"n": band_total[name],
                   "recall": band_hits[name] / band_total[name]
                   if band_total[name] else float("nan")}
            for name, _, _ in INTENSITY_BANDS},
    }


def tune_threshold(model, index, track, device, cache, cache_offset,
                   grid=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) -> float:
    # pick the peak threshold on validation, never on test
    best, best_f1 = grid[0], -1.0
    for thr in grid:
        r = evaluate(model, index, track, device, threshold=thr,
                     cache=cache, cache_offset=cache_offset, with_baseline=False)
        print(f"    threshold {thr:.2f}  F1 {r['f1']:.3f}  "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}")
        if r["f1"] > best_f1:
            best, best_f1 = thr, r["f1"]
    return best


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"CHAKRAVAT T1 - detection and centre-fixing  [{device}]")
    print("=" * 78)

    track = build_track(str(RAW))
    index = build_index(GRIDSAT, track)
    index = index[index["n_storms"] > 0].reset_index(drop=True)
    print(f"\nscenes on disk with a storm: {len(index):,}")
    print(f"storm fixes covered: {int(index['n_storms'].sum()):,}")
    if len(index) < 40:
        raise SystemExit("not enough GridSat scenes yet - let the download finish")

    splits = season_split(index)
    for name, part in splits.items():
        print(f"  {name:<6} {len(part):>5,} scenes  "
              f"seasons {sorted(part['season'].unique()) if len(part) else '-'}")
    if min(len(p) for p in splits.values()) == 0:
        raise SystemExit("a split is empty - wait for more seasons to download")

    # pre-render every scene once. Reading netCDF per sample left the GPU at
    # 0% utilisation and put a 40-epoch run at seven hours.
    from vision.detect import build_scene_cache
    cache_path = ROOT / "data" / "processed" / "basin_scenes.npy"
    ordered = pd.concat([splits[k] for k in ("train", "val", "test")],
                        ignore_index=True)
    if not cache_path.exists() or len(np.load(cache_path, mmap_mode="r")) != len(ordered):
        print(f"\nrendering {len(ordered):,} scenes to cache ...")
        build_scene_cache(ordered, GRIDSAT, cache_path)
    cache = np.load(cache_path, mmap_mode="r")

    offsets, at = {}, 0
    for k in ("train", "val", "test"):
        offsets[k] = at
        at += len(splits[k])

    loaders = {
        name: DataLoader(BasinScenes(part, track, GRIDSAT,
                                     train=(name == "train"),
                                     cache=cache, cache_offset=offsets[name]),
                         batch_size=BATCH, shuffle=(name == "train"),
                         num_workers=0, pin_memory=(device == "cuda"))
        for name, part in splits.items()
    }

    model = HeatmapNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    print(f"\nparameters: {sum(p.numel() for p in model.parameters()):,}")
    print("training ...")
    best, best_state, bad = np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, heat, wmap, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            wmap = wmap.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(x)
            loss = focal_loss(logits, heat, wmap)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += float(loss.detach()) * len(x); seen += len(x)
        sched.step()

        model.eval()
        vtot, vseen = 0.0, 0
        with torch.no_grad():
            for x, heat, wmap, _ in loaders["val"]:
                x, heat, wmap = x.to(device), heat.to(device), wmap.to(device)
                vtot += float(focal_loss(model(x), heat, wmap).detach()) * len(x)
                vseen += len(x)
        vloss = vtot / max(vseen, 1)

        flag = ""
        if vloss < best - 1e-4:
            best, bad, flag = vloss, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  epoch {epoch:>2}  train {total/max(seen,1):6.3f}  "
              f"val {vloss:6.3f}  ({time.time()-t0:.0f}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    print("\ntuning the peak threshold on validation ...")
    threshold = tune_threshold(model, splits["val"], track, device,
                               cache, offsets["val"])
    print(f"  chosen: {threshold:.2f}")

    print("\nevaluating on held-out seasons ...")
    res = evaluate(model, splits["test"], track, device, threshold=threshold,
                   cache=cache, cache_offset=offsets["test"])

    print("\n" + "=" * 78)
    print("T1 RESULTS - held-out seasons")
    print("=" * 78)
    print(f"  scenes {res['n_scenes']:,}   storm fixes {res['n_storms']:,}")
    print(f"\n  coldest-pixel baseline   F1 {res['baseline_f1']:.3f}   "
          f"median fix {res['baseline_median_km']:.0f} km")
    print(f"  chakravat heatmap        F1 {res['f1']:.3f}   "
          f"median fix {res['median_km']:.0f} km")
    print(f"    precision {res['precision']:.3f}  recall {res['recall']:.3f}  "
          f"mean fix {res['mean_km']:.0f} km  (threshold {res['threshold']:.2f})")

    print("\n  recall by IMD intensity - detection is not equally hard across these")
    for name, band in res["recall_by_intensity"].items():
        if band["n"]:
            bar = "#" * int(round(20 * band["recall"]))
            print(f"    {name:<10} n={band['n']:>4}  recall {band['recall']:.3f}  {bar}")
    # scored again on the systems IMD actually names. Pooling over everything
    # lets Depressions - 56% of the fixes, and frequently with no organised
    # infrared signature - dominate a single number, which then says nothing
    # about performance on the storms that carry warning value.
    named = evaluate(model, splits["test"], track, device, threshold=threshold,
                     cache=cache, cache_offset=offsets["test"],
                     with_baseline=False, min_kt=34.0)
    print("\n  scored on Cyclonic Storm and above (34 kt+), which IMD names")
    print(f"    n={named['n_storms']}  F1 {named['f1']:.3f}  "
          f"precision {named['precision']:.3f}  recall {named['recall']:.3f}  "
          f"median fix {named['median_km']:.0f} km")
    res["named_only"] = named

    print(f"\n  targets: F1 >= 0.90, centre-fix <= 40 km")

    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ART / "detector.pt")
    # an intensity band with no test examples yields NaN, which json.dumps
    # happily writes and every strict JSON reader - including the dashboard's
    # own /api/skill - then rejects. Write null instead.
    def _finite(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        return o

    (OUT / "t1_detection.json").write_text(json.dumps(_finite(res), indent=2))
    print(f"\nSaved {ART / 'detector.pt'}")
    print(f"Wrote {OUT / 't1_detection.json'}")


if __name__ == "__main__":
    main()
