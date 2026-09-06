# variance-restoring recalibration for the intensity model

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision.scenes import storm_split                    # noqa: E402
from train_intensity_gridsat import IntensityPatches     # noqa: E402

CACHE = ROOT / "data" / "processed" / "scenes"
ART, OUT = ROOT / "artifacts", ROOT / "reports"
BANDS = [("<34", 0, 34), ("34-48", 34, 48), ("48-64", 48, 64),
         ("64-90", 64, 90), ("90+", 90, 999)]


def predict(model, patches, meta, idx, ckpt, device):
    p, t = [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(IntensityPatches(patches, meta, idx), batch_size=32):
            out = model(x.to(device)).squeeze(-1)
            p.append(out.cpu().numpy() * ckpt["y_std"] + ckpt["y_mean"])
            t.append(y.numpy())
    return np.concatenate(p), np.concatenate(t)


def report(name, pred, truth):
    err = pred - truth
    print(f"\n{name}")
    print(f"  RMSE {np.sqrt(np.mean(err**2)):.2f} kt   MAE {np.mean(np.abs(err)):.2f}"
          f"   bias {np.mean(err):+.2f}   corr {np.corrcoef(pred, truth)[0,1]:.3f}")
    print(f"  {'band':<8} {'n':>4} {'bias':>8} {'RMSE':>7}")
    for lab, lo, hi in BANDS:
        k = (truth >= lo) & (truth < hi)
        if k.sum() >= 3:
            print(f"  {lab:<8} {int(k.sum()):>4} {np.mean(err[k]):>+8.1f} "
                  f"{np.sqrt(np.mean(err[k]**2)):>7.1f}")
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
            "bands": {lab: {"n": int(((truth >= lo) & (truth < hi)).sum()),
                            "bias": float(np.mean(err[(truth >= lo) & (truth < hi)]))
                            if ((truth >= lo) & (truth < hi)).sum() >= 3 else None}
                      for lab, lo, hi in BANDS}}


def main() -> None:
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patches = np.load(CACHE / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(CACHE / "patches.csv")
    keep = meta["vmax_kt"].notna() & (meta["vmax_kt"] > 0)
    meta = meta[keep].reset_index(drop=True)
    patches = patches[np.flatnonzero(keep.to_numpy())]
    splits = storm_split(meta)

    ckpt = torch.load(ART / "intensity_gridsat.pt", map_location=device,
                      weights_only=False)
    model = timm.create_model(ckpt["backbone"], pretrained=False,
                              num_classes=1, in_chans=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("=" * 74)
    print("INTENSITY RECALIBRATION - fitted on validation, applied to test")
    print("=" * 74)

    vp, vt = predict(model, patches, meta, splits["val"], ckpt, device)
    tp, tt = predict(model, patches, meta, splits["test"], ckpt, device)

    slope, intercept = np.polyfit(vt, vp, 1)
    print(f"\nvalidation fit: predicted = {slope:.3f} x truth + {intercept:.1f}")
    print(f"  slope below 1 means shrinkage toward the mean; inverting restores scale")

    def apply(p):
        return np.clip((p - intercept) / max(slope, 1e-6), 0.0, None)

    before = report("BEFORE - raw model on test", tp, tt)
    after = report("AFTER  - recalibrated on test", apply(tp), tt)

    print("\n" + "=" * 74)
    d_rmse = after["rmse"] - before["rmse"]
    print(f"RMSE {before['rmse']:.2f} -> {after['rmse']:.2f} kt ({d_rmse:+.2f})")
    print("band bias, absolute:")
    for lab, _, _ in BANDS:
        b, a = before["bands"][lab]["bias"], after["bands"][lab]["bias"]
        if b is None:
            continue
        mark = "better" if abs(a) < abs(b) else "worse"
        print(f"  {lab:<8} {b:>+7.1f} -> {a:>+7.1f}   {mark}")

    ckpt["calibration"] = {"slope": float(slope), "intercept": float(intercept)}
    torch.save(ckpt, ART / "intensity_gridsat.pt")
    (OUT / "t3_calibration.json").write_text(json.dumps(
        {"slope": float(slope), "intercept": float(intercept),
         "before": before, "after": after}, indent=2))
    print(f"\nsaved calibration into {ART / 'intensity_gridsat.pt'}")


if __name__ == "__main__":
    main()
