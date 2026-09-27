# variance-restoring recalibration for the intensity model
#
#     python src/calibrate_intensity.py [--checkpoint artifacts/intensity_gridsat.pt]
#
# refits the linear recalibration on validation for an existing checkpoint,
# against the same target the checkpoint was trained on, and reports test
# before and after. train_intensity_gridsat.py already does this at the end of
# training; this is for refitting without retraining.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_intensity_gridsat import (apply_calibration, fit_calibration,  # noqa: E402
                                     load_patches, predict_idx, summarise)
from vision.scenes import storm_split                                     # noqa: E402

ART, OUT = ROOT / "artifacts", ROOT / "reports"


def report(name, s):
    print(f"\n{name}")
    print(f"  RMSE {s['rmse']:.2f} kt   MAE {s['mae']:.2f}   bias {s['bias']:+.2f}")
    for lab, b in s["bands"].items():
        if b["bias"] is not None:
            print(f"  {lab:<8} n={b['n']:>4}  bias {b['bias']:>+6.1f}  RMSE {b['rmse']:>5.1f}")


def main() -> None:
    import timm

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ART / "intensity_gridsat.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(args.checkpoint)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # checkpoints from before the best-track fix carry no target key; they
    # were trained on ADT's wind.
    target = ckpt.get("target", "adt")
    patches, meta, y = load_patches(target)
    splits = storm_split(meta)

    model = timm.create_model(ckpt["backbone"], pretrained=False,
                              num_classes=1, in_chans=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scale = (ckpt["y_mean"], ckpt["y_std"])

    print("=" * 74)
    print(f"INTENSITY RECALIBRATION - {path.name}, truth = {target}")
    print("=" * 74)
    cal = fit_calibration(predict_idx(model, patches, y, splits["val"], scale, device),
                          y[splits["val"]])
    print(f"\nvalidation fit: predicted = {cal['slope']:.3f} x truth + {cal['intercept']:.2f}")

    test_raw = predict_idx(model, patches, y, splits["test"], scale, device)
    before = summarise(apply_calibration(test_raw, None), y[splits["test"]])
    after = summarise(apply_calibration(test_raw, cal), y[splits["test"]])
    report("BEFORE - raw model on test", before)
    report("AFTER  - recalibrated on test", after)

    ckpt["calibration"] = cal
    torch.save(ckpt, path)
    (OUT / "t3_calibration.json").write_text(json.dumps(
        {"target": target, "applied": True, **cal, "before": before, "after": after},
        indent=2))
    print(f"\nsaved calibration into {path}")


if __name__ == "__main__":
    main()
