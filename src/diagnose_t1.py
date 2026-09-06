# where does detection actually fail? Evaluation only, no training

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build as build_track                # noqa: E402
from vision.detect import build_index, season_split            # noqa: E402
from train_detect import (                                     # noqa: E402
    HeatmapNet, evaluate, INTENSITY_BANDS, GRIDSAT,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = build_track(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    index = build_index(GRIDSAT, track)
    index = index[index["n_storms"] > 0].reset_index(drop=True)
    splits = season_split(index)

    cache_path = ROOT / "data" / "processed" / "basin_scenes.npy"
    cache = np.load(cache_path, mmap_mode="r")
    offsets, at = {}, 0
    for k in ("train", "val", "test"):
        offsets[k] = at
        at += len(splits[k])

    ckpt = torch.load(ROOT / "artifacts" / "detector.pt", map_location=device,
                      weights_only=False)
    model = HeatmapNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("=" * 74)
    print("T1 DIAGNOSIS - held-out seasons, no retraining")
    print("=" * 74)

    print("\nthreshold sweep on TEST (for shape only - the deployed threshold")
    print("is chosen on validation):")
    print(f"  {'thr':>5} {'F1':>7} {'prec':>7} {'recall':>7}")
    best = None
    for thr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
        r = evaluate(model, splits["test"], track, device, threshold=thr,
                     cache=cache, cache_offset=offsets["test"],
                     with_baseline=False)
        print(f"  {thr:>5.2f} {r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f}")
        if best is None or r["f1"] > best[1]["f1"]:
            best = (thr, r)

    thr, r = best
    print(f"\nbest F1 on test at threshold {thr:.2f}: {r['f1']:.3f}")
    print("\nrecall by IMD intensity at that threshold:")
    print(f"  {'band':<12} {'n':>4} {'recall':>8}")
    for name, band in r["recall_by_intensity"].items():
        if band["n"]:
            bar = "#" * int(round(30 * band["recall"]))
            print(f"  {name:<12} {band['n']:>4} {band['recall']:>8.3f}  {bar}")

    print(f"\ncentre-fix error on matched storms: "
          f"median {r['median_km']:.0f} km, mean {r['mean_km']:.0f} km")

    # how much of the test set is even plausibly detectable?
    t = track[track["ISO_TIME"].isin(splits["test"]["time"])]
    weak = float((t["vmax_kt"] < 34).mean())
    print(f"\nshare of test fixes below Cyclonic Storm (34 kt): {100*weak:.0f}%")


if __name__ == "__main__":
    main()
