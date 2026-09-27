# A detector has one model and more than one job.
#
#     python src/t1_operating_points.py
#
# The served threshold is tuned for F1, which is the right objective when a
# detection becomes a track and a track becomes a forecast: a false storm costs
# a wrong forecast. It is the wrong objective for asking "is anything forming
# out there", where a miss costs more than a false alarm, and it shows - recall
# on depressions is 0.39 while recall on named storms is 0.76.
#
# So the same checkpoint gets a second, more sensitive operating point, chosen
# on validation by F2 (recall weighted four times precision, the standard way of
# saying a miss is worse than a false alarm). Nothing is retrained. The warn
# tier keeps the threshold it always had, and the watch tier is reported beside
# it with the cost printed rather than buried.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build as build_track                # noqa: E402
from vision.detect import build_index, season_split            # noqa: E402
from train_detect import HeatmapNet, evaluate, GRIDSAT         # noqa: E402

GRID = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
WEAK = ("D <28", "DD 28-33")          # the bands a watch tier exists for


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0


def weak_recall(res: dict) -> tuple[float, int]:
    hits = total = 0
    for name in WEAK:
        band = res["recall_by_intensity"].get(name)
        if band and band["n"]:
            hits += band["recall"] * band["n"]
            total += band["n"]
    return (hits / total if total else float("nan")), total


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = build_track(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    index = build_index(GRIDSAT, track)
    index = index[index["n_storms"] > 0].reset_index(drop=True)
    splits = season_split(index)

    cache = np.load(ROOT / "data" / "processed" / "basin_scenes.npy", mmap_mode="r")
    offsets, at = {}, 0
    for k in ("train", "val", "test"):
        offsets[k] = at
        at += len(splits[k])

    ckpt = torch.load(ROOT / "artifacts" / "detector.pt", map_location=device,
                      weights_only=False)
    model = HeatmapNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    served = float(json.loads((ROOT / "reports" / "t1_detection.json").read_text())["threshold"])

    def run(split: str, thr: float) -> dict:
        return evaluate(model, splits[split], track, device, threshold=thr,
                        cache=cache, cache_offset=offsets[split], with_baseline=False)

    print("=" * 78)
    print("T1 OPERATING POINTS - one checkpoint, two jobs")
    print("=" * 78)
    print(f"\nsweep on VALIDATION ({len(splits['val'])} scenes). the watch tier is "
          f"chosen here, never on test.")
    print(f"  {'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'F2':>7} {'weak R':>8}")

    sweep, best = [], None
    for thr in GRID:
        r = run("val", thr)
        f2 = fbeta(r["precision"], r["recall"], 2.0)
        wr, wn = weak_recall(r)
        print(f"  {thr:>5.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {f2:>7.3f} {wr:>8.3f}")
        sweep.append({"threshold": thr, "precision": r["precision"], "recall": r["recall"],
                      "f1": r["f1"], "f2": f2, "weak_recall": wr, "weak_n": wn})
        if best is None or f2 > best["f2"]:
            best = sweep[-1]

    watch = best["threshold"]
    print(f"\n  watch tier: threshold {watch:.2f}, chosen by F2 on validation")
    print(f"  warn tier : threshold {served:.2f}, the served one, chosen by F1")
    if watch >= served:
        print("  note: F2 did not prefer a more sensitive point than the served one.")

    print(f"\nscored on TEST ({len(splits['test'])} scenes), both tiers:")
    out = {"detector": "detector.pt", "source": "gridsat",
           "validation_sweep": sweep,
           "chosen_by": {"warn": "F1 on validation (unchanged)",
                         "watch": "F2 on validation"}}
    for tier, thr in (("warn", served), ("watch", watch)):
        r = run("test", thr)
        named = evaluate(model, splits["test"], track, device, threshold=thr,
                         cache=cache, cache_offset=offsets["test"],
                         with_baseline=False, min_kt=34.0)
        wr, wn = weak_recall(r)
        out[tier] = {
            "threshold": thr, "precision": r["precision"], "recall": r["recall"],
            "false_alarms_per_scene": (r["n_pred"] - r["hits"]) / max(r["n_scenes"], 1),
            "f1": r["f1"], "f2": fbeta(r["precision"], r["recall"], 2.0),
            "median_km": r["median_km"],
            "weak_recall": wr, "weak_n": wn,
            "recall_by_intensity": r["recall_by_intensity"],
            "named_only": {"precision": named["precision"], "recall": named["recall"],
                           "f1": named["f1"], "median_km": named["median_km"]},
        }
        print(f"  {tier:<6} thr {thr:.2f}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}  weak recall {wr:.3f} of {wn}  "
              f"named F1 {named['f1']:.3f}  "
              f"{(r['n_pred'] - r['hits']) / max(r['n_scenes'], 1):.2f} false/scene")

    print("\nceiling probe: the most permissive threshold on the grid, on TEST.")
    print("if weak systems are still missed here, the misses are not a tuning choice.")
    floor_thr = min(GRID)
    fr = run("test", floor_thr)
    fwr, _ = weak_recall(fr)
    print(f"  thr {floor_thr:.2f}  precision {fr['precision']:.3f}  "
          f"weak recall {fwr:.3f}  "
          f"{(fr['n_pred'] - fr['hits']) / max(fr['n_scenes'], 1):.2f} false/scene")
    out["ceiling"] = {
        "threshold": floor_thr, "precision": fr["precision"], "weak_recall": fwr,
        "false_alarms_per_scene": (fr["n_pred"] - fr["hits"]) / max(fr["n_scenes"], 1),
        "reading": "even here a large share of weak systems is missed, so the "
                   "misses are structural rather than a threshold artefact",
    }

    w, n = out["watch"], out["warn"]
    out["cost_of_the_watch_tier"] = {
        "weak_recall_gain": w["weak_recall"] - n["weak_recall"],
        "precision_loss": w["precision"] - n["precision"],
        "false_alarms_per_scene_extra": (w["false_alarms_per_scene"]
                                         - n["false_alarms_per_scene"]),
    }
    path = ROOT / "reports" / "t1_operating_points.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nweak-system recall {n['weak_recall']:.3f} -> {w['weak_recall']:.3f}, "
          f"precision {n['precision']:.3f} -> {w['precision']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
