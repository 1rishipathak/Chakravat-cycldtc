# cross-validated score for the T2 hybrid that actually ships
#
#     python src/eval/t2_hybrid.py [--aug reflect]
#
# cv_vision.py scores the CNN alone. what serves is the CNN's probabilities
# averaged with a logistic model on cold-cloud statistics, so this refits that
# logistic model on the same folds (cheap, no GPU), averages it with the CNN's
# out-of-fold probabilities and scores the three side by side.
#
# why the average is worth it: ADT assigns scene type from rules on cloud-top
# temperature, which the statistics measure directly, while the CNN reads
# spatial structure. they fail on different scenes - the CNN is far better on
# EYE, the statistics on SHEAR and IRRCDO.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import train_scene as t2                       # noqa: E402
from cv_vision import storm_bootstrap          # noqa: E402

OUT = ROOT / "reports" / "cv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="reflect")
    args = ap.parse_args()

    patches, meta, classes, y, _ = t2.load_scenes(rebuild_check=False)
    oof = pd.read_csv(OUT / f"t2_{args.aug}_oof.csv")
    if len(oof) != len(meta) or not (oof["storm"].to_numpy() == meta["storm"].to_numpy()).all():
        raise SystemExit("out-of-fold file does not line up with the patch table - rerun cv_vision")

    folds = oof["fold"].to_numpy()
    cnn = oof[[f"p_{c}" for c in classes]].to_numpy()
    stat = np.zeros_like(cnn)
    for f in np.unique(folds):
        # the same fit the served model uses, on this fold's training storms
        train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        model = t2.fit_stat_model(patches, y, train)
        stat[test] = model.predict_proba(t2.stats_features(patches, test))

    storms = meta["storm"].to_numpy()
    n = len(classes)
    hybrid = (1 - t2.BLEND) * cnn + t2.BLEND * stat
    result = {"augmentation": args.aug, "blend": t2.BLEND, "classes": classes,
              "patches": int(len(y)), "storms": int(meta["storm"].nunique()),
              "folds": int(len(np.unique(folds)))}
    preds = {}
    for name, probs in (("cnn", cnn), ("cold_cloud_logistic", stat), ("hybrid", hybrid)):
        p = probs.argmax(1)
        preds[name] = p
        result[name] = {
            "macro_f1": t2.macro_f1(y, p, n),
            "macro_f1_ci95": storm_bootstrap(storms, lambda i: t2.macro_f1(y[i], p[i], n)),
            "accuracy": float((p == y).mean()),
            "per_class_f1": dict(zip(classes, t2.per_class_f1(y, p, n))),
        }
        print(f"{name:<20} macro-F1 {result[name]['macro_f1']:.3f} "
              f"[{result[name]['macro_f1_ci95'][0]:.3f}, {result[name]['macro_f1_ci95'][1]:.3f}]  "
              f"acc {result[name]['accuracy']:.3f}")
    for other in ("cnn", "cold_cloud_logistic"):
        h, o = preds["hybrid"], preds[other]
        result[f"hybrid_minus_{other}_ci95"] = storm_bootstrap(
            storms, lambda i: t2.macro_f1(y[i], h[i], n) - t2.macro_f1(y[i], o[i], n))
        print(f"hybrid minus {other}: 95% interval "
              f"{[round(v, 3) for v in result[f'hybrid_minus_{other}_ci95']]}")
    result["support"] = dict(zip(classes, np.bincount(y, minlength=n).tolist()))
    result["confusion_hybrid"] = [[int(np.sum((y == i) & (preds["hybrid"] == j)))
                                   for j in range(n)] for i in range(n)]
    (OUT / f"t2_hybrid_{args.aug}.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {(OUT / f't2_hybrid_{args.aug}.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
