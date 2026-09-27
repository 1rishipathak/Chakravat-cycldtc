# is scene typing worse on INSAT because of the sensor, or because the INSAT
# rows are a different set of scenes?
#
#     python src/eval_t2_sensors.py
#
# the two archives are on the same grid at the same slots, so for some storms
# the same (storm, time) exists in both. scoring on exactly those pairs removes
# scene composition from the comparison and leaves only the sensor.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import train_scene as t2                      # noqa: E402
from cv_vision import storm_bootstrap         # noqa: E402

oof = pd.read_csv(ROOT / "reports" / "cv" / "sensors_t2_gridsat_oof.csv")
classes = ["EYE", "CRVBND", "SHEAR", "EMBC", "IRRCDO"]
n = len(classes)
lookup = {c: i for i, c in enumerate(classes)}
oof["y"] = oof["scene"].map(lookup)
cnn = oof[[f"cnn_{c}" for c in classes]].to_numpy()
stat = oof[[f"stat_{c}" for c in classes]].to_numpy()
hyb = 0.5 * cnn + 0.5 * stat
oof["pred_hyb"] = hyb.argmax(1)
oof["pred_cnn"] = cnn.argmax(1)

key = ["storm", "time"]
g = oof[oof.sensor == "gridsat"].set_index(key)
i = oof[oof.sensor == "insat"].set_index(key)
both = g.index.intersection(i.index)
print(f"gridsat rows {len(g)}   insat rows {len(i)}   exactly paired {len(both)}")

gp, ip = g.loc[both].reset_index(), i.loc[both].reset_index()
assert (gp["y"].to_numpy() == ip["y"].to_numpy()).all(), "same scene, different label?"


def score(df, col):
    y, p, st = df["y"].to_numpy(), df[col].to_numpy(), df["storm"].to_numpy()
    f1 = t2.macro_f1(y, p, n)
    ci = storm_bootstrap(st, lambda k: t2.macro_f1(y[k], p[k], n))
    return f1, ci


print(f"\n{'':<34}{'hybrid':>24}{'CNN alone':>24}")
for label, df in (("GridSat, paired scenes", gp), ("INSAT, same scenes", ip)):
    h, hci = score(df, "pred_hyb")
    c, cci = score(df, "pred_cnn")
    print(f"{label:<34}{h:>8.3f} [{hci[0]:.3f},{hci[1]:.3f}]{c:>10.3f} [{cci[0]:.3f},{cci[1]:.3f}]")

# and how often do the two sensors disagree with each other on the same scene?
agree = (gp["pred_hyb"].to_numpy() == ip["pred_hyb"].to_numpy()).mean()
gright = (gp["pred_hyb"].to_numpy() == gp["y"].to_numpy())
iright = (ip["pred_hyb"].to_numpy() == ip["y"].to_numpy())
print(f"\nsame prediction from both sensors : {agree:.1%}")
print(f"both right                        : {(gright & iright).mean():.1%}")
print(f"GridSat right, INSAT wrong        : {(gright & ~iright).mean():.1%}")
print(f"INSAT right, GridSat wrong        : {(~gright & iright).mean():.1%}")
print(f"both wrong                        : {(~gright & ~iright).mean():.1%}")

print(f"\nper-class F1 on the paired scenes")
print(f"{'class':<10}{'GridSat':>10}{'INSAT':>10}{'n':>6}")
for j, c in enumerate(classes):
    m = gp["y"].to_numpy() == j
    gf = t2.per_class_f1(gp["y"].to_numpy(), gp["pred_hyb"].to_numpy(), n)[j]
    jf = t2.per_class_f1(ip["y"].to_numpy(), ip["pred_hyb"].to_numpy(), n)[j]
    print(f"{c:<10}{gf:>10.2f}{jf:>10.2f}{m.sum():>6}")


result = {
    "paired_scenes": int(len(both)),
    "note": "every INSAT patch has a GridSat twin at the same storm and time with "
            "the same ADT label, so the sensor is the only thing that differs",
    "agreement": {
        "same_prediction": float(agree),
        "both_right": float((gright & iright).mean()),
        "gridsat_right_insat_wrong": float((gright & ~iright).mean()),
        "insat_right_gridsat_wrong": float((~gright & iright).mean()),
        "both_wrong": float((~gright & ~iright).mean()),
    },
}
for _label, _df in (("gridsat", gp), ("insat", ip)):
    _h, _hci = score(_df, "pred_hyb")
    _c, _cci = score(_df, "pred_cnn")
    result[_label] = {
        "hybrid_macro_f1": _h, "hybrid_macro_f1_ci95": [float(v) for v in _hci],
        "cnn_macro_f1": _c, "cnn_macro_f1_ci95": [float(v) for v in _cci],
        "per_class_f1": {c: float(v) for c, v in
                         zip(classes, t2.per_class_f1(_df["y"].to_numpy(),
                                                      _df["pred_hyb"].to_numpy(), n))},
    }
result["support"] = {c: int((gp["y"].to_numpy() == j).sum()) for j, c in enumerate(classes)}

import json
_out = ROOT / "reports" / "t2_sensor_paired.json"
_out.write_text(json.dumps(result, indent=2))
print(f"\nwrote {_out}")
