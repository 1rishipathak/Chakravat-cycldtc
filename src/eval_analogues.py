# Are the analogues any good as a forecast?
#
#     python src/eval_analogues.py
#
# The dashboard presents analogues as context for reading a forecast, not as one.
# That framing should be earned rather than asserted, so this scores them as if
# they were a forecast: take the mean displacement of the five nearest earlier
# storms and see how far off it lands.
#
# No held-out split is needed, because the analogue rule already supplies one.
# The pool is storms that had ended before the current storm began, so every
# analogue forecast is out of sample by construction. Scoring on the same
# seasons the models are held out on keeps the numbers comparable.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports" / "analogue_skill.json"
HELD_OUT = range(2020, 2026)
LEADS = (24, 48, 72)


def main() -> None:
    from dataset import load_dataset
    from eval.metrics import great_circle_km
    from models import analogues as an
    from models.landfall import add_landfall_targets

    data = add_landfall_targets(load_dataset(verbose=False))
    test = data[data["SEASON"].isin(HELD_OUT)]
    sids = list(dict.fromkeys(test["SID"]))

    print("=" * 74)
    print("ANALOGUE SKILL - the five nearest earlier storms, as a forecast")
    print("=" * 74)
    print(f"{len(sids)} storms in the held-out seasons\n")

    errs = {h: [] for h in LEADS}
    kt_errs = {h: [] for h in LEADS}
    persistence = {h: [] for h in LEADS}
    used = 0

    for n, sid in enumerate(sids, 1):
        storm = data[data["SID"] == sid].sort_values("ISO_TIME")
        for _, row in storm.iterrows():
            if row["vmax_kt"] < 34:
                continue
            t0 = pd.Timestamp(row["ISO_TIME"])
            found = an.find(data, sid, t0, k=5)
            if not found.get("analogues"):
                continue
            cons = an.consensus(found, float(row["LAT"]), float(row["LON"]))
            if not cons:
                continue
            used += 1
            for h in LEADS:
                truth = storm[storm["ISO_TIME"] == t0 + pd.Timedelta(hours=h)]
                c = cons.get(f"{h}h")
                if truth.empty or not c:
                    continue
                tlat = float(truth["LAT"].iloc[0])
                tlon = float(truth["LON"].iloc[0])
                errs[h].append(float(great_circle_km(tlat, tlon, c["lat"], c["lon"])))
                kt_errs[h].append(abs(c["vmax_kt"] - float(truth["vmax_kt"].iloc[0])))
                # persistence: the storm simply stays where it is
                persistence[h].append(
                    float(great_circle_km(tlat, tlon, float(row["LAT"]),
                                          float(row["LON"]))))
        if n % 10 == 0 or n == len(sids):
            print(f"  {n}/{len(sids)} storms, {used} fixes with analogues", flush=True)

    served = {24: 126.2, 48: 259.2, 72: 388.3}      # reports/forecast_skill_ci.json
    cliper = {24: 142.1, 48: 285.6, 72: 406.5}

    print(f"\n{'lead':>6}{'analogue':>11}{'persistence':>13}{'CLIPER':>9}"
          f"{'ours':>8}{'n':>7}")
    rows = []
    for h in LEADS:
        if not errs[h]:
            continue
        a = float(np.mean(errs[h]))
        pz = float(np.mean(persistence[h]))
        rows.append({"lead_h": h, "analogue_km": a, "persistence_km": pz,
                     "cliper_km": cliper[h], "served_km": served[h],
                     "analogue_intensity_mae_kt": float(np.mean(kt_errs[h])),
                     "n": len(errs[h])})
        print(f"{h:>5}h{a:>11.0f}{pz:>13.0f}{cliper[h]:>9.0f}{served[h]:>8.0f}"
              f"{len(errs[h]):>7}")

    out = {"storms": len(sids), "fixes_with_analogues": used, "leads": rows,
           "pool_rule": "storms that had already ended when the current storm "
                        "began, so every analogue forecast is out of sample",
           "note": "CLIPER and served errors are from forecast_skill_ci.json and "
                   "are computed over a different set of fixes, so the comparison "
                   "is indicative rather than paired"}
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORT}")

    if rows:
        a24 = rows[0]["analogue_km"]
        print("\nreading")
        if a24 > served[24]:
            print(f"  analogues land {a24 - served[24]:.0f} km further out at 24 h "
                  f"than the model does.")
            print("  which is the expected result and the reason they are presented as")
            print("  context rather than as a forecast. five neighbours cannot beat a")
            print("  fitted ensemble; what they add is a storm with a name attached.")
        else:
            print(f"  analogues beat the served model at 24 h ({a24:.0f} against "
                  f"{served[24]:.0f} km). that would be a finding worth chasing.")


if __name__ == "__main__":
    main()
