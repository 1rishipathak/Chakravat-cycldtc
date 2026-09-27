# replay storms through the alerting decision and measure how much it talks
#
# the only honest way to test an alerting system is to run real storms through
# it fix by fix and read what it decided to send. a system that fires on every
# tick is useless and a system that stays silent through a category jump is
# dangerous, and you cannot tell which one you built by reading the code. the
# first version of the rule sent on 25 of Amphan's 25 forecast cycles and on
# all 22 cycles of a 32 kt system that never became a cyclone; both numbers
# came from here, not from inspection.
#
#   python src/replay_alerts.py --sid 2020136N10088     # one storm, in detail
#   python src/replay_alerts.py --suite                 # the whole suite
#
# the suite writes reports/alert_cadence.json, which is what the dashboard and
# the deck read, so the claim and the measurement cannot drift apart.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8010"
ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "alerts"
REPORT = ROOT / "reports" / "alert_cadence.json"

# a spread of intensities, because the question is whether the rule
# discriminates - a weak system must be quiet and a super cyclone must not be
SUITE = [
    "2025295N09068",   # unnamed, 25 kt, below the alerting floor
    "2022230N18095",   # unnamed, 33 kt deep depression
    "2024329N04089",   # FENGAL, 45 kt
    "2021267N18094",   # GULAB/SHAHEEN, 60 kt
    "2023156N10067",   # BIPARJOY, 90 kt, long-lived
    "2020136N10088",   # AMPHAN, 130 kt, super cyclonic storm
]

# how each reason is classified, for the mix in the report
ROUTINE = "routine"


def classify(reason: str) -> str:
    if reason.startswith("routine"):
        return ROUTINE
    if "intensified" in reason or "peak raised" in reason:
        return "intensity change"
    if "rapid intensification" in reason:
        return "rapid intensification"
    if "crossed the coast" in reason:
        return "landfall"
    if "landfall" in reason:
        return "landfall forecast change"
    if "within" in reason:
        return "new place threatened"
    return "first alert"


def one(base: str, sid: str, every: int, verbose: bool) -> dict:
    detail = requests.get(f"{base}/api/storm/{sid}", timeout=30).json()
    fixes = [p["time"] for p in detail["track"]][::every]
    name = detail.get("name") or sid
    peak = detail["peak_kt"]
    if verbose:
        print(f"{name} ({sid}): {len(fixes)} forecast cycles, "
              f"peak {peak:.0f} kt, every {every * 3} h\n")

    mix, sent = Counter(), 0
    for t in fixes:
        r = requests.post(f"{base}/api/alerts/evaluate",
                          params={"sid": sid, "time": t}, timeout=120)
        if r.status_code != 200:
            print(f"  {t}  HTTP {r.status_code}  {r.text[:120]}")
            continue
        d = r.json()
        if d["issued"]:
            sent += 1
            mix[classify(d["reason"])] += 1
            if verbose:
                a = d["alert"]
                print(f"  {t}  {a['msgType']:<7} {a['severity']:<8} {d['reason']}")
        elif verbose:
            print(f"  {t}  {'-':<7} {'':<8} {d['reason']}")

    # span comes from the timestamps, never from assuming a fix interval. the
    # first version computed 3 h * cycles, but IBTrACS fixes here are 6-hourly
    # as often as 3-hourly, which silently doubled every alerts-per-day figure
    # it reported. alerts per day is the number that means anything - "share of
    # cycles" is really a statement about the archive's sampling, not about how
    # much the system talks - so it had to be the one that is right.
    t0, t1 = datetime.fromisoformat(fixes[0]), datetime.fromisoformat(fixes[-1])
    span_h = (t1 - t0).total_seconds() / 3600.0
    step_h = span_h / max(len(fixes) - 1, 1)
    return {"sid": sid, "name": name, "peak_kt": peak, "cycles": len(fixes),
            "alerts": sent, "share": round(sent / max(len(fixes), 1), 3),
            "span_h": round(span_h, 1), "fix_interval_h": round(step_h, 1),
            "per_day": round(24 * sent / max(span_h, 1), 2),
            "reasons": dict(mix)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--every", type=int, default=1, help="use every Nth 3-hourly fix")
    ap.add_argument("--keep", action="store_true", help="do not clear the alert store")
    a = ap.parse_args()

    if not a.keep and STORE.exists():
        shutil.rmtree(STORE)
        print(f"cleared {STORE}\n")

    try:
        if a.suite:
            rows = [one(a.base, s, a.every, verbose=False) for s in SUITE]
            print(f"{'storm':<18}{'peak':>6}{'fix h':>7}{'cycles':>8}"
                  f"{'alerts':>8}{'per day':>9}")
            for r in rows:
                print(f"{r['name'][:17]:<18}{r['peak_kt']:>5.0f}"
                      f"{r['fix_interval_h']:>7.1f}{r['cycles']:>8}"
                      f"{r['alerts']:>8}{r['per_day']:>9.1f}")
            mix = Counter()
            for r in rows:
                mix.update(r["reasons"])
            total = sum(mix.values())
            print("\nwhy alerts were sent")
            for why, n in mix.most_common():
                print(f"  {why:<26}{n:>4}  {100 * n / max(total, 1):>4.0f}%")
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(
                {"storms": rows, "reason_mix": dict(mix), "total_alerts": total},
                indent=1), encoding="utf-8")
            print(f"\nwrote {REPORT.relative_to(ROOT)}")
        else:
            r = one(a.base, a.sid or SUITE[-1], a.every, verbose=True)
            print(f"\n{r['alerts']} alerts from {r['cycles']} forecast cycles "
                  f"({100 * r['share']:.0f}%), {r['per_day']:.1f} per day")
    except requests.ConnectionError:
        sys.exit(f"no API at {a.base} - start it with uvicorn api.main:app --port 8010")


if __name__ == "__main__":
    main()
