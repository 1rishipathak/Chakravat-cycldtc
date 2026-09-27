# Drive the live loop through a real storm, one cycle at a time.
#
#     python src/rehearse_live.py --storm BIPARJOY --season 2023
#     python src/rehearse_live.py --storm TAUKTAE --season 2021 --cycles 8
#
# The live path cannot be tested by waiting for a cyclone, and replay_alerts.py
# does not test it either: that replays the alerting *rule* using best track, so
# the storm arrives with a name and an identity already attached. On the live
# feed there is no best track. A detection is the only evidence there is, the
# system has to invent an identity for it, and three hours later it has to
# recognise the same storm from position alone. Nothing in the test suite
# touched that until this.
#
# So: point the same endpoint the fetch loop calls at archive INSAT scans,
# oldest first, and watch what happens to the storm's identity. One real storm
# should collect one identity. If it collects six, the system would have issued
# six unrelated alert threads for one cyclone, and no reader could follow it.
#
# The alert store is snapshotted before and restored after, so a rehearsal never
# disturbs what has actually been published.

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

STORE = ROOT / "data" / "alerts"
REPORT = ROOT / "reports" / "live_rehearsal.json"
BASE = "http://127.0.0.1:8021"


def scan_times(season: int, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    # the INSAT scans on disk inside the storm's life, oldest first
    out = []
    for f in sorted((ROOT / "data" / "insat" / "grid" / str(season)).glob("nio_*.nc")):
        stem = f.stem.split("_")                       # nio_YYYYMMDD_HH
        t = pd.Timestamp(f"{stem[1][:4]}-{stem[1][4:6]}-{stem[1][6:]} {stem[2]}:00")
        if start <= t <= end:
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storm", default="BIPARJOY")
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--cycles", type=int, default=0, help="stop after N cycles")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--level", type=float, default=0.67)
    args = ap.parse_args()

    from eval.metrics import great_circle_km
    from ingest.ibtracs import build

    ATTRIB_KM = 300.0          # closer than this and we call it that system

    track = build(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    storm = track[(track["SEASON"] == args.season)
                  & (track["NAME"].str.upper() == args.storm.upper())].sort_values("ISO_TIME")
    if storm.empty:
        sys.exit(f"{args.storm} {args.season} is not in the best track")

    start, end = storm["ISO_TIME"].min(), storm["ISO_TIME"].max()
    times = scan_times(args.season, start, end)
    if args.cycles:
        times = times[:args.cycles]
    if not times:
        sys.exit(f"no INSAT scans on disk between {start} and {end}")

    print("=" * 78)
    print(f"LIVE REHEARSAL - {args.storm} {args.season}, {len(times)} INSAT scans")
    print("=" * 78)
    print(f"best track: {start} to {end}, peak {storm['vmax_kt'].max():.0f} kt")
    print("\nthe storm arrives with no name and no identity. each cycle the system")
    print("has to decide whether what it sees is something it has seen before.\n")
    print(f"  {'scan':<17}{'found':>6}{'identity':>18}{'new?':>6}{'alert':>8}"
          f"   {'nearest real system':<22}{'off it':>8}")

    snapshot = Path(tempfile.mkdtemp(prefix="chakravat_alerts_")) / "alerts"
    if STORE.exists():
        shutil.copytree(STORE, snapshot)
    rows, sids, alerts = [], Counter(), 0
    try:
        for t in times:
            when = t.strftime("%Y-%m-%d %H:%M:%S")
            try:
                r = requests.post(f"{args.base}/api/live/rehearse",
                                  params={"time": when, "source": "insat",
                                          "level": args.level}, timeout=900).json()
            except Exception as exc:                        # noqa: BLE001
                print(f"  {when:<17}  request failed: {type(exc).__name__}")
                continue

            if not r.get("available") or not r.get("results"):
                print(f"  {when:<17}{0:>6}{'-':>18}{'-':>6}{'-':>8}{'-':>16}")
                rows.append({"scan": when, "found": 0})
                continue

            # every system in the basin at this hour, not only the one we came for
            here = track[(track["ISO_TIME"] - t).abs() <= pd.Timedelta("3h")]
            for res in r["results"]:
                lat, lon = res.get("lat"), res.get("lon")
                who, km = None, float("nan")
                if lat is not None and len(here):
                    d = here.apply(lambda row: great_circle_km(row["LAT"], row["LON"],
                                                              lat, lon), axis=1)
                    j = d.idxmin()
                    if d[j] <= ATTRIB_KM:
                        who, km = str(here.loc[j, "NAME"]).upper(), float(d[j])
                issued = res.get("issued")
                is_target = who == args.storm.upper()
                if is_target:
                    sids[res["sid"]] += 1
                alerts += bool(issued)
                label = who if who else "no system within 300 km"
                print(f"  {when:<17}{r['tracks_found']:>6}{res['sid']:>18}"
                      f"{('no' if res.get('sid_was_matched') else 'NEW'):>6}"
                      f"{(res.get('alert', {}).get('msgType', '-') if issued else '-'):>8}"
                      f"   {label:<22}"
                      f"{('' if km != km else f'{km:.0f} km'):>8}")
                rows.append({"scan": when, "found": r["tracks_found"], "sid": res["sid"],
                             "matched": bool(res.get("sid_was_matched")),
                             "issued": bool(issued), "reason": res.get("reason"),
                             "attributed_to": who, "km_from_that_system": float(km),
                             "is_target": is_target})
    finally:
        if snapshot.exists():
            if STORE.exists():
                shutil.rmtree(STORE)
            shutil.copytree(snapshot, STORE)
            shutil.rmtree(snapshot.parent, ignore_errors=True)
            print("\nalert store restored to what was published before the rehearsal")

    on_target = [r for r in rows if r.get("is_target")]
    unattributed = [r for r in rows if r.get("sid") and not r.get("attributed_to")]
    cycles_with_target = len({r["scan"] for r in on_target})
    errs = [r["km_from_that_system"] for r in on_target
            if r["km_from_that_system"] == r["km_from_that_system"]]
    out = {
        "storm": args.storm, "season": args.season,
        "cycles": len(times),
        "cycles_that_found_the_target": cycles_with_target,
        "identities_given_to_the_target": len(sids),
        "identities": dict(sids),
        "alerts_issued": alerts,
        "detections_matching_no_known_system": len(unattributed),
        "median_km_from_the_target": float(pd.Series(errs).median()) if errs else None,
        "attribution_radius_km": ATTRIB_KM,
        "rows": rows,
        "reading": "one real storm should collect one identity. more than one means "
                   "the system would have opened unrelated alert threads for the "
                   "same cyclone, and nobody reading the feed could follow it.",
    }
    REPORT.write_text(json.dumps(out, indent=2))

    print(f"\n{cycles_with_target} of {len(times)} cycles found {args.storm}")
    print(f"identities it was given: {len(sids)}  {dict(sids)}")
    if errs:
        print(f"median distance from its best track: {pd.Series(errs).median():.0f} km")
    print(f"alerts issued across all systems: {alerts}")
    print(f"detections matching no system in best track: {len(unattributed)}"
          f"  (the basin-wide false alarm rate is 0.33 per scene at this threshold)")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
