# Is there a microwave overpass when we need one?
#
#     python src/scout_microwave.py [--sample 150] [--window 90]
#
# 89 GHz sees the eyewall through the cirrus that hides it from infrared, which
# is the mechanism behind two failures we have measured and cannot fix: the
# severe-storm under-read, and the collapse of EMBC and IRRCDO on INSAT. It is
# the one instrument that would move those numbers.
#
# The catch is that microwave radiometers fly on polar orbiters with narrow
# swaths, not on a geostationary satellite staring at the basin. A storm gets
# looked at when an orbit happens to pass, not when we ask. So before any of
# this is built, the question is how often a pass coincides with a moment we
# already have a labelled patch for.
#
# NASA's metadata search is anonymous even though the data behind it is not, so
# this can be answered before anybody registers for anything. Asked here across
# the whole GPM constellation - GMI, AMSR2 and the three SSMIS flights - which
# share one intercalibrated Level-1C format.

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
REPORT = ROOT / "reports" / "microwave_scout.json"
CACHE = ROOT / "data" / "processed" / "scenes"

# one intercalibrated brightness-temperature product per radiometer
SENSORS = {
    "GMI": ("C2259345484-GES_DISC", 2014),
    "AMSR2": ("C2264132976-GES_DISC", 2012),
    "SSMIS-F16": ("C2264132881-GES_DISC", 2005),
    "SSMIS-F17": ("C2264132902-GES_DISC", 2008),
    "SSMIS-F18": ("C2264132936-GES_DISC", 2010),
}


def granules(when: pd.Timestamp, lat: float, lon: float, window_min: int,
             box_deg: float = 3.0) -> list[dict]:
    lo = (when - pd.Timedelta(minutes=window_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (when + pd.Timedelta(minutes=window_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = [("collection_concept_id", cid) for cid, _ in SENSORS.values()]
    params += [("temporal", f"{lo},{hi}"),
               ("bounding_box",
                f"{lon-box_deg},{lat-box_deg},{lon+box_deg},{lat+box_deg}"),
               ("page_size", "50")]
    url = CMR + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "chakravat-research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["feed"]["entry"]


def sensor_of(title: str) -> str:
    # match the specific flight first: "SSMIS" alone is in all three of them,
    # and checking it first labelled every SSMIS granule as F16
    for name in SENSORS:
        if "-" in name and name.split("-")[-1] in title:
            return name
    for name in SENSORS:
        if "-" not in name and name in title:
            return name
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--window", type=int, default=90,
                    help="minutes either side of the patch time")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(CACHE / "patches.csv")
    meta = meta[meta["bt_vmax_kt"].notna()].copy()
    meta["time"] = pd.to_datetime(meta["time"])
    meta = meta[meta["time"].dt.year >= 2005]          # first SSMIS in the set
    pick = meta.sample(min(args.sample, len(meta)), random_state=args.seed)

    print("=" * 76)
    print(f"MICROWAVE SCOUT - is there a pass within {args.window} min of a patch?")
    print("=" * 76)
    print(f"{len(meta)} labelled patches from 2005 on; sampling {len(pick)}\n")

    rows, t0 = [], time.time()
    for n, r in enumerate(pick.itertuples(), 1):
        try:
            ents = granules(pd.Timestamp(r.time), float(r.lat), float(r.lon),
                            args.window)
        except Exception as exc:                        # noqa: BLE001
            print(f"  query failed ({type(exc).__name__}), skipping")
            continue
        seen = {}
        for e in ents:
            s = sensor_of(e.get("title", ""))
            seen[s] = seen.get(s, 0) + 1
        rows.append({"time": str(r.time), "year": int(pd.Timestamp(r.time).year),
                     "lat": float(r.lat), "lon": float(r.lon),
                     "vmax_kt": float(r.bt_vmax_kt),
                     "passes": len(ents), "by_sensor": seen})
        if n % 25 == 0 or n == len(pick):
            got = sum(1 for x in rows if x["passes"])
            print(f"  {n}/{len(pick)}  {got} with a pass  {time.time()-t0:.0f}s",
                  flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no queries succeeded")

    any_pass = float((df.passes > 0).mean())
    print(f"\n{(df.passes > 0).sum()} of {len(df)} sampled patches have at least "
          f"one pass within {args.window} min  ({any_pass:.0%})")
    print(f"mean passes when there is one: {df.loc[df.passes>0,'passes'].mean():.1f}")

    print(f"\n{'era':<16}{'patches':>9}{'with a pass':>13}{'share':>8}")
    for label, lo, hi in (("2005-2011", 2005, 2011), ("2012-2013", 2012, 2013),
                          ("2014-2025", 2014, 2025)):
        m = df.year.between(lo, hi)
        if m.sum():
            print(f"{label:<16}{int(m.sum()):>9}{int((df[m].passes>0).sum()):>13}"
                  f"{(df[m].passes>0).mean():>8.0%}")

    by = {}
    for x in rows:
        for s, k in x["by_sensor"].items():
            by[s] = by.get(s, 0) + k
    print(f"\npasses by instrument: {dict(sorted(by.items(), key=lambda t: -t[1]))}")

    strong = df[df.vmax_kt >= 64]
    if len(strong):
        print(f"\nat 64 kt and above, the band we read low: "
              f"{(strong.passes>0).sum()} of {len(strong)} have a pass "
              f"({(strong.passes>0).mean():.0%})")

    out = {"window_min": args.window, "sampled": len(df),
           "with_a_pass": int((df.passes > 0).sum()), "share": any_pass,
           "by_instrument": by,
           "severe_sampled": int(len(strong)),
           "severe_with_a_pass": int((strong.passes > 0).sum()) if len(strong) else 0,
           "note": "a granule intersecting the box means the orbit came near; "
                   "whether the swath actually covered the storm centre needs "
                   "the granule geometry, so this is an upper bound",
           "rows": rows}
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORT}")
    print("\nthis is an upper bound: a granule whose orbit crosses the box may")
    print("still have the storm outside its swath. the real number comes from")
    print("the granule geometry, and only matters if this one is large enough.")


if __name__ == "__main__":
    main()
