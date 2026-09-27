# Is a storm surge model validatable with the data that exists?
#
#     python src/scout_tide_gauges.py
#
# Surge is validated one way: observed sea level minus predicted tide is the
# residual, and that residual is what a surge model has to reproduce. So the
# question is not whether we can write a surge model - anyone can - but whether
# there is a tide gauge near enough to a landfall, working at the time, for
# enough of our storms to say anything with a straight face.
#
# This answers that before any surge code is written. It reads the UHSLC station
# catalogue (free, global, no login), finds the gauges on the North Indian Ocean
# coast, and counts how many of our storms made landfall near a gauge that was
# recording at the time.
#
# If that count is thin, the honest answer is that we do not build a surge
# model, because an unvalidated number in this project would be the only one.

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

META = "https://uhslc.soest.hawaii.edu/data/meta.geojson"
REPORT = ROOT / "reports" / "tide_gauge_scout.json"

# the basin, generously drawn
BOX = dict(lat=(-5.0, 31.0), lon=(30.0, 100.0))
NEAR_KM = 200.0          # a gauge further than this sees little of the surge
HELD_OUT = range(2020, 2026)


def stations() -> pd.DataFrame:
    req = urllib.request.Request(META, headers={"User-Agent": "chakravat-research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        raw = json.load(r)
    rows = []
    for f in raw["features"]:
        p, g = f["properties"], f.get("geometry") or {}
        c = (g.get("coordinates") or [None, None])
        rows.append({"name": p.get("name"), "country": p.get("country"),
                     "uhslc_id": p.get("uhslc_id"),
                     "lon": c[0], "lat": c[1],
                     "fd_span": p.get("fd_span"), "rq_span": p.get("rq_span")})
    return pd.DataFrame(rows).dropna(subset=["lat", "lon"])


def span_years(span) -> tuple[int, int] | None:
    # spans arrive as {"oldest": "...", "latest": "..."} or a string pair
    if not span:
        return None
    if isinstance(span, dict):
        lo, hi = span.get("oldest"), span.get("latest")
    elif isinstance(span, (list, tuple)) and len(span) == 2:
        lo, hi = span
    else:
        return None
    try:
        return int(str(lo)[:4]), int(str(hi)[:4])
    except (TypeError, ValueError):
        return None


def main() -> None:
    from eval.metrics import great_circle_km
    from ingest.ibtracs import build

    st = stations()
    here = st[(st.lat.between(*BOX["lat"])) & (st.lon.between(*BOX["lon"]))].copy()
    print("=" * 78)
    print("TIDE GAUGE SCOUT - can a surge model be validated here?")
    print("=" * 78)
    print(f"{len(st)} UHSLC stations worldwide, {len(here)} inside the basin box\n")
    print(f"  {'station':<26}{'country':<10}{'lat':>7}{'lon':>8}   {'research quality':<18}"
          f"{'fast delivery'}")
    for _, r in here.sort_values("country").iterrows():
        rq, fd = span_years(r.rq_span), span_years(r.fd_span)
        print(f"  {str(r['name'])[:25]:<26}{str(r['country'])[:9]:<10}"
              f"{r.lat:>7.2f}{r.lon:>8.2f}   "
              f"{(f'{rq[0]}-{rq[1]}' if rq else 'none'):<18}"
              f"{(f'{fd[0]}-{fd[1]}' if fd else 'none')}")

    # every landfall in the archive: first fix with the centre over land
    track = build(str(ROOT / "data" / "raw" / "ibtracs.NI.csv"))
    track["DIST2LAND"] = pd.to_numeric(track.get("DIST2LAND"), errors="coerce")
    landfalls = []
    for sid, g in track.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        ashore = g[g["DIST2LAND"] == 0]
        if ashore.empty:
            continue
        row = ashore.iloc[0]
        landfalls.append({"sid": sid, "name": str(row["NAME"]), "season": int(row["SEASON"]),
                          "time": row["ISO_TIME"], "lat": float(row["LAT"]),
                          "lon": float(row["LON"]), "vmax_kt": float(row["vmax_kt"])})
    lf = pd.DataFrame(landfalls)
    print(f"\n{len(lf)} landfalls in the archive, "
          f"{int((lf.season.isin(HELD_OUT)).sum())} of them in the held-out seasons")

    # pair each landfall with any gauge close enough and recording at the time
    pairs = []
    for _, L in lf.iterrows():
        for _, S in here.iterrows():
            km = float(great_circle_km(L["lat"], L["lon"], S["lat"], S["lon"]))
            if km > NEAR_KM:
                continue
            yr = int(pd.Timestamp(L["time"]).year)
            rq, fd = span_years(S.rq_span), span_years(S.fd_span)
            covered = [k for k, sp in (("research", rq), ("fast", fd))
                       if sp and sp[0] <= yr <= sp[1]]
            if covered:
                pairs.append({"sid": L["sid"], "storm": L["name"], "season": L["season"],
                              "vmax_kt": L["vmax_kt"], "station": S["name"],
                              "country": S["country"], "km": round(km),
                              "datasets": covered})
    pr = pd.DataFrame(pairs)
    strong = pr[pr.vmax_kt >= 48] if len(pr) else pr

    print(f"\nlandfalls with a gauge within {NEAR_KM:.0f} km that was recording:")
    print(f"  any intensity                {len(pr):>4} pairs, "
          f"{pr.sid.nunique() if len(pr) else 0} distinct storms")
    print(f"  severe cyclonic storm and up {len(strong):>4} pairs, "
          f"{strong.sid.nunique() if len(strong) else 0} distinct storms")
    if len(pr):
        held = pr[pr.season.isin(HELD_OUT)]
        print(f"  in held-out seasons 2020-25  {len(held):>4} pairs, "
              f"{held.sid.nunique()} distinct storms")
        print("\n  the strongest pairings:")
        for _, r in strong.sort_values("vmax_kt", ascending=False).head(12).iterrows():
            print(f"    {r.storm:<12} {r.season}  {r.vmax_kt:>3.0f} kt  "
                  f"{str(r.station)[:22]:<23}{r.km:>4} km  {','.join(r.datasets)}")

    out = {"stations_in_basin": int(len(here)), "landfalls": int(len(lf)),
           "pairs_any": int(len(pr)), "pairs_severe": int(len(strong)),
           "storms_any": int(pr.sid.nunique()) if len(pr) else 0,
           "storms_severe": int(strong.sid.nunique()) if len(strong) else 0,
           "near_km": NEAR_KM,
           "stations": here.drop(columns=["fd_span", "rq_span"]).to_dict("records"),
           "pairs": pr.to_dict("records") if len(pr) else [],
           "caveat": "a gauge inside its published span is not a guarantee of data "
                     "during the storm: gauges fail, go off scale, or lose power at "
                     "exactly the peak, which is when surge matters",
           }
    REPORT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {REPORT}")

    n = strong.sid.nunique() if len(strong) else 0
    print("\nverdict")
    if n >= 15:
        print(f"  {n} severe landfalls near a working gauge. that is enough to fit and")
        print("  verify an empirical surge relation honestly. worth building.")
    elif n >= 6:
        print(f"  {n} severe landfalls near a working gauge. thin: enough to sanity")
        print("  check a published relation, not enough to fit one of our own.")
    else:
        print(f"  only {n} severe landfalls near a working gauge. not enough to")
        print("  validate anything. a surge number here would be the one unverified")
        print("  figure in the project, and it should not be built.")


if __name__ == "__main__":
    main()
