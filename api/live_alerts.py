# turning a live INSAT detection into an alert
#
# the historical path starts from an IBTrACS row: a named storm with an agreed
# position, an agreed intensity and a distance to land. a live detection has
# none of that. it is a blob the detector found in an image taken an hour ago,
# and everything about it - where it is, how strong it is, whether it is even
# a cyclone - is our own model's opinion. so this module exists to adapt the
# pipeline's output into the shapes the alerting code already speaks, and to
# be honest in the two places where the live path genuinely knows less:
#
#   it has no name, because naming is IMD's and only IMD's. live systems are
#     called by their detection id.
#
#   it has no distance to land, so it cannot know the centre is already
#     inland. over_land is left false and the landfall probability is the
#     model's, unpatched. the historical path corrects that from IBTrACS and
#     this one cannot, which is a real difference and is labelled as one.
#
# identity is the other problem. the pipeline numbers its detections DET01,
# DET02 per run, which is useless for an alert lifecycle - an Update has to
# reference the alert it supersedes, so the same physical storm must keep the
# same id between runs. so a detection is matched to the last live system
# alerted near it, and only gets a new id when nothing is close enough.

from __future__ import annotations

import math

import pandas as pd

MATCH_KM = 300.0        # a storm moves maybe 25 km/h, so this covers ~12 h
MATCH_H = 24.0          # older than this and we call it a new system
MIN_SCORE = 0.0         # detector confidence floor, tuned threshold applies upstream


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def match_sid(entries: list[dict], lat: float, lon: float,
              when: pd.Timestamp) -> str | None:
    # the live system already being alerted that this detection continues
    best, best_km = None, MATCH_KM
    for e in entries:
        if not e["sid"].startswith("live."):
            continue
        st = e.get("state") or {}
        if st.get("lat") is None:
            continue
        age = abs((when - pd.Timestamp(e["sent"]).tz_convert("UTC").tz_localize(None))
                  .total_seconds()) / 3600.0
        if age > MATCH_H:
            continue
        km = _haversine_km(st["lat"], st["lon"], lat, lon)
        if km < best_km:
            best, best_km = e["sid"], km
    return best


def new_sid(entries: list[dict], when: pd.Timestamp) -> str:
    day = f"{pd.Timestamp(when):%Y%m%d}"
    used = {e["sid"] for e in entries if e["sid"].startswith(f"live.{day}.")}
    return f"live.{day}.{len(used) + 1:02d}"


def adapt(storm: dict, level: float) -> tuple[dict, dict] | None:
    # pipeline storm -> the (forecast, landfall) pair the alerting code wants
    from ingest.ibtracs import imd_category

    pred = storm.get("prediction")
    if not pred:
        return None                      # too few track points to forecast from

    fc = {
        "issued_at": pred["issued_at"],
        # deliberately not a name: naming a cyclone is IMD's call, not ours
        "name": storm["id"],
        "cone_level": pred.get("level", level),
        "current": {"lat": storm["lat"], "lon": storm["lon"],
                    "vmax_kt": storm["vmax_kt"],
                    "category": imd_category(storm["vmax_kt"])},
        "forecast": pred["forecast"],
        "rapid_intensification": pred["rapid_intensification"],
    }

    lfp = pred.get("landfall") or {}
    lf = {
        "probability_within_72h": lfp.get("probability_within_72h", 0.0),
        # no IBTrACS distance-to-land on the live path, so we cannot assert the
        # centre is inland and do not pretend to
        "over_land": False,
    }
    if lfp.get("hours") is not None:
        lf.update({
            "hours_to_landfall": lfp["hours"],
            "lat": lfp["lat"], "lon": lfp["lon"],
            "vmax_kt": lfp["vmax_kt"],
            "category_at_landfall": imd_category(lfp["vmax_kt"]),
            # the regression path, so the wider of the two position errors
            "typical_error": {"timing_h": 9.0, "position_km": 234.0,
                              "intensity_kt": 11.8},
        })
    return fc, lf
