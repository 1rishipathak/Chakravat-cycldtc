# the difference between a CAP file and an alerting system
#
# api/alerts.py renders one CAP document. that is a format, not a system. an
# alerting system has to answer three more questions, and each one is a way
# real warning systems fail:
#
#   when is a change worth sending?  send on every model tick and you train
#     people to ignore you. alert fatigue is the documented killer of public
#     warning systems, not missed detections. so we re-issue only when the
#     picture changes by more than our own measured error can resolve - a
#     landfall time that moved 2 h is noise when our timing error is 9 h.
#
#   how does a consumer know this alert replaces that one?  CAP answers with
#     msgType and references: Alert, then Update carrying the identifier of
#     what it supersedes, then Cancel. without it a consumer sees fifty
#     unrelated alerts for one storm instead of one storm's story.
#
#   where does a consumer get them?  a stable feed it can poll, holding alerts
#     at permanent URLs. that is what this store is for.
#
# everything written here stays status "Exercise". this is a prototype, and
# the whole point of speaking a format that reaches phones is that you must
# never be mistaken for the authority that actually uses it.

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "alerts"
INDEX = STORE / "index.json"

SENDER = "chakravat.prototype"

# re-issue thresholds, set at half of our own measured error so that we never
# re-alert on a change the forecast cannot actually resolve. landfall timing
# MAE 9.0 h and landfall position median error 114 km, both from
# reports/landfall_results.json on held-out storms.
TIMING_H = 4.5
POSITION_KM = 57.0

# ...but a threshold alone is not enough, and the first version of this file
# proved it: replaying Amphan it sent an alert on all 25 forecast cycles. the
# reasons were things like a landfall point that "moved" 472 km between two
# cycles and Lhasa entering the threat zone. that is the 72 h cone wobbling,
# not the storm changing. so two more rules, both of them how IMD actually
# works rather than anything clever:
#
#   only react to what the forecast can see. a landfall point at 70 h lead is
#     noise, so landfall shifts only count inside LANDFALL_TRIGGER_H, and a new
#     place only counts if the storm reaches it inside NEAR_H.
#
#   only interrupt for bad news. an upgrade is worth breaking in for; a
#     downgrade can wait for the next routine bulletin. warning systems that
#     blast people when a storm weakens get switched off.
NEAR_H = 36.0                 # a place is news only this close
LANDFALL_TRIGGER_H = 48.0     # landfall shifts only count inside this lead
PROB_IN, PROB_OUT = 0.60, 0.35   # hysteresis band, so the flag cannot flicker

# routine bulletin cadence, by how urgent things are. this is the floor: a
# material change sends immediately whatever the cadence says.
CADENCE_IMMINENT_H = 3.0      # centre inland, or landfall within 12 h
CADENCE_ACTIVE_H = 6.0        # landfall within 24 h, or a hurricane-force storm
CADENCE_WATCH_H = 12.0        # everything else

# ...and severity comes before proximity. replaying a 32 kt system that never
# became a cyclone, the first cadence sent on all 22 of its forecast cycles,
# because it was near a coast and the rule read proximity first. a depression
# near land is not a super cyclone near land. India gets many depressions and
# alerting on each one at cyclone tempo is how you get ignored.
CYCLONE_KT = 34.0             # IMD: cyclonic storm begins here
HURRICANE_KT = 64.0           # very severe cyclonic storm

# a floor on how often anything at all can go out. replaying a 33 kt monsoon
# depression, the rules above still sent on all 22 cycles: its predicted
# landfall point jumped 584 km between two of them, its category oscillated
# deep depression - depression - deep depression, and it crossed the whole
# gangetic plain putting a new city in range each cycle. those triggers were
# not wrong, they were faithfully reporting a genuinely unstable forecast. the
# answer to that is a policy, stated in the open rather than hidden in a
# threshold: routine news waits, and only the things you would wake someone
# for can break the gap.
MIN_GAP_WEAK_H = 6.0
MIN_GAP_CYCLONE_H = 3.0

# alerting floor: IMD's deep depression. below this it is weather, not a storm.
MIN_ALERT_KT = 28.0

# IMD's scale, weakest first, for telling an upgrade from a downgrade
CATEGORY_ORDER = [
    "Low Pressure Area", "Depression", "Deep Depression", "Cyclonic Storm",
    "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm",
]


def rank(category: str) -> int:
    return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else -1


def expires_h(state: dict) -> float:
    # an alert must stay valid until the next routine one is due, or a consumer
    # sees a gap where the storm apparently stopped existing
    return cadence_h(state) + 1.0


def tier_kt(state: dict) -> float:
    # how loud this storm is allowed to be. it reads the *current* intensity,
    # not the forecast peak: gating tempo on the forecast peak meant a 25 kt
    # depression whose model happened to predict 44 kt got the same 3-hourly
    # treatment as a super cyclone, which is how two weak systems in the suite
    # kept alerting on every cycle. the forecast peak still sets CAP severity,
    # where being cautious is the right bias - it just does not set the tempo.
    return state.get("vmax_kt", state.get("peak_kt", 0))


def cadence_h(state: dict) -> float:
    peak = tier_kt(state)
    h = state.get("landfall_h")
    imminent = bool(state.get("over_land")) or (h is not None and h <= 12)
    if peak < CYCLONE_KT:
        # a depression or deep depression: worth saying, not worth shouting
        return CADENCE_ACTIVE_H if imminent else CADENCE_WATCH_H
    if imminent:
        return CADENCE_IMMINENT_H
    if (h is not None and h <= 24) or peak >= HURRICANE_KT:
        return CADENCE_ACTIVE_H
    return CADENCE_WATCH_H


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def material_state(fc: dict, lf: dict, places: list[dict]) -> dict:
    # the part of a forecast that decides whether people need telling again.
    # deliberately not the whole forecast: a centre that moved 3 km and a wind
    # that changed 1 kt are not news.
    peak = max([fc["current"]] + fc["forecast"], key=lambda p: p["vmax_kt"])
    return {
        # where the centre was when this went out. the decision never reads it;
        # the live path does, to recognise that a detection near here is the
        # same system it alerted on an hour ago and not a new one.
        "lat": round(float(fc["current"]["lat"]), 3),
        "lon": round(float(fc["current"]["lon"]), 3),
        "vmax_kt": round(float(fc["current"]["vmax_kt"])),
        "category": fc["current"]["category"],
        "peak_category": peak["category"],
        "peak_kt": round(float(peak["vmax_kt"])),
        "over_land": bool(lf.get("over_land")),
        "landfall_h": (None if lf.get("hours_to_landfall") is None
                       else round(float(lf["hours_to_landfall"]), 1)),
        "landfall_prob": round(float(lf.get("probability_within_72h", 0.0)), 3),
        "landfall_lat": (None if lf.get("lat") is None else round(float(lf["lat"]), 3)),
        "landfall_lon": (None if lf.get("lon") is None else round(float(lf["lon"]), 3)),
        "ri_flagged": bool(fc["rapid_intensification"]["flagged"]),
        # the whole zone for the record, and the near field for deciding: a
        # town the storm reaches in 12 h is news, one it might reach in 70 h
        # is the cone breathing
        "places": sorted(p["name"] for p in places),
        "near_places": sorted(p["name"] for p in places
                              if p["first_inside_h"] <= NEAR_H),
        # latched once true, so the flag cannot flicker back and forth across
        # the probability threshold
        "landfall_expected": bool(lf.get("over_land"))
        or float(lf.get("probability_within_72h", 0.0)) >= PROB_IN,
    }


def decide(prev: dict | None, new: dict, issued: pd.Timestamp,
           prev_sent: pd.Timestamp | None) -> tuple[str | None, str]:
    # returns (msgType, reason). msgType None means nothing worth sending.
    # intensity decides whether there is anything to say. the first version
    # also required "no landfall expected", which meant a 25 kt low near a
    # coast - and every system near a coast has a landfall estimate - was never
    # quiet, so nothing was ever below the floor.
    quiet = new["peak_kt"] < MIN_ALERT_KT and not new["over_land"]

    if prev is None:
        if quiet:
            return None, "below alerting threshold, nothing issued"
        new["announced"] = list(new["near_places"])
        return "Alert", f"first alert for this storm ({new['category']})"

    if quiet:
        return "Cancel", "storm no longer meets the alerting threshold"

    # hysteresis, applied here because it needs the previous answer: the flag
    # goes up at PROB_IN and only comes down below PROB_OUT, so a probability
    # hovering near the threshold cannot flip the alert on and off. the latched
    # value is written back so it is what the next cycle compares against.
    p = new["landfall_prob"]
    new["landfall_expected"] = bool(
        new["over_land"] or (True if p >= PROB_IN else
                             (False if p < PROB_OUT else prev["landfall_expected"])))

    # reasons carry a flag: urgent ones are the four things you would wake
    # someone for, and only those may break the minimum gap below
    reasons: list[tuple[str, bool]] = []

    # intensity compares against the strongest this storm has ever been, not
    # against last cycle. a storm that wobbles deep depression - depression -
    # deep depression has not intensified the second time, and announcing that
    # it has is how the 33 kt system earned three of its alerts.
    prev_max = prev.get("max_rank", rank(prev["category"]))
    prev_peak_max = prev.get("max_peak_rank", rank(prev["peak_category"]))
    new["max_rank"] = max(prev_max, rank(new["category"]))
    new["max_peak_rank"] = max(prev_peak_max, rank(new["peak_category"]))
    if rank(new["category"]) > prev_max:
        reasons.append((f"intensified to {new['category']}", True))
    if rank(new["peak_category"]) > prev_peak_max:
        reasons.append((f"forecast peak raised to {new['peak_category']}", True))
    if new["over_land"] and not prev["over_land"]:
        reasons.append(("centre has crossed the coast", True))
    if new["ri_flagged"] and not prev["ri_flagged"]:
        reasons.append(("rapid intensification now flagged", True))
    if new["landfall_expected"] and not prev["landfall_expected"]:
        reasons.append((f"landfall now expected within 72 h "
                        f"(probability {100 * p:.0f}%)", True))

    # landfall shifts only count once landfall is close enough for the forecast
    # to mean anything, and only for a real cyclone. for weaker systems the
    # landfall point is the least reliable thing we produce - it moved 584 km
    # between two cycles of a monsoon depression - so it is not news.
    near = (new["landfall_h"] is not None and new["landfall_h"] <= LANDFALL_TRIGGER_H
            and tier_kt(new) >= CYCLONE_KT)
    a, b = prev["landfall_h"], new["landfall_h"]
    if near and a is not None and abs(a - b) > TIMING_H:
        reasons.append((f"landfall time moved {abs(a - b):.0f} h "
                        f"(more than half our {2 * TIMING_H:.0f} h typical error)", False))
    if near and prev["landfall_lat"] is not None and new["landfall_lat"] is not None:
        km = _haversine_km(prev["landfall_lat"], prev["landfall_lon"],
                           new["landfall_lat"], new["landfall_lon"])
        if km > POSITION_KM:
            reasons.append((f"landfall point moved {km:.0f} km (more than half our "
                            f"{2 * POSITION_KM:.0f} km typical error)", False))

    # a place is news the first time the storm threatens it, once. the set is
    # cumulative for the whole storm, so a town that drops out of the zone and
    # comes back does not get announced twice - which is most of the churn.
    announced = set(prev.get("announced") or prev["near_places"])
    fresh = sorted(set(new["near_places"]) - announced)
    new["announced"] = sorted(announced | set(new["near_places"]))

    # ...but only while this is still a cyclone threat. a moving storm sweeps
    # new towns into its 36 h zone forever, so a per-place trigger never stops
    # firing - the 33 kt system in the suite alerted its way across the
    # gangetic plain one city at a time, Allahabad then Kanpur then Agra. once
    # the centre is inland and below cyclone strength the thing people need is
    # a rainfall and flood warning, which is IMD's product and not ours, so
    # new towns ride the routine bulletin instead of interrupting. the set is
    # still updated, so nothing is announced twice if it re-intensifies.
    spent = new["over_land"] and tier_kt(new) < CYCLONE_KT
    if fresh and not spent:
        shown = ", ".join(fresh[:4]) + (" and others" if len(fresh) > 4 else "")
        reasons.append((f"within {NEAR_H:.0f} h of {shown}", False))

    gap_h = None if prev_sent is None else (issued - prev_sent).total_seconds() / 3600
    floor = (MIN_GAP_CYCLONE_H if tier_kt(new) >= CYCLONE_KT else MIN_GAP_WEAK_H)

    if reasons:
        text = "; ".join(r for r, _ in reasons)
        if any(urgent for _, urgent in reasons) or gap_h is None or gap_h >= floor:
            return "Update", text
        # real, but it can ride the next bulletin rather than interrupt
        return None, f"held for the {floor:.0f} h minimum gap: {text}"

    # nothing to report: fall back to the routine bulletin cadence, which
    # tightens as the storm gets closer
    due = cadence_h(new)
    if gap_h is not None and gap_h >= due:
        return "Update", f"routine {due:.0f}-hourly bulletin"
    return None, "no material change since the last alert"


def load_index() -> list[dict]:
    if not INDEX.exists():
        return []
    return json.loads(INDEX.read_text(encoding="utf-8"))


def latest_for(sid: str, entries: list[dict] | None = None) -> dict | None:
    entries = load_index() if entries is None else entries
    mine = [e for e in entries if e["sid"] == sid]
    return max(mine, key=lambda e: e["sent"]) if mine else None


def publish(sid: str, identifier: str, sent, msg_type: str, reason: str,
            state: dict, headline: str, severity: str, urgency: str,
            certainty: str, references: str | None, cap_xml: str) -> dict:
    # one alert becomes permanent here: the CAP document at a URL that will
    # not change, and a line in the index the feed is built from.
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / f"{identifier}.xml").write_text(cap_xml, encoding="utf-8")
    sent = pd.Timestamp(sent)
    entry = {
        "identifier": identifier, "sid": sid,
        "sent": sent.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "expires": (sent + timedelta(hours=expires_h(state))
                    ).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "msgType": msg_type, "references": references, "reason": reason,
        "headline": headline, "severity": severity, "urgency": urgency,
        "certainty": certainty, "state": state,
    }
    entries = [e for e in load_index() if e["identifier"] != identifier]
    entries.append(entry)
    entries.sort(key=lambda e: e["sent"])
    INDEX.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    return entry


def reference_string(entry: dict) -> str:
    # CAP 1.2: "sender,identifier,sent", space separated for several
    return f"{SENDER},{entry['identifier']},{entry['sent']}"


def cap_path(identifier: str) -> Path:
    return STORE / f"{identifier}.xml"
