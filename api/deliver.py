# pushing a published alert to the systems that subscribed to it
#
# the Atom feed is a pull: a consumer polls it when it feels like it. that is
# right for a national aggregator, which polls hundreds of sources on its own
# schedule. it is wrong for a district control room that wants the alert the
# second it exists. so alerts are also pushed, as an HTTP POST of the CAP
# document to whoever asked for one.
#
# what this is NOT is a way to reach the public. in India cyclone warnings to
# the public are IMD's to issue and NDMA's channels to carry - cell broadcast,
# location-based SMS, the SACHET app. bulk SMS here needs TRAI DLT
# registration, cell broadcast needs telco access, and a prototype should have
# neither. so the last mile here stops at systems, not phones. see
# limitations.md.
#
# subscribers are configured in a file on disk and deliberately not through an
# API: an endpoint that lets a caller name a URL we will then POST to is a
# server-side request forgery waiting to happen.

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSCRIBERS = ROOT / "data" / "alerts" / "subscribers.json"
TIMEOUT_S = 5

SEVERITY_ORDER = ["Unknown", "Minor", "Moderate", "Severe", "Extreme"]

EXAMPLE = [
    {"name": "example district EOC", "url": "http://127.0.0.1:9099/cap",
     "min_severity": "Severe", "sid": None, "enabled": False},
]


def subscribers() -> list[dict]:
    if not SUBSCRIBERS.exists():
        SUBSCRIBERS.parent.mkdir(parents=True, exist_ok=True)
        SUBSCRIBERS.write_text(json.dumps(EXAMPLE, indent=1), encoding="utf-8")
        return []
    subs = json.loads(SUBSCRIBERS.read_text(encoding="utf-8"))
    return [s for s in subs if s.get("enabled")]


def _wanted(sub: dict, entry: dict) -> bool:
    if sub.get("sid") and sub["sid"] != entry["sid"]:
        return False
    floor = sub.get("min_severity")
    if floor and floor in SEVERITY_ORDER:
        have = entry.get("severity", "Unknown")
        if SEVERITY_ORDER.index(have) < SEVERITY_ORDER.index(floor):
            return False
    return True


def fan_out(entry: dict, cap_xml: str) -> list[dict]:
    # deliver to every matching subscriber and report what happened to each.
    # a subscriber that is down must never stop the alert being published -
    # publishing is the record, delivery is best effort.
    import requests

    results = []
    for sub in subscribers():
        if not _wanted(sub, entry):
            results.append({"name": sub["name"], "status": "skipped",
                            "detail": "filtered out by this subscriber's rules"})
            continue
        try:
            r = requests.post(
                sub["url"], data=cap_xml.encode("utf-8"), timeout=TIMEOUT_S,
                headers={"Content-Type": "application/cap+xml",
                         "X-Chakravat-Identifier": entry["identifier"],
                         "X-Chakravat-MsgType": entry["msgType"],
                         "X-Chakravat-Status": "Exercise"})
            results.append({"name": sub["name"], "status": "delivered"
                            if r.ok else "rejected", "detail": f"HTTP {r.status_code}"})
        except Exception as exc:                       # noqa: BLE001 - report, never raise
            results.append({"name": sub["name"], "status": "failed",
                            "detail": f"{type(exc).__name__}: {exc}"[:140]})
    return results
