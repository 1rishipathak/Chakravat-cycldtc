# hit every endpoint against a running server and check the shape of what comes back
#
#     python src/smoke_api.py --base http://127.0.0.1:8000
#
# not a unit test suite: it is the check to run before a demo or a submission,
# when the question is "does the whole surface still answer, and does it answer
# with the fields the dashboard and the frontend contract expect". the pipeline
# and imagery calls need the GPU and the satellite archive; skip them with
# --no-imagery on a machine that has neither.

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

AMPHAN = "2020136N10088"
MANDOUS = "2022338N05100"
BEFORE_LANDFALL = "2020-05-19 00:00:00"

PASS, FAIL = "ok  ", "FAIL"


def get(base: str, path: str, text: bool = False):
    t0 = time.time()
    with urllib.request.urlopen(base + path, timeout=900) as r:
        raw = r.read()
        ctype = r.headers.get("Content-Type", "")
        headers = dict(r.headers)
    if "json" in ctype:
        body = json.loads(raw.decode("utf-8"))
    elif text and ctype.startswith(("text/", "application/xml")):
        body = raw.decode("utf-8")
    else:
        body = raw                      # an image stays bytes
    return body, time.time() - t0, headers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-imagery", action="store_true")
    args = ap.parse_args()
    q = urllib.parse.quote

    # (label, path, text?, check) - check raises or returns a note
    checks = [
        ("seasons", "/api/seasons", False,
         lambda b: f"{len(b['seasons'])} seasons" if b["seasons"] else _fail("empty")),
        ("storms", "/api/storms?season=2020", False,
         lambda b: f"{b['count']} storms"),
        ("storm detail", f"/api/storm/{AMPHAN}", False,
         lambda b: f"{b['name']}, {len(b['track'])} fixes"),
        ("forecast", f"/api/storm/{AMPHAN}/forecast?level=0.67", False,
         lambda b: _forecast(b)),
        ("forecast at a chosen time", f"/api/storm/{AMPHAN}/forecast?time={q(BEFORE_LANDFALL)}", False,
         lambda b: b["issued_at"] if b["issued_at"] == BEFORE_LANDFALL else _fail(b["issued_at"])),
        ("landfall, lands", f"/api/storm/{MANDOUS}/landfall", False,
         lambda b: f"{b['hours_to_landfall']:.0f} h, {b['position_method']}"
         if b.get("hours_to_landfall") is not None else _fail("no landfall branch")),
        ("landfall, over land", f"/api/storm/{AMPHAN}/landfall", False,
         lambda b: "over_land flagged" if b.get("over_land") else _fail("expected over_land")),
        ("exposure", f"/api/storm/{AMPHAN}/exposure?time={q(BEFORE_LANDFALL)}", False,
         lambda b: f"{len(b['places'])} places, first {b['places'][0]['name']}"
         if b["places"] else _fail("no places")),
        ("CAP alert", f"/api/storm/{AMPHAN}/cap.xml?time={q(BEFORE_LANDFALL)}", True,
         lambda b: "Exercise, 2 languages" if "<status>Exercise</status>" in b
         and b.count("<info>") == 2 else _fail("status or info blocks wrong")),
        ("alert index", "/api/alerts", False,
         lambda b: f"{b['count']} published, feed at {b['feed']}"),
        ("alert feed (atom)", "/api/alerts/feed.atom", True, lambda b: _feed(b)),
        ("alert wall page", "/live", True,
         lambda b: f"{len(b)//1024} KB" if "EXERCISE ONLY" in b
         else _fail("the not-an-IMD-warning banner is missing")),
        ("bulletin", f"/api/storm/{AMPHAN}/bulletin", True,
         lambda b: f"{len(b.splitlines())} lines" if "CHAKRAVAT" in b else _fail("no header")),
        ("bulletin, hindi", f"/api/storm/{AMPHAN}/bulletin?lang=hi", True,
         lambda b: "devanagari present" if "चक्रवात" in b
         else _fail("no devanagari")),
        ("sms alert", f"/api/storm/{AMPHAN}/sms", False,
         lambda b: f"{b['encoding']} {b['units']}/{b['budget']} units, {b['segments']} seg"
         if "not an IMD warning" in b["text"] else _fail("the disclaimer was trimmed away")),
        ("sms alert, hindi", f"/api/storm/{AMPHAN}/sms?lang=hi", False,
         lambda b: f"{b['encoding']} {b['units']} units, {b['segments']} seg"
         if "IMD" in b["text"] and "चक्रवात" in b["text"]
         else _fail("no devanagari or no disclaimer")),
        ("why this forecast", f"/api/storm/{AMPHAN}/explain_forecast?horizon=24", False,
         lambda b: f"{b['track'][0]['label']} moves it {b['track'][0]['track_shift_km']:.0f} km"
         if b["track"] else _fail("nothing ranked")),
        ("skill", "/api/skill", False, lambda b: _skill(b)),
        ("live status", "/api/live/status", False,
         lambda b: f"{len(b['scenes'])} scenes, latest {b['latest']}, {b['age_hours']:.1f} h old"
         if b["available"] else "no live scenes yet (run src/ingest/insat_live.py)"),
        ("vision status", "/api/vision/status", False,
         lambda b: f"{b['device']}, {b['gridsat_scenes']} scenes"),
    ]
    if not args.no_imagery:
        checks += [
            ("scene (T2)", f"/api/storm/{AMPHAN}/scene?time={q(BEFORE_LANDFALL)}", False,
             lambda b: f"{b['scene']} {b['confidence']:.2f} ({b.get('method', 'cnn')})"
             if b["available"] else _fail(b.get("reason"))),
            ("scene (T2) on INSAT", f"/api/storm/{AMPHAN}/scene?time={q(BEFORE_LANDFALL)}&source=insat", False,
             lambda b: (f"{b['scene']}, class F1 {b['reliability']['class_f1_here']} on INSAT"
                        if b.get("reliability") else _fail("no reliability reported"))
             if b["available"] else _fail(b.get("reason"))),
            ("intensity (T3)", f"/api/storm/{AMPHAN}/intensity_from_image?time={q(BEFORE_LANDFALL)}", False,
             lambda b: f"{b['vmax_kt']:.0f} kt vs best track {b['best_track_kt']:.0f}"
             if b["available"] else _fail(b.get("reason"))),
            ("intensity interval", f"/api/storm/{AMPHAN}/intensity_from_image?time={q(BEFORE_LANDFALL)}&level=0.9", False,
             lambda b: (f"{b['vmax_kt']:.0f} kt [{b['interval']['lower_kt']:.0f}-"
                        f"{b['interval']['upper_kt']:.0f}], coverage "
                        f"{b['interval']['measured_coverage']:.2f}")
             if b.get("interval") else _fail("no interval returned")),
            ("attention map", f"/api/storm/{AMPHAN}/explain.png?task=intensity&time={q(BEFORE_LANDFALL)}",
             True, lambda b: f"png, {len(b)/1024:.0f} KB" if b[1:4] == b"PNG"
             else _fail("not a png")),
            ("detect", f"/api/vision/detect?time={q(BEFORE_LANDFALL)}", False,
             lambda b: f"{len(b['detections'])} detections at threshold {b['threshold']:.2f}"
             if b["available"] else _fail(b.get("reason"))),
            ("detect, watch tier", f"/api/vision/detect?time={q(BEFORE_LANDFALL)}&tier=watch", False,
             lambda b: (f"thr {b['threshold']:.2f}, weak recall "
                        f"{b['tier_measured']['weak_recall']:.2f}, "
                        f"{b['tier_measured']['false_alarms_per_scene']:.2f} false/scene")
             if b.get("tier_measured") else _fail("the tier reported no measured cost")),
            ("pipeline", f"/api/vision/pipeline?time={q('2020-05-19 12:00')}", False,
             lambda b: f"{b['tracks_found']} tracks from {b['scenes_examined']} scenes"
             if b["available"] else _fail(b.get("reason"))),
        ]

    print(f"smoke test against {args.base}\n")
    bad = 0
    for label, path, text, check in checks:
        try:
            body, secs, _ = get(args.base, path, text)
            note = check(body)
            print(f"  {PASS} {label:<26} {secs:5.1f}s  {note}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"  {FAIL} {label:<26}        {type(exc).__name__}: {str(exc)[:90]}")
    print(f"\n{len(checks) - bad}/{len(checks)} endpoints healthy")
    raise SystemExit(1 if bad else 0)


def _fail(msg):
    raise AssertionError(msg)


def _forecast(b):
    horizons = [p["horizon_h"] for p in b["forecast"]]
    if horizons != [6, 12, 24, 48, 72]:
        _fail(f"horizons {horizons}")
    if not all(p["radius_km"] > 0 for p in b["forecast"]):
        _fail("a cone radius is not positive")
    return f"{b['name']}, {b['current']['vmax_kt']:.0f} kt, cone to {b['forecast'][-1]['radius_km']:.0f} km"


def _feed(b):
    # an empty store is a valid feed, so check the shape rather than a count
    import xml.etree.ElementTree as ET
    root = ET.fromstring(b)
    ns = "{http://www.w3.org/2005/Atom}"
    if not root.tag.endswith("feed"):
        _fail(f"root is {root.tag}, not an atom feed")
    entries = root.findall(f"{ns}entry")
    links = [e.find(f"{ns}link") for e in entries]
    if entries and not all(l is not None and l.get("type") == "application/cap+xml"
                           for l in links):
        _fail("an entry does not link to a CAP document")
    return f"{len(entries)} entries, each linking to CAP"


def _skill(b):
    missing = [k for k in ("final", "forecast_ci", "cone", "landfall", "detection",
                           "cv_scene", "cv_intensity") if not b.get(k)]
    if missing:
        _fail(f"missing keys {missing}")
    return f"{len(b)} report groups"


if __name__ == "__main__":
    main()
