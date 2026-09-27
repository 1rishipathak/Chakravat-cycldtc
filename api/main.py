# chakravat decision-support API

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_dataset                        # noqa: E402
from features.build import HORIZONS                     # noqa: E402
from ingest.ibtracs import imd_category                 # noqa: E402
from models.landfall import add_landfall_targets        # noqa: E402

ART = ROOT / "artifacts"
WEB = ROOT / "web"

app = FastAPI(title="Chakravat", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class Store:
    # loads data and models once, at startup

    def __init__(self) -> None:
        self.data = add_landfall_targets(load_dataset(verbose=False))
        bundle = joblib.load(ART / "prediction_models.joblib")
        self.ri = bundle["ri"]
        self.ri_threshold = bundle["ri_threshold"]
        self.ensemble = joblib.load(ART / "ensemble_cone.joblib")
        self.landfall = joblib.load(ART / "landfall_model.joblib")
        from features.build import model_features
        self.features = model_features(self.data)
        self.medians = self.data[self.features].median(numeric_only=True)

    def storm(self, sid: str) -> pd.DataFrame:
        s = self.data[self.data["SID"] == sid].sort_values("ISO_TIME")
        if s.empty:
            raise HTTPException(404, f"unknown storm {sid}")
        return s


store: Store | None = None
_CHAIN = None


@app.on_event("startup")
def _startup() -> None:
    global store
    store = Store()


@app.get("/api/storms")
def list_storms(season: int | None = None, min_peak_kt: float = 0.0):
    df = store.data
    if season is not None:
        df = df[df["SEASON"] == season]
    rows = (df.groupby("SID")
              .agg(name=("NAME", "first"), season=("SEASON", "first"),
                   peak_kt=("vmax_kt", "max"), fixes=("vmax_kt", "size"),
                   start=("ISO_TIME", "min"), end=("ISO_TIME", "max"),
                   sub_basin=("sub_basin", "first"))
              .reset_index())
    rows = rows[rows["peak_kt"] >= min_peak_kt]
    rows["peak_category"] = rows["peak_kt"].map(imd_category)
    rows = rows.sort_values(["season", "peak_kt"], ascending=[False, False])
    return {"count": len(rows), "storms": [
        {**r, "start": str(r["start"]), "end": str(r["end"])}
        for r in rows.to_dict("records")]}


@app.get("/api/seasons")
def list_seasons():
    counts = store.data.groupby("SEASON")["SID"].nunique()
    return {"seasons": [{"season": int(s), "storms": int(n)}
                        for s, n in counts.items()][::-1]}


@app.get("/api/storm/{sid}")
def storm_detail(sid: str):
    s = store.storm(sid)
    return {
        "sid": sid,
        "name": str(s["NAME"].iloc[0]),
        "season": int(s["SEASON"].iloc[0]),
        "peak_kt": float(s["vmax_kt"].max()),
        "peak_category": imd_category(float(s["vmax_kt"].max())),
        "track": [
            {"time": str(r.ISO_TIME), "lat": float(r.LAT), "lon": float(r.LON),
             "vmax_kt": float(r.vmax_kt), "category": imd_category(float(r.vmax_kt)),
             "pressure": None if pd.isna(r.pres_hpa) else float(r.pres_hpa),
             "dist2land_km": None if pd.isna(r.DIST2LAND) else float(r.DIST2LAND)}
            for r in s.itertuples()
        ],
    }


def _row_at(sid: str, time: str | None) -> pd.DataFrame:
    s = store.storm(sid)
    if not time:        # absent or empty ?time= both mean "latest usable fix"
        # default to the last fix from which a 24 h forecast is verifiable.
        usable = s[s["y_dlat_24"].notna()]
        row = (usable if len(usable) else s).iloc[[-1]]
    else:
        match = s[s["ISO_TIME"] == pd.Timestamp(time)]
        if match.empty:
            raise HTTPException(404, f"no fix at {time}")
        row = match.iloc[[0]]
    return row


@app.get("/api/storm/{sid}/forecast")
def forecast(sid: str, time: str | None = None, level: float = 0.67):
    row = _row_at(sid, time)
    base = row.iloc[0]

    points = []
    for h in HORIZONS:
        cone = store.ensemble.track_cone(row, h, level)
        band = store.ensemble.intensity_band(row, h, level)
        vmax = float(band["vmax"][0])
        points.append({
            "horizon_h": h,
            "lat": float(cone["lat"][0]), "lon": float(cone["lon"][0]),
            "radius_km": float(cone["radius_km"][0]),
            "vmax_kt": vmax,
            "vmax_lower": float(band["lower"][0]),
            "vmax_upper": float(band["upper"][0]),
            "category": imd_category(vmax),
        })

    ri_prob = float(store.ri.predict_proba(row)[0])
    return {
        "sid": sid, "name": str(base["NAME"]),
        "issued_at": str(base["ISO_TIME"]),
        "current": {"lat": float(base["LAT"]), "lon": float(base["LON"]),
                    "vmax_kt": float(base["vmax_kt"]),
                    "category": imd_category(float(base["vmax_kt"]))},
        "cone_level": level,
        "forecast": points,
        "rapid_intensification": {
            "probability": ri_prob,
            "threshold": float(store.ri_threshold),
            "flagged": bool(ri_prob >= store.ri_threshold),
            "definition": "+30 kt within 24 h",
        },
    }


@app.get("/api/storm/{sid}/landfall")
def landfall(sid: str, time: str | None = None):
    row = _row_at(sid, time)
    prob = float(store.landfall.predict_will_land(row)[0])
    # the model answers "will it make landfall later", which is false once the
    # centre is already inland - Amphan at 20 May 12 UTC read as 4%. IBTrACS
    # puts distance to land at 0 over land, so say that instead.
    d2l = row.iloc[0].get("DIST2LAND")
    over_land = bool(d2l is not None and np.isfinite(d2l) and d2l <= 0)
    out = {"sid": sid, "issued_at": str(row.iloc[0]["ISO_TIME"]),
           "probability_within_72h": prob, "over_land": over_land}
    if prob >= 0.5:
        # position comes from where the forecast track crosses the coast when
        # it does, and from the snapped regression when it does not. Timing
        # stays with the regressor: the crossing implies a time too, but it
        # scored 11.5 h against the model's 8.1 h on the same fixes, so the
        # better number is kept rather than the more elegant one.
        base = row.iloc[0]
        track = [(float(base["LAT"]), float(base["LON"]))]
        for h in HORIZONS:
            c = store.ensemble.track_cone(row, h)
            track.append((float(c["lat"][0]), float(c["lon"][0])))
        pos = store.landfall.predict_position(row, track, list(HORIZONS))

        out.update({
            "hours_to_landfall": float(store.landfall.predict(row, "lf_hours")[0]),
            "lat": pos["lat"], "lon": pos["lon"],
            "position_method": pos["method"],
            "vmax_kt": float(store.landfall.predict(row, "lf_vmax")[0]),
        })
        out["category_at_landfall"] = imd_category(out["vmax_kt"])
        # skill measured on held-out seasons; surfaced so the number is never
        # read as more precise than it is. Position error is quoted for the
        # method actually used on this forecast.
        out["typical_error"] = {
            "timing_h": 9.0,
            "position_km": 160.0 if pos["method"] == "track_crossing" else 234.0,
            "intensity_kt": 11.8,
        }
    return out


def _in_cone(fc: dict) -> tuple[list[dict], object]:
    # places the swept forecast cone covers, and the cone itself
    from models.impact import cone_polygon, places_in_cone, track_nodes
    nodes = track_nodes(fc["current"], fc["forecast"])
    return places_in_cone(nodes), cone_polygon(nodes)


@app.get("/api/storm/{sid}/exposure")
def exposure(sid: str, time: str | None = None, level: float = 0.67):
    # populated places inside the forecast cone, and how soon
    fc = forecast(sid, time, level)
    places, cone = _in_cone(fc)
    return {
        "sid": sid, "issued_at": fc["issued_at"], "cone_level": level,
        "places": places,
        "cone_polygon": [[round(lon, 3), round(lat, 3)] for lon, lat in cone.exterior.coords],
        "note": ("Places of 100,000 and up (Natural Earth metro estimates) inside the "
                 "forecast cone widened by the typical gale-force wind radius for the "
                 "forecast intensity. Says who is in the path, not how badly they "
                 "will be affected."),
    }


@app.get("/api/storm/{sid}/cap.xml")
def cap_xml(sid: str, time: str | None = None, level: float = 0.67):
    # CAP 1.2 alert, status Exercise - the format NDMA's SACHET system carries
    from api import alerts
    fc = forecast(sid, time, level)
    lf = landfall(sid, time)
    places, cone = _in_cone(fc)
    return Response(alerts.cap_alert(sid, fc, lf, places, cone),
                    media_type="application/xml")


def _evaluate(sid: str, fc: dict, lf: dict, origin: str) -> dict:
    # one decision-and-publish path, shared by the historical storms and the
    # live feed. the two differ only in where the forecast came from, and
    # having them share this is the reason a live alert is a real CAP alert
    # with a real lifecycle rather than a second, weaker kind of thing.
    from api import alert_store, alerts, deliver, events

    places, cone = _in_cone(fc)
    issued = pd.Timestamp(fc["issued_at"])
    identifier = alerts.identifier_for(sid, issued)

    entries = alert_store.load_index()
    already = next((e for e in entries if e["identifier"] == identifier), None)
    if already is not None:
        # same storm, same issue time: the alert exists and must not be minted
        # a second time under a different identifier
        return {"issued": False, "reason": "this alert has already been published",
                "sid": sid, "alert": already}

    prev_entry = alert_store.latest_for(sid, entries)
    prev_sent = (None if prev_entry is None else
                 pd.Timestamp(prev_entry["sent"]).tz_convert("UTC").tz_localize(None))
    state = alert_store.material_state(fc, lf, places)
    state["origin"] = origin
    msg_type, reason = alert_store.decide(
        prev_entry["state"] if prev_entry else None, state, issued, prev_sent)

    if msg_type is None:
        return {"issued": False, "reason": reason, "sid": sid,
                "issued_at": fc["issued_at"],
                "last_alert": prev_entry["identifier"] if prev_entry else None}

    refs = alert_store.reference_string(prev_entry) if prev_entry else None
    xml = alerts.cap_alert(sid, fc, lf, places, cone, msg_type=msg_type, references=refs,
                           expires_h=alert_store.expires_h(state))
    head = alerts.summary(fc, lf)
    entry = alert_store.publish(sid, identifier, issued, msg_type, reason, state,
                                head["headline"], head["severity"], head["urgency"],
                                head["certainty"], refs, xml)
    # publishing is the record; delivery is best effort and never blocks it
    delivered = deliver.fan_out(entry, xml)
    events.publish("alert", {"alert": entry, "origin": origin,
                             "cap_url": f"/api/alerts/{identifier}.xml",
                             "delivered": delivered,
                             "places": [p["name"] for p in places[:8]]})
    return {"issued": True, "reason": reason, "alert": entry,
            "cap_url": f"/api/alerts/{identifier}.xml", "delivered": delivered}


@app.post("/api/alerts/evaluate")
def evaluate_alert(sid: str, time: str | None = None, level: float = 0.67):
    # decide whether this forecast is worth telling anyone about, and publish it
    # if it is. cheap to call on a timer, and silent unless something changed.
    return _evaluate(sid, forecast(sid, time, level), landfall(sid, time), "best_track")


@app.get("/api/alerts")
def alert_index(sid: str | None = None):
    from api import alert_store
    entries = alert_store.load_index()
    if sid:
        entries = [e for e in entries if e["sid"] == sid]
    return {"count": len(entries), "feed": "/api/alerts/feed.atom",
            "alerts": sorted(entries, key=lambda e: e["sent"], reverse=True)}


@app.get("/api/alerts/feed.atom")
def alert_feed(request: Request, sid: str | None = None):
    # the one URL a consumer subscribes to
    from api import alert_store, feed
    entries = alert_store.load_index()
    if sid:
        entries = [e for e in entries if e["sid"] == sid]
    return Response(feed.atom(entries, str(request.base_url)),
                    media_type="application/atom+xml")


def _chain_and_alert(scan: str, source: str, hours: int, level: float,
                     threshold: float | None, origin: str,
                     announce: dict | None = None) -> dict:
    # imagery in, CAP out. this is the whole live loop, and the one path where a
    # detection is the only evidence there is - no best track, no agreed
    # position, no name, so the system has to invent an identity and then keep
    # recognising it three hours later.
    from api import alert_store, events, live_alerts

    chain = vision_pipeline(time=scan, hours=hours, threshold=threshold,
                            level=level, source=source)
    if announce is not None:
        events.publish("fetch", {**announce,
                                 "scenes_examined": chain.get("scenes_examined", 0),
                                 "tracks_found": chain.get("tracks_found", 0)})
    if not chain.get("available"):
        return {"available": False, "reason": chain.get("reason"), "latest_scan": scan}

    results = []
    for storm in chain["storms"]:
        pair = live_alerts.adapt(storm, level)
        if pair is None:
            results.append({"detection": storm["id"], "issued": False,
                            "reason": "only one fix so far, too short to forecast from"})
            continue
        fc, lf = pair
        entries = alert_store.load_index()
        when = pd.Timestamp(fc["issued_at"])
        matched = live_alerts.match_sid(entries, storm["lat"], storm["lon"], when)
        sid = matched or live_alerts.new_sid(entries, when)
        out = _evaluate(sid, fc, lf, origin)
        results.append({"detection": storm["id"], "sid": sid,
                        "sid_was_matched": bool(matched),
                        "lat": storm["lat"], "lon": storm["lon"],
                        "vmax_kt": storm.get("vmax_kt"), **out})

    return {"available": True, "latest_scan": scan,
            "scenes_examined": chain["scenes_examined"],
            "tracks_found": chain["tracks_found"], "results": results}


@app.post("/api/live/evaluate")
def evaluate_live(hours: int = 48, level: float = 0.67,
                  threshold: float | None = None):
    # what a fetch triggers: run the chain on the newest live INSAT scan and
    # alert on whatever it finds, pushed to every open page.
    from api import events

    status = live_status()
    if not status["available"]:
        events.publish("fetch", {"available": False,
                                 "detail": "no live scenes on disk yet"})
        return {"available": False, "reason": "no live scenes yet - "
                "run python src/ingest/insat_live.py"}

    return _chain_and_alert(
        status["latest"], "live", hours, level, threshold, "live_insat",
        announce={"available": True, "latest_scan": status["latest"],
                  "age_hours": status["age_hours"],
                  "fetched_at": status["fetched_at"],
                  "satellite": (status["scenes"][-1].get("satellite")
                                if status["scenes"] else None)})


@app.post("/api/live/rehearse")
def rehearse_live(time: str, hours: int = 48, level: float = 0.67,
                  threshold: float | None = None, source: str = "insat"):
    # the same loop pointed at an archive scan instead of today's.
    #
    # the live path cannot be tested by waiting for a cyclone, and the part of
    # it most likely to be wrong is the part no archive test touches: a storm
    # detected at one cycle has to be recognised as the same storm at the next,
    # from position alone. src/rehearse_live.py drives this cycle by cycle and
    # counts how many identities one real storm was given.
    return _chain_and_alert(time, _source(source), hours, level, threshold,
                            "rehearsal")


@app.get("/api/stream")
def stream():
    # server-sent events: an HTTP response that never ends. every page that
    # opens it is told about each fetch and each alert as it happens.
    import asyncio

    from fastapi.responses import StreamingResponse

    from api import events

    async def pump():
        q = events.subscribe()
        try:
            yield events.sse({"kind": "hello", "seq": 0,
                              "at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "watching": events.subscriber_count()})
            for past in events.history():
                yield events.sse(past)
            idle = 0.0
            while True:
                try:
                    yield events.sse(q.get_nowait())
                    idle = 0.0
                except Exception:       # noqa: BLE001 - empty queue
                    await asyncio.sleep(0.25)
                    idle += 0.25
                    # proxies and load balancers close a silent connection, so
                    # say something harmless every 20 s to keep it open
                    if idle >= 20:
                        idle = 0.0
                        yield events.sse({"kind": "heartbeat", "seq": 0,
                                          "at": pd.Timestamp.utcnow()
                                          .strftime("%Y-%m-%dT%H:%M:%SZ")})
        finally:
            events.unsubscribe(q)

    return StreamingResponse(pump(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive",
        "X-Accel-Buffering": "no"})       # nginx buffers SSE to death without this


@app.get("/api/alerts/{identifier}.xml")
def alert_cap(identifier: str):
    # a published alert stays exactly as it was sent, forever
    from api import alert_store
    path = alert_store.cap_path(identifier)
    if not path.exists():
        raise HTTPException(404, f"no alert {identifier}")
    return Response(path.read_text(encoding="utf-8"), media_type="application/cap+xml")


@app.get("/api/storm/{sid}/explain_forecast")
def explain_forecast(sid: str, time: str | None = None, horizon: int = 24,
                     level: float = 0.67, top: int = 6):
    # the forecast's equivalent of a Grad-CAM: which of the 33 inputs the
    # answer rested on, measured by removing them one at a time
    from api import explain_forecast as ef
    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {list(HORIZONS)}")
    out = ef.explain(store.ensemble, _row_at(sid, time), store.medians,
                     horizon, level, top)
    out["sentence"] = ef.sentence(out)
    return out


@app.get("/api/storm/{sid}/sms")
def sms_alert(sid: str, time: str | None = None, lang: str = "en",
              segments: int | None = None):
    # the same forecast state, cut to what a phone actually receives. CAP goes
    # to SACHET and SACHET sends SMS, where the budget is 160 GSM-7 characters
    # or 70 if the text is Devanagari, so the two languages are written
    # separately rather than translated.
    from api import alerts
    if lang not in ("en", "hi"):
        raise HTTPException(400, "lang must be en or hi")
    return alerts.sms(forecast(sid, time), landfall(sid, time), lang, segments)


@app.get("/api/storm/{sid}/bulletin", response_class=PlainTextResponse)
def bulletin(sid: str, time: str | None = None, lang: str = "en"):
    # advisory text in the shape IMD publishes, generated from model state
    fc = forecast(sid, time)
    lf = landfall(sid, time)
    if lang == "hi":
        from api import alerts
        return alerts.bulletin_hindi(sid, fc, lf, _in_cone(fc)[0])
    from api.alerts import ist as alerts_ist
    cur, name = fc["current"], fc["name"] or "UNNAMED"

    lines = [
        "=" * 66,
        f"CHAKRAVAT DECISION SUPPORT - {name} ({sid})",
        f"ISSUED {alerts_ist(fc['issued_at'])}",
        "=" * 66,
        "",
        f"PRESENT POSITION : {cur['lat']:.1f} N  {cur['lon']:.1f} E",
        f"PRESENT INTENSITY: {cur['vmax_kt']:.0f} kt (3-min sustained)",
        f"CLASSIFICATION   : {cur['category'].upper()}",
        "",
        "FORECAST TRACK AND INTENSITY",
        f"  {'LEAD':>6}  {'POSITION':<20} {'INTENSITY':<22} {'CONE'}",
    ]
    for p in fc["forecast"]:
        pos = f"{p['lat']:.1f}N {p['lon']:.1f}E"
        inten = f"{p['vmax_kt']:.0f} kt ({p['vmax_lower']:.0f}-{p['vmax_upper']:.0f})"
        lines.append(f"  {p['horizon_h']:>4} h  {pos:<20} {inten:<22} "
                     f"+/-{p['radius_km']:.0f} km")

    ri = fc["rapid_intensification"]
    lines += ["", "RAPID INTENSIFICATION",
              f"  Probability of {ri['definition']}: {100*ri['probability']:.0f}%"
              f"{'   *** WATCH ***' if ri['flagged'] else ''}"]

    lines += ["", "LANDFALL"]
    if lf.get("over_land"):
        lines.append("  Centre is over land - landfall has already occurred.")
    elif lf.get("hours_to_landfall") is not None:
        lines += [
            f"  Probability within 72 h : {100*lf['probability_within_72h']:.0f}%",
            f"  Expected in             : {lf['hours_to_landfall']:.0f} h "
            f"(+/- {lf['typical_error']['timing_h']:.0f} h)",
            f"  Expected near           : {lf['lat']:.1f} N {lf['lon']:.1f} E "
            f"(+/- {lf['typical_error']['position_km']:.0f} km)",
            f"  Intensity at coast      : {lf['vmax_kt']:.0f} kt "
            f"({lf['category_at_landfall'].upper()})",
        ]
    else:
        lines.append(f"  No landfall indicated within 72 h "
                     f"(probability {100*lf['probability_within_72h']:.0f}%)")

    lines += [
        "",
        "-" * 66,
        "Cone radii and intensity ranges are calibrated to the stated",
        f"confidence level ({100*fc['cone_level']:.0f}%), verified on held-out",
        "seasons 2020-2025.",
        "",
        "NOT AN IMD PRODUCT. Decision support only; official warnings are",
        "issued by RSMC New Delhi.",
        "=" * 66,
    ]
    return "\n".join(lines)


# ---- imagery tasks (T1 detection, T2 scene, T3 intensity) ---------------

def _source(source: str) -> str:
    # which imagery archive a vision call reads: gridsat, insat or live
    from api import vision
    if source not in vision.SOURCES:
        raise HTTPException(400, f"source must be one of {sorted(vision.SOURCES)}")
    return source


@app.get("/api/vision/status")
def vision_status():
    from api import vision
    return vision.MODELS.status()


@app.get("/api/vision/detect")
def vision_detect(time: str, threshold: float | None = None, source: str = "gridsat",
                  tier: str = "warn"):
    # find every cyclone in a whole basin scene - no storm id needed.
    #
    # tier picks how eager the detector is. "warn" is the served point and what
    # every reported number is scored at; "watch" is more sensitive, finds more
    # weak systems and produces about twice the false alarms. an explicit
    # threshold overrides both.
    from api import vision
    if tier not in ("warn", "watch"):
        raise HTTPException(400, "tier must be warn or watch")
    if threshold is None:
        threshold = vision.tier_threshold(tier, _source(source))
    out = vision.detect(pd.Timestamp(time), threshold, _source(source))
    out["tier"] = tier
    points = vision.operating_points()
    if tier in points:
        out["tier_measured"] = points[tier]
    return out


@app.get("/api/storm/{sid}/scene")
def storm_scene(sid: str, time: str | None = None, source: str = "gridsat"):
    # source picks the archive the patch is cut from. it matters: the answer
    # carries the measured reliability of that class on that sensor, and two of
    # the five classes read much worse on INSAT.
    from api import vision
    row = _row_at(sid, time).iloc[0]
    out = vision.classify_scene(pd.Timestamp(row["ISO_TIME"]),
                                float(row["LAT"]), float(row["LON"]), _source(source))
    out["sid"] = sid
    out["source"] = source
    return out


@app.get("/api/storm/{sid}/intensity_from_image")
def storm_intensity_image(sid: str, time: str | None = None, source: str = "gridsat",
                          level: float = 0.67):
    # T3 estimate, alongside best track so the two can be compared
    from api import vision
    row = _row_at(sid, time).iloc[0]
    out = vision.estimate_intensity(pd.Timestamp(row["ISO_TIME"]),
                                    float(row["LAT"]), float(row["LON"]),
                                    _source(source), level)
    out["sid"] = sid
    out["source"] = source
    out["best_track_kt"] = float(row["vmax_kt"])
    if out.get("available"):
        out["error_kt"] = out["vmax_kt"] - out["best_track_kt"]
    return out


@app.get("/api/storm/{sid}/explain.png")
def storm_explain_png(sid: str, task: str = "scene", time: str | None = None):
    # Grad-CAM: where T2 (its CNN half) or T3 looked on this storm's patch
    from api import vision
    if task not in ("scene", "intensity"):
        raise HTTPException(400, "task must be scene or intensity")
    row = _row_at(sid, time).iloc[0]
    got = vision.explain(pd.Timestamp(row["ISO_TIME"]), float(row["LAT"]),
                         float(row["LON"]), task)
    if got is None:
        raise HTTPException(404, "no usable patch or model for this fix")
    png, meta = got
    return Response(content=png, media_type="image/png",
                    headers={"X-Share-In-Core": f"{meta['share_in_core']:.3f}",
                             "X-Core-Area-Share": f"{meta['core_area_share']:.3f}",
                             "Access-Control-Expose-Headers": "X-Share-In-Core, X-Core-Area-Share"})


@app.get("/api/vision/scene.png")
def vision_scene_png(time: str, enhance: bool = True, source: str = "gridsat"):
    # the satellite scene itself, as a georeferenced PNG for map overlay
    from api import vision
    got = vision.scene_image(pd.Timestamp(time), enhance, _source(source))
    if got is None:
        raise HTTPException(404, f"no {source} scene near {time}")
    png, meta = got
    return Response(content=png, media_type="image/png",
                    headers={"X-Scene-Time": meta["scene_time"],
                             "X-Bounds": str(meta["bounds"]),
                             "Cache-Control": "public, max-age=86400"})


@app.get("/api/vision/scene_meta")
def vision_scene_meta(time: str, source: str = "gridsat"):
    from api import vision
    got = vision.scene_bounds(pd.Timestamp(time), _source(source))
    if got is None:
        raise HTTPException(404, f"no {source} scene near {time}")
    return got


@app.get("/api/vision/pipeline")
def vision_pipeline(time: str, hours: int = 48, threshold: float | None = None,
                    level: float = 0.67, source: str = "gridsat"):
    # the whole chain: imagery in, detected storms and forecasts out
    global _CHAIN
    if _CHAIN is None:
        sys.path.insert(0, str(ROOT))
        from pipeline import Chakravat
        _CHAIN = Chakravat(verbose=False)
    return _CHAIN.run(pd.Timestamp(time), hours=hours,
                      threshold=threshold, level=level, source=_source(source))


@app.get("/api/live/status")
def live_status():
    # what the live INSAT feed holds; filled by src/ingest/insat_live.py
    import json
    from api import vision
    path = vision.SOURCES["live"] / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    latest = status.get("latest")
    age = None
    if latest:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        age = (now - pd.Timestamp(latest)).total_seconds() / 3600.0
    return {"available": bool(latest), "latest": latest, "age_hours": age,
            "fetched_at": status.get("fetched_at"),
            "scenes": [{"time": k, **v} for k, v in sorted(status.get("scenes", {}).items())],
            "refresh": "python src/ingest/insat_live.py"}


@app.get("/api/skill")
def skill():
    # verification numbers, so the dashboard can show its own track record
    import json
    import math

    def clean(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    out = {}
    for key, name in (("final", "final_results.json"),
                      ("cone", "cone_calibration.json"),
                      ("landfall", "landfall_results.json"),
                      ("ablation", "era5_ablation.json"),
                      ("detection", "t1_detection.json"),
                      ("detection_by_sensor", "t1_detection_by_sensor.json"),
                      ("scene", "t2_scene.json"),
                      ("intensity", "t3_intensity_gridsat.json"),
                      ("intensity_vision", "t3_intensity_vision.json"),
                      ("adt", "adt_vs_best_track.json"),
                      # what the API actually serves, with intervals, and the
                      # cross-validated scores behind the served configurations
                      ("forecast_ci", "forecast_skill_ci.json"),
                      ("cv_scene", "cv/t2_hybrid_reflect.json"),
                      ("cv_intensity", "cv/t3_best_track_rotate_joint.json"),
                      # the two detector operating points, and the sensor gap
                      # measured on scenes carrying both instruments
                      ("detection_tiers", "t1_operating_points.json"),
                      ("scene_by_sensor", "t2_sensor_paired.json")):
        path = ROOT / "reports" / name
        if path.exists():
            try:
                out[key] = clean(json.loads(path.read_text()))
            except (ValueError, OSError):
                # One unreadable report must not cost the whole tab.
                out[key] = None
    return out


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))

    @app.get("/live")
    def live_page():
        # the public alert wall, deliberately its own page: the dashboard is
        # for exploring and this is for watching
        return FileResponse(str(WEB / "live.html"))
