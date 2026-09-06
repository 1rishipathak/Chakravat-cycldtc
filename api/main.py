# chakravat decision-support API

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
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
    if time is None:
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
    out = {"sid": sid, "issued_at": str(row.iloc[0]["ISO_TIME"]),
           "probability_within_72h": prob}
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


@app.get("/api/storm/{sid}/bulletin", response_class=PlainTextResponse)
def bulletin(sid: str, time: str | None = None):
    # advisory text in the shape IMD publishes, generated from model state
    fc = forecast(sid, time)
    lf = landfall(sid, time)
    cur, name = fc["current"], fc["name"] or "UNNAMED"

    lines = [
        "=" * 66,
        f"CHAKRAVAT DECISION SUPPORT - {name} ({sid})",
        f"ISSUED {fc['issued_at']} UTC",
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
    if lf.get("hours_to_landfall") is not None:
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

@app.get("/api/vision/status")
def vision_status():
    from api import vision
    return vision.MODELS.status()


@app.get("/api/vision/detect")
def vision_detect(time: str, threshold: float = 0.30):
    # find every cyclone in a whole basin scene - no storm id needed
    from api import vision
    return vision.detect(pd.Timestamp(time), threshold)


@app.get("/api/storm/{sid}/scene")
def storm_scene(sid: str, time: str | None = None):
    from api import vision
    row = _row_at(sid, time).iloc[0]
    out = vision.classify_scene(pd.Timestamp(row["ISO_TIME"]),
                                float(row["LAT"]), float(row["LON"]))
    out["sid"] = sid
    return out


@app.get("/api/storm/{sid}/intensity_from_image")
def storm_intensity_image(sid: str, time: str | None = None):
    # T3 estimate, alongside best track so the two can be compared
    from api import vision
    row = _row_at(sid, time).iloc[0]
    out = vision.estimate_intensity(pd.Timestamp(row["ISO_TIME"]),
                                    float(row["LAT"]), float(row["LON"]))
    out["sid"] = sid
    out["best_track_kt"] = float(row["vmax_kt"])
    if out.get("available"):
        out["error_kt"] = out["vmax_kt"] - out["best_track_kt"]
    return out


@app.get("/api/vision/scene.png")
def vision_scene_png(time: str, enhance: bool = True):
    # the satellite scene itself, as a georeferenced PNG for map overlay
    from fastapi import Response
    from api import vision
    got = vision.scene_image(pd.Timestamp(time), enhance)
    if got is None:
        raise HTTPException(404, f"no GridSat scene near {time}")
    png, meta = got
    return Response(content=png, media_type="image/png",
                    headers={"X-Scene-Time": meta["scene_time"],
                             "X-Bounds": str(meta["bounds"]),
                             "Cache-Control": "public, max-age=86400"})


@app.get("/api/vision/scene_meta")
def vision_scene_meta(time: str):
    from api import vision
    got = vision.scene_bounds(pd.Timestamp(time))
    if got is None:
        raise HTTPException(404, f"no GridSat scene near {time}")
    return got


@app.get("/api/vision/pipeline")
def vision_pipeline(time: str, hours: int = 48, threshold: float = 0.30,
                    level: float = 0.67):
    # the whole chain: imagery in, detected storms and forecasts out
    global _CHAIN
    if _CHAIN is None:
        sys.path.insert(0, str(ROOT))
        from pipeline import Chakravat
        _CHAIN = Chakravat(verbose=False)
    return _CHAIN.run(pd.Timestamp(time), hours=hours,
                      threshold=threshold, level=level)


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
                      ("scene", "t2_scene.json"),
                      ("intensity_vision", "t3_intensity_vision.json")):
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
