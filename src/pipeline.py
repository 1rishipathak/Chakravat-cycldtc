# end-to-end pipeline: satellite imagery in, forecast out

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# both are needed: the models live under src/, the vision endpoints under api/.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
GRIDSAT = ROOT / "data" / "gridsat"

# A cyclone does not move faster than this, so two detections further apart
# than one step at this speed are different storms.
MAX_STORM_SPEED_KMH = 45.0
MIN_TRACK_POINTS = 2


@dataclass
class Detection:
    time: pd.Timestamp
    lat: float
    lon: float
    score: float
    scene: str | None = None
    scene_confidence: float | None = None
    vmax_kt: float | None = None


@dataclass
class Track:
    points: list[Detection] = field(default_factory=list)

    @property
    def last(self) -> Detection:
        return self.points[-1]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ISO_TIME": p.time, "LAT": p.lat, "LON": p.lon,
            "vmax_kt": p.vmax_kt if p.vmax_kt is not None else np.nan,
            "scene": p.scene,
        } for p in self.points])


def associate(per_time: dict[pd.Timestamp, list[Detection]]) -> list[Track]:
    # link detections across consecutive timesteps into tracks
    tracks: list[Track] = []
    for when in sorted(per_time):
        dets = list(per_time[when])
        open_tracks = [t for t in tracks
                       if (when - t.last.time) <= pd.Timedelta(hours=12)]
        for track in sorted(open_tracks, key=lambda t: -t.last.score):
            if not dets:
                break
            gap_h = (when - track.last.time).total_seconds() / 3600.0
            limit_km = MAX_STORM_SPEED_KMH * gap_h
            best, best_d = None, np.inf
            for d in dets:
                dist = _km(track.last.lat, track.last.lon, d.lat, d.lon)
                if dist < best_d:
                    best, best_d = d, dist
            if best is not None and best_d <= limit_km:
                track.points.append(best)
                dets.remove(best)
        for d in dets:                      # unmatched detections start tracks
            tracks.append(Track(points=[d]))
    return [t for t in tracks if len(t.points) >= MIN_TRACK_POINTS]


def _km(lat1, lon1, lat2, lon2) -> float:
    from eval.metrics import great_circle_km
    return float(great_circle_km(lat1, lon1, lat2, lon2))


class Chakravat:
    # the whole system behind one call

    def __init__(self, verbose: bool = True) -> None:
        import joblib
        from api import vision

        self.vision = vision
        self.verbose = verbose
        bundle = joblib.load(ART / "prediction_models.joblib")
        self.ri = bundle["ri"]
        self.ri_threshold = bundle["ri_threshold"]
        self.ensemble = joblib.load(ART / "ensemble_cone.joblib")
        self.landfall = joblib.load(ART / "landfall_model.joblib")

    # - stage 1 and 2 ---------------------------------------------------

    def detect_window(self, end: pd.Timestamp, hours: int = 48,
                      step_h: int = 6, threshold: float = 0.30
                      ) -> dict[pd.Timestamp, list[Detection]]:
        # run T1 over a window of scenes ending at end
        out: dict[pd.Timestamp, list[Detection]] = {}
        steps = int(hours // step_h) + 1
        for k in range(steps - 1, -1, -1):
            when = pd.Timestamp(end) - pd.Timedelta(hours=step_h * k)
            res = self.vision.detect(when, threshold)
            if not res.get("available"):
                continue
            slot = pd.Timestamp(res["scene_time"])
            out[slot] = [Detection(slot, d["lat"], d["lon"], d["score"])
                         for d in res["detections"]]
        return out

    def describe(self, det: Detection) -> Detection:
        # T2 and T3 on one detection
        scene = self.vision.classify_scene(det.time, det.lat, det.lon)
        if scene.get("available"):
            det.scene = scene["scene"]
            det.scene_confidence = scene["confidence"]
        inten = self.vision.estimate_intensity(det.time, det.lat, det.lon)
        if inten.get("available"):
            det.vmax_kt = inten["vmax_kt"]
        return det

    # - stage 3 ---------------------------------------------------------

    def _feature_row(self, track: Track) -> pd.DataFrame | None:
        # build the row T4 expects from an imagery-derived track
        from features.build import add_history, add_targets

        frame = track.to_frame()
        if frame["vmax_kt"].isna().all() or len(frame) < 2:
            return None

        # put the detected track on a regular 6-hourly grid before deriving
        # motion. Two reasons it is not already on one: GridSat slots are
        # 3-hourly and our schedule lands on 03/09/15/21 as often as on
        # synoptic hours, and a scene the detector misses leaves a gap. The
        # lag features join on exact timestamps, so a single 12-hour gap makes
        # the most recent row unusable and the forecast silently disappears --
        # which is exactly what happened on Amphan.
        frame = frame.set_index("ISO_TIME").sort_index()
        grid = pd.date_range(frame.index.min().ceil("6h"),
                             frame.index.max().floor("6h"), freq="6h")
        if len(grid) < 2:
            grid = pd.date_range(frame.index.min(), frame.index.max(), freq="6h")
        if len(grid) < 2:
            return None
        numeric = frame[["LAT", "LON", "vmax_kt"]].astype(float)
        resampled = (numeric.reindex(numeric.index.union(grid))
                            .interpolate(method="time", limit_direction="both")
                            .reindex(grid))

        # intensity from a single frame is jumpy - the detected centre moves a
        # little each scene, so the patch shifts and the estimate with it. A
        # short rolling median keeps the trend without chasing one bad frame,
        # which is the same reason operational Dvorak time-averages.
        resampled["vmax_kt"] = (resampled["vmax_kt"]
                                .rolling(3, center=True, min_periods=1).median())

        frame = resampled.reset_index().rename(columns={"index": "ISO_TIME"})
        frame["scene"] = None

        # pressure and distance-to-land are not readable from infrared, so they
        # go in as missing rather than invented.
        frame["pres_hpa"] = np.nan
        frame["DIST2LAND"] = np.nan
        frame["SID"] = "LIVE"
        frame["SEASON"] = frame["ISO_TIME"].dt.year
        frame["NAME"] = "DETECTED"
        frame["sub_basin"] = np.where(frame["LON"] >= 80.0, "BOB", "AS")
        frame["NEWDELHI_CI"] = np.nan

        built = add_targets(add_history(frame))

        # T4 was trained with ERA5 environment, so the row has to carry it or
        # the model rejects the feature set outright. Reanalysis is available
        # for any past time and, operationally, would come from the forecast
        # fields - so unlike pressure and distance-to-land this is a genuine
        # input rather than something imagery cannot supply.
        era5 = ROOT / "data" / "era5"
        if era5.exists():
            from features.environment import attach
            built = attach(built, era5)

        row = built.iloc[[-1]]
        return row if row["dlat_prev6"].notna().all() else None

    def forecast(self, track: Track, level: float = 0.67) -> dict | None:
        from features.build import HORIZONS
        from ingest.ibtracs import imd_category

        row = self._feature_row(track)
        if row is None:
            return None

        points = []
        for h in HORIZONS:
            cone = self.ensemble.track_cone(row, h, level)
            band = self.ensemble.intensity_band(row, h, level)
            vmax = float(band["vmax"][0])
            points.append({"horizon_h": h,
                           "lat": float(cone["lat"][0]),
                           "lon": float(cone["lon"][0]),
                           "radius_km": float(cone["radius_km"][0]),
                           "vmax_kt": vmax,
                           "vmax_lower": float(band["lower"][0]),
                           "vmax_upper": float(band["upper"][0]),
                           "category": imd_category(vmax)})

        ri_prob = float(self.ri.predict_proba(row)[0])
        lf_prob = float(self.landfall.predict_will_land(row)[0])
        # the forecast is valid from the resampled synoptic time, not from the
        # raw detection timestamp. Those differ - GridSat slots often land on
        # 03/09/15/21 while the model and best track work on 00/06/12/18 - and
        # leaving callers to infer it makes every verification silently
        # misaligned by up to three hours.
        out = {"issued_at": str(pd.Timestamp(row["ISO_TIME"].iloc[0])),
               "level": level,
               "forecast": points,
               "rapid_intensification": {
                   "probability": ri_prob,
                   "flagged": bool(ri_prob >= self.ri_threshold)},
               "landfall": {"probability_within_72h": lf_prob}}
        if lf_prob >= 0.5:
            out["landfall"].update({
                "hours": float(self.landfall.predict(row, "lf_hours")[0]),
                "lat": float(self.landfall.predict(row, "lf_lat")[0]),
                "lon": float(self.landfall.predict(row, "lf_lon")[0]),
                "vmax_kt": float(self.landfall.predict(row, "lf_vmax")[0])})
        return out

    # - the whole chain -------------------------------------------------

    def run(self, end: pd.Timestamp, hours: int = 48,
            threshold: float = 0.30, level: float = 0.67) -> dict:
        per_time = self.detect_window(end, hours, threshold=threshold)
        if not per_time:
            return {"available": False,
                    "reason": "no GridSat scenes in this window"}

        for dets in per_time.values():
            for d in dets:
                self.describe(d)

        tracks = associate(per_time)
        storms = []
        for i, track in enumerate(tracks, 1):
            last = track.last
            storms.append({
                "id": f"DET{i:02d}",
                "detected_at": str(last.time),
                "lat": last.lat, "lon": last.lon,
                "detection_score": last.score,
                "track_points": len(track.points),
                "scene": last.scene,
                "scene_confidence": last.scene_confidence,
                "vmax_kt": last.vmax_kt,
                "history": [{"time": str(p.time), "lat": p.lat, "lon": p.lon,
                             "vmax_kt": p.vmax_kt, "scene": p.scene}
                            for p in track.points],
                "prediction": self.forecast(track, level),
            })

        return {"available": True,
                "window_end": str(end),
                "window_hours": hours,
                "scenes_examined": len(per_time),
                "tracks_found": len(tracks),
                "storms": storms,
                "note": "detected from imagery alone; pressure and "
                        "distance-to-land are unavailable and passed as missing"}


if __name__ == "__main__":
    import json

    when = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2020-05-19 12:00")
    chain = Chakravat()
    result = chain.run(when)
    print(json.dumps(result, indent=2, default=str)[:4000])
