# landfall forecasting: when, where, and how strong at the coast

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from features.build import model_features

MAX_LEAD_H = 72          # only forecast landfalls within three days
KM_PER_DEG = 111.32

LANDFALL_TARGETS = ["lf_hours", "lf_lat", "lf_lon", "lf_vmax"]

# coastline-derived columns. These are added to the *landfall* feature list
# only, never to `FEATURES` in features/build.py - the track and intensity
# forecasters are trained and verified, and widening their input would force a
# retrain that could move numbers we are not trying to move.
COAST_FEATURES = ["coast_dist_km", "coast_bearing_rel",
                  "coast_lat", "coast_lon"]


def add_coast_features(df: pd.DataFrame) -> pd.DataFrame:
    # attach where the nearest coast is, relative to where the storm is going
    from models import coastline as cl

    out = df.copy()
    if not cl.available():
        for c in COAST_FEATURES:
            out[c] = np.nan
        return out

    c_lat, c_lon, dist, bearing = cl.nearest_coast_bulk(
        out["LAT"].to_numpy(), out["LON"].to_numpy())

    heading = (np.degrees(np.arctan2(
        out["dlon_prev6"].to_numpy()
        * np.cos(np.radians(out["LAT"].to_numpy())),
        out["dlat_prev6"].to_numpy())) + 360.0) % 360.0

    rel = (bearing - heading + 180.0) % 360.0 - 180.0

    out["coast_dist_km"] = dist
    out["coast_bearing_rel"] = np.abs(rel)     # symmetric: left/right is not signal
    out["coast_lat"] = c_lat
    out["coast_lon"] = c_lon
    return out


def landfall_features(df: pd.DataFrame) -> list[str]:
    # track-model features plus the coastline columns
    return model_features(df) + [c for c in COAST_FEATURES if c in df.columns]


def add_landfall_targets(df: pd.DataFrame) -> pd.DataFrame:
    # attach each pre-landfall fix to its storm's landfall event
    out = df.copy()
    for col in LANDFALL_TARGETS + ["lf_will_land"]:
        out[col] = np.nan

    for sid, storm in out.groupby("SID", sort=False):
        storm = storm.sort_values("ISO_TIME")
        on_land = storm["DIST2LAND"].to_numpy() == 0
        offshore_first = (~on_land).any()
        if not (on_land.any() and offshore_first):
            out.loc[storm.index, "lf_will_land"] = 0.0
            continue

        # first landing that follows an offshore fix.
        idx = None
        for pos in range(1, len(storm)):
            if on_land[pos] and not on_land[:pos].all():
                idx = pos
                break
        if idx is None:
            out.loc[storm.index, "lf_will_land"] = 0.0
            continue

        event = storm.iloc[idx]
        pre = storm.index[:idx]
        lead = (event["ISO_TIME"] - storm["ISO_TIME"].iloc[:idx]).dt.total_seconds() / 3600.0

        out.loc[storm.index, "lf_will_land"] = 0.0
        usable = pre[(lead.to_numpy() <= MAX_LEAD_H) & (lead.to_numpy() > 0)]
        out.loc[usable, "lf_will_land"] = 1.0
        out.loc[usable, "lf_hours"] = lead[lead <= MAX_LEAD_H][lead > 0].to_numpy()
        out.loc[usable, "lf_lat"] = event["LAT"]
        out.loc[usable, "lf_lon"] = event["LON"]
        out.loc[usable, "lf_vmax"] = event["vmax_kt"]

    return out


class LandfallForecaster:
    # Three regressors for the landfall event, plus a will-it-land classifier

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.classifier: HistGradientBoostingClassifier | None = None

    def _regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=self.seed,
        )

    def fit(self, train: pd.DataFrame) -> "LandfallForecaster":
        for target in LANDFALL_TARGETS:
            sub = train[train[target].notna()]
            if len(sub) < 50:
                continue
            model = self._regressor()
            model.fit(sub[landfall_features(sub)], sub[target])
            self.models[target] = model

        sub = train[train["lf_will_land"].notna()]
        if len(sub) >= 50:
            self.classifier = HistGradientBoostingClassifier(
                max_depth=3, max_iter=250, learning_rate=0.05,
                min_samples_leaf=25, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                random_state=self.seed,
            )
            self.classifier.fit(sub[landfall_features(sub)], sub["lf_will_land"])
        return self

    @staticmethod
    def _ensure_coast(df: pd.DataFrame) -> pd.DataFrame:
        # attach coastline columns if the caller has not
        if all(c in df.columns for c in COAST_FEATURES):
            return df
        return add_coast_features(df)

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        if target not in self.models:
            return np.full(len(df), np.nan)
        df = self._ensure_coast(df)
        return self.models[target].predict(df[landfall_features(df)])

    def predict_will_land(self, df: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            return np.full(len(df), np.nan)
        df = self._ensure_coast(df)
        return self.classifier.predict_proba(df[landfall_features(df)])[:, 1]

    def predict_position(self, df: pd.DataFrame,
                         track: list[tuple[float, float]] | None = None,
                         horizons_h: list[float] | None = None) -> dict:
        # landfall position, preferring the forecast track's coast crossing
        from models import coastline as cl

        lat = float(self.predict(df, "lf_lat")[0])
        lon = float(self.predict(df, "lf_lon")[0])
        s_lat, s_lon = snap_to_coast([lat], [lon])
        out = {"lat": float(s_lat[0]), "lon": float(s_lon[0]),
               "method": "regression_snapped", "crossing_hours": None}

        if track and len(track) >= 2 and cl.available():
            hit = cl.first_crossing(track)
            if hit is not None:
                out["lat"], out["lon"] = hit[0], hit[1]
                out["method"] = "track_crossing"
                if horizons_h:
                    out["crossing_hours"] = cl.crossing_time_h(hit[2], horizons_h)
        return out


def snap_to_coast(lats, lons):
    # project predicted landfall positions onto the nearest coastline point
    from models import coastline as cl

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if not cl.available():
        return lats, lons
    c_lat, c_lon, _, _ = cl.nearest_coast_bulk(lats, lons)
    ok = np.isfinite(c_lat)
    return np.where(ok, c_lat, lats), np.where(ok, c_lon, lons)


# ---- hand-computable baselines -------------------------------------------

def baseline_hours(df: pd.DataFrame) -> np.ndarray:
    # distance to land divided by current speed toward it
    speed_kmh = np.hypot(
        df["dlat_prev6"].to_numpy() * KM_PER_DEG,
        df["dlon_prev6"].to_numpy() * KM_PER_DEG
        * np.cos(np.radians(df["LAT"].to_numpy())),
    ) / 6.0
    speed_kmh = np.clip(speed_kmh, 5.0, None)      # avoid dividing by a stalled storm
    return df["DIST2LAND"].to_numpy() / speed_kmh


def baseline_position(df: pd.DataFrame, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # extrapolate the current 6-hour motion out to the predicted landfall time
    steps = hours / 6.0
    return (df["LAT"].to_numpy() + df["dlat_prev6"].to_numpy() * steps,
            df["LON"].to_numpy() + df["dlon_prev6"].to_numpy() * steps)


def baseline_intensity(df: pd.DataFrame) -> np.ndarray:
    # persistence: the storm arrives as strong as it is now
    return df["vmax_kt"].to_numpy()
