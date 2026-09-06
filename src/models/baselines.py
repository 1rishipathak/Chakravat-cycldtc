# reference forecasts every learned model must beat to claim skill

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# climatology-and-persistence predictors, following the classic CLIPER recipe:
# where the storm is, where it is going, how strong it is, how that is trending,
# and the time of year.
CLIPER_FEATURES = [
    "LAT", "LON", "vmax_kt",
    "dlat_prev6", "dlon_prev6", "dv_prev6",
    "dlat_prev12", "dlon_prev12", "dv_prev12",
    "u_deg_h", "v_deg_h", "doy_sin", "doy_cos", "is_bob",
]


class Persistence:
    # extrapolate the last observed 6-hour motion; hold intensity constant

    @staticmethod
    def predict_track(df: pd.DataFrame, horizon_h: int) -> tuple[np.ndarray, np.ndarray]:
        steps = horizon_h / 6.0
        pred_lat = df["LAT"].to_numpy() + df["dlat_prev6"].to_numpy() * steps
        pred_lon = df["LON"].to_numpy() + df["dlon_prev6"].to_numpy() * steps
        return pred_lat, pred_lon

    @staticmethod
    def predict_intensity(df: pd.DataFrame, horizon_h: int) -> np.ndarray:
        return np.zeros(len(df))


class DecayPersistence:
    # extrapolate the recent intensity trend with exponential decay

    def __init__(self, e_folding_h: float = 24.0):
        self.e_folding_h = e_folding_h

    def predict_intensity(self, df: pd.DataFrame, horizon_h: int) -> np.ndarray:
        rate = df["dv_prev6"].to_numpy() / 6.0                  # kt per hour
        tau = self.e_folding_h
        # integral of rate * exp(-t/tau) from 0 to horizon.
        return rate * tau * (1.0 - np.exp(-horizon_h / tau))


class Cliper:
    # ridge regression on climatology + persistence predictors

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[str, object] = {}

    def _pipe(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        )

    def fit(self, train: pd.DataFrame, targets: list[str]) -> "Cliper":
        for target in targets:
            sub = train[train[target].notna()]
            if len(sub) < 30:
                continue
            model = self._pipe()
            model.fit(sub[CLIPER_FEATURES], sub[target])
            self.models[target] = model
        return self

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        if target not in self.models:
            return np.zeros(len(df))
        return self.models[target].predict(df[CLIPER_FEATURES])
