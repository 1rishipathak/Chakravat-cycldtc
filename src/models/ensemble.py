# ensemble forecaster with a calibrated uncertainty cone

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.metrics import great_circle_km
from models.residual import ResidualForecaster

DEFAULT_LEVELS = (0.50, 0.67, 0.90)


class EnsembleForecaster:
    # bootstrap ensemble of residual forecasters, with cone calibration

    def __init__(self, n_members: int = 8, seed: int = 0):
        self.n_members = n_members
        self.seed = seed
        self.members: list[ResidualForecaster] = []
        # multipliers[(task, horizon, level)] -> float
        self.multipliers: dict[tuple[str, int, float], float] = {}

    def fit(self, train: pd.DataFrame, targets: list[str]) -> "EnsembleForecaster":
        rng = np.random.default_rng(self.seed)
        storms = train["SID"].unique()

        for member in range(self.n_members):
            picked = rng.choice(storms, size=len(storms), replace=True)
            # bootstrap by storm, keeping every fix of a chosen storm together.
            counts = pd.Series(picked).value_counts()
            frames = [train[train["SID"] == sid] for sid, k in counts.items()
                      for _ in range(k)]
            sample = pd.concat(frames, ignore_index=True)
            self.members.append(
                ResidualForecaster(seed=self.seed + member).fit(sample, targets)
            )
        return self

    def predict(self, df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
        # ensemble mean and standard deviation for one target
        stack = np.stack([m.predict(df, target) for m in self.members])
        return stack.mean(axis=0), stack.std(axis=0, ddof=1)

    # ---- track cone -------------------------------------------------------

    def _track_spread_km(self, df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, ...]:
        mlat, slat = self.predict(df, f"y_dlat_{horizon}")
        mlon, slon = self.predict(df, f"y_dlon_{horizon}")
        lat0, lon0 = df["LAT"].to_numpy(), df["LON"].to_numpy()
        pred_lat, pred_lon = lat0 + mlat, lon0 + mlon

        # convert degree spread to km in a storm-relative sense; longitude
        # degrees shrink with latitude, which matters across a 0-30N basin.
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(pred_lat))
        spread = np.hypot(slat * km_per_deg_lat, slon * km_per_deg_lon)
        return pred_lat, pred_lon, spread

    def calibrate(self, val: pd.DataFrame, horizons: list[int],
                  levels: tuple[float, ...] = DEFAULT_LEVELS) -> "EnsembleForecaster":
        for h in horizons:
            sub = val[val[f"y_dlat_{h}"].notna() & val[f"y_dlon_{h}"].notna()]
            if len(sub) >= 30:
                pred_lat, pred_lon, spread = self._track_spread_km(sub, h)
                err = great_circle_km(sub[f"true_lat_{h}"], sub[f"true_lon_{h}"],
                                      pred_lat, pred_lon)
                ratio = err / np.clip(spread, 1e-6, None)
                for lv in levels:
                    self.multipliers[("track", h, lv)] = float(np.quantile(ratio, lv))

            sub = val[val[f"y_dv_{h}"].notna()]
            if len(sub) >= 30:
                mean, sd = self.predict(sub, f"y_dv_{h}")
                err = np.abs(sub[f"y_dv_{h}"].to_numpy() - mean)
                ratio = err / np.clip(sd, 1e-6, None)
                for lv in levels:
                    self.multipliers[("intensity", h, lv)] = float(np.quantile(ratio, lv))
        return self

    def track_cone(self, df: pd.DataFrame, horizon: int,
                   level: float = 0.67) -> dict[str, np.ndarray]:
        pred_lat, pred_lon, spread = self._track_spread_km(df, horizon)
        mult = self.multipliers.get(("track", horizon, level), 1.0)
        return {"lat": pred_lat, "lon": pred_lon,
                "spread_km": spread, "radius_km": spread * mult}

    def intensity_band(self, df: pd.DataFrame, horizon: int,
                       level: float = 0.67) -> dict[str, np.ndarray]:
        mean, sd = self.predict(df, f"y_dv_{horizon}")
        mult = self.multipliers.get(("intensity", horizon, level), 1.0)
        base = df["vmax_kt"].to_numpy()
        half = sd * mult
        centre = np.clip(base + mean, 0.0, None)
        # A wind speed cannot be negative. At 72 h the band is wide enough that
        # the lower edge crosses zero, which printed as "-0" in the bulletin.
        return {"vmax": centre, "half_width_kt": half,
                "lower": np.clip(centre - half, 0.0, None),
                "upper": centre + half}
