# gradient-boosted forecasters for track, intensity, and rapid

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from features.build import model_features


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    # shallow and heavily regularised on purpose - the training set is small.
    return HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=400,
        learning_rate=0.05,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


class GbmForecaster:
    # One regressor per (target, horizon); one classifier for RI

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.ri_model: HistGradientBoostingClassifier | None = None

    def fit(self, train: pd.DataFrame, targets: list[str]) -> "GbmForecaster":
        for target in targets:
            sub = train[train[target].notna()]
            if len(sub) < 50:
                continue
            model = _regressor(self.seed)
            model.fit(sub[model_features(sub)], sub[target])
            self.models[target] = model
        return self

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        if target not in self.models:
            return np.zeros(len(df))
        return self.models[target].predict(df[model_features(df)])

    def fit_ri(self, train: pd.DataFrame) -> "GbmForecaster":
        sub = train[train["y_ri"].notna()]
        y = sub["y_ri"].to_numpy()
        # RI is ~3% of cases. Without class weighting the model learns to
        # always answer "no", which is accurate and useless.
        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        weights = np.where(y == 1, pos_weight, 1.0)
        self.ri_model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.seed,
        )
        self.ri_model.fit(sub[model_features(sub)], y, sample_weight=weights)
        return self

    def predict_ri(self, df: pd.DataFrame) -> np.ndarray:
        if self.ri_model is None:
            return np.full(len(df), np.nan)
        return self.ri_model.predict_proba(df[model_features(df)])[:, 1]
