# residual boosting over CLIPER

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from features.build import model_features
from models.baselines import Cliper, CLIPER_FEATURES


def _residual_regressor(seed: int) -> HistGradientBoostingRegressor:
    # deliberately weaker than a standalone GBM: it is only correcting a
    # residual, and over-capacity here is what caused the original regression.
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=200,
        learning_rate=0.03,
        min_samples_leaf=40,
        l2_regularization=5.0,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=seed,
    )


class ResidualForecaster:
    # CLIPER + a regularised GBM fitted to CLIPER's out-of-fold residuals

    def __init__(self, seed: int = 0, n_splits: int = 5):
        self.seed = seed
        self.n_splits = n_splits
        self.cliper = Cliper()
        self.residual_models: dict[str, HistGradientBoostingRegressor] = {}

    def fit(self, train: pd.DataFrame, targets: list[str]) -> "ResidualForecaster":
        self.cliper.fit(train, targets)

        for target in targets:
            sub = train[train[target].notna()]
            if len(sub) < 100:
                continue

            # out-of-fold CLIPER predictions, grouped by storm so a storm's own
            # fixes never inform its baseline. In-sample residuals would be
            # optimistically small and the correction would learn noise.
            groups = sub["SID"].to_numpy()
            n_splits = min(self.n_splits, len(np.unique(groups)))
            if n_splits < 2:
                continue
            oof = cross_val_predict(
                self.cliper._pipe(),
                sub[CLIPER_FEATURES],
                sub[target],
                groups=groups,
                cv=GroupKFold(n_splits=n_splits),
            )
            residual = sub[target].to_numpy() - oof

            model = _residual_regressor(self.seed)
            model.fit(sub[model_features(sub)], residual)
            self.residual_models[target] = model
        return self

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        base = self.cliper.predict(df, target)
        if target not in self.residual_models:
            return base
        return base + self.residual_models[target].predict(df[model_features(df)])
