# rapid intensification classifier with calibrated probabilities

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from features.build import FEATURES


class RiClassifier:
    # unweighted booster + Platt calibration on a held-out season block

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model: CalibratedClassifierCV | None = None
        self.base_rate: float = float("nan")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "RiClassifier":
        tr = train[train["y_ri"].notna()]
        va = val[val["y_ri"].notna()]
        self.base_rate = float(tr["y_ri"].mean())

        base = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=250,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.seed,
        )
        base.fit(tr[FEATURES], tr["y_ri"].to_numpy())

        # sigmoid rather than isotonic: the validation block holds only a few
        # dozen RI events, far too few for a non-parametric calibrator.
        # FrozenEstimator keeps `base` fitted on train while the calibrator
        # learns only from the validation seasons (sklearn >= 1.6 spelling of
        # the old cv="prefit").
        self.model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        self.model.fit(va[FEATURES], va["y_ri"].to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), np.nan)
        return self.model.predict_proba(df[FEATURES])[:, 1]

    def pick_threshold(self, val: pd.DataFrame, min_pod: float = 0.5) -> float:
        # lowest threshold meeting a recall floor, to minimise false alarms
        va = val[val["y_ri"].notna()]
        y = va["y_ri"].to_numpy()
        prob = self.predict_proba(va)
        best, best_far = 0.5, np.inf
        for thr in np.linspace(0.02, 0.9, 89):
            pred = (prob >= thr).astype(float)
            hits = np.sum((pred == 1) & (y == 1))
            misses = np.sum((pred == 0) & (y == 1))
            fa = np.sum((pred == 1) & (y == 0))
            pod = hits / (hits + misses) if (hits + misses) else 0.0
            far = fa / (hits + fa) if (hits + fa) else 1.0
            if pod >= min_pod and far < best_far:
                best, best_far = float(thr), float(far)
        return best
