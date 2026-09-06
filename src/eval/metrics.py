# forecast metrics used

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    # Haversine distance, helpful in getting position errors
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def track_error(true_lat, true_lon, pred_lat, pred_lon) -> dict[str, float]:
    err = great_circle_km(true_lat, true_lon, pred_lat, pred_lon)
    return {
        "n": int(len(err)),
        "mean_km": float(np.mean(err)),
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
    }


def intensity_error(true_dv, pred_dv) -> dict[str, float]:
    true_dv, pred_dv = np.asarray(true_dv), np.asarray(pred_dv)
    err = pred_dv - true_dv
    return {
        "n": int(len(err)),
        "mae_kt": float(np.mean(np.abs(err))),
        "rmse_kt": float(np.sqrt(np.mean(err ** 2))),
        "bias_kt": float(np.mean(err)),
    }


def brier_skill_score(y_true, y_prob, climatology: float | None = None) -> dict[str, float]:
    # Brier score 
    y_true, y_prob = np.asarray(y_true, float), np.asarray(y_prob, float)
    if climatology is None:
        climatology = float(np.mean(y_true))
    bs = float(np.mean((y_prob - y_true) ** 2))
    bs_ref = float(np.mean((climatology - y_true) ** 2))
    return {
        "n": int(len(y_true)),
        "base_rate": climatology,
        "brier": bs,
        "brier_climatology": bs_ref,
        "bss": float(1 - bs / bs_ref) if bs_ref > 0 else float("nan"),
    }


def contingency(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    # helpful in finding actual hit rate and false alarms
    y_true = np.asarray(y_true, float)
    pred = (np.asarray(y_prob, float) >= threshold).astype(float)
    hits = float(np.sum((pred == 1) & (y_true == 1)))
    misses = float(np.sum((pred == 0) & (y_true == 1)))
    false_alarms = float(np.sum((pred == 1) & (y_true == 0)))
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    csi_denom = hits + misses + false_alarms
    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "pod": pod, "far": far,
        "csi": hits / csi_denom if csi_denom else float("nan"),
    }
