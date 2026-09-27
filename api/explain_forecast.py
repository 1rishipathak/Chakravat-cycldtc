"""Why the forecast said what it said.

The imagery models get Grad-CAM, which shows where the network looked. The
forecast is gradient-boosted trees over 33 named numbers, so the equivalent
question is which of those numbers the answer rested on.

We answer it by ablation: replace a feature with its median over the training
data, re-run the ensemble, and measure how far the forecast moved. That reads
as a sentence a forecaster would accept - "if we had not known the steering
flow, this 24 h position would move 80 km" - and it needs no extra dependency
and no surrogate model standing in for the real one.

Two things make the naive version of this useless, and both are handled here.

**Correlated encodings have to move together.** Time of year is stored as a
sine and a cosine; wind shear as a magnitude and two components. Ablating one
at a time lets its twin carry the information, and everything looks
unimportant. They are ablated as groups.

**Present position and intensity are excluded from the ranking.** Substituting
the median latitude describes a different storm somewhere else, not the same
storm with less information, so it always dominates and always says the same
uninformative thing. They are reported separately as the state the forecast
starts from.

What this is not: it is not a Shapley value. One-at-a-time ablation misses
interactions, and features that remain correlated after grouping still cover
for each other, so a small number means "the forecast did not need this given
the others", not "this does not matter".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# the state the forecast starts from. ablating these asks about a different
# storm, so they are described rather than ranked.
STATE = ["LAT", "LON", "vmax_kt"]

# things that are one piece of information stored as several numbers
GROUPS = {
    "time of year": ["doy_sin", "doy_cos"],
    "steering flow": ["env_steer_u", "env_steer_v"],
    "wind shear": ["env_shear_ms", "env_shear_u", "env_shear_v"],
    "speed and heading": ["u_deg_h", "v_deg_h"],
    "motion, last 6 h": ["dlat_prev6", "dlon_prev6"],
    "motion, last 12 h": ["dlat_prev12", "dlon_prev12"],
    "motion, last 24 h": ["dlat_prev24", "dlon_prev24"],
    "intensity trend": ["dv_prev6", "dv_prev12", "dv_prev24"],
    "mid-level humidity": ["env_rh700", "env_rh500"],
    "intensity so far": ["vmax_max_so_far", "vmax_mean_so_far"],
}

SINGLES = {
    "pres_hpa": "central pressure",
    "DIST2LAND": "distance to land",
    "age_h": "hours since first fix",
    "is_bob": "Bay of Bengal or Arabian Sea",
    "NEWDELHI_CI": "RSMC Dvorak number",
    "env_sst_c": "sea surface temperature",
    "env_tcwv": "total column water vapour",
    "env_div200": "divergence aloft",
}


def _km(lat1, lon1, lat2, lon2) -> float:
    from eval.metrics import great_circle_km
    return float(great_circle_km(lat1, lon1, lat2, lon2))


def _blocks(columns) -> list[tuple[str, list[str]]]:
    out = [(label, [c for c in cols if c in columns]) for label, cols in GROUPS.items()]
    out += [(label, [c]) for c, label in SINGLES.items() if c in columns]
    return [(label, cols) for label, cols in out if cols]


def explain(ensemble, row: pd.DataFrame, medians: pd.Series, horizon: int,
            level: float = 0.67, top: int = 6) -> dict:
    base_cone = ensemble.track_cone(row, horizon, level)
    base_band = ensemble.intensity_band(row, horizon, level)
    base_lat, base_lon = float(base_cone["lat"][0]), float(base_cone["lon"][0])
    base_vmax = float(base_band["vmax"][0])

    out = []
    for label, cols in _blocks(row.columns):
        usable = [c for c in cols
                  if c in medians.index
                  and not pd.isna(row.iloc[0][c]) and not pd.isna(medians[c])]
        if not usable:
            continue                      # already missing: nothing to remove
        probe = row.copy()
        for c in usable:
            probe.loc[probe.index[0], c] = medians[c]
        cone = ensemble.track_cone(probe, horizon, level)
        band = ensemble.intensity_band(probe, horizon, level)
        out.append({
            "label": label,
            "features": usable,
            "track_shift_km": _km(base_lat, base_lon,
                                  float(cone["lat"][0]), float(cone["lon"][0])),
            "intensity_shift_kt": float(band["vmax"][0]) - base_vmax,
        })

    return {
        "horizon_h": horizon,
        "baseline": {"lat": base_lat, "lon": base_lon, "vmax_kt": base_vmax},
        "state": {k: float(row.iloc[0][k]) for k in STATE if k in row.columns},
        "track": sorted(out, key=lambda d: -d["track_shift_km"])[:top],
        "intensity": sorted(out, key=lambda d: -abs(d["intensity_shift_kt"]))[:top],
        "method": "group ablation against the training median, holding present "
                  "position and intensity fixed",
        "caveat": "Not Shapley values: these do not sum to the forecast, and a small "
                  "shift means the forecast did not need that input given the others.",
    }


def medians_for(data: pd.DataFrame, features: list[str]) -> pd.Series:
    return data[features].median(numeric_only=True)


def sentence(ex: dict) -> str:
    """The one line a forecaster would actually read."""
    if not ex["track"]:
        return "Nothing beyond the storm's present position moved this forecast measurably."
    t = ex["track"][0]
    bits = [f"beyond where the storm already is, the {ex['horizon_h']} h position rests "
            f"most on {t['label']} (removing it moves the forecast "
            f"{t['track_shift_km']:.0f} km)"]
    i = ex["intensity"][0]
    if abs(i["intensity_shift_kt"]) >= 0.5:
        bits.append(f"the intensity most on {i['label']} ({i['intensity_shift_kt']:+.1f} kt)")
    line = "; ".join(bits)
    return line[:1].upper() + line[1:] + "."     # capitalize() would lowercase RSMC
