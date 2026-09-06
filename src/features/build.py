# helpful in creating a dataset that can be used for forecast by joining tables on common timestamps

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [6, 12, 24, 48, 72]   # hours
LAGS = [6, 12, 24]               # hours of history used as predictors
RI_THRESHOLD_KT = 30             # +30 kt in 24 h is the operational RI definition
RI_HORIZON_H = 24


def _shifted(df: pd.DataFrame, hours: int, cols: list[str], suffix: str) -> pd.DataFrame:
    other = df[["SID", "ISO_TIME"] + cols].copy()
    other["ISO_TIME"] = other["ISO_TIME"] - pd.Timedelta(hours=hours)
    other = other.rename(columns={c: f"{c}{suffix}" for c in cols})
    return df.merge(other, on=["SID", "ISO_TIME"], how="left")


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in LAGS:
        out = _shifted(out, -lag, ["LAT", "LON", "vmax_kt"], f"_m{lag}")
        out[f"dlat_prev{lag}"] = out["LAT"] - out[f"LAT_m{lag}"]
        out[f"dlon_prev{lag}"] = out["LON"] - out[f"LON_m{lag}"]
        out[f"dv_prev{lag}"] = out["vmax_kt"] - out[f"vmax_kt_m{lag}"]
        out = out.drop(columns=[f"LAT_m{lag}", f"LON_m{lag}", f"vmax_kt_m{lag}"])

    # zonal/meridional speed over the last 6 h, in degrees per hour.
    out["u_deg_h"] = out["dlon_prev6"] / 6.0
    out["v_deg_h"] = out["dlat_prev6"] / 6.0

    grp = out.groupby("SID")
    out["age_h"] = (out["ISO_TIME"] - grp["ISO_TIME"].transform("min")).dt.total_seconds() / 3600.0
    out["vmax_max_so_far"] = grp["vmax_kt"].cummax()
    out["vmax_mean_so_far"] = grp["vmax_kt"].transform(lambda s: s.expanding().mean())

    doy = out["ISO_TIME"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["is_bob"] = (out["sub_basin"] == "BOB").astype(int)
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    # forecast targets
    out = df.copy()
    for h in HORIZONS:
        out = _shifted(out, h, ["LAT", "LON", "vmax_kt"], f"_p{h}")
        out[f"y_dlat_{h}"] = out[f"LAT_p{h}"] - out["LAT"]
        out[f"y_dlon_{h}"] = out[f"LON_p{h}"] - out["LON"]
        out[f"y_dv_{h}"] = out[f"vmax_kt_p{h}"] - out["vmax_kt"]
        out = out.rename(columns={f"LAT_p{h}": f"true_lat_{h}", f"LON_p{h}": f"true_lon_{h}"})
        out = out.drop(columns=[f"vmax_kt_p{h}"])

    out["y_ri"] = (out[f"y_dv_{RI_HORIZON_H}"] >= RI_THRESHOLD_KT).astype("float")
    out.loc[out[f"y_dv_{RI_HORIZON_H}"].isna(), "y_ri"] = np.nan
    return out


FEATURES = [
    "LAT", "LON", "vmax_kt", "pres_hpa", "DIST2LAND",
    "dlat_prev6", "dlon_prev6", "dv_prev6",
    "dlat_prev12", "dlon_prev12", "dv_prev12",
    "dlat_prev24", "dlon_prev24", "dv_prev24",
    "u_deg_h", "v_deg_h", "age_h",
    "vmax_max_so_far", "vmax_mean_so_far",
    "doy_sin", "doy_cos", "is_bob", "NEWDELHI_CI",
]


def build_dataset(track: pd.DataFrame) -> pd.DataFrame:
    # feature + target dataset construction
    df = add_history(track)
    df = add_targets(df)
    # forecast needs at least one prior fix to establish motion.
    return df[df["dlat_prev6"].notna()].reset_index(drop=True)


def model_features(df: pd.DataFrame) -> list[str]:
    # model feature list which can use ERA5 columns if present.
    # its dynamic so the pipeline runs identically with or without ERA5 since we treat environmental factors as good to have but not necessary for model

    from features.environment import ENV_FEATURES

    return FEATURES + [c for c in ENV_FEATURES if c in df.columns]
