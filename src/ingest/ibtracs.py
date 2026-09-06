# load and clean the IBTrACS North Indian Ocean best-track archive

from __future__ import annotations

import numpy as np
import pandas as pd

# IMD reports 3-minute sustained wind; JTWC reports 1-minute. Mixing them
# without conversion injects a systematic high bias of several knots into the
# labels. 0.93 is the conventional 1-min -> 3-min factor.
ONE_MIN_TO_THREE_MIN = 0.93

# IMD intensity scale, in knots (3-minute sustained). Upper bound exclusive.
IMD_CATEGORIES = [
    ("Low Pressure Area", 0, 17),
    ("Depression", 17, 28),
    ("Deep Depression", 28, 34),
    ("Cyclonic Storm", 34, 48),
    ("Severe Cyclonic Storm", 48, 64),
    ("Very Severe Cyclonic Storm", 64, 90),
    ("Extremely Severe Cyclonic Storm", 90, 120),
    ("Super Cyclonic Storm", 120, 999),
]

# the basin this system covers: North Indian Ocean, equator to 30N, 40E to 100E.
# IBTrACS files are organised by the basin a storm *entered*, not where it spent
# its life, so the NI file also carries Western Pacific typhoons that were
# tracked across basins - Kajiki, Hato, Bualoi, Damrey and 21 others sit
# entirely east of 100E. Training on them mixes a different basin's dynamics
# into a model built for IMD, so they are filtered out.
# (north, west, south, east)
NIO_BOX = (30.0, 40.0, 0.0, 100.0)

KEEP = [
    "SID", "SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME", "ISO_TIME",
    "NATURE", "LAT", "LON", "WMO_WIND", "WMO_PRES", "TRACK_TYPE",
    "DIST2LAND", "LANDFALL", "USA_WIND", "USA_PRES", "USA_RMW", "USA_SSHS",
    "NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_GRADE", "NEWDELHI_WIND",
    "NEWDELHI_PRES", "NEWDELHI_CI",
]

NUMERIC = [
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "LANDFALL",
    "USA_WIND", "USA_PRES", "USA_RMW", "NEWDELHI_LAT", "NEWDELHI_LON",
    "NEWDELHI_WIND", "NEWDELHI_PRES", "NEWDELHI_CI",
]


def imd_category(vmax_kt: float) -> str:
    # map a 3-minute sustained wind in knots to the IMD category name
    if not np.isfinite(vmax_kt):
        return "Unknown"
    for name, lo, hi in IMD_CATEGORIES:
        if lo <= vmax_kt < hi:
            return name
    return "Unknown"


def load_raw(path: str) -> pd.DataFrame:
    # read the IBTrACS CSV, dropping its units row and coercing numerics
    df = pd.read_csv(path, skiprows=[1], usecols=KEEP, low_memory=False)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df.dropna(subset=["ISO_TIME", "LAT", "LON"])


def unify_intensity(df: pd.DataFrame) -> pd.DataFrame:
    # build one vmax_kt column plus a provenance flag for every row
    imd = df["NEWDELHI_WIND"]
    wmo = df["WMO_WIND"]
    jtwc = df["USA_WIND"] * ONE_MIN_TO_THREE_MIN

    df["vmax_kt"] = imd.where(imd.notna(), wmo.where(wmo.notna(), jtwc))
    df["vmax_source"] = np.where(
        imd.notna(), "imd",
        np.where(wmo.notna(), "wmo", np.where(df["USA_WIND"].notna(), "jtwc", "none")),
    )
    df["pres_hpa"] = df["NEWDELHI_PRES"].where(
        df["NEWDELHI_PRES"].notna(), df["WMO_PRES"].where(df["WMO_PRES"].notna(), df["USA_PRES"])
    )
    return df


def to_synoptic(df: pd.DataFrame) -> pd.DataFrame:
    # keep the four main synoptic hours, which is where best track is densest
    return df[df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])].copy()


def restrict_to_basin(df: pd.DataFrame, box: tuple = NIO_BOX,
                      max_outside: float = 0.5) -> pd.DataFrame:
    # drop cross-basin storms, then trim stray fixes outside the basin
    north, west, south, east = box
    outside = (
        (df["LON"] > east) | (df["LON"] < west)
        | (df["LAT"] > north) | (df["LAT"] < south)
    )
    share = df.assign(_o=outside).groupby("SID")["_o"].transform("mean")
    df = df[share <= max_outside]

    outside = (
        (df["LON"] > east) | (df["LON"] < west)
        | (df["LAT"] > north) | (df["LAT"] < south)
    )
    return df[~outside]


def build(path: str, min_season: int = 1990, min_points: int = 8,
          basin_only: bool = True) -> pd.DataFrame:
    # full ingest: read, unify intensity, restrict to the reliable era, clean
    df = load_raw(path)
    df = unify_intensity(df)
    df = to_synoptic(df)

    df = df[df["SEASON"] >= min_season]
    df = df[df["vmax_kt"].notna() & (df["vmax_kt"] > 0)]
    # keep only tropical/subtropical stages; extratropical transition follows
    # different dynamics and would blur the intensity relationships.
    df = df[df["NATURE"].isin(["TS", "SS", "DS", "NR"])]

    if basin_only:
        df = restrict_to_basin(df)

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # applied after the basin filter, so a storm whose North Indian segment is
    # too short to forecast from is dropped rather than kept as a stub.
    counts = df.groupby("SID")["ISO_TIME"].transform("size")
    df = df[counts >= min_points].copy()

    df["category"] = df["vmax_kt"].apply(imd_category)
    # Bay of Bengal east of 80E, Arabian Sea west of it - the standard split.
    df["sub_basin"] = np.where(df["LON"] >= 80.0, "BOB", "AS")
    return df.reset_index(drop=True)
