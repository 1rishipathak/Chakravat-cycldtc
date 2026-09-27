"""What has this storm looked like before?

Forecasters reason by analogue constantly - "this is Phailin at the same stage"
- and it is the one form of explanation a meteorologist trusts without being
taught how the model works. It also carries information a metric cannot: what
*happened* to the storms that looked like this one.

The rule for which storms may be analogues is the operational one: only storms
that had already finished when this one began. A forecaster in October 2013
could not have used Fani as an analogue, and neither can we. That rule costs
nothing in the dashboard, where the storm being looked at is historical anyway,
and it means the verification is honest by construction rather than by a split
we have to remember to apply.

Similarity is over position, motion, intensity and season, each standardised so
none dominates by unit. Equal weights, deliberately: a weighting that made the
analogues look better would have been chosen by looking at the answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["LAT", "LON", "vmax_kt", "u_deg_h", "v_deg_h", "doy_sin", "doy_cos"]
LOOKAHEAD = (24, 48, 72)


def _standardise(pool: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mu = pool[cols].mean().to_numpy(dtype=float)
    sd = pool[cols].std(ddof=0).to_numpy(dtype=float)
    return mu, np.where(sd > 1e-9, sd, 1.0)


def _outcome(storm: pd.DataFrame, t0: pd.Timestamp) -> dict:
    """What became of that storm after the moment we matched."""
    after = storm[storm["ISO_TIME"] >= t0]
    if after.empty:
        return {}
    start = after.iloc[0]
    out = {"peak_kt_after": float(after["vmax_kt"].max()),
           "hours_tracked_after": float(
               (after["ISO_TIME"].max() - t0).total_seconds() / 3600.0)}
    for h in LOOKAHEAD:
        row = after[after["ISO_TIME"] == t0 + pd.Timedelta(hours=h)]
        if not row.empty:
            out[f"at_{h}h"] = {"lat": float(row["LAT"].iloc[0]),
                               "lon": float(row["LON"].iloc[0]),
                               "vmax_kt": float(row["vmax_kt"].iloc[0])}
    ashore = after[pd.to_numeric(after.get("DIST2LAND"), errors="coerce") == 0]
    if not ashore.empty:
        first = ashore.iloc[0]
        out["landfall"] = {
            "hours_after": float((first["ISO_TIME"] - t0).total_seconds() / 3600.0),
            "lat": float(first["LAT"]), "lon": float(first["LON"]),
            "vmax_kt": float(first["vmax_kt"])}
    return out


def find(data: pd.DataFrame, sid: str, when: pd.Timestamp, k: int = 5) -> dict:
    """The k most similar earlier moments, and what happened to them."""
    storm = data[data["SID"] == sid].sort_values("ISO_TIME")
    if storm.empty:
        return {"analogues": [], "reason": f"unknown storm {sid}"}
    here = storm[storm["ISO_TIME"] == when]
    if here.empty:
        return {"analogues": [], "reason": "no fix at that time"}

    began = storm["ISO_TIME"].min()
    ends = data.groupby("SID")["ISO_TIME"].max()
    eligible = set(ends[ends < began].index)
    pool = data[data["SID"].isin(eligible)].copy()
    cols = [c for c in FEATURES if c in pool.columns and c in here.columns]
    pool = pool.dropna(subset=cols)
    if pool.empty:
        return {"analogues": [], "pool_storms": 0,
                "reason": "nothing in the archive finished before this storm began"}

    mu, sd = _standardise(pool, cols)
    x = (here[cols].to_numpy(dtype=float)[0] - mu) / sd
    P = (pool[cols].to_numpy(dtype=float) - mu) / sd
    dist = np.sqrt(((P - x) ** 2).sum(axis=1))

    order = np.argsort(dist)
    seen, picks = set(), []
    for i in order:
        row = pool.iloc[i]
        if row["SID"] in seen:          # one moment per storm, the closest
            continue
        seen.add(row["SID"])
        other = data[data["SID"] == row["SID"]].sort_values("ISO_TIME")
        picks.append({
            "sid": row["SID"], "name": str(row.get("NAME") or "UNNAMED"),
            "season": int(row["SEASON"]), "time": str(row["ISO_TIME"]),
            "similarity_km_equivalent": None,
            "distance": float(dist[i]),
            "lat": float(row["LAT"]), "lon": float(row["LON"]),
            "vmax_kt": float(row["vmax_kt"]),
            "outcome": _outcome(other, pd.Timestamp(row["ISO_TIME"])),
        })
        if len(picks) >= k:
            break

    return {"analogues": picks, "pool_storms": len(eligible),
            "pool_fixes": int(len(pool)), "features": cols,
            "rule": "only storms that had already ended when this one began, "
                    "which is what a forecaster at the time could have used"}


def consensus(found: dict, lat0: float, lon0: float) -> dict:
    """Where the analogues went, as a crude forecast of their own.

    Reported so the analogues can be judged rather than admired. It is a
    nearest-neighbour forecast over a handful of storms and it is not expected
    to beat the model; if it did, that would be the finding.
    """
    out = {}
    for h in LOOKAHEAD:
        dlat, dlon, kt = [], [], []
        for a in found.get("analogues", []):
            at = a["outcome"].get(f"at_{h}h")
            if not at:
                continue
            dlat.append(at["lat"] - a["lat"])
            dlon.append(at["lon"] - a["lon"])
            kt.append(at["vmax_kt"])
        if dlat:
            out[f"{h}h"] = {"lat": lat0 + float(np.mean(dlat)),
                            "lon": lon0 + float(np.mean(dlon)),
                            "vmax_kt": float(np.mean(kt)), "n": len(dlat)}
    return out
