"""Probability that a place sees damaging wind, not just where the centre goes.

The cone answers "where might the eye be". Nobody is hurt by an eye. The
question a district officer asks is whether *this* town gets gale-force wind,
and the cone cannot answer it for two reasons: it describes the centre, and it
describes one position per lead time rather than the whole passage.

This answers it the way operational centres do. Each of the eight ensemble
members is a plausible future, already fitted by bootstrapping whole storms.
Sweep each member's own wind field along its own track, hour by hour, and ask
how many of the eight cover a given point at any time in the window. Eight
members give probabilities in eighths - coarse, and honestly reported as such.

The wind field size comes from reports/wind_radii.json, measured from JTWC wind
radii in IBTrACS and binned on the same 3-minute wind the forecast produces.
The field is taken as circular. It is not: the right-forward quadrant is
normally the largest because the storm's own motion adds to its circulation.
Using the quadrant mean spreads that asymmetry evenly, which makes the swath a
little too wide on one side and too narrow on the other, and the verification
is what says whether that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.impact import gale_radius_km

THRESHOLDS = (34, 50, 64)
STEP_H = 1.0                 # how finely a member's track is walked
KM_PER_DEG = 111.32


def member_tracks(ensemble, row: pd.DataFrame, horizons) -> list[list[tuple]]:
    """Each member's own future: (hours, lat, lon, vmax) including the start.

    The ensemble normally collapses to a mean and a spread. The swath needs the
    members themselves, because a wind field has to be swept along a coherent
    track - averaging first and sweeping after would draw one fat storm instead
    of eight plausible ones.
    """
    lat0 = float(row["LAT"].iloc[0])
    lon0 = float(row["LON"].iloc[0])
    v0 = float(row["vmax_kt"].iloc[0])

    tracks = []
    for m in ensemble.members:
        pts = [(0.0, lat0, lon0, v0)]
        for h in horizons:
            dlat = float(m.predict(row, f"y_dlat_{h}")[0])
            dlon = float(m.predict(row, f"y_dlon_{h}")[0])
            dv = float(m.predict(row, f"y_dv_{h}")[0])
            pts.append((float(h), lat0 + dlat, lon0 + dlon, max(v0 + dv, 0.0)))
        tracks.append(pts)
    return tracks


def densify(pts: list[tuple], step_h: float = STEP_H) -> list[tuple]:
    """Walk a member's track at a finer step than the forecast lead times.

    Without this the swath is a string of beads: a storm moving 25 km/h travels
    300 km between the 24 and 36 hour nodes and everything between them would
    be missed.
    """
    out = []
    for (h0, la0, lo0, v0), (h1, la1, lo1, v1) in zip(pts, pts[1:]):
        n = max(int(round((h1 - h0) / step_h)), 1)
        for k in range(n):
            f = k / n
            out.append((h0 + f * (h1 - h0), la0 + f * (la1 - la0),
                        lo0 + f * (lo1 - lo0), v0 + f * (v1 - v0)))
    out.append(pts[-1])
    return out


def basin_grid(deg: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arange(-5.0, 31.0 + 1e-9, deg)
    lon = np.arange(34.0, 100.0 + 1e-9, deg)
    return lat, lon


def probability_field(tracks: list[list[tuple]], lat: np.ndarray, lon: np.ndarray,
                      thresholds=THRESHOLDS, step_h: float = STEP_H) -> dict:
    """Share of members whose wind field covers each point at any time.

    Distances use an equirectangular approximation rather than haversine: the
    radii involved are under 250 km, where the two agree to well inside the
    precision of the radii themselves, and this runs over the whole basin grid
    in one array operation per step.
    """
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cos_lat = np.cos(np.radians(LA))
    counts = {int(t): np.zeros(LA.shape, dtype=np.int32) for t in thresholds}

    for pts in tracks:
        hit = {int(t): np.zeros(LA.shape, dtype=bool) for t in thresholds}
        for _h, la, lo, v in densify(pts, step_h):
            radii = {int(t): gale_radius_km(v, int(t)) for t in thresholds}
            biggest = max(radii.values())
            if biggest <= 0:
                continue
            dy = (LA - la) * KM_PER_DEG
            dx = (LO - lo) * KM_PER_DEG * cos_lat
            d2 = dy * dy + dx * dx
            for t, r in radii.items():
                if r > 0:
                    hit[t] |= d2 <= r * r
        for t in counts:
            counts[t] += hit[t]

    n = max(len(tracks), 1)
    return {t: counts[t] / n for t in counts}


def contours(prob: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             levels=(0.125, 0.25, 0.5, 0.75)) -> list[dict]:
    """Probability contours as GeoJSON-ready rings.

    Levels default to eighths because eight members cannot resolve anything
    finer, and a contour at 0.1 would imply a precision we do not have.
    """
    from contourpy import contour_generator

    gen = contour_generator(x=lon, y=lat, z=prob, fill_type="OuterCode",
                            line_type="SeparateCode")
    out = []
    for lv in levels:
        if prob.max() < lv:
            continue
        lines, _codes = gen.lines(lv)
        rings = [[[round(float(x), 3), round(float(y), 3)] for x, y in seg]
                 for seg in lines if len(seg) >= 4]
        if rings:
            out.append({"level": float(lv), "rings": rings})
    return out


def summarise(field: dict, lat: np.ndarray, lon: np.ndarray) -> dict:
    """The shape of the answer, without shipping the whole grid."""
    deg = float(lat[1] - lat[0])
    out = {}
    for t, p in field.items():
        # area is latitude dependent; a degree of longitude shrinks polewards
        cell = (deg * KM_PER_DEG) ** 2 * np.cos(np.radians(lat))[:, None]
        out[str(t)] = {
            "max_probability": float(p.max()),
            "area_km2_at_25pct": float((cell * (p >= 0.25)).sum()),
            "area_km2_at_50pct": float((cell * (p >= 0.50)).sum()),
        }
    return out
