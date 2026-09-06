# coastline geometry: where a track meets land, and how far land is

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COAST_GEOJSON = ROOT / "data" / "static" / "coastline_nio.geojson"

EARTH_R_KM = 6371.0088


# ---------------------------------------------------------------------------
# loading


@lru_cache(maxsize=1)
def _vertices() -> tuple[np.ndarray, np.ndarray]:
    # every coastline vertex as (lat, lon) arrays, radians and degrees
    if not COAST_GEOJSON.exists():
        raise FileNotFoundError(
            f"{COAST_GEOJSON} missing - run `python src/ingest/coastline.py`")
    fc = json.loads(COAST_GEOJSON.read_text(encoding="utf-8"))
    pts = [c for f in fc["features"] for c in f["geometry"]["coordinates"]]
    arr = np.asarray(pts, dtype=np.float64)          # (n, 2) as lon, lat
    return arr[:, 1].copy(), arr[:, 0].copy()        # lat, lon


@lru_cache(maxsize=1)
def _segments() -> list:
    # coastline as shapely LineStrings, for track intersection
    from shapely.geometry import LineString

    fc = json.loads(COAST_GEOJSON.read_text(encoding="utf-8"))
    return [LineString(f["geometry"]["coordinates"]) for f in fc["features"]]


def available() -> bool:
    # true when the vendored coastline is present and loadable
    try:
        _vertices()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# distance


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    # great-circle distance in km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_coast_point(lat: float, lon: float) -> tuple[float, float, float]:
    # nearest coastline vertex: (lat, lon, distance_km)
    clat, clon = _vertices()
    d = haversine_km(lat, lon, clat, clon)
    i = int(np.argmin(d))
    return float(clat[i]), float(clon[i]), float(d[i])


def distance_to_coast(lat: float, lon: float) -> float:
    return nearest_coast_point(lat, lon)[2]


def nearest_coast_bulk(lats, lons, chunk: int = 400):
    # vectorised nearest_coast_point over many points
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    clat, clon = _vertices()

    n = len(lats)
    out_lat = np.full(n, np.nan)
    out_lon = np.full(n, np.nan)
    out_d = np.full(n, np.nan)

    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        ok = np.isfinite(lats[i:j]) & np.isfinite(lons[i:j])
        if not ok.any():
            continue
        d = haversine_km(lats[i:j, None], lons[i:j, None],
                         clat[None, :], clon[None, :])
        d[~ok] = np.inf
        k = np.argmin(d, axis=1)
        out_lat[i:j] = np.where(ok, clat[k], np.nan)
        out_lon[i:j] = np.where(ok, clon[k], np.nan)
        out_d[i:j] = np.where(ok, d[np.arange(j - i), k], np.nan)

    p1, p2 = np.radians(lats), np.radians(out_lat)
    dl = np.radians(out_lon - lons)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return out_lat, out_lon, out_d, bearing


def bearing_to_coast(lat: float, lon: float) -> float:
    # initial bearing to the nearest coast point, degrees clockwise from N
    clat, clon, _ = nearest_coast_point(lat, lon)
    p1, p2 = np.radians(lat), np.radians(clat)
    dl = np.radians(clon - lon)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


# ---------------------------------------------------------------------------
# track intersection


def first_crossing(points: list[tuple[float, float]]
                   ) -> tuple[float, float, float] | None:
    # where an ordered track first meets the coast
    try:
        from shapely.geometry import LineString, Point
    except ImportError:
        # distance features work without shapely; only the intersection needs
        # it. Returning None sends the caller to the snapped-regression
        # fallback rather than taking the whole landfall module down.
        return None

    if len(points) < 2:
        return None

    track = LineString([(lon, lat) for lat, lon in points])
    if track.length == 0:
        return None

    best_frac, best_pt = None, None
    for seg in _segments():
        if not track.intersects(seg):
            continue
        hit = track.intersection(seg)
        for geom in getattr(hit, "geoms", [hit]):
            # A tangential overlap yields a LineString; take its start.
            pt = Point(geom.coords[0]) if geom.geom_type != "Point" else geom
            frac = track.project(pt, normalized=True)
            if best_frac is None or frac < best_frac:
                best_frac, best_pt = frac, pt

    if best_pt is None:
        return None
    return float(best_pt.y), float(best_pt.x), float(best_frac)


def crossing_time_h(frac: float, horizons_h: list[float]) -> float:
    # interpolate hours-to-landfall from a crossing fraction along the track
    stops = np.asarray([0.0] + list(horizons_h), dtype=float)
    n = len(stops)
    if n < 2:
        return float("nan")
    # uniform in vertex index is the best we can do without re-measuring each
    # leg; callers wanting more should pass denser forecast points.
    pos = float(np.clip(frac, 0.0, 1.0)) * (n - 1)
    lo = int(np.floor(pos))
    if lo >= n - 1:
        return float(stops[-1])
    w = pos - lo
    return float(stops[lo] * (1 - w) + stops[lo + 1] * w)
