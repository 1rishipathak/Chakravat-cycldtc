# who a forecast reaches: the threat zone as a polygon, and the places in it
#
# the cone the dashboard draws is a circle per lead time. what matters on the
# ground is the area between them too - a town can sit between the 24 h and
# 48 h circles and still be in the storm's path. so the zone here is swept:
# position and radius are interpolated along the forecast track.
#
# a cone only says where the storm's *centre* may go. the first version stopped
# there and missed Kolkata for Amphan with the centre 52 km away, because a
# storm is not a point. so each node's radius is the cone radius plus the
# typical radius of gale-force winds for the forecast intensity, measured below.
#
# places are natural earth populated places around the basin rim, with its
# metro population estimates. they say who is in the path; they are not a
# damage or exposure model and are not quoted as one.

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
PLACES = ROOT / "data" / "static" / "places_nio.geojson"
STEPS = 24            # samples per segment between lead times
MIN_POP = 100_000

# median radius of 34 kt winds by intensity band, from JTWC wind radii in
# IBTrACS for North Indian Ocean storms 2004-2025 (2,478 fixes, 107 storms,
# mean of the four quadrants). below 34 kt there are no gales to extend by.
GALE_RADIUS_KM = [(34, 0.0), (48, 106.0), (64, 130.0), (90, 162.0), (999, 201.0)]


def gale_radius_km(vmax_kt: float) -> float:
    for upper, radius in GALE_RADIUS_KM:
        if vmax_kt < upper:
            return radius
    return GALE_RADIUS_KM[-1][1]


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def track_nodes(current: dict, forecast: list[dict]) -> list[dict]:
    # issue position (no position uncertainty yet), then every lead time;
    # each radius is cone radius plus the gale radius for that intensity
    nodes = [{"h": 0.0, "lat": current["lat"], "lon": current["lon"],
              "r": gale_radius_km(current["vmax_kt"])}]
    for p in forecast:
        nodes.append({"h": float(p["horizon_h"]), "lat": p["lat"], "lon": p["lon"],
                      "r": float(p["radius_km"]) + gale_radius_km(p["vmax_kt"])})
    return nodes


def _circle(lat: float, lon: float, km: float, n: int = 48) -> Polygon:
    km = max(km, 1.0)
    dlat = km / 110.574
    dlon = km / (111.320 * max(math.cos(math.radians(lat)), 0.05))
    return Polygon([(lon + dlon * math.cos(2 * math.pi * i / n),
                     lat + dlat * math.sin(2 * math.pi * i / n)) for i in range(n)])


def cone_polygon(nodes: list[dict], max_points: int = 120) -> Polygon:
    # union of hulls between consecutive circles, in lon/lat degrees
    circles = [_circle(n["lat"], n["lon"], n["r"]) for n in nodes]
    hulls = [circles[i].union(circles[i + 1]).convex_hull for i in range(len(circles) - 1)]
    shape = unary_union(hulls)
    if shape.geom_type != "Polygon":
        shape = shape.convex_hull
    tol = 0.01
    while len(shape.exterior.coords) > max_points and tol < 1.0:
        shape = shape.simplify(tol, preserve_topology=True)
        tol *= 2
    return shape


@lru_cache(maxsize=1)
def _places() -> list[dict]:
    if not PLACES.exists():
        return []
    fc = json.loads(PLACES.read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"]
        out.append({**f["properties"], "lat": lat, "lon": lon})
    return out


def places_in_cone(nodes: list[dict], min_pop: int = MIN_POP) -> list[dict]:
    # every place the swept cone covers, with the first lead time it does
    hits = []
    for place in _places():
        if place["pop"] < min_pop:
            continue
        first, closest = None, math.inf
        for a, b in zip(nodes, nodes[1:]):
            for k in range(STEPS + 1):
                t = k / STEPS
                lat = a["lat"] + t * (b["lat"] - a["lat"])
                lon = a["lon"] + t * (b["lon"] - a["lon"])
                r = a["r"] + t * (b["r"] - a["r"])
                d = _haversine_km(place["lat"], place["lon"], lat, lon)
                closest = min(closest, d)
                if first is None and d <= r:
                    first = a["h"] + t * (b["h"] - a["h"])
        if first is not None:
            hits.append({"name": place["name"], "country": place["country"],
                         "region": place["region"], "population": place["pop"],
                         "lat": place["lat"], "lon": place["lon"],
                         "first_inside_h": round(first, 1),
                         "closest_approach_km": round(closest)})
    hits.sort(key=lambda p: (p["first_inside_h"], -p["population"]))
    return hits
