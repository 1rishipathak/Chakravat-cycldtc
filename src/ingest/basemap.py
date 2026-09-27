# offline basemap: country boundaries as India depicts them, and places
#
#     python src/ingest/basemap.py
#
# the dashboard used OpenStreetMap raster tiles. two problems: it needed the
# network, and OSM draws boundaries by who administers the ground, which is not
# how the Government of India depicts its own boundary - a real problem on a
# screen shown to a MoES panel. natural earth publishes point-of-view editions
# of its country file, and the IND edition draws the boundary as India claims
# it. we vendor a clipped, simplified copy so the map needs no network at all.
#
# populated places come from the same source, for map labels and for the list
# of places a forecast cone covers.

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import Point, box, mapping, shape

ROOT = Path(__file__).resolve().parents[2]

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
COUNTRIES_URL = BASE + "ne_10m_admin_0_countries_ind.geojson"
PLACES_URL = BASE + "ne_10m_populated_places_simple.geojson"

WEB_COUNTRIES = ROOT / "web" / "vendor" / "countries_ind.geojson"
WEB_LABELS = ROOT / "web" / "vendor" / "places_labels.geojson"
PLACES = ROOT / "data" / "static" / "places_nio.geojson"

# what the map can pan to: the basin with room either side
VIEW = box(20.0, -25.0, 125.0, 50.0)
# the rim of the basin, where a forecast cone can reach a populated place
RIM = box(38.0, -5.0, 102.0, 32.0)
SIMPLIFY_DEG = 0.01           # ~1 km, invisible at dashboard zoom
LABEL_MIN_POP = 2_000_000
PLACE_MIN_POP = 50_000

# places that must fall inside India's polygon if this really is the India
# point-of-view edition: Gilgit, Aksai Chin, Arunachal Pradesh.
INDIA_CHECKS = [("Gilgit", 74.31, 35.92), ("Aksai Chin", 79.40, 35.20),
                ("Tawang", 91.87, 27.59)]


def fetch(url: str) -> dict:
    print(f"fetching {url.rsplit('/', 1)[-1]} ...")
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def _round(obj, nd=3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    return obj


def countries() -> None:
    raw = fetch(COUNTRIES_URL)
    india = [f for f in raw["features"] if f["properties"].get("ADM0_A3") == "IND"]
    if not india:
        raise SystemExit("no IND feature - wrong file?")
    geom = shape(india[0]["geometry"])
    for name, lon, lat in INDIA_CHECKS:
        inside = geom.contains(Point(lon, lat))
        print(f"  {name:<11} inside India's polygon: {inside}")
        if not inside:
            raise SystemExit(f"{name} is outside India here - not the India point of view")

    kept = []
    for f in raw["features"]:
        g = shape(f["geometry"])
        if not g.intersects(VIEW):
            continue
        g = g.intersection(VIEW).simplify(SIMPLIFY_DEG, preserve_topology=True)
        if g.is_empty:
            continue
        kept.append({"type": "Feature",
                     "properties": {"name": f["properties"].get("NAME_EN")
                                    or f["properties"].get("NAME"),
                                    "iso": f["properties"].get("ADM0_A3")},
                     "geometry": _round(mapping(g))})
    WEB_COUNTRIES.parent.mkdir(parents=True, exist_ok=True)
    WEB_COUNTRIES.write_text(json.dumps({"type": "FeatureCollection", "features": kept},
                                        separators=(",", ":")), encoding="utf-8")
    print(f"  {len(kept)} countries -> {WEB_COUNTRIES.relative_to(ROOT)} "
          f"({WEB_COUNTRIES.stat().st_size / 1024:.0f} KB)")


def places() -> None:
    raw = fetch(PLACES_URL)
    rim, labels = [], []
    for f in raw["features"]:
        p = f["properties"]
        lon, lat = float(p["longitude"]), float(p["latitude"])
        pop = int(p.get("pop_max") or 0)
        feat = {"type": "Feature",
                "properties": {"name": p["name"], "country": p.get("adm0name"),
                               "region": p.get("adm1name"), "pop": pop,
                               "capital": p.get("featurecla", "")},
                "geometry": {"type": "Point", "coordinates": [round(lon, 3), round(lat, 3)]}}
        if RIM.contains(Point(lon, lat)) and pop >= PLACE_MIN_POP:
            rim.append(feat)
        if VIEW.contains(Point(lon, lat)) and (pop >= LABEL_MIN_POP
                                               or "Admin-0 capital" in feat["properties"]["capital"]):
            labels.append(feat)
    PLACES.parent.mkdir(parents=True, exist_ok=True)
    PLACES.write_text(json.dumps({"type": "FeatureCollection", "features": rim},
                                 separators=(",", ":")), encoding="utf-8")
    WEB_LABELS.write_text(json.dumps({"type": "FeatureCollection", "features": labels},
                                     separators=(",", ":")), encoding="utf-8")
    print(f"  {len(rim)} rim places -> {PLACES.relative_to(ROOT)} "
          f"({PLACES.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(labels)} map labels -> {WEB_LABELS.relative_to(ROOT)}")


if __name__ == "__main__":
    countries()
    places()
    sys.exit(0)
