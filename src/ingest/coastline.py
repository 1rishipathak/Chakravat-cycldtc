# coastline for the North Indian Ocean basin

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.ibtracs import NIO_BOX  # noqa: E402

OUT = ROOT / "data" / "static" / "coastline_nio.geojson"

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_10m_coastline.geojson")

# A margin so a storm that lands just outside the box still meets a coast, and
# so line segments crossing the boundary are not clipped into open ends.
MARGIN_DEG = 3.0


def _bounds() -> tuple[float, float, float, float]:
    north, west, south, east = NIO_BOX      # CDS ordering, as in ingest.era5
    return (west - MARGIN_DEG, south - MARGIN_DEG,
            east + MARGIN_DEG, north + MARGIN_DEG)


def _segments_in_box(coords, w, s, e, n):
    # split a LineString at box exits, keeping the parts inside
    runs, cur = [], []
    for lon, lat in coords:
        if w <= lon <= e and s <= lat <= n:
            cur.append([lon, lat])
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def main() -> None:
    w, s, e, n = _bounds()
    print(f"basin box with {MARGIN_DEG}deg margin: "
          f"{s:.0f}-{n:.0f}N, {w:.0f}-{e:.0f}E")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL.rsplit('/', 1)[-1]} ...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        raw = json.loads(r.read().decode("utf-8"))
    print(f"  {len(raw['features'])} global coastline features")

    kept = []
    for feat in raw["features"]:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            continue
        for part in parts:
            for run in _segments_in_box(part, w, s, e, n):
                kept.append({"type": "Feature", "properties": {},
                             "geometry": {"type": "LineString",
                                          "coordinates": run}})

    if not kept:
        raise SystemExit("no coastline inside the basin box - check NIO_BOX")

    fc = {"type": "FeatureCollection", "features": kept}
    OUT.write_text(json.dumps(fc), encoding="utf-8")

    verts = sum(len(f["geometry"]["coordinates"]) for f in kept)
    print(f"  {len(kept)} segments, {verts} vertices")
    print(f"wrote {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
