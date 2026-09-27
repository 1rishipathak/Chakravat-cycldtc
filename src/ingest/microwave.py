"""89 GHz brightness temperature, storm-centred, for every patch that has a pass.

    python src/ingest/microwave.py [--workers 4] [--limit 0]

Infrared sees the top of the cloud. At 89 GHz the atmosphere is largely
transparent and what the radiometer sees instead is ice scattering: precipitation
sized ice aloft knocks the brightness temperature down, so a deep eyewall prints
as a cold ring even when a cirrus canopy has hidden it completely from the
infrared window. That is the mechanism behind the two failures we have measured
and cannot fix by any other means - the severe-storm under-read, and the collapse
of the EMBC and IRRCDO scene classes on INSAT.

The polarisation corrected temperature, PCT = 1.818 V - 0.818 H, is the standard
way to use it: over ocean the surface is strongly polarised and cloud is not, so
the combination suppresses the sea and leaves the scattering signal.

Getting the data is the hard part, and not for the reason you would guess. These
are polar orbiters with narrow swaths, so a granule is a full orbit of a few
hundred megabytes and we need hundreds of them. Downloading whole granules would
be something like 100 GB. Instead each one is subset twice through OPeNDAP:
once at stride 10 to find which scan line crosses the storm, costing about 70 kB,
then once at full resolution for a window of scans around it, costing under a
megabyte. That turns the job from 100 GB into well under one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
CACHE = ROOT / "data" / "processed" / "scenes"
OUT = ROOT / "data" / "processed" / "microwave"

# group and channel indices of the ~89 GHz V and H pair in each 1C product,
# read off the DMRs rather than assumed
SENSORS = {
    "GMI": ("C2259345484-GES_DISC", "S1", 7, 8),
    "AMSR2": ("C2264132976-GES_DISC", "S5", 0, 1),
    "SSMIS-F16": ("C2264132881-GES_DISC", "S4", 0, 1),
    "SSMIS-F17": ("C2264132902-GES_DISC", "S4", 0, 1),
    "SSMIS-F18": ("C2264132936-GES_DISC", "S4", 0, 1),
}

WINDOW_MIN = 90          # a pass this far from the patch time still describes it
GRID_PX = 128
GRID_DEG = 0.05          # about 5.5 km, near the resolution of the instrument
HALF = GRID_PX * GRID_DEG / 2
MAX_CENTRE_DEG = 0.6     # if the swath misses the centre by more than this, skip
STRIDE = 10


def sensor_of(title: str) -> str | None:
    for name in SENSORS:
        if "-" in name and name.split("-")[-1] in title:
            return name
    for name in SENSORS:
        if "-" not in name and name in title:
            return name
    return None


def find_passes(when: pd.Timestamp, lat: float, lon: float) -> list[dict]:
    lo = (when - pd.Timedelta(minutes=WINDOW_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (when + pd.Timedelta(minutes=WINDOW_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = [("collection_concept_id", c) for c, _, _, _ in SENSORS.values()]
    params += [("temporal", f"{lo},{hi}"),
               ("bounding_box", f"{lon-3},{lat-3},{lon+3},{lat+3}"),
               ("page_size", "20")]
    req = urllib.request.Request(CMR + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": "chakravat-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        entries = json.load(r)["feed"]["entry"]

    out = []
    for e in entries:
        name = sensor_of(e.get("title", ""))
        link = next((l["href"] for l in e.get("links", [])
                     if "opendap" in l.get("href", "")), None)
        if not name or not link:
            continue
        start = pd.Timestamp(e["time_start"].replace("Z", ""))
        out.append({"sensor": name, "url": link, "start": start,
                    "gap_min": abs((start - when).total_seconds()) / 60.0})
    return sorted(out, key=lambda d: d["gap_min"])


def _read(session, url: str, ce: str):
    import h5py
    r = session.get(f"{url}.dap.nc4?dap4.ce=" + urllib.parse.quote(ce, safe=""),
                    timeout=300)
    if r.status_code != 200:
        return None
    fd, path = tempfile.mkstemp(suffix=".nc4")
    os.write(fd, r.content)
    os.close(fd)
    try:
        with h5py.File(path, "r") as f:
            return {k: f[k][:] for k in _walk(f)}
    finally:
        os.unlink(path)


def _walk(g, prefix=""):
    import h5py
    for k in g:
        item = g[k]
        if isinstance(item, h5py.Dataset):
            yield f"{prefix}/{k}"
        else:
            yield from _walk(item, f"{prefix}/{k}")


_DIMS: dict[str, tuple[int, int]] = {}


def dims_of(session, url: str, group: str) -> tuple[int, int] | None:
    """Scan and pixel counts for a group, from the granule's own metadata.

    DAP4 rejects an index past the end rather than clamping it, so every
    constraint expression has to carry real bounds and the only place to get
    them is the DMR. It is 40 kB and cached per granule, which matters because
    one granule often serves several patches of the same storm.
    """
    import re

    key = f"{url}|{group}"
    if key in _DIMS:
        return _DIMS[key]
    r = session.get(url + ".dmr", timeout=180)
    if r.status_code != 200:
        return None
    text = r.text
    m = re.search(rf'<Group name="{group}">', text)
    if not m:
        return None
    nxt = text.find('<Group name=', m.end())
    body = text[m.end(): nxt if nxt > 0 else len(text)]
    lat = re.search(r'<Float32 name="Latitude">(.*?)</Float32>', body, re.S)
    if not lat:
        return None
    sizes = [int(x) for x in re.findall(r'<Dim size="(\d+)"/>', lat.group(1))]
    if len(sizes) < 2:
        return None
    _DIMS[key] = (sizes[0], sizes[1])
    return _DIMS[key]


def grid_pass(session, url: str, group: str, iv: int, ih: int,
              lat0: float, lon0: float) -> np.ndarray | None:
    """One overpass resampled onto a storm-centred grid of PCT, or None."""
    from scipy.spatial import cKDTree

    shape = dims_of(session, url, group)
    if not shape:
        return None
    nscan, npix = shape[0] - 1, shape[1] - 1

    coarse = _read(session, url,
                   f"/{group}/Latitude[0:{STRIDE}:{nscan}][0:{STRIDE}:{npix}];"
                   f"/{group}/Longitude[0:{STRIDE}:{nscan}][0:{STRIDE}:{npix}]")
    if not coarse:
        return None
    lat = next(v for k, v in coarse.items() if k.endswith("Latitude"))
    lon = next(v for k, v in coarse.items() if k.endswith("Longitude"))
    d = np.hypot(lat - lat0, (lon - lon0) * np.cos(np.radians(lat0)))
    if not np.isfinite(d).any() or np.nanmin(d) > MAX_CENTRE_DEG * 3:
        return None
    i, _ = np.unravel_index(np.nanargmin(d), d.shape)
    centre = i * STRIDE
    lo, hi = max(centre - 200, 0), min(centre + 200, nscan)

    scan_time = ";".join(
        f"/{group}/ScanTime/{f}[{lo}:1:{hi}]"
        for f in ("Year", "Month", "DayOfMonth", "Hour", "Minute", "Second"))
    fine = _read(session, url,
                 f"/{group}/Latitude[{lo}:1:{hi}][0:1:{npix}];"
                 f"/{group}/Longitude[{lo}:1:{hi}][0:1:{npix}];"
                 f"/{group}/Tc[{lo}:1:{hi}][0:1:{npix}][{iv}:1:{ih}];"
                 + scan_time)
    if not fine:
        return None
    flat = next(v for k, v in fine.items() if k.endswith("Latitude"))
    flon = next(v for k, v in fine.items() if k.endswith("Longitude"))
    tc = next(v for k, v in fine.items() if k.endswith("Tc"))
    if tc.ndim != 3 or tc.shape[-1] < 2:
        return None

    v, h = tc[..., 0].astype(np.float32), tc[..., 1].astype(np.float32)
    v[v < 50], h[h < 50] = np.nan, np.nan
    pct = 1.818 * v - 0.818 * h                 # suppress the polarised sea

    good = np.isfinite(pct) & np.isfinite(flat) & np.isfinite(flon)
    if good.sum() < 500:
        return None

    # which scan actually looked at the storm, and when
    dd = np.hypot(flat - lat0, (flon - lon0) * np.cos(np.radians(lat0)))
    dd = np.where(np.isfinite(dd), dd, np.inf)
    si, _ = np.unravel_index(np.argmin(dd), dd.shape)
    observed = None
    try:
        parts = {f: next(v for k, v in fine.items() if k.endswith("/" + f))
                 for f in ("Year", "Month", "DayOfMonth", "Hour", "Minute", "Second")}
        observed = pd.Timestamp(
            year=int(parts["Year"][si]), month=int(parts["Month"][si]),
            day=int(parts["DayOfMonth"][si]), hour=int(parts["Hour"][si]),
            minute=int(parts["Minute"][si]), second=int(parts["Second"][si]))
    except Exception:                                   # noqa: BLE001
        observed = None
    py, px = flat[good], flon[good]
    if np.hypot(py - lat0, (px - lon0) * np.cos(np.radians(lat0))).min() > MAX_CENTRE_DEG:
        return None                              # swath came near but missed

    gy = lat0 + (np.arange(GRID_PX) - GRID_PX / 2 + 0.5) * GRID_DEG
    gx = lon0 + (np.arange(GRID_PX) - GRID_PX / 2 + 0.5) * GRID_DEG
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    tree = cKDTree(np.column_stack([py, px * np.cos(np.radians(lat0))]))
    dist, idx = tree.query(np.column_stack([GY.ravel(),
                                            GX.ravel() * np.cos(np.radians(lat0))]))
    out = pct[good][idx].astype(np.float32)
    out[dist > 0.12] = np.nan                    # do not invent data past a gap
    return out.reshape(GRID_PX, GRID_PX), observed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=str(CACHE))
    args = ap.parse_args()

    from ingest.earthdata import session

    cache = Path(args.cache)
    meta = pd.read_csv(cache / "patches.csv")
    meta = meta[meta["bt_vmax_kt"].notna()].reset_index(drop=True)
    meta["time"] = pd.to_datetime(meta["time"])
    meta = meta[meta["time"].dt.year >= 2005].reset_index(drop=True)
    if args.limit:
        meta = meta.head(args.limit)

    OUT.mkdir(parents=True, exist_ok=True)
    done_path = OUT / "index.csv"
    done = (pd.read_csv(done_path) if done_path.exists()
            else pd.DataFrame(columns=["row", "storm", "time", "sensor",
                                       "gap_min", "observed", "file",
                                       "pct_min", "filled"]))
    have = set(done["row"]) if len(done) else set()
    todo = [r for r in meta.itertuples() if r.Index not in have]
    print(f"{len(meta)} labelled patches from 2005 on, {len(todo)} still to fetch")

    lock_rows: list[dict] = []
    t0 = time.time()

    def one(r) -> dict | None:
        s = session()
        try:
            passes = find_passes(pd.Timestamp(r.time), float(r.lat), float(r.lon))
        except Exception:                                   # noqa: BLE001
            return None
        for p in passes[:4]:
            _, group, iv, ih = SENSORS[p["sensor"]]
            try:
                got = grid_pass(s, p["url"], group, iv, ih,
                                float(r.lat), float(r.lon))
            except Exception:                               # noqa: BLE001
                continue
            if got is None:
                continue
            grid, observed = got
            if grid is None or not np.isfinite(grid).any():
                continue
            gap = (abs((observed - pd.Timestamp(r.time)).total_seconds()) / 60.0
                   if observed is not None else p["gap_min"])
            if gap > WINDOW_MIN:
                continue          # the orbit was in range, this scan was not
            name = f"{r.storm}_{pd.Timestamp(r.time):%Y%m%d%H%M}_{p['sensor']}.npy"
            np.save(OUT / name, grid)
            return {"row": int(r.Index), "storm": r.storm, "time": str(r.time),
                    "sensor": p["sensor"], "gap_min": round(gap, 1),
                    "observed": str(observed) if observed is not None else "",
                    "file": name, "pct_min": float(np.nanmin(grid)),
                    "filled": float(np.isfinite(grid).mean())}
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, r): r.Index for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                got = fut.result()
            except Exception:                               # noqa: BLE001
                got = None
            if got:
                lock_rows.append(got)
            if n % 20 == 0 or n == len(todo):
                hit = len(lock_rows)
                print(f"  {n}/{len(todo)}  {hit} with imagery  "
                      f"{time.time()-t0:.0f}s", flush=True)
                if lock_rows:
                    pd.concat([done, pd.DataFrame(lock_rows)]).to_csv(
                        done_path, index=False)

    final = pd.concat([done, pd.DataFrame(lock_rows)]) if lock_rows else done
    final.to_csv(done_path, index=False)
    print(f"\n{len(final)} patches now have 89 GHz imagery, in {OUT}")
    if len(final):
        print(f"median time from the patch: {final['gap_min'].median():.0f} min")
        print(f"by instrument: {final['sensor'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
