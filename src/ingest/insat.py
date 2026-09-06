"""fetch INSAT-3DR imagery from MOSDAC, cropped to the basin.

    python src/ingest/insat.py --probe
    python src/ingest/insat.py --start 2020-05-16 --end 2020-05-20
"""

# own client rather than MOSDAC's mdapi.py - theirs is fine but takes creds
# through a config.json in the working dir, and a password inside the repo is
# one `git add -A` from being public. same endpoints, creds stay in memory.
#
# product is 3RIMG_L1C_ASIA_MER, already geo-corrected and Mercator projected,
# ~1.1 GB/day. the raw L1B full disk is 52 GB/day for the same coverage. sector
# runs 44.5-110E so it misses 40-44.5E of our box - empty ocean off Somalia.
#
# 180 files a day is an 8 min cadence. subsampled to GridSat's 3-hourly grid,
# which drops a day to ~50 MB and keeps the two archives comparable.
#
# ONE auth attempt ever. MOSDAC locks the account for an hour after three
# consecutive failures, so nothing here retries a login.

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.credentials import DEFAULT_MOSDAC_PATH, load, require  # noqa: E402
from ingest.ibtracs import NIO_BOX                                  # noqa: E402

TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"

DATASET = "3RIMG_L1C_ASIA_MER"
RAW_DIR = ROOT / "data" / "insat" / "raw"

# synoptic slots, matching the GridSat 3-hourly grid.
CADENCE_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

# INSAT-3DR repeats a six-scan pattern roughly every half hour - granules land
# at :02, :06, :15, :19, :23 and :28 past, not on the hour. A fixed 10-minute
# window round each slot therefore threw away most of the archive and, worse,
# sometimes matched a granule to the *next* hour. We take the nearest granule to
# each slot instead, rejecting it only when it is more than half a cadence step
# away, which means the slot is genuinely absent from the archive.
SLOT_TOLERANCE_MIN = 40
PAGE_SIZE = 100                # the search endpoint's hard cap per request

_TIME_RE = re.compile(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_")


def bbox_param() -> str:
    # MOSDAC wants west,south,east,north; NIO_BOX is north,west,south,east
    north, west, south, east = NIO_BOX
    return f"{west},{south},{east},{north}"


def search(start: str, end: str, dataset: str = DATASET) -> list[dict]:
    # catalogue query, paginated
    entries: list[dict] = []
    start_index, total = 1, None

    while True:
        r = requests.get(SEARCH_URL, timeout=90, params={
            "datasetId": dataset, "startTime": start, "endTime": end,
            "boundingBox": bbox_param(),
            "startIndex": start_index, "count": PAGE_SIZE})
        if r.status_code != 200:
            try:
                msg = r.json().get("message")
            except ValueError:
                msg = r.text[:200]
            if entries:      # partial results beat none
                print(f"  [warn] page at {start_index} failed ({r.status_code}): {msg}")
                break
            raise RuntimeError(f"search failed ({r.status_code}): {msg}")

        body = r.json()
        page = body.get("entries", []) or []
        if total is None:
            total = int(body.get("totalResults", 0) or 0)
        entries.extend(page)
        if len(page) < PAGE_SIZE or len(entries) >= total:
            break
        start_index += len(page)

    # the same granule can appear on a page boundary; key on the file id.
    seen, unique = set(), []
    for e in entries:
        if e.get("id") in seen:
            continue
        seen.add(e.get("id"))
        unique.append(e)
    return unique


def entry_time(entry: dict) -> datetime | None:
    m = _TIME_RE.search(entry.get("identifier", ""))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")
    except ValueError:
        return None


def subsample(entries: list[dict]) -> list[dict]:
    # keep the file closest to each 3-hourly slot, within tolerance
    dated = [(entry_time(e), e) for e in entries]
    dated = [(t, e) for t, e in dated if t is not None]
    if not dated:
        return []

    # group candidates by (date, nearest slot), then keep the single closest.
    # choosing per slot rather than per granule means a slot yields one file
    # even when several sit inside the tolerance, and none when it is a gap.
    best: dict[tuple, tuple[int, dict]] = {}
    for t, e in dated:
        minutes = t.hour * 60 + t.minute
        slot = min(CADENCE_HOURS, key=lambda h: abs(minutes - h * 60))
        gap = abs(minutes - slot * 60)
        if gap > SLOT_TOLERANCE_MIN:
            continue
        key = (t.date(), slot)
        if key not in best or gap < best[key][0]:
            best[key] = (gap, e)

    return [e for _, (_, e) in sorted(best.items())]


def get_token(creds_path: Path = DEFAULT_MOSDAC_PATH) -> str:
    # authenticate
    creds = load(creds_path)
    user, pwd = require(creds, "username", "password")
    r = requests.post(TOKEN_URL, json={"username": user, "password": pwd},
                      timeout=60)
    if r.status_code != 200:
        try:
            detail = r.json()
            detail = detail.get("message") or detail.get("error") or detail
        except ValueError:
            detail = r.text[:200]
        raise RuntimeError(
            f"MOSDAC auth failed ({r.status_code}): {detail}\n"
            f"NOT retrying - three consecutive failures lock the account for "
            f"an hour. Check the credentials in {creds_path} by hand.")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("auth returned 200 but no access_token")
    return token


def download(entry: dict, token: str, out_dir: Path = RAW_DIR) -> Path | None:
    # fetch one granule
    out_dir.mkdir(parents=True, exist_ok=True)
    name = entry["identifier"]
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return None

    r = requests.get(DOWNLOAD_URL, stream=True, timeout=120,
                     headers={"Authorization": f"Bearer {token}"},
                     params={"id": entry["id"]})
    if r.status_code != 200:
        try:
            detail = r.json()
        except ValueError:
            detail = r.text[:200]
        raise RuntimeError(f"download failed for {name} ({r.status_code}): {detail}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--probe", action="store_true",
                    help="download a single granule to prove the path")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--dry-run", action="store_true",
                    help="search and report only; never authenticates")
    args = ap.parse_args()

    start = args.start or "2020-05-18"
    end = args.end or start

    print(f"dataset {args.dataset}   {start} .. {end}")
    print(f"basin box (W,S,E,N): {bbox_param()}")

    # MOSDAC's date filter is not a clean UTC day - asking for 2020-05-18
    # returns 08:02 to 23:58 and nothing before08:00. Rather than reverse
    # engineer the binning, over-query by a day each side and filter on the
    # timestamp in the filename, which is unambiguous.
    lo = datetime.strptime(start, "%Y-%m-%d").date()
    hi = datetime.strptime(end, "%Y-%m-%d").date()
    pad_lo = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    pad_hi = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    entries = search(pad_lo, pad_hi, args.dataset)
    picked = [e for e in subsample(entries)
              if (t := entry_time(e)) and lo <= t.date() <= hi]
    days = (hi - lo).days + 1
    print(f"  {len(entries)} granules in the padded window, "
          f"{len(picked)} on the 3-hourly grid over {days} day(s) "
          f"({len(picked) / days:.1f}/day of a possible 8)")
    if not picked:
        print("  nothing to do")
        return

    if args.probe:
        picked = picked[:1]
    if args.dry_run:
        for e in picked[:10]:
            print(f"    {e['identifier']}")
        print(f"  dry run - no authentication, nothing downloaded")
        return

    print("authenticating (one attempt only) ...")
    token = get_token()
    print("  token acquired")

    total = 0
    for i, e in enumerate(picked, 1):
        path = download(e, token)
        if path is None:
            print(f"  [{i}/{len(picked)}] {e['identifier']}  already present")
            continue
        mb = path.stat().st_size / 1e6
        total += mb
        print(f"  [{i}/{len(picked)}] {path.name}  {mb:.1f} MB")
    print(f"done - {total:.1f} MB into {RAW_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
