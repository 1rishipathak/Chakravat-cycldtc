"""mirror the GridSat archive with INSAT full-sector scans at the same slots.

    python src/ingest/insat_archive.py --satellite 3R --dry-run
    python src/ingest/insat_archive.py --satellite 3R
    python src/ingest/insat_archive.py --satellite 3D
"""

# every GridSat scene on disk inside a satellite's era gets the INSAT full
# sector scan nearest its slot. same storms, same times, two sensors, so any
# difference between a GridSat-trained and an INSAT-trained model is the
# sensor and nothing else.
#
# 3DR covers 2017-2025. 3D goes back to 2014 and extends the INSAT record by
# three seasons. both are ~26-30 MB a granule for the full sector.
#
# the catalogue is searched once and cached in a manifest, so a rerun after an
# interrupted download only authenticates and fetches what is missing. one
# login per run, never retried - three failures lock the account. an expired
# access token is renewed with the refresh token, which needs no password.

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.insat import (RAW_DIR, SLOT_TOLERANCE_MIN, TokenExpired,  # noqa: E402
                          download, entry_time, get_tokens, is_full_scan, logout,
                          refresh_tokens, search)

GRIDSAT = ROOT / "data" / "gridsat"
NETWORK_RETRIES = 3
SATELLITES = {
    "3R": {"dataset": "3RIMG_L1C_ASIA_MER", "years": (2017, 2025)},
    "3D": {"dataset": "3DIMG_L1C_ASIA_MER", "years": (2014, 2016)},
}
AVG_MB = 27.0


def gridsat_slots(years: tuple[int, int]) -> list[datetime]:
    out = []
    for path in GRIDSAT.rglob("nio_*.nc"):
        date, hour = path.stem.replace("nio_", "").split("_")
        t = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), int(hour))
        if years[0] <= t.year <= years[1]:
            out.append(t)
    return sorted(out)


def day_runs(slots: list[datetime]) -> list[tuple[datetime, datetime]]:
    # contiguous runs of calendar days that hold at least one slot
    days = sorted({s.date() for s in slots})
    runs, start, prev = [], None, None
    for d in days:
        if start is None:
            start = prev = d
        elif (d - prev).days <= 1:
            prev = d
        else:
            runs.append((start, prev))
            start = prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


def build_manifest(sat: str, slots: list[datetime], path: Path, log) -> dict:
    manifest = json.loads(path.read_text()) if path.exists() else {}
    dataset = SATELLITES[sat]["dataset"]
    runs = day_runs(slots)
    log(f"{len(slots)} GridSat slots in {len(runs)} day runs")

    for k, (d0, d1) in enumerate(runs, 1):
        in_run = [s for s in slots if d0 <= s.date() <= d1]
        if all(s.isoformat() in manifest for s in in_run):
            continue
        # MOSDAC's date filter is not a clean UTC day, so pad a day each side
        # and match on the timestamp in the file name instead.
        lo = (d0 - timedelta(days=1)).isoformat()
        hi = (d1 + timedelta(days=1)).isoformat()
        try:
            entries = [e for e in search(lo, hi, dataset) if is_full_scan(e)]
        except Exception as exc:  # noqa: BLE001
            log(f"  run {k}/{len(runs)} {d0}..{d1}: search failed ({exc}); will retry next run")
            continue
        dated = [(entry_time(e), e) for e in entries]
        found = 0
        for s in in_run:
            best = min(dated, key=lambda te: abs(te[0] - s), default=None)
            if best and abs(best[0] - s) <= timedelta(minutes=SLOT_TOLERANCE_MIN):
                manifest[s.isoformat()] = {"identifier": best[1]["identifier"],
                                           "id": best[1]["id"]}
                found += 1
            else:
                manifest[s.isoformat()] = None
        log(f"  run {k}/{len(runs)} {d0}..{d1}: {len(entries)} full scans, "
            f"{found}/{len(in_run)} slots matched")
        path.write_text(json.dumps(manifest, indent=1))
        time.sleep(0.5)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--satellite", choices=sorted(SATELLITES), default="3R")
    ap.add_argument("--dry-run", action="store_true",
                    help="search and report only; never authenticates")
    args = ap.parse_args()

    sat = SATELLITES[args.satellite]
    manifest_path = ROOT / "data" / "insat" / f"manifest_{args.satellite}.json"
    log_path = ROOT / "data" / "insat" / f"archive_{args.satellite}.log"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== {sat['dataset']} for GridSat seasons {sat['years'][0]}-{sat['years'][1]}")
    slots = gridsat_slots(sat["years"])
    manifest = build_manifest(args.satellite, slots, manifest_path, log)

    wanted = [v for k, v in sorted(manifest.items()) if v]
    have = [v for v in wanted if (RAW_DIR / v["identifier"]).exists()]
    todo = [v for v in wanted if not (RAW_DIR / v["identifier"]).exists()]
    missing = sum(1 for v in manifest.values() if v is None)
    log(f"slots {len(manifest)}   matched {len(wanted)}   no full scan within "
        f"{SLOT_TOLERANCE_MIN} min {missing}")
    log(f"already on disk {len(have)}   to download {len(todo)}  "
        f"(~{len(todo) * AVG_MB / 1000:.1f} GB)")
    if args.dry_run or not todo:
        log("dry run - no authentication, nothing downloaded" if args.dry_run else "nothing to do")
        return

    log("authenticating (one attempt only) ...")
    tokens = get_tokens()
    log("token acquired")
    t0, total, refreshes = time.time(), 0.0, 0
    try:
        for i, entry in enumerate(todo, 1):
            try:
                path = None
                for attempt in range(NETWORK_RETRIES + 1):
                    try:
                        path = download(entry, tokens["access_token"])
                        break
                    except TokenExpired:
                        # swap the refresh token for a new pair - no password,
                        # so this can't count toward the lockout
                        tokens = refresh_tokens(tokens)
                        refreshes += 1
                    except requests.RequestException as exc:
                        # a dropped connection is not an auth failure; retrying
                        # the file is safe. anything else stops the run.
                        if attempt == NETWORK_RETRIES:
                            raise
                        log(f"  network error on {entry['identifier']} ({type(exc).__name__}), "
                            f"retry {attempt + 1}/{NETWORK_RETRIES}")
                        time.sleep(10 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                log(f"stopping at {i}/{len(todo)}, not re-authenticating: {exc}")
                log("rerun the same command later; finished files are skipped")
                return
            if path is not None:
                total += path.stat().st_size / 1e6
            if i % 25 == 0 or i == len(todo):
                rate = total / max(time.time() - t0, 1)
                log(f"  {i}/{len(todo)}  {total / 1000:.1f} GB  {rate:.1f} MB/s  "
                    f"(token refreshed {refreshes}x)")
    finally:
        logout(tokens)
    log(f"done - {total / 1000:.1f} GB into {RAW_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
