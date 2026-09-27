"""fetch the latest INSAT-3DS full-sector scans onto the model grid.

    python src/ingest/insat_live.py                 last 48 h, once
    python src/ingest/insat_live.py --hours 24
    python src/ingest/insat_live.py --watch 60      refresh every 60 minutes

    python src/ingest/insat_live.py --watch 60 --notify http://127.0.0.1:8010
        the operational shape: fetch, then have the API run the chain on the
        new imagery and push any alert it raises to every open /live page
"""

# the models were trained on GridSat, a delayed archive, so on its own the
# system could only ever describe the past. this is what makes it run on
# today's sky: ISRO's INSAT-3DS L1C through MOSDAC, the full-sector scan at
# each 3-hourly slot the pipeline works on, regridded to the same 0.07 degree
# grid and file naming as GridSat. every model and endpoint then reads it
# unchanged with source=live.
#
# a slot INSAT-3DS doesn't have falls back to INSAT-3DR. MOSDAC publishes a
# scan roughly an hour after it is taken, so the newest slot is often not
# there yet and is simply picked up on the next run.
#
# one login per fetch and a refresh token for the rest. in watch mode a failed
# login ends the loop rather than trying again an hour later - three failures
# lock the account.

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.insat import (SLOT_TOLERANCE_MIN, TokenExpired, download,  # noqa: E402
                          entry_time, get_tokens, is_full_scan, logout,
                          refresh_tokens, search)
from ingest.insat_regrid import to_gridsat                                 # noqa: E402
from vision.detect import gridsat_path                                     # noqa: E402

SATELLITES = [("3SIMG_L1C_ASIA_MER", "INSAT-3DS"), ("3RIMG_L1C_ASIA_MER", "INSAT-3DR")]
LIVE_DIR = ROOT / "data" / "insat" / "live"
RAW_LIVE = ROOT / "data" / "insat" / "raw_live"
STATUS = LIVE_DIR / "status.json"
NETWORK_RETRIES = 3


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def wanted_slots(hours: int, now: datetime) -> list[datetime]:
    # every 3-hourly synoptic slot inside the window, oldest first
    last = now.replace(minute=0, second=0, microsecond=0)
    last -= timedelta(hours=last.hour % 3)
    slots, t = [], last
    while t > now - timedelta(hours=hours):
        slots.append(t)
        t -= timedelta(hours=3)
    return sorted(slots)


def write_scene(raw_path: Path, slot: datetime, satellite: str) -> Path:
    ds = to_gridsat(raw_path)
    ds = ds.assign_coords(time=[np.datetime64(slot, "s")])
    ds.attrs.update({"satellite": satellite, "slot": slot.isoformat(),
                     "note": f"{satellite} L1C full-sector scan resampled to the GridSat 0.07deg grid"})
    out = gridsat_path(LIVE_DIR, slot)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("irwin_cdr", "irwvp")}
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(out)
    return out


def fetch_once(hours: int, log) -> dict:
    now = utcnow()
    slots = wanted_slots(hours, now)
    need = [s for s in slots if not gridsat_path(LIVE_DIR, s).exists()]
    log(f"window {slots[0]:%d %b %H:%M} to {slots[-1]:%d %b %H:%M} UTC: "
        f"{len(slots)} slots, {len(need)} not on disk")

    plan: dict[datetime, tuple[dict, str]] = {}
    if need:
        lo = (min(need) - timedelta(days=1)).date().isoformat()
        hi = (now + timedelta(days=1)).date().isoformat()
        for dataset, label in SATELLITES:
            open_slots = [s for s in need if s not in plan]
            if not open_slots:
                break
            entries = [(entry_time(e), e) for e in search(lo, hi, dataset) if is_full_scan(e)]
            for s in open_slots:
                best = min(entries, key=lambda te: abs(te[0] - s), default=None)
                if best and abs(best[0] - s) <= timedelta(minutes=SLOT_TOLERANCE_MIN):
                    plan[s] = (best[1], label)
        log(f"catalogue has {len(plan)} of them "
            f"({sum(1 for v in plan.values() if v[1] == 'INSAT-3DS')} from INSAT-3DS)")

    if plan:
        log("authenticating (one attempt only) ...")
        tokens = get_tokens()
        try:
            for slot, (entry, label) in sorted(plan.items()):
                raw = None
                for attempt in range(NETWORK_RETRIES + 1):
                    try:
                        download(entry, tokens["access_token"], RAW_LIVE)
                        raw = RAW_LIVE / entry["identifier"]
                        break
                    except TokenExpired:
                        tokens = refresh_tokens(tokens)
                    except requests.RequestException as exc:
                        if attempt == NETWORK_RETRIES:
                            raise
                        log(f"  network error ({type(exc).__name__}), retry {attempt + 1}")
                        time.sleep(10 * (attempt + 1))
                if raw is None:
                    continue
                out = write_scene(raw, slot, label)
                log(f"  {slot:%d %b %H:%M} <- {entry['identifier']}  -> {out.name}")
        finally:
            logout(tokens)

    status = json.loads(STATUS.read_text()) if STATUS.exists() else {"scenes": {}}
    for slot, (entry, label) in plan.items():
        if gridsat_path(LIVE_DIR, slot).exists():
            status["scenes"][slot.isoformat()] = {"satellite": label, "file": entry["identifier"]}
    have = sorted(k for k in status["scenes"]
                  if gridsat_path(LIVE_DIR, datetime.fromisoformat(k)).exists())
    status["scenes"] = {k: status["scenes"][k] for k in have}
    status["latest"] = have[-1] if have else None
    status["fetched_at"] = now.isoformat(timespec="seconds")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=1))
    log(f"live scenes on disk: {len(have)}, latest {status['latest']}")
    return status


def notify(base: str, log) -> None:
    # tell the API a fetch happened, so it runs the chain on the new imagery
    # and pushes whatever it finds to every open alert page. a failure here is
    # never fatal: the imagery is on disk either way, and the fetch loop has to
    # keep its schedule whether or not anything is listening.
    import requests
    try:
        d = requests.post(f"{base.rstrip('/')}/api/live/evaluate", timeout=600).json()
        if not d.get("available"):
            log(f"evaluated: {d.get('reason')}")
            return
        issued = [x for x in d.get("results", []) if x.get("issued")]
        log(f"evaluated {d['scenes_examined']} scene(s), {d['tracks_found']} system(s), "
            f"{len(issued)} alert(s) published")
        for x in issued:
            log(f"  {x['alert']['msgType']} {x['sid']}: {x['reason']}")
    except Exception as exc:                      # noqa: BLE001
        log(f"could not notify {base}: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--watch", type=int, default=0, metavar="MINUTES",
                    help="keep refreshing every N minutes; stops on any login failure")
    ap.add_argument("--notify", metavar="URL", default=None,
                    help="API to run the chain and publish alerts on after each "
                         "fetch, e.g. http://127.0.0.1:8010")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    while True:
        try:
            fetch_once(args.hours, log)
            if args.notify:
                notify(args.notify, log)
        except RuntimeError as exc:
            if "auth" in str(exc).lower() or "refresh" in str(exc).lower():
                log(f"stopping: {exc}")
                sys.exit(1)
            log(f"fetch failed, will try next cycle: {exc}")
        if not args.watch:
            break
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
