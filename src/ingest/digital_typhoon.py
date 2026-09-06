"""selective download of the Digital Typhoon pretraining corpus."""

# the mirrors are all-or-nothing - AU is 32 GB on Kaggle, WP is 89 GB. three
# knobs cut it to a couple of GB without hurting the pretraining signal:
#
# PNG not HDF5. the H5 files hold calibrated brightness temperature; the PNGs
# are the same imagery, much smaller. a vision trunk learns cloud morphology,
# which survives the conversion. radiometric precision only matters when
# fine-tuning on INSAT-3D, and that comes from MOSDAC.
#
# subsample in time. Digital Typhoon is hourly and consecutive frames of one
# cyclone are near duplicates - the same redundancy that makes random splits
# leak. every 6th frame matches our synoptic cadence and cuts volume ~6x.
#
# restrict the years. storm diversity matters more than frame count.
#
# use AU. smaller mirror but the only one with per-storm metadata, so the only
# one with labels - WP ships exactly 2x its image count, so H5 + PNG and
# nothing else. AU cyclones spin clockwise, so mirror vertically to match North
# Indian rotation. one line of augmentation, cheaper than having no labels.
#
# reads the per-storm metadata CSVs (~9 kB each) instead of enumerating 140k
# files 200 at a time. labels come across too, so the pretraining set supports
# storm-level splits like the North Indian data does.
#
# auth is required even though the data is public. anonymous requests succeed
# about half the time on files that definitely exist, and pacing makes it
# worse, so it is not throttling. authenticated was 8/8 in testing. put a
# kaggle.json (username AND key) in ~/.kaggle/.

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

BASINS = {
    "AU": "bwandowando/digital-typhoon-dataset-au-1979-2024",
    "WP": "bwandowando/digital-typhoon-dataset-western-pacific-wp",
}
YEAR_RANGE = {"AU": (1979, 2024), "WP": (1978, 2023)}

BYTES_PER_PNG = 150_000          # dry-run estimate only
MAX_STORMS_PER_SEASON = 20
# stop a season only after several consecutive misses AND enough of the
# numbering scanned. Breaking at the first small gap truncated seasons badly:
# an early version found 12 storms across four seasons in a basin that averages
# about ten per season.
MISSES_BEFORE_NEXT_SEASON = 5
MIN_SCAN_PER_SEASON = 14

_local = threading.local()
_quota_lock = threading.Lock()
RATE_LIMIT_PAUSE = 90        # seconds to wait out an HTTP 429


def _api():
    # One authenticated client per thread; KaggleApi is not thread-safe
    api = getattr(_local, "api", None)
    if api is not None:
        return api
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("pip install kaggle")
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Kaggle authentication failed: {exc}\n\n"
            "Kaggle Settings -> API -> Create New Token, then save kaggle.json to\n"
            f"  {Path.home() / '.kaggle' / 'kaggle.json'}"
        )
    _local.api = api
    return api


class _DropBanners:
    # filter the Kaggle client's per-call chatter out of stdout

    NOISE = ("Dataset URL:", "Downloading ", "Resuming ")

    def __init__(self, wrapped):
        self._w = wrapped
        self._suppressed = False

    def write(self, text):
        if any(text.lstrip().startswith(n) for n in self.NOISE):
            self._suppressed = True
            return
        # swallow the bare newline the client emits after a banner, otherwise
        # the run is padded with blank lines.
        if self._suppressed and text.strip() == "":
            self._suppressed = False
            return
        self._suppressed = False
        self._w.write(text)

    def __getattr__(self, name):
        return getattr(self._w, name)


def _fetch(dataset: str, remote: str, dest_dir: Path,
           attempts: int = 3) -> Path | None:
    # download one file; returns its local path, or None if it truly is absent
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(remote).name
    for attempt in range(attempts):
        try:
            _api().dataset_download_file(dataset, remote, path=str(dest_dir),
                                         force=True, quiet=True)
        except Exception as exc:  # noqa: BLE001
            # 429 is a quota wall, not a missing file. Retrying quickly just
            # extends the block, and treating it as absence silently produces
            # empty results - an earlier run "found" 20 storms and then got
            # metadata for 3 of the same ids, purely because it was throttled.
            if "429" in str(exc) or "Too Many Requests" in str(exc):
                with _quota_lock:
                    print(f"  rate limited by Kaggle - pausing "
                          f"{RATE_LIMIT_PAUSE}s", flush=True)
                    time.sleep(RATE_LIMIT_PAUSE)
                continue
        if local.exists() and local.stat().st_size > 0:
            return local
        if attempt < attempts - 1:
            time.sleep(1.0 * (attempt + 1))
    return None


def read_metadata(dataset: str, storm: str) -> list[dict]:
    # frame rows for one storm, dropping interpolated entries with no image
    with TemporaryDirectory() as tmp:
        path = _fetch(dataset, f"metadata/metadata/{storm}.csv", Path(tmp))
        if path is None:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")

    if not text.lstrip().startswith("year,month,day"):
        return []

    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        h5 = (row.get("file_1") or "").strip()
        if not h5:                      # intp == 1: interpolated, no frame
            continue
        rows.append({
            "storm_id": storm,
            "time": f"{row['year']}-{int(row['month']):02d}-{int(row['day']):02d} "
                    f"{int(row['hour']):02d}:00",
            "year": int(row["year"]),
            "lat": row["lat"], "lng": row["lng"],
            "grade": row["grade"], "wind_kt": row["wind"], "pressure": row["pressure"],
            "remote": f"image_png/image_png/{storm}/{h5[:-3]}.png",
            "local": f"images/{storm}/{h5[:-3]}.png",
        })
    return rows


def candidate_ids(basin: str, year_from: int, year_to: int) -> list[str]:
    # storm ids are YYYYNN, numbered sequentially within a season
    lo, hi = YEAR_RANGE[basin]
    return [f"{y}{n:02d}"
            for y in range(max(lo, year_from), min(hi, year_to) + 1)
            for n in range(1, MAX_STORMS_PER_SEASON + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--basin", choices=sorted(BASINS), default="AU",
                    help="AU ships per-storm metadata (labels); WP is images only")
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--to-year", type=int, default=2024)
    ap.add_argument("--stride", type=int, default=6,
                    help="keep every Nth frame; 6 matches our synoptic cadence")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection and estimated size, download nothing")
    args = ap.parse_args()

    dataset = BASINS[args.basin]
    root = Path(args.out or Path(__file__).resolve().parents[2]
                / "data" / "digital_typhoon" / args.basin)

    sys.stdout = _DropBanners(sys.stdout)

    print(f"Digital Typhoon {args.basin}  ({dataset})")
    print(f"  years {args.from_year}-{args.to_year}, stride {args.stride}, PNG only\n")
    _api()
    print("  authenticated\n")

    storms = candidate_ids(args.basin, args.from_year, args.to_year)
    print(f"  reading metadata for {len(storms):,} candidate storm ids ...")
    print("  (ids that do not exist come back empty, so there is no separate "
          "probe pass - that halves the requests, which matters against "
          "Kaggle's rate limit)")
    frames: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(read_metadata, dataset, s) for s in storms]
        for i, fut in enumerate(as_completed(futures), 1):
            frames.extend(fut.result())
            if i % 25 == 0:
                print(f"    {i:,}/{len(storms):,} storms, {len(frames):,} frames",
                      end="\r", flush=True)
    real = len({f["storm_id"] for f in frames})
    print(f"    {real:,} real storms of {len(storms):,} candidates, "
          f"{len(frames):,} frames total          ")
    if not frames:
        print("no storms found - check the basin and year range")
        return 1

    # subsample within each storm so every storm keeps even time coverage.
    by_storm: dict[str, list[dict]] = {}
    for f in frames:
        by_storm.setdefault(f["storm_id"], []).append(f)
    chosen: list[dict] = []
    for rows in by_storm.values():
        rows.sort(key=lambda r: r["time"])
        chosen.extend(rows[::args.stride])

    print(f"\n  selected {len(chosen):,} frames from {len(by_storm):,} storms")
    print(f"  estimated ~{len(chosen) * BYTES_PER_PNG / 1e9:.1f} GB "
          f"(whole mirror is {'89' if args.basin == 'WP' else '32'} GB)")

    if args.dry_run:
        print("\ndry run - nothing downloaded. Drop --dry-run to fetch.")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    labels = root / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chosen[0].keys()))
        w.writeheader()
        w.writerows(chosen)
    print(f"\n  wrote {labels}")

    def grab(frame: dict) -> bool:
        target = root / frame["local"]
        if target.exists() and target.stat().st_size > 0:
            return True
        return _fetch(dataset, frame["remote"], target.parent) is not None

    print(f"  downloading {len(chosen):,} frames to {root} ...\n")
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(grab, f) for f in chosen]
        for i, fut in enumerate(as_completed(futures), 1):
            ok, bad = (ok + 1, bad) if fut.result() else (ok, bad + 1)
            if i % 50 == 0:
                print(f"    [{i:,}/{len(chosen):,}] {ok:,} ok, {bad:,} failed",
                      end="\r", flush=True)
    print(f"\n  finished: {ok:,} frames on disk, {bad:,} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
