from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = "https://tropic.ssec.wisc.edu/real-time/adt/archive{year}/"
NIO_SUFFIXES = ("A", "B")

# the scene vocabulary is the anchor for parsing - see parse_history().
SCENE_TYPES = {
    "EYE", "EYE/P", "EYE/L", "EYE/R", "EYE/LP", "EYE/RP",
    "CRVBND", "SHEAR", "EMBC", "IRRCDO", "CDO", "UNIFRM", "N/A",
}

# the six Dvorak scene classes worth modelling. UNIFRM and N/A are not scenes:
# UNIFRM means the algorithm found no organised pattern, N/A means it assigned
# none at all. Both are dropped from the classification target.
DVORAK_CLASSES = ["EYE", "CRVBND", "SHEAR", "EMBC", "IRRCDO", "CDO"]

DATE_RE = re.compile(r"^(\d{4})([A-Z]{3})(\d{2})\s+(\d{6})")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def list_storms(year: int, session: requests.Session | None = None,
                suffixes: tuple[str, ...] = NIO_SUFFIXES) -> list[str]:
    # list of storm ids present in year
    get = (session or requests).get
    try:
        html = get(ARCHIVE.format(year=year), timeout=60).text
    except requests.RequestException:
        return []
    ids = re.findall(r'href="(\d{2}[A-Z])-list\.txt"', html)
    return sorted({s for s in ids if s[-1] in suffixes})


def parse_history(text: str) -> pd.DataFrame:
    # parse one `<storm>-list.txt` into records
    rows = []
    for line in text.splitlines():
        m = DATE_RE.match(line)
        if not m:
            continue
        year, mon, day, hhmmss = m.groups()
        parts = line.split()

        idx = next((i for i, tok in enumerate(parts)
                    if tok in SCENE_TYPES and i >= 5), None)
        if idx is None:
            continue

        def num(tok: str):
            try:
                return float(tok)
            except ValueError:
                return float("nan")

        # position cannot be counted forward from the scene type, because the
        # EstRMW field between them is one token when it reads "N/A" but two
        # when an eye is measured ("8 IR") which shifted eye scenes, the
        # most valuable class, straight out of the basin. Nor can it be counted
        # back from the end: the archive spans three ADT generations, and the
        # 2003 files stop at longitude while v9 adds method, satellite and view
        # angle after it.
        #
        # what is stable across all of them is geography. Scan the float pairs
        # that follow the scene token and keep the last one that could actually
        # be a North Indian Ocean position. Temperatures and scores fail the
        # test; the true fix passes it uniquely.
        after = [(i, num(t)) for i, t in enumerate(parts) if i > idx]
        lat = lon = float("nan")
        for (i, a), (j, b) in zip(after, after[1:]):
            if j != i + 1 or a != a or b != b:
                continue
            if -60.0 <= a <= 60.0 and 20.0 <= -b <= 120.0:
                lat, lon = a, b
        if lat != lat:
            continue

        ci, mslp, vmax = num(parts[2]), num(parts[3]), num(parts[4])
        if not (0.0 <= ci <= 8.5):
            ci = float("nan")
        if not (850.0 <= mslp <= 1030.0):
            mslp = float("nan")
        if not (0.0 <= vmax <= 200.0):
            vmax = float("nan")

        method = next((t for t in parts[idx + 1:]
                       if t.isalpha() and t.upper() == t and len(t) >= 4
                       and t not in SCENE_TYPES), "")
        satellite = next((t for t in reversed(parts)
                          if re.fullmatch(r"[A-Z]{2,4}\d?[A-Z]?", t)
                          and t not in SCENE_TYPES and t != method), "")

        rows.append({
            "time": pd.Timestamp(int(year), MONTHS[mon], int(day),
                                 int(hhmmss[:2]), int(hhmmss[2:4])),
            "ci": ci, "mslp": mslp, "vmax_kt": vmax,
            "scene": parts[idx],
            "lat": lat,
            # ADT reports longitude west-positive; the Bay of Bengal appears as
            # -86.4 rather than +86.4. Negate for the usual east-positive frame.
            "lon": -lon,
            "method": method,
            "satellite": satellite,
        })
    return pd.DataFrame(rows)


def fetch_season(year: int, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    frames = []
    for sid in list_storms(year, sess):
        try:
            text = sess.get(ARCHIVE.format(year=year) + f"{sid}-list.txt",
                            timeout=60).text
        except requests.RequestException:
            continue
        df = parse_history(text)
        if df.empty:
            continue
        df.insert(0, "storm", f"{year}{sid}")
        df.insert(1, "season", year)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(out_path: Path, years: range) -> pd.DataFrame:
    sess = requests.Session()
    sess.headers["User-Agent"] = "chakravat-ingest/1.0"
    frames = []
    for year in years:
        df = fetch_season(year, sess)
        n_scene = int(df["scene"].isin(DVORAK_CLASSES).sum()) if len(df) else 0
        print(f"  {year}: {df['storm'].nunique() if len(df) else 0:>2} storms, "
              f"{len(df):>5,} records, {n_scene:>5,} with a Dvorak scene")
        if len(df):
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(all_df):
        # sanity: North Indian storms must sit in the basin box. Anything
        # outside means the longitude convention was misread.
        bad = ((all_df["lat"] < -5) | (all_df["lat"] > 35)
               | (all_df["lon"] < 30) | (all_df["lon"] > 110))
        if bad.any():
            print(f"  warning: {int(bad.sum()):,} records fall outside the "
                  f"North Indian box and were dropped")
            all_df = all_df[~bad]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(out_path, index=False)
    return all_df


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 2003
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    print(f"CIMSS ADT archive - North Indian Ocean, {lo}-{hi}\n")

    df = build(root / "data" / "adt" / "adt_nio.csv", range(lo, hi + 1))
    if df.empty:
        raise SystemExit("nothing parsed")

    print(f"\ntotal: {len(df):,} records, {df['storm'].nunique()} storms")
    print(f"lat {df['lat'].min():.1f}-{df['lat'].max():.1f}N   "
          f"lon {df['lon'].min():.1f}-{df['lon'].max():.1f}E")
    print("\nDvorak scene distribution:")
    counts = df["scene"].value_counts()
    for scene in DVORAK_CLASSES + ["UNIFRM", "N/A"]:
        if scene in counts:
            tag = "  <- target class" if scene in DVORAK_CLASSES else ""
            print(f"  {scene:<8} {counts[scene]:>6,}{tag}")
    print(f"\ntrainable scenes: {int(df['scene'].isin(DVORAK_CLASSES).sum()):,}")
    print(f"wrote {root / 'data' / 'adt' / 'adt_nio.csv'}")
