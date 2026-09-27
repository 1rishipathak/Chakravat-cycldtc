# crosswalk from ADT analyses to the IBTrACS best track
#
#     python src/vision/best_track.py
#     python src/vision/best_track.py --cache data/processed/scenes_insat
#
# the ADT archive keys storms by its own id (200501B) at its own analysis times
# (01:00, 11:30), IBTrACS by SID at 3-hourly fixes, so the two share no key and
# an exact join finds zero rows. the join here is spatio-temporal instead:
# interpolate every best track to the ADT time and keep the nearest storm
# within MAX_KM.
#
# why it exists: patches.csv carried ADT's own wind as `vmax_kt`, and T3 was
# trained and scored on it as if it were the truth. ADT follows the Dvorak
# tables, which give 1-minute winds. IMD's best track is 3-minute. over the
# whole archive ADT reads about 7.6 kt higher than IMD, so T3 learned ADT's
# scale and then got mapped onto IMD's categories and fed into T4, which was
# trained on IMD winds. the best-track columns added here fix the target.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ingest.ibtracs import load_raw, restrict_to_basin, unify_intensity  # noqa: E402

IBTRACS = ROOT / "data" / "raw" / "ibtracs.NI.csv"
ADT = ROOT / "data" / "adt" / "adt_nio.csv"
CACHE = ROOT / "data" / "processed" / "scenes"
OUT = ROOT / "reports"

# a best-track fix every 3 h is normal; a 6 h gap is the most we interpolate
# across before calling the time uncovered.
MAX_GAP_H = 6.0
# ADT and best track centres sit a median 28 km apart. 200 km still separates
# two storms reliably - the basin rarely holds two within 500 km.
MAX_KM = 200.0
BANDS = [("<34", 0, 34), ("34-48", 34, 48), ("48-64", 48, 64),
         ("64-90", 64, 90), ("90+", 90, 999)]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _hours(times) -> np.ndarray:
    return (pd.DatetimeIndex(pd.to_datetime(times)).asi8 // 10**9) / 3600.0


def load_tracks(path: Path = IBTRACS, min_season: int = 2002) -> dict[str, dict]:
    # every best-track fix with a wind, at full 3-hourly density
    bt = restrict_to_basin(unify_intensity(load_raw(str(path))))
    bt = bt[(bt["SEASON"] >= min_season) & bt["vmax_kt"].notna() & (bt["vmax_kt"] > 0)]
    bt = bt.sort_values(["SID", "ISO_TIME"])
    return {sid: {"t": _hours(g["ISO_TIME"]), "lat": g["LAT"].to_numpy(),
                  "lon": g["LON"].to_numpy(), "v": g["vmax_kt"].to_numpy(),
                  "src": g["vmax_source"].to_numpy()}
            for sid, g in bt.groupby("SID")}


def match(times, lats, lons, tracks: dict[str, dict]) -> pd.DataFrame:
    # best-track wind interpolated to each record, from the nearest storm
    th = _hours(times)
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    spans = [(sid, tr["t"][0], tr["t"][-1]) for sid, tr in tracks.items()]
    out = {"bt_sid": [], "bt_vmax_kt": [], "bt_offset_km": [], "bt_source": []}

    for i in range(len(th)):
        best = (np.inf, "", np.nan, "")
        for sid, t0, t1 in spans:
            if not (t0 <= th[i] <= t1):
                continue
            tr = tracks[sid]
            j = int(np.searchsorted(tr["t"], th[i]))
            if j < len(tr["t"]) and tr["t"][j] == th[i]:
                la, lo, v, src = tr["lat"][j], tr["lon"][j], tr["v"][j], tr["src"][j]
            else:
                a, b = j - 1, j
                if tr["t"][b] - tr["t"][a] > MAX_GAP_H:
                    continue
                f = (th[i] - tr["t"][a]) / (tr["t"][b] - tr["t"][a])
                la = tr["lat"][a] + f * (tr["lat"][b] - tr["lat"][a])
                lo = tr["lon"][a] + f * (tr["lon"][b] - tr["lon"][a])
                v = tr["v"][a] + f * (tr["v"][b] - tr["v"][a])
                src = tr["src"][a] if tr["src"][a] == tr["src"][b] else "mixed"
            d = float(haversine_km(lats[i], lons[i], la, lo))
            if d < best[0]:
                best = (d, sid, float(v), str(src))
        ok = best[0] <= MAX_KM
        out["bt_sid"].append(best[1] if ok else "")
        out["bt_vmax_kt"].append(best[2] if ok else np.nan)
        out["bt_offset_km"].append(best[0] if ok else np.nan)
        out["bt_source"].append(best[3] if ok else "")
    return pd.DataFrame(out)


def error_summary(pred, truth) -> dict:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    k = np.isfinite(pred) & np.isfinite(truth)
    e = pred[k] - truth[k]
    bands = {}
    for lab, lo, hi in BANDS:
        b = (truth[k] >= lo) & (truth[k] < hi)
        bands[lab] = {"n": int(b.sum()),
                      "bias": float(e[b].mean()) if b.sum() >= 3 else None}
    return {"n": int(k.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()), "bands": bands}


def add_columns(path: Path, tracks: dict) -> pd.DataFrame:
    # match every row of a patch table and write the bt_ columns back into it
    meta = pd.read_csv(path)
    meta = meta.drop(columns=[c for c in meta.columns if c.startswith("bt_")])
    meta = pd.concat([meta, match(meta["time"], meta["lat"], meta["lon"], tracks)], axis=1)
    meta.to_csv(path, index=False)
    got = meta["bt_vmax_kt"].notna()
    print(f"  {path.parent.name}: {int(got.sum())} of {len(meta)} rows matched to best track; "
          f"wrote bt_ columns into {path.relative_to(ROOT)}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE),
                    help="patch table directory; any cache other than the GridSat "
                         "one only gets its columns added, no archive report")
    args = ap.parse_args()
    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = ROOT / cache

    print("loading best track ...")
    tracks = load_tracks()
    print(f"  {len(tracks)} storms with winds, 2002 onwards")
    if cache.resolve() != CACHE.resolve():
        add_columns(cache / "patches.csv", tracks)
        return

    # ---- the whole ADT archive: how far is ADT's wind scale from IMD's ----
    adt = pd.read_csv(ADT, parse_dates=["time"])
    adt = adt[adt["vmax_kt"].notna() & (adt["vmax_kt"] > 0)].reset_index(drop=True)
    m = pd.concat([adt, match(adt["time"], adt["lat"], adt["lon"], tracks)], axis=1)
    m = m[m["bt_vmax_kt"].notna()]
    imd = m[m["bt_source"] == "imd"]
    strong = imd[imd["bt_vmax_kt"] >= 34]
    archive = {
        "adt_records_with_wind": int(len(adt)),
        "matched": int(len(m)),
        "median_offset_km": float(m["bt_offset_km"].median()),
        "best_track_source": m["bt_source"].value_counts().to_dict(),
        "adt_vs_best_track": error_summary(m["vmax_kt"], m["bt_vmax_kt"]),
        "adt_vs_best_track_imd_rows": error_summary(imd["vmax_kt"], imd["bt_vmax_kt"]),
        "adt_x0.93_vs_best_track_imd_rows": error_summary(0.93 * imd["vmax_kt"],
                                                          imd["bt_vmax_kt"]),
        "mean_ratio_adt_to_best_track_34kt_up": float(strong["vmax_kt"].mean()
                                                      / strong["bt_vmax_kt"].mean()),
    }
    s = archive["adt_vs_best_track"]
    print(f"  ADT archive: {len(m):,} of {len(adt):,} records matched, "
          f"median offset {archive['median_offset_km']:.0f} km")
    print(f"  ADT minus best track: bias {s['bias']:+.2f} kt, RMSE {s['rmse']:.2f}, "
          f"ratio at 34 kt+ {archive['mean_ratio_adt_to_best_track_34kt_up']:.3f}")

    # ---- the patch table T2 and T3 train from ----
    path = CACHE / "patches.csv"
    meta = add_columns(path, tracks)

    got = meta["bt_vmax_kt"].notna()
    print(f"\n  patches: {got.sum()} of {len(meta)} matched to best track "
          f"(median offset {meta.loc[got, 'bt_offset_km'].median():.0f} km)")
    print(f"  with an ADT wind: {int((meta['vmax_kt'] > 0).sum())}, "
          f"with a best-track wind: {int(got.sum())}")

    # one ADT storm should map to one best-track storm. more than one means
    # two systems were close together and the nearest-storm rule was tested.
    per = meta[got].groupby("storm")["bt_sid"].agg(lambda s: s.value_counts().to_dict())
    split = {k: v for k, v in per.items() if len(v) > 1}
    print(f"  ADT storms mapping to more than one SID: {len(split)}")
    for k, v in list(split.items())[:10]:
        print(f"    {k}: {v}")

    both = meta[got & (meta["vmax_kt"] > 0)]
    archive["patches"] = {
        "rows": int(len(meta)), "matched": int(got.sum()),
        "with_adt_wind": int((meta["vmax_kt"] > 0).sum()),
        "adt_storms_split_across_sids": {k: v for k, v in split.items()},
        "adt_vs_best_track": error_summary(both["vmax_kt"], both["bt_vmax_kt"]),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "adt_vs_best_track.json").write_text(json.dumps(archive, indent=2))
    print(f"  wrote {(OUT / 'adt_vs_best_track.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
