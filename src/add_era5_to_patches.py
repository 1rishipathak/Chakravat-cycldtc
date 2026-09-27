# Give the imagery models the environment the forecast already sees.
#
#     python src/add_era5_to_patches.py
#
# T2 and T3 look at a picture and nothing else. That is the right experiment to
# have run first - it says what a satellite image alone is worth - but it is not
# how intensity works. A storm in 20 m/s of shear over a cold sea is not the same
# storm as the one with the identical cloud top over 30 C water, and T4 has known
# that since it was built: shear, steering flow, humidity, SST and divergence are
# already in the forecast feature list.
#
# So this writes those same ERA5 columns for every patch, sampled at the patch's
# own time and centre. It goes to a sidecar, data/processed/scenes*/era5.csv,
# keyed on (storm, time), rather than into patches.csv - that table has been
# wiped by a cache rebuild before and the columns went with it. A sidecar cannot
# be lost that way and cannot break anything that does not ask for it.

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHES = {"gridsat": ROOT / "data" / "processed" / "scenes",
          "insat": ROOT / "data" / "processed" / "scenes_insat"}
ERA5 = ROOT / "data" / "era5"


def main() -> None:
    from features.environment import ENV_FEATURES, Era5Environment

    env = Era5Environment(ERA5)
    for name, cache in CACHES.items():
        table = cache / "patches.csv"
        if not table.exists():
            print(f"{name}: no patch table at {table}, skipped")
            continue
        df = pd.read_csv(table)
        # sorted by time so the yearly ERA5 file is opened once, not per row
        order = df.sort_values("time").index
        t0, records = time.time(), {}
        for n, i in enumerate(order, 1):
            r = df.loc[i]
            records[i] = env.extract(pd.Timestamp(r["time"]),
                                     float(r["lat"]), float(r["lon"]))
            if n % 200 == 0 or n == len(order):
                print(f"  {name} {n}/{len(order)}  {time.time() - t0:.0f}s", flush=True)

        out = pd.DataFrame.from_dict(records, orient="index")[ENV_FEATURES]
        out = pd.concat([df[["storm", "time"]], out], axis=1)
        path = cache / "era5.csv"
        out.to_csv(path, index=False)
        have = out[ENV_FEATURES].notna().all(axis=1).mean()
        print(f"{name}: wrote {path.name}, {len(out)} rows, "
              f"{have:.1%} with every feature present")
        for f in ENV_FEATURES:
            col = out[f]
            print(f"    {f:<14} {col.notna().mean():>6.1%} present  "
                  f"mean {col.mean():>8.2f}  sd {col.std():>7.2f}")


if __name__ == "__main__":
    main()
