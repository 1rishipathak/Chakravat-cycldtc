# storm-centred INSAT patches for T2 and T3, paired with the GridSat ones
#
#     python src/build_insat_patches.py
#
# the same ADT records, the same 3-hourly slots and the same 224 px window as
# the GridSat patch cache, cut from the INSAT scenes insat_grid.py wrote. the
# best-track columns are added afterwards the same way, so T3 has one target
# for both sensors.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.adt import DVORAK_CLASSES      # noqa: E402
from vision.scenes import build_cache      # noqa: E402

ADT = ROOT / "data" / "adt" / "adt_nio.csv"
INSAT_GRID = ROOT / "data" / "insat" / "grid"
OUT = ROOT / "data" / "processed" / "scenes_insat"


def main() -> None:
    meta = build_cache(ADT, INSAT_GRID, OUT, DVORAK_CLASSES)
    print(f"  {len(meta)} INSAT patches from {meta['storm'].nunique()} storms, "
          f"seasons {meta['season'].min()}-{meta['season'].max()}")
    subprocess.run([sys.executable, str(ROOT / "src" / "vision" / "best_track.py"),
                    "--cache", str(OUT)], check=True)


if __name__ == "__main__":
    main()
