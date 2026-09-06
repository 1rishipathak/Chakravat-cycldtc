# profile the ingested best-track data before any modelling decisions

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest.ibtracs import build, IMD_CATEGORIES  # noqa: E402

pd.set_option("display.width", 130)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ibtracs.NI.csv"

df = build(str(RAW))

print("=" * 68)
print("INGESTED BEST TRACK - North Indian Ocean, synoptic hours, 1990+")
print("=" * 68)
print(f"rows            {len(df):,}")
print(f"storms          {df['SID'].nunique():,}")
print(f"seasons         {int(df['SEASON'].min())} - {int(df['SEASON'].max())}")
print(f"storms/season   {df['SID'].nunique() / (df['SEASON'].max() - df['SEASON'].min() + 1):.1f}")

print("\n--- intensity label provenance ---")
print(df["vmax_source"].value_counts().to_string())

print("\n--- Dvorak CI number (NEWDELHI_CI) availability ---")
ci = df["NEWDELHI_CI"]
print(f"rows with CI    {ci.notna().sum():,} ({100 * ci.notna().mean():.1f}%)")
if ci.notna().any():
    print(f"CI range        {ci.min():.1f} - {ci.max():.1f}")

print("\n--- IMD category distribution (rows) ---")
order = [c[0] for c in IMD_CATEGORIES]
vc = df["category"].value_counts().reindex(order).fillna(0).astype(int)
for name, n in vc.items():
    bar = "#" * int(48 * n / max(vc.max(), 1))
    print(f"  {name:<34} {n:>6,}  {bar}")

print("\n--- storms per sub-basin ---")
print(df.groupby("sub_basin")["SID"].nunique().to_string())

print("\n--- storms reaching each threshold (peak intensity) ---")
peak = df.groupby("SID")["vmax_kt"].max()
for thr in [34, 48, 64, 90, 120]:
    print(f"  peak >= {thr:>3} kt : {(peak >= thr).sum():>4} storms")

print("\n--- track length distribution (synoptic points per storm) ---")
lens = df.groupby("SID").size()
print(lens.describe().to_string())

print("\n--- seasons histogram (storms) ---")
per_season = df.groupby("SEASON")["SID"].nunique()
for season, n in per_season.items():
    print(f"  {int(season)}  {'*' * int(n):<16} {n}")

print("\n--- rapid intensification base rate (+30 kt in 24 h) ---")
g = df.sort_values("ISO_TIME").groupby("SID")
fut = g["vmax_kt"].shift(-4)          # 4 synoptic steps = 24 h
delta = fut - df["vmax_kt"]
valid = delta.notna()
print(f"  usable 24 h pairs : {valid.sum():,}")
print(f"  RI events         : {(delta >= 30).sum():,}  ({100 * (delta[valid] >= 30).mean():.2f}%)")
