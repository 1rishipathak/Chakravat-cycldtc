# single entry point for the modelling dataset

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
ENV_CACHE = ROOT / "data" / "processed" / "dataset_with_env.pkl"


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    if ENV_CACHE.exists():
        df = pd.read_pickle(ENV_CACHE)
        if verbose:
            from features.environment import ENV_FEATURES
            have = [c for c in ENV_FEATURES if c in df.columns]
            cov = df[have].notna().all(axis=1).mean() if have else 0.0
            print(f"dataset: {len(df):,} rows, {df['SID'].nunique()} storms "
                  f"| ERA5 environment on {100 * cov:.1f}% of rows")
        return df

    from ingest.ibtracs import build
    from features.build import build_dataset
    df = build_dataset(build(str(RAW)))
    if verbose:
        print(f"dataset: {len(df):,} rows, {df['SID'].nunique()} storms "
              f"| no ERA5 (run src/train_with_env.py to build the cache)")
    return df
