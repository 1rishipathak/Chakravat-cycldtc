# train/validation/test splitting.

from __future__ import annotations

import numpy as np
import pandas as pd

# frozen before any model was trained. Do not move these to chase a number.
TRAIN_MAX_SEASON = 2015
VAL_MAX_SEASON = 2019
# test is everything after VAL_MAX_SEASON (2020-2025).


def storm_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # chronological split by season, so no storm appears in two sets
    train = df[df["SEASON"] <= TRAIN_MAX_SEASON]
    val = df[(df["SEASON"] > TRAIN_MAX_SEASON) & (df["SEASON"] <= VAL_MAX_SEASON)]
    test = df[df["SEASON"] > VAL_MAX_SEASON]
    return {"train": train.copy(), "val": val.copy(), "test": test.copy()}


def frame_split(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    # random row-level split - LEAKY. Included only for the leakage demo
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.7 * len(df)), int(0.15 * len(df))
    return {
        "train": df.iloc[idx[:n_train]].copy(),
        "val": df.iloc[idx[n_train:n_train + n_val]].copy(),
        "test": df.iloc[idx[n_train + n_val:]].copy(),
    }


def describe(splits: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, part in splits.items():
        seasons = ""
        if len(part):
            seasons = f"  seasons {int(part['SEASON'].min())}-{int(part['SEASON'].max())}"
        lines.append(
            f"  {name:<6} {len(part):>5,} rows  {part['SID'].nunique():>4} storms{seasons}"
        )
    return "\n".join(lines)
