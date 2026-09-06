# Digital Typhoon image dataset for intensity regression (task T3)

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 224
# ImageNet statistics. The trunks are ImageNet-pretrained and expect inputs
# normalised this way; feeding raw [0,1] leaves the activations off-distribution
# from the first layer.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# grade in the Digital Typhoon metadata; 2 and above are tropical systems.
# grade 6 is extratropical transition, which follows different dynamics.
VALID_GRADES = {2, 3, 4, 5}


def load_labels(root, basin: str = "AU", min_wind: float = 0.0) -> pd.DataFrame:
    # read labels.csv and keep frames whose image is actually on disk
    from pathlib import Path

    root = Path(root)
    df = pd.read_csv(root / "labels.csv")
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df[df["wind_kt"].notna() & (df["wind_kt"] > min_wind)]
    df = df[df["grade"].isin(VALID_GRADES)]

    df["path"] = df["local"].map(lambda p: root / p)
    df = df[df["path"].map(lambda p: p.exists() and p.stat().st_size > 0)]

    df["season"] = df["storm_id"].astype(str).str[:4].astype(int)
    return df.reset_index(drop=True)


def storm_split(df: pd.DataFrame, val_frac: float = 0.15,
                test_frac: float = 0.2, seed: int = 0) -> dict[str, pd.DataFrame]:
    # split by storm id, never by frame
    rng = np.random.default_rng(seed)
    storms = df["storm_id"].unique()
    rng.shuffle(storms)

    n_test = int(len(storms) * test_frac)
    n_val = int(len(storms) * val_frac)
    groups = {
        "test": set(storms[:n_test]),
        "val": set(storms[n_test:n_test + n_val]),
        "train": set(storms[n_test + n_val:]),
    }
    return {k: df[df["storm_id"].isin(v)].reset_index(drop=True)
            for k, v in groups.items()}


class TyphoonFrames(Dataset):
    # One storm-centred infrared frame, one intensity in knots

    def __init__(self, frame: pd.DataFrame, train: bool = False,
                 mirror_hemisphere: bool = True, size: int = IMAGE_SIZE):
        self.df = frame.reset_index(drop=True)
        self.train = train
        self.mirror = mirror_hemisphere
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("L").resize(
            (self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.mirror:
            # southern Hemisphere -> Northern: flip the sense of rotation.
            arr = arr[::-1, :]

        if self.train:
            k = np.random.randint(4)
            if k:
                arr = np.rot90(arr, k)
            if np.random.rand() < 0.5:
                # A mirror flips rotation sense, so pair it with a 180-degree
                # turn to keep the cyclonic sense intact while still varying
                # the view.
                arr = np.rot90(arr[:, ::-1], 2)

        arr = np.ascontiguousarray(arr)
        # single channel repeated to three so ImageNet-pretrained stems apply.
        x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        x = (x - torch.from_numpy(IMAGENET_MEAN).view(3, 1, 1))             / torch.from_numpy(IMAGENET_STD).view(3, 1, 1)
        y = torch.tensor(float(row["wind_kt"]), dtype=torch.float32)
        return x, y
