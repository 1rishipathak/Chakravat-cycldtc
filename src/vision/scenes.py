# storm-centred GridSat patches labelled with ADT Dvorak scene types

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

PATCH_PX = 224
GRID_DEG = 0.07
# half-width of the patch: ~7.8 degrees, about 870 km, which holds the central
# dense overcast and the banding features around it.
HALF_DEG = PATCH_PX * GRID_DEG / 2

# brightness temperature limits for scaling. 180 K is colder than any
# tropical cloud top; 310 K is warmer than the sea surface here.
BT_MIN, BT_MAX = 180.0, 310.0
# ImageNet statistics. The trunks are ImageNet-pretrained and expect inputs
# normalised this way; feeding raw [0,1] leaves the activations off-distribution
# from the first layer.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



def gridsat_path(root: Path, when: pd.Timestamp) -> Path:
    return (root / f"{when.year}" /
            f"nio_{when.year}{when.month:02d}{when.day:02d}_{when.hour:02d}.nc")


def _scale(a: np.ndarray) -> np.ndarray:
    return np.clip((a - BT_MIN) / (BT_MAX - BT_MIN), 0.0, 1.0)


def extract_patch(ds: xr.Dataset, lat: float, lon: float) -> np.ndarray | None:
    # A PATCH_PX square centred on the storm, as uint8 HxWx3
    lats = ds["lat"].values
    lons = ds["lon"].values
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon).argmin())
    half = PATCH_PX // 2
    if i - half < 0 or i + half > len(lats) or j - half < 0 or j + half > len(lons):
        return None                      # too close to the crop edge

    sl = (slice(i - half, i + half), slice(j - half, j + half))
    ir = ds["irwin_cdr"].isel(time=0).values[sl]
    wv = (ds["irwvp"].isel(time=0).values[sl]
          if "irwvp" in ds.variables else ir)
    if not np.isfinite(ir).any():
        return None

    ir_s, wv_s = _scale(ir), _scale(wv)
    # IR minus WV, recentred: near zero where convection is deepest.
    diff = np.clip(0.5 + (ir - wv) / 40.0, 0.0, 1.0)
    stack = np.stack([ir_s, wv_s, diff], axis=-1)
    stack = np.nan_to_num(stack, nan=1.0)
    return (stack * 255).astype(np.uint8)


def build_cache(adt_csv: Path, gridsat_root: Path, out_dir: Path,
                classes: list[str]) -> pd.DataFrame:
    # extract every labelled scene whose GridSat timestep is on disk
    adt = pd.read_csv(adt_csv, parse_dates=["time"])
    adt = adt[adt["scene"].isin(classes)].copy()
    adt["slot"] = pd.DatetimeIndex(adt["time"]).round("3h")

    # One scene per storm per slot: ADT runs every 30 min, GridSat every 3 h,
    # so six ADT records map to the same image. Keep the closest in time,
    # otherwise the same picture appears six times with possibly different
    # labels and the split leaks within a timestep.
    adt["gap"] = (adt["time"] - adt["slot"]).abs()
    adt = (adt.sort_values("gap")
              .drop_duplicates(subset=["storm", "slot"], keep="first")
              .sort_values(["storm", "slot"]).reset_index(drop=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    patches, rows = [], []
    missing = edge = 0
    cache: dict[Path, xr.Dataset] = {}

    for slot, group in adt.groupby("slot", sort=True):
        path = gridsat_path(gridsat_root, slot)
        if not path.exists():
            missing += len(group)
            continue
        try:
            ds = xr.open_dataset(path)
        except Exception:  # noqa: BLE001
            missing += len(group)
            continue
        for r in group.itertuples():
            patch = extract_patch(ds, r.lat, r.lon)
            if patch is None:
                edge += 1
                continue
            patches.append(patch)
            rows.append({"storm": r.storm, "season": r.season,
                         "time": r.time, "scene": r.scene,
                         "lat": r.lat, "lon": r.lon,
                         "ci": r.ci, "vmax_kt": r.vmax_kt})
        ds.close()

    if not patches:
        raise SystemExit("no patches built - is GridSat downloaded yet?")

    arr = np.stack(patches)
    np.save(out_dir / "patches.npy", arr)
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "patches.csv", index=False)
    print(f"  built {len(arr):,} patches  {arr.nbytes/1e6:.0f} MB")
    print(f"  skipped: {missing:,} without imagery, {edge:,} too near the edge")
    return meta


def storm_split(meta: pd.DataFrame, val_frac: float = 0.15,
                test_frac: float = 0.2, seed: int = 0) -> dict[str, np.ndarray]:
    # split by storm, assigned by a hash of the storm id
    def bucket(storm_id: str) -> int:
        h = hashlib.md5(f"{seed}:{storm_id}".encode()).hexdigest()
        return int(h[:8], 16) % 100

    cut_test = int(test_frac * 100)
    cut_val = cut_test + int(val_frac * 100)

    def which(storm_id: str) -> str:
        b = bucket(storm_id)
        return "test" if b < cut_test else ("val" if b < cut_val else "train")

    assigned = meta["storm"].map(which)
    return {k: meta.index[assigned == k].to_numpy()
            for k in ("train", "val", "test")}


class ScenePatches(Dataset):
    def __init__(self, patches: np.ndarray, meta: pd.DataFrame,
                 idx: np.ndarray, classes: list[str], train: bool = False):
        self.patches, self.meta, self.idx = patches, meta, idx
        self.classes, self.train = classes, train
        self.lookup = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = self.idx[k]
        a = self.patches[i].astype(np.float32) / 255.0
        if self.train:
            # cyclones have no canonical orientation, so rotation is free
            # signal. A plain mirror would reverse the sense of rotation, so
            # it is paired with a 180-degree turn to keep the storm cyclonic.
            r = np.random.randint(4)
            if r:
                a = np.rot90(a, r, axes=(0, 1))
            if np.random.rand() < 0.5:
                a = np.rot90(a[:, ::-1], 2, axes=(0, 1))
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        y = self.lookup[self.meta.iloc[i]["scene"]]
        return torch.from_numpy(a), torch.tensor(y, dtype=torch.long)
