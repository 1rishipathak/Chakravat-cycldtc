# basin-wide cyclone detection and centre-fixing (task T1)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

SCENE_H, SCENE_W = 320, 544      # basin crop resized, aspect preserved
STRIDE = 4                        # heatmap is 1/4 resolution
HEAT_H, HEAT_W = SCENE_H // STRIDE, SCENE_W // STRIDE
SIGMA_PX = 2.5                    # Gaussian radius on the heatmap

# loss weight as a function of intensity. Every storm currently contributes an
# identical target peak, so a 20 kt depression and a 130 kt super cyclonic
# storm pull equally hard - and depressions are 56% of the training signal
# while being frequently indistinguishable from ordinary monsoon convection in
# infrared. Most of the loss was therefore spent chasing systems the sensor may
# not resolve. This tilts it toward the storms that carry warning value without
# discarding the weak ones.
WEIGHT_REF_KT = 34.0              # Cyclonic Storm threshold
WEIGHT_POWER = 1.5
WEIGHT_RANGE = (0.3, 3.0)
BT_MIN, BT_MAX = 180.0, 310.0
PEAK_THRESHOLD = 0.30


def gridsat_path(root: Path, when: pd.Timestamp) -> Path:
    return (root / f"{when.year}" /
            f"nio_{when.year}{when.month:02d}{when.day:02d}_{when.hour:02d}.nc")


def build_scene_cache(index: pd.DataFrame, root: Path, cache: Path) -> None:
    # pre-render every scene to a uint8 memmap, once
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = len(index)
    arr = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8,
                                    shape=(n, 3, SCENE_H, SCENE_W))
    for i, row in enumerate(index.itertuples()):
        when = pd.Timestamp(row.time)
        img = _render(gridsat_path(root, when))
        if img is not None:
            arr[i] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if (i + 1) % 100 == 0:
            print(f"    cached {i + 1:,}/{n:,} scenes", flush=True)
    arr.flush()
    print(f"    cached {n:,} scenes -> {cache.name} "
          f"({cache.stat().st_size / 1e6:.0f} MB)")


def _render(path: Path) -> np.ndarray | None:
    # GridSat file -> the 3-channel scene the model sees, float in [0,1]
    import torch as _t

    try:
        with xr.open_dataset(path) as ds:
            ir = ds["irwin_cdr"].isel(time=0).values.astype(np.float32)
            wv = (ds["irwvp"].isel(time=0).values.astype(np.float32)
                  if "irwvp" in ds.variables else ir)
    except Exception:  # noqa: BLE001
        return None
    scale = lambda a: np.clip((a - BT_MIN) / (BT_MAX - BT_MIN), 0, 1)  # noqa: E731
    diff = np.clip(0.5 + (ir - wv) / 40.0, 0, 1)
    img = np.nan_to_num(np.stack([scale(ir), scale(wv), diff], axis=0), nan=1.0)
    t = _t.from_numpy(img).unsqueeze(0)
    t = _t.nn.functional.interpolate(t, size=(SCENE_H, SCENE_W),
                                     mode="bilinear", align_corners=False)[0]
    return t.numpy()


class BasinScenes(Dataset):
    # One GridSat basin scene, with a heatmap of every storm centre in it

    def __init__(self, index: pd.DataFrame, track: pd.DataFrame,
                 root: Path, train: bool = False,
                 cache: np.ndarray | None = None,
                 cache_offset: int = 0):
        self.index = index.reset_index(drop=True)
        self.track = track
        self.root = Path(root)
        self.train = train
        self.cache = cache
        self.cache_offset = cache_offset
        self._geo: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _latlon(self, when: pd.Timestamp):
        # grid coordinates only - cheap, and identical for every file
        if self._geo is None:
            with xr.open_dataset(gridsat_path(self.root, when)) as ds:
                self._geo = (ds["lat"].values.astype(np.float32),
                             ds["lon"].values.astype(np.float32))
        return self._geo

    def _read(self, when: pd.Timestamp):
        with xr.open_dataset(gridsat_path(self.root, when)) as ds:
            ir = ds["irwin_cdr"].isel(time=0).values.astype(np.float32)
            wv = (ds["irwvp"].isel(time=0).values.astype(np.float32)
                  if "irwvp" in ds.variables else ir)
            lats = ds["lat"].values.astype(np.float32)
            lons = ds["lon"].values.astype(np.float32)
        return ir, wv, lats, lons

    @staticmethod
    def intensity_weight(vmax_kt: float) -> float:
        # how hard this storm should pull on the loss
        if not np.isfinite(vmax_kt) or vmax_kt <= 0:
            return 1.0
        w = (vmax_kt / WEIGHT_REF_KT) ** WEIGHT_POWER
        return float(np.clip(w, *WEIGHT_RANGE))

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        when = pd.Timestamp(row["time"])

        if self.cache is not None:
            t = torch.from_numpy(
                self.cache[self.cache_offset + i].astype(np.float32) / 255.0)
            lats, lons = self._latlon(when)
        else:
            ir, wv, lats, lons = self._read(when)
            t = torch.from_numpy(_render_from(ir, wv))

        # storm centres at this timestep -> heatmap peaks.
        here = self.track[self.track["ISO_TIME"] == when]
        heat = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)
        # background weight is 1; each storm raises it in its own neighbourhood.
        wmap = np.ones((HEAT_H, HEAT_W), dtype=np.float32)
        # tracks which storm currently dominates each pixel, so the nearest one
        # sets the weight. Combining with a maximum instead would silently drop
        # every downweight: a depression's weight of 0.3 is below the
        # background 1.0, so max() would always discard it and only the
        # upweighting half of this scheme would ever take effect.
        influence = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)
        targets = []
        for r in here.itertuples():
            y, x = self.to_pixel(r.LAT, r.LON, lats, lons)
            if y is None:
                continue
            hy, hx = y / STRIDE, x / STRIDE
            if not (0 <= hy < HEAT_H and 0 <= hx < HEAT_W):
                continue
            self._splat(heat, hy, hx)
            self._splat_weight(wmap, influence, hy, hx,
                               self.intensity_weight(getattr(r, "vmax_kt", np.nan)))
            targets.append((hy, hx))

        heat_t = torch.from_numpy(heat).unsqueeze(0)
        w_t = torch.from_numpy(wmap).unsqueeze(0)
        if self.train:
            t, heat_t, w_t = self._augment(t, heat_t, w_t)
        return t, heat_t, w_t, torch.tensor(len(targets), dtype=torch.long)

    @staticmethod
    def _splat_weight(wmap: np.ndarray, influence: np.ndarray,
                      cy: float, cx: float, weight: float,
                      sigma: float = SIGMA_PX) -> None:
        # set the loss weight around one storm; nearest storm wins on overlap
        rad = int(4 * sigma)
        y0, y1 = max(0, int(cy) - rad), min(wmap.shape[0], int(cy) + rad + 1)
        x0, x1 = max(0, int(cx) - rad), min(wmap.shape[1], int(cx) + rad + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (1.5 * sigma) ** 2))
        sub_w, sub_i = wmap[y0:y1, x0:x1], influence[y0:y1, x0:x1]
        closer = g > sub_i
        sub_w[closer] = 1.0 + (weight - 1.0) * g[closer]
        sub_i[closer] = g[closer]

    @staticmethod
    def _augment(img: torch.Tensor, heat: torch.Tensor, wmap: torch.Tensor):
        # flips, shifts and brightness jitter, applied to image and target alike
        if torch.rand(1).item() < 0.5:
            img, heat, wmap = (torch.flip(img, [-1]), torch.flip(heat, [-1]),
                               torch.flip(wmap, [-1]))
        if torch.rand(1).item() < 0.5:
            img, heat, wmap = (torch.flip(img, [-2]), torch.flip(heat, [-2]),
                               torch.flip(wmap, [-2]))

        # shift in whole heatmap cells so image and target stay registered.
        sy = int(torch.randint(-6, 7, (1,)).item())
        sx = int(torch.randint(-10, 11, (1,)).item())
        if sy or sx:
            img = torch.roll(img, shifts=(sy * STRIDE, sx * STRIDE), dims=(-2, -1))
            heat = torch.roll(heat, shifts=(sy, sx), dims=(-2, -1))
            wmap = torch.roll(wmap, shifts=(sy, sx), dims=(-2, -1))

        # mild radiometric jitter: sensor calibration differs across the
        # satellites GridSat merges, so the model should not lean on absolute
        # brightness.
        img = torch.clamp(img * (1.0 + 0.06 * (torch.rand(1).item() - 0.5))
                          + 0.04 * (torch.rand(1).item() - 0.5), 0.0, 1.0)
        return img, heat, wmap

    @staticmethod
    def to_pixel(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray):
        # map a geographic position to resized-scene pixel coordinates
        if not (lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()):
            return None, None
        fy = (lat - lats[0]) / (lats[-1] - lats[0])
        fx = (lon - lons[0]) / (lons[-1] - lons[0])
        return fy * SCENE_H, fx * SCENE_W

    @staticmethod
    def from_pixel(y: float, x: float, lats: np.ndarray, lons: np.ndarray):
        lat = lats[0] + (y / SCENE_H) * (lats[-1] - lats[0])
        lon = lons[0] + (x / SCENE_W) * (lons[-1] - lons[0])
        return float(lat), float(lon)

    @staticmethod
    def _splat(heat: np.ndarray, cy: float, cx: float,
               sigma: float = SIGMA_PX) -> None:
        # draw a Gaussian, keeping the maximum where storms overlap
        rad = int(3 * sigma)
        y0, y1 = max(0, int(cy) - rad), min(heat.shape[0], int(cy) + rad + 1)
        x0, x1 = max(0, int(cx) - rad), min(heat.shape[1], int(cx) + rad + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
        # the true centre falls between grid cells, so the sampled Gaussian
        # peaks just below 1. The focal loss identifies positives by an exact
        # value of 1, so pin the nearest cell - otherwise a scene contains no
        # positive pixels at all and the model only ever learns background.
        py, px = int(round(cy)), int(round(cx))
        if 0 <= py < heat.shape[0] and 0 <= px < heat.shape[1]:
            heat[py, px] = 1.0


def _render_from(ir: np.ndarray, wv: np.ndarray) -> np.ndarray:
    scale = lambda a: np.clip((a - BT_MIN) / (BT_MAX - BT_MIN), 0, 1)  # noqa: E731
    diff = np.clip(0.5 + (ir - wv) / 40.0, 0, 1)
    img = np.nan_to_num(np.stack([scale(ir), scale(wv), diff], axis=0), nan=1.0)
    t = torch.from_numpy(img).unsqueeze(0)
    return torch.nn.functional.interpolate(
        t, size=(SCENE_H, SCENE_W), mode="bilinear", align_corners=False)[0].numpy()


def build_index(root: Path, track: pd.DataFrame) -> pd.DataFrame:
    # GridSat timesteps present on disk, with how many storms each contains
    rows = []
    for path in sorted(Path(root).rglob("nio_*.nc")):
        stem = path.stem.replace("nio_", "")
        date, hour = stem.split("_")
        when = pd.Timestamp(int(date[:4]), int(date[4:6]), int(date[6:8]), int(hour))
        n = int((track["ISO_TIME"] == when).sum())
        rows.append({"time": when, "season": when.year, "n_storms": n})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def season_split(index: pd.DataFrame, val_seasons=(2019, 2020),
                 test_seasons=(2021, 2022, 2023, 2024, 2025)) -> dict:
    # split by season, as elsewhere in the project - never within a storm
    test = index[index["season"].isin(test_seasons)]
    val = index[index["season"].isin(val_seasons)]
    train = index[~index["season"].isin(set(val_seasons) | set(test_seasons))]
    return {"train": train, "val": val, "test": test}


def find_peaks(heat: np.ndarray, threshold: float = PEAK_THRESHOLD,
               window: int = 5) -> list[tuple[float, float, float]]:
    # local maxima above threshold, as (y, x, score) in heatmap pixels
    import scipy.ndimage as ndi

    pooled = ndi.maximum_filter(heat, size=window, mode="constant")
    mask = (heat == pooled) & (heat >= threshold)
    ys, xs = np.nonzero(mask)
    return sorted([(float(y), float(x), float(heat[y, x])) for y, x in zip(ys, xs)],
                  key=lambda p: -p[2])
