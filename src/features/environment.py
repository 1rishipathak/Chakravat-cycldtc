# environmental predictors extracted from ERA5 (CDS api has been used)

# each predictor is an area statistic in a storm-relative frame, following
# operational practice:

# shear is the vector difference between the 200 and 850 hPa area-mean winds
# over a 200-800 km annulus. The annulus deliberately excludes the core - the
# storm's own circulation would otherwise dominate the mean and the "shear"
# would mostly measure the cyclone rather than its environment.
# steering flow is a deep-layer mean over the inner 500 km, which is what
# actually advects the vortex.
#SST is averaged over the inner 200 km, the water the storm is drawing from.
#Humidity** is annulus-averaged, since dry-air intrusion is an environmental
#process happening outside the core.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS_KM = 6371.0088

INNER_KM = 200.0        # SST disc
STEERING_KM = 500.0     # steering / divergence disc
ANNULUS_KM = (200.0, 800.0)   # shear and humidity annulus

ENV_FEATURES = [
    "env_shear_ms", "env_shear_u", "env_shear_v",
    "env_steer_u", "env_steer_v",
    "env_rh700", "env_rh500",
    "env_sst_c", "env_tcwv", "env_div200",
]


def _pick(ds: xr.Dataset, *candidates: str) -> str:
    # finding first name present in candidates so that even older cds dataset can be used
    for name in candidates:
        if name in ds.variables or name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"none of {candidates} found in {list(ds.variables)[:20]}")


def _distance_grid(lats: np.ndarray, lons: np.ndarray,
                   clat: float, clon: float) -> np.ndarray:
    # haversin distance from clat clon to grid points
    lon2d, lat2d = np.meshgrid(lons, lats)
    p1, p2 = np.radians(clat), np.radians(lat2d)
    dp = np.radians(lat2d - clat)
    dl = np.radians(lon2d - clon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _masked_mean(field: np.ndarray, mask: np.ndarray) -> float:
    sel = field[mask]
    sel = sel[np.isfinite(sel)]
    return float(sel.mean()) if sel.size else np.nan


def _divergence(u: np.ndarray, v: np.ndarray,
                lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    # horizontal divergence on that lat lon grid
    dlat = np.gradient(lats) * np.pi / 180.0 * EARTH_RADIUS_KM * 1000.0
    coslat = np.cos(np.radians(lats))[:, None]
    dlon = np.gradient(lons) * np.pi / 180.0 * EARTH_RADIUS_KM * 1000.0
    du_dx = np.gradient(u, axis=1) / (dlon[None, :] * coslat)
    dv_dy = np.gradient(v, axis=0) / dlat[:, None]
    return du_dx + dv_dy


class Era5Environment:
    # opening one year of era5 and extracting storm relevant stats

    def __init__(self, era5_dir: Path):
        self.dir = Path(era5_dir)
        self._year: int | None = None
        self._pl: xr.Dataset | None = None
        self._sl: xr.Dataset | None = None

    def _load(self, year: int) -> bool:
        if self._year == year:
            return self._pl is not None
        for ds in (self._pl, self._sl):
            if ds is not None:
                ds.close()
        self._pl = self._sl = None
        self._year = year

        pl_path = self.dir / f"era5_pl_{year}.nc"
        sl_path = self.dir / f"era5_sl_{year}.nc"
        if not pl_path.exists():
            return False
        self._pl = xr.open_dataset(pl_path)
        if sl_path.exists():
            self._sl = xr.open_dataset(sl_path)
        return True

    def _at_time(self, ds: xr.Dataset, when: pd.Timestamp) -> xr.Dataset | None:
        tname = _pick(ds, "valid_time", "time")
        try:
            snap = ds.sel({tname: np.datetime64(when)}, method="nearest", tolerance=np.timedelta64(3, "h"))
        except (KeyError, IndexError):
            return None
        return snap

    def extract(self, when: pd.Timestamp, lat: float, lon: float) -> dict[str, float]:
        blank = dict.fromkeys(ENV_FEATURES, np.nan)
        if not self._load(when.year):
            return blank

        pl = self._at_time(self._pl, when)
        if pl is None:
            return blank

        latname = _pick(pl, "latitude", "lat")
        lonname = _pick(pl, "longitude", "lon")
        levname = _pick(pl, "pressure_level", "level", "isobaricInhPa")
        lats = pl[latname].values
        lons = pl[lonname].values

        dist = _distance_grid(lats, lons, lat, lon)
        inner = dist <= INNER_KM
        steer = dist <= STEERING_KM
        annulus = (dist >= ANNULUS_KM[0]) & (dist <= ANNULUS_KM[1])

        uname, vname, rname = _pick(pl, "u"), _pick(pl, "v"), _pick(pl, "r")

        def level(var: str, hpa: int) -> np.ndarray:
            return pl[var].sel({levname: hpa}).values

        out = dict(blank)
        try:
            u200, v200 = level(uname, 200), level(vname, 200)
            u850, v850 = level(uname, 850), level(vname, 850)

            su = _masked_mean(u200, annulus) - _masked_mean(u850, annulus)
            sv = _masked_mean(v200, annulus) - _masked_mean(v850, annulus)
            out["env_shear_u"], out["env_shear_v"] = su, sv
            out["env_shear_ms"] = float(np.hypot(su, sv))

            # deep-layer steering: simple mean of 850/700/500/200, which is a
            # good approximation to the mass-weighted mean for this purpose.
            us = [_masked_mean(level(uname, p), steer) for p in (850, 700, 500, 200)]
            vs = [_masked_mean(level(vname, p), steer) for p in (850, 700, 500, 200)]
            out["env_steer_u"] = float(np.nanmean(us))
            out["env_steer_v"] = float(np.nanmean(vs))

            out["env_rh700"] = _masked_mean(level(rname, 700), annulus)
            out["env_rh500"] = _masked_mean(level(rname, 500), annulus)

            div = _divergence(u200, v200, lats, lons)
            out["env_div200"] = _masked_mean(div, steer)
        except (KeyError, ValueError):
            pass

        if self._sl is not None:
            sl = self._at_time(self._sl, when)
            if sl is not None:
                try:
                    sst = sl[_pick(sl, "sst")].values
                    out["env_sst_c"] = _masked_mean(sst, inner) - 273.15
                except KeyError:
                    pass
                try:
                    out["env_tcwv"] = _masked_mean(sl[_pick(sl, "tcwv")].values, steer)
                except KeyError:
                    pass
        return out


def attach(df: pd.DataFrame, era5_dir: Path) -> pd.DataFrame:
    # add the environmental predictor columns to a storm table.
    env = Era5Environment(era5_dir)
    records = [
        env.extract(pd.Timestamp(row.ISO_TIME), float(row.LAT), float(row.LON))
        for row in df.itertuples()
    ]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
