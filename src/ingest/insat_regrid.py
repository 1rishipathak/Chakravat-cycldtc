# put an INSAT-3DR granule onto the GridSat grid

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

FILL_COUNT = 1023
EARTH_R_M = 6378137.0          # WGS84 semi-major; the file declares WGS84

# the GridSat crop every model was trained on.
GRID_DEG = 0.07
GRID_LAT = np.round(np.arange(-5.95, 35.98 + 1e-9, GRID_DEG), 4)
GRID_LON = np.round(np.arange(34.06, 105.95 + 1e-9, GRID_DEG), 4)


def _mercator_lat(y_m: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan(np.exp(y_m / EARTH_R_M)) - np.pi / 2.0)


def _channel(h5, counts_key: str, lut_key: str) -> np.ndarray:
    # counts to brightness temperature in kelvin, fill masked to NaN
    counts = np.asarray(h5[counts_key][0])
    lut = np.asarray(h5[lut_key][:], dtype=np.float32)
    idx = np.clip(counts, 0, len(lut) - 1)
    kelvin = lut[idx].astype(np.float32)
    kelvin[counts >= FILL_COUNT] = np.nan          # the trap; see module docstring
    kelvin[~np.isfinite(kelvin)] = np.nan
    return kelvin


def valid_bounds(path: Path | str) -> tuple[float, float] | None:
    # (south, north) of the granule's populated strip, or None if empty
    import h5py

    with h5py.File(path, "r") as f:
        counts = np.asarray(f["IMG_TIR1"][0])
        y = np.asarray(f["Y"][:])
        rows = np.flatnonzero((counts < FILL_COUNT).any(axis=1))
        if rows.size == 0:
            return None
        lats = _mercator_lat(y[rows])
        return float(lats.min()), float(lats.max())


def covers(path: Path | str, lat: float, margin_deg: float = 1.0) -> bool:
    # true when the granule's strip contains lat with room to spare
    b = valid_bounds(path)
    return b is not None and (b[0] + margin_deg) <= lat <= (b[1] - margin_deg)


def to_gridsat(path: Path | str) -> xr.Dataset:
    # regrid one granule to the GridSat lat/lon grid and variable names
    import h5py

    with h5py.File(path, "r") as f:
        ir = _channel(f, "IMG_TIR1", "IMG_TIR1_TEMP")
        wv = _channel(f, "IMG_WV", "IMG_WV_TEMP")
        x = np.asarray(f["X"][:])
        y = np.asarray(f["Y"][:])
        lon0 = float(f["Projection_Information"].attrs[
            "longitude_of_projection_origin"][0])

    # Mercator is separable, so latitude depends only on Y and longitude only
    # on X - no 2-D interpolation needed, just two 1-D coordinate vectors.
    lat = _mercator_lat(y)
    lon = lon0 + np.degrees(x / EARTH_R_M)

    # xarray interpolation wants ascending coordinates; Y runs north to south.
    if lat[0] > lat[-1]:
        lat, ir, wv = lat[::-1], ir[::-1, :], wv[::-1, :]

    src = xr.Dataset(
        {"irwin_cdr": (("lat", "lon"), ir), "irwvp": (("lat", "lon"), wv)},
        coords={"lat": lat, "lon": lon},
    )
    out = src.interp(lat=GRID_LAT, lon=GRID_LON, method="linear",
                     kwargs={"fill_value": np.nan})

    # downstream code indexes .isel(time=0); give it the axis it expects.
    out = out.expand_dims(time=[np.datetime64("1970-01-01")])
    out.attrs["source"] = str(Path(path).name)
    out.attrs["note"] = "INSAT-3DR L1C resampled to the GridSat 0.07deg grid"
    return out
