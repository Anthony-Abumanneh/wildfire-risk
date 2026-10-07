"""gridMET daily weather -> weekly fire-weather features per gridMET pixel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from config import BBOX, PROC

GRIDMET = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{}_1979_CurrentYear_CONUS.nc"
VARS = {"tmmx": "tmax", "rmin": "rhmin", "vs": "wind", "vpd": "vpd", "pr": "pr"}
RAIN_MM = 2.5


def download_gridmet(start: str, end: str | None = None) -> pd.DataFrame:
    """Daily gridMET for the county bbox -> wide frame indexed by (date, lat, lon)."""
    lon0, lat0, lon1, lat1 = BBOX
    out = {}
    for code, name in VARS.items():
        ds = xr.open_dataset(GRIDMET.format(code))
        da = ds[list(ds.data_vars)[0]].sel(lat=slice(lat1 + 0.05, lat0 - 0.05),
                                           lon=slice(lon0 - 0.05, lon1 + 0.05),
                                           day=slice(start, end))
        years = pd.DatetimeIndex(da.day.values).year.unique()
        da = xr.concat([da.sel(day=str(y)).load() for y in years], dim="day")
        out[name] = da.to_series()
        print(f"  {name}: {da.sizes['day']} days")
    df = pd.DataFrame(out).rename_axis(["date", "lat", "lon"])
    df["tmax"] -= 273.15  # K -> °C
    return df


def pixel_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Land pixels (non-NaN) with an integer id."""
    first = daily.xs(daily.index.get_level_values("date")[0], level="date")
    px = first.dropna().reset_index()[["lat", "lon"]]
    px["pixel"] = np.arange(len(px))
    return px


def map_cells_to_pixels(cells: gpd.GeoDataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Nearest valid gridMET pixel for each H3 cell centroid (handles coastal NaNs)."""
    d = (cells.lat.values[:, None] - px.lat.values) ** 2 + \
        ((cells.lon.values[:, None] - px.lon.values) * np.cos(np.radians(33))) ** 2
    return pd.DataFrame({"h3": cells.h3.values, "pixel": px.pixel.values[d.argmin(1)]})


def to_arrays(daily: pd.DataFrame, px: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict]:
    """Daily long frame -> {var: (days x pixels) array}."""
    d = daily.reset_index().merge(px, on=["lat", "lon"])
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    arrs = {v: d.pivot(index="date", columns="pixel", values=v).reindex(dates)
            .reindex(columns=px.pixel).to_numpy(float) for v in VARS.values()}
    return dates, arrs


def weekly_features(dates: pd.DatetimeIndex, arrs: dict, anchors) -> pd.DataFrame:
    """Fire-weather features for the 7 days starting at each anchor date, plus
    antecedent precipitation from the days *before* the anchor (no leakage)."""
    pos = dates.get_indexer(pd.DatetimeIndex(anchors))
    keep = (pos >= 365) & (pos + 7 <= len(dates))
    anchors, pos = pd.DatetimeIndex(anchors)[keep], pos[keep]

    win = {v: sliding_window_view(a, 7, axis=0)[pos] for v, a in arrs.items()}  # (A, P, 7)
    feats = {
        "tmax_max": win["tmax"].max(-1), "tmax_mean": win["tmax"].mean(-1),
        "rhmin_min": win["rhmin"].min(-1), "rhmin_mean": win["rhmin"].mean(-1),
        "wind_max": win["wind"].max(-1), "wind_mean": win["wind"].mean(-1),
        "vpd_max": win["vpd"].max(-1), "vpd_mean": win["vpd"].mean(-1),
        "pr_week": win["pr"].sum(-1),
    }
    cum = np.vstack([np.zeros((1, arrs["pr"].shape[1])), np.nancumsum(arrs["pr"], 0)])
    for n in (30, 90, 365):
        feats[f"pr_{n}d"] = cum[pos] - cum[pos - n]

    rain_idx = np.where(arrs["pr"] >= RAIN_MM, np.arange(len(dates))[:, None], np.nan)
    last_rain = pd.DataFrame(rain_idx).ffill().to_numpy()
    feats["days_since_rain"] = np.minimum(pos[:, None] - 1 - np.nan_to_num(last_rain[pos - 1], nan=-1e4), 365)

    A, P = len(anchors), arrs["pr"].shape[1]
    out = pd.DataFrame({k: v.reshape(-1) for k, v in feats.items()})
    out.insert(0, "week", np.repeat(anchors.values, P))
    out.insert(1, "pixel", np.tile(np.arange(P), A))
    return out


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    daily = download_gridmet("2000-01-01")
    daily.to_parquet(PROC / "gridmet_daily.parquet")
    px = pixel_table(daily)
    px.to_parquet(PROC / "gridmet_pixels.parquet")
    map_cells_to_pixels(cells, px).to_parquet(PROC / "cell_pixel.parquet")

    dates, arrs = to_arrays(daily, px)
    mondays = pd.date_range("2001-01-01", dates[-1], freq="W-MON")
    wk = weekly_features(dates, arrs, mondays)
    wk.to_parquet(PROC / "weather_weekly.parquet")
    print(f"{len(px)} pixels, {wk.week.nunique()} weeks ({wk.week.min():%Y-%m-%d} to {wk.week.max():%Y-%m-%d})")
