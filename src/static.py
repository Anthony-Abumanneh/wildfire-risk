"""Terrain (Copernicus DEM) and land cover (ESA WorldCover) aggregated to H3 cells."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
from odc.stac import load

from config import BBOX, H3_RES, PROC

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
RES = 0.001  # degrees (~100 m)
LANDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare"}


def catalog():
    return pystac_client.Client.open(STAC, modifier=pc.sign_inplace)


def pixel_cells(da) -> np.ndarray:
    """H3 cell id for every pixel center of a lat/lon raster."""
    lat, lon = np.meshgrid(da.latitude.values, da.longitude.values, indexing="ij")
    return np.array([h3.latlng_to_cell(a, o, H3_RES) for a, o in zip(lat.ravel(), lon.ravel())])


def terrain(cat) -> pd.DataFrame:
    items = cat.search(collections=["cop-dem-glo-30"], bbox=BBOX).item_collection()
    dem = load(items, bands=["data"], crs="EPSG:4326", resolution=RES, bbox=BBOX,
               resampling="bilinear").data.isel(time=0)
    z = dem.values.astype(float)
    lat = dem.latitude.values[:, None]
    dy = RES * 111_320
    dx = RES * 111_320 * np.cos(np.radians(lat))
    gy, gx = np.gradient(z)
    sz, sx = -gy / dy, gx / dx            # rows run north -> south
    slope = np.degrees(np.arctan(np.hypot(sx, sz)))
    aspect = np.arctan2(-sx, -sz)         # downslope direction, 0 = north
    df = pd.DataFrame({"h3": pixel_cells(dem), "elev": z.ravel(), "slope": slope.ravel(),
                       "northness": np.cos(aspect).ravel()})
    return df.groupby("h3").agg(elev_mean=("elev", "mean"), elev_std=("elev", "std"),
                                slope_mean=("slope", "mean"), northness=("northness", "mean"))


def landcover(cat) -> pd.DataFrame:
    items = cat.search(collections=["esa-worldcover"], bbox=BBOX,
                       query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).item_collection()
    lc = load(items, bands=["map"], crs="EPSG:4326", resolution=RES, bbox=BBOX,
              resampling="mode")["map"].isel(time=0)
    df = pd.DataFrame({"h3": pixel_cells(lc), "cls": lc.values.ravel()})
    frac = pd.crosstab(df.h3, df.cls, normalize="index")
    out = pd.DataFrame(index=frac.index)
    for code, name in LANDCOVER.items():
        out[f"frac_{name}"] = frac[code] if code in frac else 0.0
    return out


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    cat = catalog()
    static = cells[["h3"]].set_index("h3").join(terrain(cat)).join(landcover(cat))
    static.reset_index().to_parquet(PROC / "static.parquet")
    print(static.describe().T[["mean", "min", "max"]].round(2))
    print("missing:", static.isna().sum().sum())
