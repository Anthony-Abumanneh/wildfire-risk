"""Build the H3 prediction grid over San Diego County."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from config import BLOCK_RES, COUNTY_FIPS, H3_RES, PROC, STATE_FIPS

COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"


def load_county() -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTY_URL)
    county = counties[(counties.STATEFP == STATE_FIPS) & (counties.COUNTYFP == COUNTY_FIPS)]
    return county.to_crs(4326)[["NAME", "geometry"]].reset_index(drop=True)


def build_grid(county: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cells = h3.geo_to_cells(county.geometry.iloc[0], H3_RES)
    rows = []
    for c in cells:
        lat, lon = h3.cell_to_latlng(c)
        boundary = [(lng, lt) for lt, lng in h3.cell_to_boundary(c)]
        rows.append({"h3": c, "lat": lat, "lon": lon,
                     "block": h3.cell_to_parent(c, BLOCK_RES), "geometry": Polygon(boundary)})
    return gpd.GeoDataFrame(rows, crs=4326).sort_values("h3").reset_index(drop=True)


if __name__ == "__main__":
    county = load_county()
    county.to_parquet(PROC / "county.parquet")
    grid = build_grid(county)
    grid.to_parquet(PROC / "cells.parquet")
    print(f"{len(grid)} cells, {grid.block.nunique()} CV blocks")
