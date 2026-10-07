"""Download CAL FIRE (FRAP) perimeters and turn them into weekly cell labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import requests

from config import BBOX, PROC

FRAP_URL = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
            "California_Historic_Fire_Perimeters/FeatureServer/0/query")


def download_perimeters() -> gpd.GeoDataFrame:
    params = {
        "where": "YEAR_ >= 1950",
        "geometry": ",".join(map(str, BBOX)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FIRE_NAME,YEAR_,ALARM_DATE,CONT_DATE,CAUSE,GIS_ACRES",
        "f": "geojson", "resultRecordCount": 1000,
    }
    pages, offset = [], 0
    while True:
        r = requests.get(FRAP_URL, params={**params, "resultOffset": offset}, timeout=120)
        r.raise_for_status()
        page = gpd.GeoDataFrame.from_features(r.json()["features"], crs=4326)
        if page.empty:
            break
        pages.append(page)
        offset += len(page)
    fires = pd.concat(pages, ignore_index=True)
    fires["alarm_date"] = pd.to_datetime(fires.ALARM_DATE, unit="ms").dt.normalize()
    fires = fires[fires.geometry.notna()]
    fires["geometry"] = fires.geometry.make_valid()
    return fires


def cell_burns(fires: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per (cell, fire) where the fire perimeter touches the cell."""
    hit = gpd.sjoin(cells[["h3", "geometry"]], fires[["FIRE_NAME", "alarm_date", "geometry"]],
                    predicate="intersects")
    burns = hit[["h3", "FIRE_NAME", "alarm_date"]].dropna(subset=["alarm_date"])
    burns["week"] = burns.alarm_date.dt.to_period("W-SUN").dt.start_time
    return burns.drop_duplicates(["h3", "FIRE_NAME", "alarm_date"]).reset_index(drop=True)


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    fires = download_perimeters()
    fires.to_parquet(PROC / "fires.parquet")
    burns = cell_burns(fires, cells)
    burns.to_parquet(PROC / "burns.parquet")
    print(f"{len(fires)} perimeters ({fires.YEAR_.min()}-{fires.YEAR_.max()}), "
          f"{len(burns)} cell-burns, {burns.alarm_date.isna().sum()} undated")
