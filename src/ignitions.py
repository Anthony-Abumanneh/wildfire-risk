"""Wildfire ignition points (where and when fires started, any size).

- FPA-FOD 6th ed. (USFS, 1992-2020): every reported wildfire with point of origin and size.
- WFIGS incident locations (NIFC/IRWIN, 2021+): same fields for recent years.
Output: data/processed/ignitions.parquet with [date, lat, lon, acres, cause, source, h3].
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import h3
import pandas as pd
import requests

from config import BBOX, H3_RES, PROC, RAW

FOD_ZIP = RAW / "fpa_fod.zip"
WFIGS_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
             "WFIGS_Incident_Locations/FeatureServer/0/query")
FOD_LAST_YEAR = 2020


def load_fpa_fod() -> pd.DataFrame:
    """Extract the GeoPackage, keep the county bbox, delete the big files."""
    gpkg = next(RAW.glob("**/*.gpkg"), None)
    if gpkg is None:
        subprocess.run(["unzip", "-o", "-q", str(FOD_ZIP), "-d", str(RAW / "fpa_fod")], check=True)
        gpkg = next(RAW.glob("**/*.gpkg"))
    df = gpd.read_file(gpkg, bbox=BBOX, ignore_geometry=True)
    out = pd.DataFrame({
        "date": pd.to_datetime(df.DISCOVERY_DATE).dt.tz_localize(None).dt.normalize(),
        "lat": df.LATITUDE, "lon": df.LONGITUDE, "acres": df.FIRE_SIZE,
        "cause": df.NWCG_GENERAL_CAUSE, "name": df.FIRE_NAME, "source": "FPA-FOD"})
    return out


def load_wfigs(first_year: int) -> pd.DataFrame:
    params = {
        "where": f"IncidentTypeCategory = 'WF' AND FireDiscoveryDateTime >= DATE '{first_year}-01-01'",
        "geometry": ",".join(map(str, BBOX)), "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FireDiscoveryDateTime,IncidentSize,FinalAcres,DiscoveryAcres,"
                     "FireCauseGeneral,IncidentName,UniqueFireIdentifier",
        "f": "json", "resultRecordCount": 2000,
    }
    rows, offset = [], 0
    while True:
        r = requests.get(WFIGS_URL, params={**params, "resultOffset": offset}, timeout=120)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        rows += [{**f["attributes"], "lon": f["geometry"]["x"], "lat": f["geometry"]["y"]} for f in feats]
        offset += len(feats)
    df = pd.DataFrame(rows).drop_duplicates("UniqueFireIdentifier")
    acres = df[["FinalAcres", "IncidentSize", "DiscoveryAcres"]].bfill(axis=1).iloc[:, 0]
    return pd.DataFrame({
        "date": pd.to_datetime(df.FireDiscoveryDateTime, unit="ms", utc=True)
                  .dt.tz_convert("America/Los_Angeles").dt.tz_localize(None).dt.normalize(),
        "lat": df.lat, "lon": df.lon, "acres": acres.fillna(0),
        "cause": df.FireCauseGeneral, "name": df.IncidentName, "source": "WFIGS"})


def assign_cells(ign: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    ign = ign.dropna(subset=["date", "lat", "lon"]).copy()
    ign["h3"] = [h3.latlng_to_cell(a, o, H3_RES) for a, o in zip(ign.lat, ign.lon)]
    ign = ign[ign.h3.isin(set(cells.h3))]
    ign["week"] = ign.date.dt.to_period("W-SUN").dt.start_time
    return ign.reset_index(drop=True)


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    fod = load_fpa_fod()
    wf = load_wfigs(2014)
    overlap = pd.concat([fod.assign(y=fod.date.dt.year), wf.assign(y=wf.date.dt.year)])
    print("annual counts (bbox):\n", overlap[overlap.y >= 2014].groupby(["y", "source"]).size()
          .unstack().to_string())

    ign = pd.concat([fod, wf[wf.date.dt.year > FOD_LAST_YEAR]], ignore_index=True)
    ign = assign_cells(ign, cells)
    ign.to_parquet(PROC / "ignitions.parquet")
    print(f"{len(ign):,} ignitions in county {ign.date.min():%Y}-{ign.date.max():%Y}; "
          f">=100 ac: {(ign.acres >= 100).sum()}")
