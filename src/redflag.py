"""NWS Red Flag Warnings (San Diego office, SGX) from the Iowa Environmental Mesonet archive,
mapped to (cell, week) for benchmarking. A cell-week is flagged if any zone-based warning
covering the cell center was in effect at any time that week."""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import requests

from config import PROC, RAW

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
FIRST_YEAR = 2005  # IEM's VTEC archive is complete from ~2005


def download_year(year: int) -> gpd.GeoDataFrame:
    params = {"year1": year, "month1": 1, "day1": 1, "hour1": 0, "minute1": 0,
              "year2": year + 1, "month2": 1, "day2": 1, "hour2": 0, "minute2": 0,
              "phenomena": "FW", "significance": "W", "location_group": "wfo", "wfo": "SGX"}
    cache = RAW / f"rfw_sgx_{year}.zip"
    if not cache.exists() or year >= pd.Timestamp.today().year:
        r = requests.get(IEM, params=params, timeout=300)
        r.raise_for_status()
        cache.write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(cache.read_bytes())) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
        tmp = PROC / "_rfw_tmp"
        z.extractall(tmp)
    g = gpd.read_file(tmp / shp)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    g = g[g.GTYPE == "C"]  # zone-based warnings
    g["start"] = pd.to_datetime(g.ISSUED, format="%Y%m%d%H%M").dt.tz_localize("UTC") \
        .dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    g["end"] = pd.to_datetime(g.EXPIRED, format="%Y%m%d%H%M").dt.tz_localize("UTC") \
        .dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    return g[["start", "end", "NWS_UGC", "geometry"]].to_crs(4326)


def cell_weeks(rfw: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    ctr = gpd.GeoDataFrame(cells[["h3"]], geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    hit = gpd.sjoin(ctr, rfw, predicate="within")
    rows = []
    for _, r in hit.iterrows():
        weeks = pd.date_range(r.start.to_period("W-SUN").start_time, r.end, freq="W-MON")
        rows += [(r.h3, w) for w in weeks]
    out = pd.DataFrame(rows, columns=["h3", "week"]).drop_duplicates()
    out["rfw"] = 1
    return out


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    rfw = pd.concat([download_year(y) for y in range(FIRST_YEAR, pd.Timestamp.today().year + 1)],
                    ignore_index=True)
    cw = cell_weeks(rfw, cells)
    cw.to_parquet(PROC / "redflag.parquet")
    print(f"{len(rfw):,} zone warnings {FIRST_YEAR}-now -> {len(cw):,} flagged cell-weeks, "
          f"{cw.week.nunique()} weeks with any warning")
