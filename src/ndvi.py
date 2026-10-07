"""MODIS 16-day NDVI (MOD13Q1, 250 m) aggregated to H3 cells."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from odc.stac import load

from config import BBOX, PROC
from src.static import catalog, pixel_cells

RES = 0.0025  # degrees (~250 m)


def load_ndvi(cat, start: str, end: str, cell_ids=None) -> pd.DataFrame:
    """Mean NDVI per cell per composite; `available` = date the composite can be used."""
    items = cat.search(collections=["modis-13Q1-061"], bbox=BBOX, datetime=f"{start}/{end}").item_collection()
    items = [i for i in items if i.id.startswith("MOD13Q1")]  # Terra; `platform` tag is often blank
    if len(items) == 0:
        return pd.DataFrame(), cell_ids
    ds = load(items, bands=["250m_16_days_NDVI"], crs="EPSG:4326", resolution=RES, bbox=BBOX,
              resampling="nearest", groupby="solar_day")
    da = ds["250m_16_days_NDVI"]
    vals = da.values.astype(float).reshape(da.sizes["time"], -1)
    vals[vals == -3000] = np.nan
    vals *= 1e-4
    if cell_ids is None:
        cell_ids = pixel_cells(da)
    end_dates = {pd.Timestamp(i.datetime or i.properties["start_datetime"]).tz_localize(None).normalize():
                 pd.Timestamp(i.properties["end_datetime"]).tz_localize(None).normalize() for i in items}
    rows = []
    for t, v in zip(pd.DatetimeIndex(da.time.values), vals):
        m = pd.Series(v).groupby(cell_ids).mean()
        avail = end_dates.get(t.normalize(), t + pd.Timedelta(days=16)) + pd.Timedelta(days=1)
        rows.append(pd.DataFrame({"h3": m.index, "composite": t, "available": avail, "ndvi": m.values}))
    return pd.concat(rows, ignore_index=True), cell_ids


if __name__ == "__main__":
    cat = catalog()
    parts, ids = [], None
    years = [int(y) for y in sys.argv[1:]] or range(2000, pd.Timestamp.today().year + 1)
    if sys.argv[1:]:  # patch specific years into the existing file
        parts.append(pd.read_parquet(PROC / "ndvi.parquet"))
    for year in years:
        df, ids = load_ndvi(cat, f"{year}-01-01", f"{year}-12-31", ids)
        parts.append(df)
        print(f"  {year}: {df.composite.nunique()} composites")
    ndvi = pd.concat(parts, ignore_index=True).drop_duplicates(["h3", "composite"], keep="last")
    ndvi.to_parquet(PROC / "ndvi.parquet")
    print(f"{len(ndvi):,} rows, {ndvi.composite.nunique()} composites, missing {ndvi.ndvi.isna().mean():.1%}")
