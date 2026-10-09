"""Daily refresh: next-7-day fire probability per cell + expected impact per community.

Recent observed weather comes from gridMET (same source as training); the days ahead
come from the Open-Meteo forecast, in the same units, so features line up with training.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests

from config import MODELS, PROC, ROOT
from src.features import (FEATURES, TARGETS, assemble, cell_weather, load_static, near_pairs,
                          perimeter_points)
from src.fires import cell_burns, download_perimeters
from src.ignitions import FOD_LAST_YEAR, assign_cells, load_wfigs
from src.modeling import Ensemble  # noqa: F401  (needed to unpickle)
from src.ndvi import load_ndvi
from src.static import catalog
from src.train import Stacked  # noqa: F401  (needed to unpickle)
from src.weather import download_gridmet, load_erc_reference, to_arrays, weekly_features

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OUT = ROOT / "data" / "forecast"
HISTORY_DIR = OUT / "history"
TOP_DRIVERS = 3
MAX_OBS_LAG_DAYS = 5     # gridMET normally lags 1-2 days
MAX_NDVI_AGE_DAYS = 60   # MODIS composites can lag; older -> warning only


def _get(params: dict, tries: int = 5) -> list:
    for i in range(tries):
        r = requests.get(OPEN_METEO, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        time.sleep(30 * (i + 1))
    r.raise_for_status()


def open_meteo(px: pd.DataFrame, step: float = 0.1, batch: int = 50) -> pd.DataFrame:
    """Daily forecast for each gridMET pixel, in gridMET units. Queried on a coarser
    `step`-degree grid (forecast models are ~3-11 km anyway) to stay within rate limits."""
    px = px.assign(qlat=(px.lat / step).round() * step, qlon=(px.lon / step).round() * step)
    pts = px[["qlat", "qlon"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for i in range(0, len(pts), batch):
        chunk = pts.iloc[i:i + batch]
        data = _get({
            "latitude": ",".join(chunk.qlat.round(3).astype(str)),
            "longitude": ",".join(chunk.qlon.round(3).astype(str)),
            "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_mean,precipitation_sum,"
                     "wind_direction_10m_dominant",
            "hourly": "vapour_pressure_deficit", "wind_speed_unit": "ms",
            "past_days": 3, "forecast_days": 8, "timezone": "America/Los_Angeles"})
        for (_, p), d in zip(chunk.iterrows(), data):
            day = pd.DataFrame(d["daily"])
            vpd = pd.DataFrame(d["hourly"])
            vpd = vpd.groupby(vpd.time.str[:10]).vapour_pressure_deficit.mean()
            rows.append(pd.DataFrame({
                "date": pd.to_datetime(day.time), "qlat": p.qlat, "qlon": p.qlon,
                "tmax": day.temperature_2m_max, "rhmin": day.relative_humidity_2m_min,
                "wind": day.wind_speed_10m_mean, "vpd": day.time.map(vpd).to_numpy(),
                "pr": day.precipitation_sum, "wdir": day.wind_direction_10m_dominant}))
        time.sleep(1)
    fc = pd.concat(rows, ignore_index=True)
    return px[["lat", "lon", "qlat", "qlon"]].merge(fc, on=["qlat", "qlon"]).drop(columns=["qlat", "qlon"])


def daily_weather(px: pd.DataFrame) -> pd.DataFrame:
    """gridMET for the past ~13 months, Open-Meteo for every day after gridMET ends."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    obs = download_gridmet(start).reset_index().merge(px[["lat", "lon"]], on=["lat", "lon"])
    fc = open_meteo(px)
    fc = fc[fc.date > obs.date.max()]
    return pd.concat([obs, fc], ignore_index=True).set_index(["date", "lat", "lon"]), obs.date.max()


def latest_ndvi() -> pd.DataFrame:
    ndvi = pd.read_parquet(PROC / "ndvi.parquet")
    since = ndvi.composite.max() + pd.Timedelta(days=1)
    new, _ = load_ndvi(catalog(), f"{since:%Y-%m-%d}", f"{pd.Timestamp.today():%Y-%m-%d}")
    if len(new):
        ndvi = pd.concat([ndvi, new], ignore_index=True).drop_duplicates(["h3", "composite"])
        ndvi.to_parquet(PROC / "ndvi.parquet")
    return ndvi


def latest_ignitions(cells) -> pd.DataFrame:
    """Archived FPA-FOD years + a fresh pull of WFIGS incidents for recent years."""
    ign = pd.read_parquet(PROC / "ignitions.parquet")
    fresh = assign_cells(load_wfigs(FOD_LAST_YEAR + 1), cells)
    ign = pd.concat([ign[ign.source != "WFIGS"], fresh[fresh.date.dt.year > FOD_LAST_YEAR]], ignore_index=True)
    ign.to_parquet(PROC / "ignitions.parquet")
    return ign


class DataError(RuntimeError):
    """Raised on bad inputs so the scheduled job fails loudly (GitHub emails on failure)."""


def validate(X: pd.DataFrame, obs_end: pd.Timestamp, ndvi: pd.DataFrame, anchor) -> list[str]:
    errors, warnings = [], []
    missing = X[FEATURES].isna().mean()
    if (missing > 0).any():
        errors.append(f"missing features: {missing[missing > 0].round(3).to_dict()}")
    checks = {"tmax_max": (-20, 55), "rhmin_min": (0, 100), "wind_max": (0, 40), "pr_week": (0, 1000),
              "erc_pct_prev": (0, 1)}
    for col, (lo, hi) in checks.items():
        if not X[col].between(lo, hi).all():
            errors.append(f"{col} outside [{lo}, {hi}]: {X[col].min():.1f}..{X[col].max():.1f}")
    lag = (anchor - obs_end).days
    if lag > MAX_OBS_LAG_DAYS:
        errors.append(f"gridMET observations are {lag} days old")
    ndvi_age = (anchor - ndvi.available.max()).days
    if ndvi_age > MAX_NDVI_AGE_DAYS:
        warnings.append(f"latest NDVI composite is {ndvi_age} days old")
    if errors:
        raise DataError("; ".join(errors))
    return warnings


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)
    cells = gpd.read_parquet(PROC / "cells.parquet")
    px = pd.read_parquet(PROC / "gridmet_pixels.parquet")
    px = px[px.pixel.isin(pd.read_parquet(PROC / "cell_pixel.parquet").pixel)]
    erc_ref = load_erc_reference()

    fires = download_perimeters()
    burns = cell_burns(fires, cells)
    ign = latest_ignitions(cells)
    ndvi = latest_ndvi()

    daily, obs_end = daily_weather(px)
    dates, arrs = to_arrays(daily, px)
    anchor = pd.Timestamp.today().normalize()
    wk = weekly_features(dates, arrs, [anchor], erc_ref)
    if wk.empty:
        raise DataError("Not enough weather data to cover the next 7 days")

    base = cells[["h3"]].assign(week=anchor.to_datetime64())
    perim_near = near_pairs(perimeter_points(fires), cells)
    X = assemble(base, cell_weather(wk), load_static(), ndvi, burns, perim_near, ign, cells)
    warnings = validate(X, obs_end, ndvi, anchor)

    versions = {}
    for t in TARGETS:
        model = joblib.load(MODELS / f"ensemble_{t}.joblib")
        X[f"p_{t}"], X[f"sd_{t}"], X[f"score_{t}"] = model.predict(X)
        sv = model.shap_values(X)
        order = np.argsort(-sv, axis=1)[:, :TOP_DRIVERS]  # features pushing risk up the most
        X[f"drivers_{t}"] = [", ".join(model.features[j] for j in row) for row in order]
        versions[t] = json.loads((MODELS / f"model_card_{t}.json").read_text())["version"]

    pop = pd.read_parquet(PROC / "cell_pop.parquet")
    X = X.merge(pop, on="h3", how="left")
    X["exp_people"] = X.p_fire * X["pop"]
    X.to_parquet(OUT / "cells.parquet")
    X[["h3", *[f"{k}_{t}" for t in TARGETS for k in ("p", "score")]]] \
        .to_parquet(HISTORY_DIR / f"{anchor:%Y-%m-%d}.parquet")

    pc = pd.read_parquet(PROC / "place_cell.parquet").merge(X[["h3", "p_fire", "p_ign", "p_large"]], on="h3")
    comm = gpd.read_parquet(PROC / "communities.parquet")
    impact = pc.assign(exp_people=pc["pop"] * pc.p_fire, exp_homes=pc.housing * pc.p_fire) \
        .groupby("GEOID")[["exp_people", "exp_homes"]].sum()
    peak = pc.groupby("GEOID")[["p_fire", "p_ign", "p_large"]].max().add_prefix("max_")
    comm = comm.join(impact, on="GEOID").join(peak, on="GEOID")
    comm.drop(columns="geometry").to_parquet(OUT / "communities.parquet")

    meta = {"issued": pd.Timestamp.now().isoformat(timespec="minutes"),
            "valid_from": f"{anchor:%Y-%m-%d}",
            "valid_to": f"{anchor + pd.Timedelta(days=6):%Y-%m-%d}",
            "observed_weather_through": f"{obs_end:%Y-%m-%d}",
            "model_versions": versions, "warnings": warnings}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    print(meta)

    from src.monitor import run as monitor
    monitor(ign)


if __name__ == "__main__":
    run()
