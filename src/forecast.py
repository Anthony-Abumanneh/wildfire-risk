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
from src.features import FEATURES, assemble, cell_weather, ignition_pairs, load_static
from src.fires import cell_burns, download_perimeters
from src.ndvi import load_ndvi
from src.static import catalog
from src.train import Ensemble  # noqa: F401  (needed to unpickle)
from src.weather import download_gridmet, to_arrays, weekly_features

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OUT = ROOT / "data" / "forecast"
TOP_DRIVERS = 3


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
            "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_mean,precipitation_sum",
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
                "pr": day.precipitation_sum}))
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


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = gpd.read_parquet(PROC / "cells.parquet")
    px = pd.read_parquet(PROC / "gridmet_pixels.parquet")
    px = px[px.pixel.isin(pd.read_parquet(PROC / "cell_pixel.parquet").pixel)]

    fires = download_perimeters()
    burns = cell_burns(fires, cells)

    daily, obs_end = daily_weather(px)
    dates, arrs = to_arrays(daily, px)
    anchor = pd.Timestamp.today().normalize()
    wk = weekly_features(dates, arrs, [anchor])
    if wk.empty:
        raise RuntimeError("Not enough weather data to cover the next 7 days")

    base = cells[["h3"]].assign(week=anchor)
    X = assemble(base, cell_weather(wk), load_static(), latest_ndvi(), burns,
                 ignition_pairs(fires, cells))

    ens = joblib.load(MODELS / "ensemble.joblib")
    X["p_fire"], X["p_sd"], X["score"] = ens.predict(X)
    sv = ens.shap_values(X)
    order = np.argsort(-sv, axis=1)[:, :TOP_DRIVERS]  # features pushing risk up the most
    X["drivers"] = [", ".join(FEATURES[j] for j in row) for row in order]

    pop = pd.read_parquet(PROC / "cell_pop.parquet")
    X = X.merge(pop, on="h3", how="left")
    X["exp_people"] = X.p_fire * X["pop"]
    X.to_parquet(OUT / "cells.parquet")

    pc = pd.read_parquet(PROC / "place_cell.parquet").merge(X[["h3", "p_fire"]], on="h3")
    comm = gpd.read_parquet(PROC / "communities.parquet")
    impact = pc.assign(exp_people=pc["pop"] * pc.p_fire, exp_homes=pc.housing * pc.p_fire) \
        .groupby("GEOID")[["exp_people", "exp_homes"]].sum()
    peak = pc.groupby("GEOID").p_fire.max().rename("max_cell_p")
    comm = comm.join(impact, on="GEOID").join(peak, on="GEOID")
    comm.drop(columns="geometry").to_parquet(OUT / "communities.parquet")

    meta = {"issued": pd.Timestamp.now().isoformat(timespec="minutes"),
            "valid_from": f"{anchor:%Y-%m-%d}",
            "valid_to": f"{anchor + pd.Timedelta(days=6):%Y-%m-%d}",
            "observed_weather_through": f"{obs_end:%Y-%m-%d}"}
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    print(meta)
    print(comm.sort_values("exp_people", ascending=False)
          [["NAME", "pop", "max_cell_p", "exp_people", "escape_routes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run()
