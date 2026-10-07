"""Assemble the (cell x week) modeling panel. `assemble` is shared with the live forecast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CRS_M, PROC, TEST_START

CELL_KM2 = 5.16  # mean H3 res-7 cell area
NO_BURN_YEARS = 75
IGNITION_KM = 10
IGNITION_YEARS = 20

WEATHER = ["tmax_max", "tmax_mean", "rhmin_min", "rhmin_mean", "wind_max", "wind_mean",
           "vpd_max", "vpd_mean", "pr_week", "pr_30d", "pr_90d", "pr_365d", "days_since_rain"]
STATIC = ["elev_mean", "elev_std", "slope_mean", "northness", "frac_tree", "frac_shrub",
          "frac_grass", "frac_crop", "frac_built", "frac_bare", "log_pop_density"]
FEATURES = WEATHER + STATIC + ["ndvi", "ndvi_anom", "years_since_burn", "prior_burns",
                               "ignitions_10km_20y", "woy_sin", "woy_cos"]


def ndvi_climatology(ndvi: pd.DataFrame) -> pd.DataFrame:
    """Mean NDVI per (cell, composite day-of-year), from training years only."""
    train = ndvi[ndvi.composite < TEST_START]
    return train.groupby(["h3", train.composite.dt.dayofyear.rename("doy")]).ndvi.mean() \
        .rename("ndvi_clim").reset_index()


def ignition_pairs(fires, cells) -> pd.DataFrame:
    """(cell, alarm_date) for every fire whose origin (approximated by the perimeter's
    representative point) lies within IGNITION_KM of the cell center."""
    pts = fires.dropna(subset=["alarm_date"]).to_crs(CRS_M)
    pts = pts.set_geometry(pts.representative_point())[["alarm_date", "geometry"]]
    ctr = cells.to_crs(CRS_M)
    ctr = ctr.set_geometry(ctr.centroid)[["h3", "geometry"]]
    pairs = gpd.sjoin(ctr.assign(geometry=ctr.buffer(IGNITION_KM * 1000)), pts, predicate="contains")
    return pairs[["h3", "alarm_date"]].reset_index(drop=True)


def asof(base: pd.DataFrame, right: pd.DataFrame, right_on: str, cols: list) -> pd.DataFrame:
    """Latest `right` row per cell with right_on strictly before the week."""
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")}).sort_values(right_on)
    base = base.assign(week=base.week.astype("datetime64[ns]"))
    out = pd.merge_asof(base.sort_values("week"), right[["h3", right_on] + cols],
                        left_on="week", right_on=right_on, by="h3",
                        allow_exact_matches=False)
    return out.drop(columns=right_on)


def assemble(base: pd.DataFrame, weather: pd.DataFrame, static: pd.DataFrame,
             ndvi: pd.DataFrame, burns: pd.DataFrame, ignitions: pd.DataFrame) -> pd.DataFrame:
    """base: [h3, week]. weather: [h3, week, WEATHER...]. Returns base + FEATURES."""
    df = base.merge(weather, on=["h3", "week"], how="left").merge(static, on="h3", how="left")

    nd = ndvi.dropna(subset=["ndvi"]).copy()
    nd["doy"] = nd.composite.dt.dayofyear
    nd = nd.merge(ndvi_climatology(ndvi), on=["h3", "doy"], how="left")
    nd["ndvi_anom"] = nd.ndvi - nd.ndvi_clim
    df = asof(df, nd, "available", ["ndvi", "ndvi_anom"])

    hist = burns.drop_duplicates(["h3", "alarm_date"]).sort_values("alarm_date")
    hist["prior_burns"] = hist.groupby("h3").cumcount() + 1
    hist["last_burn"] = hist.alarm_date
    df = asof(df, hist, "alarm_date", ["last_burn", "prior_burns"])
    df["years_since_burn"] = ((df.week - df.last_burn).dt.days / 365.25).fillna(NO_BURN_YEARS)
    df["prior_burns"] = df.prior_burns.fillna(0)

    ign = ignitions.sort_values("alarm_date")
    ign["n_ign"] = ign.groupby("h3").cumcount() + 1
    now = asof(df[["h3", "week"]], ign, "alarm_date", ["n_ign"])
    past = asof(df[["h3", "week"]].assign(week=df.week - pd.DateOffset(years=IGNITION_YEARS)),
                ign, "alarm_date", ["n_ign"])
    past["week"] = past.week + pd.DateOffset(years=IGNITION_YEARS)
    df = df.merge(now.rename(columns={"n_ign": "a"}), on=["h3", "week"]) \
           .merge(past.rename(columns={"n_ign": "b"}), on=["h3", "week"])
    df["ignitions_10km_20y"] = df.pop("a").fillna(0) - df.pop("b").fillna(0)

    woy = df.week.dt.dayofyear / 365.25 * 2 * np.pi
    df["woy_sin"], df["woy_cos"] = np.sin(woy), np.cos(woy)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df.drop(columns="last_burn").sort_values(["week", "h3"]).reset_index(drop=True)


def load_static() -> pd.DataFrame:
    static = pd.read_parquet(PROC / "static.parquet")
    pop = pd.read_parquet(PROC / "cell_pop.parquet")
    static = static.merge(pop[["h3", "pop"]], on="h3", how="left")
    static["log_pop_density"] = np.log1p(static.pop("pop").fillna(0) / CELL_KM2)
    return static


def cell_weather(weekly_px: pd.DataFrame) -> pd.DataFrame:
    cp = pd.read_parquet(PROC / "cell_pixel.parquet")
    return cp.merge(weekly_px, on="pixel").drop(columns="pixel")


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    burns = pd.read_parquet(PROC / "burns.parquet")
    weather = cell_weather(pd.read_parquet(PROC / "weather_weekly.parquet"))
    last_label_week = pd.Timestamp(f"{burns.alarm_date.dt.year.max()}-12-31")
    weeks = weather.week.unique()
    weeks = weeks[weeks <= last_label_week]

    base = pd.MultiIndex.from_product([cells.h3, weeks], names=["h3", "week"]).to_frame(index=False)
    fires = gpd.read_parquet(PROC / "fires.parquet")
    panel = assemble(base, weather, load_static(), pd.read_parquet(PROC / "ndvi.parquet"), burns,
                     ignition_pairs(fires, cells))
    size = burns.groupby(["FIRE_NAME", "alarm_date"]).h3.transform("size")
    pos = burns.assign(event_cells=size).groupby(["h3", "week"]).event_cells.max() \
        .reset_index().assign(fire=1)
    panel = panel.merge(pos, on=["h3", "week"], how="left").fillna({"fire": 0, "event_cells": 0})
    panel["fire"] = panel.fire.astype("int8")
    panel = panel.merge(cells[["h3", "block"]], on="h3")
    panel.to_parquet(PROC / "panel.parquet")
    print(f"{len(panel):,} rows, {panel.fire.sum():,} positives ({panel.fire.mean():.4%}), "
          f"{panel.week.min():%Y-%m-%d} to {panel.week.max():%Y-%m-%d}")
    print("missing share:\n", panel[FEATURES].isna().mean()[lambda s: s > 0].round(4).to_string())
