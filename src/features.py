"""Assemble the (cell x week) modeling panel. `assemble` is shared with the live forecast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from config import CRS_M, PROC, TEST_START

CELL_KM2 = 5.16  # mean H3 res-7 cell area
NO_BURN_YEARS = 75
NEAR_KM = 10
LARGE_ACRES = 100

# feature groups (used by experiments.py for ablations)
WEATHER = ["tmax_max", "tmax_mean", "rhmin_min", "rhmin_mean", "wind_max", "wind_mean",
           "vpd_max", "vpd_mean", "pr_week", "pr_30d", "pr_90d", "pr_365d", "days_since_rain"]
FUELS = ["fm100_prev", "fm1000_prev", "erc_prev", "bi_prev", "erc_7d_prev", "erc_pct_prev",
         "offshore_days", "santa_ana_days", "santa_ana_wind_max"]
STATIC = ["elev_mean", "elev_std", "slope_mean", "northness", "frac_tree", "frac_shrub",
          "frac_grass", "frac_crop", "frac_built", "frac_bare", "log_pop_density"]
HUMAN = ["road_km_per_km2", "dist_major_road_km", "power_km_per_km2", "dist_power_line_km",
         "dist_campground_km", "dist_border_km", "dist_coast_km", "ring_veg", "ring_frac_shrub",
         "ring_frac_built", "ring_housing_density", "wui_intermix", "wui_interface",
         "log_housing_density"]
VEG = ["ndvi", "ndvi_anom"]
HISTORY = ["years_since_burn", "prior_burns", "perim_10km_20y"]
IGNITION_HISTORY = ["ign_cell_10y", "ign_10km_30d", "ign_10km_365d", "ign_10km_10y", "large_10km_10y"]
SEASON = ["woy_sin", "woy_cos"]
FEATURES = WEATHER + FUELS + STATIC + HUMAN + VEG + HISTORY + IGNITION_HISTORY + SEASON
TARGETS = {"ign": "Any ignition", "large": f"Fire >= {LARGE_ACRES} acres starts", "fire": "Cell burns"}
# chosen by leave-years-out CV PR-AUC on 2001-2018 (reports/experiments.csv)
TARGET_FEATURES = {"ign": FEATURES, "large": FEATURES,
                   "fire": [f for f in FEATURES if f not in IGNITION_HISTORY]}


def ndvi_climatology(ndvi: pd.DataFrame) -> pd.DataFrame:
    """Mean NDVI per (cell, composite day-of-year), from training years only."""
    train = ndvi[ndvi.composite < TEST_START]
    return train.groupby(["h3", train.composite.dt.dayofyear.rename("doy")]).ndvi.mean() \
        .rename("ndvi_clim").reset_index()


def near_pairs(pts: gpd.GeoDataFrame, cells: gpd.GeoDataFrame, km: float = NEAR_KM) -> pd.DataFrame:
    """(cell, date) for every event point within `km` of the cell center."""
    pts = pts.to_crs(CRS_M)[["date", "geometry"]]
    ctr = cells.to_crs(CRS_M)
    ctr = ctr.assign(geometry=ctr.centroid.buffer(km * 1000))[["h3", "geometry"]]
    return gpd.sjoin(ctr, pts, predicate="contains")[["h3", "date"]].reset_index(drop=True)


def perimeter_points(fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fire origins approximated by perimeter representative points (back to 1950)."""
    f = fires.dropna(subset=["alarm_date"])
    return gpd.GeoDataFrame({"date": f.alarm_date}, geometry=f.representative_point(), crs=f.crs)


def ignition_points(ign: pd.DataFrame, min_acres: float = 0) -> gpd.GeoDataFrame:
    ign = ign[ign.acres >= min_acres]
    return gpd.GeoDataFrame({"date": ign.date}, geometry=gpd.points_from_xy(ign.lon, ign.lat), crs=4326)


def count_before(df: pd.DataFrame, events: pd.DataFrame, at: pd.Series) -> np.ndarray:
    """Number of events per cell strictly before `at` (aligned with df rows)."""
    ev = events[["h3", "date"]].astype({"date": "datetime64[ns]"}).sort_values("date")
    ev["n"] = ev.groupby("h3").cumcount() + 1
    q = pd.DataFrame({"h3": df.h3.to_numpy(), "t": at.astype("datetime64[ns]").to_numpy(),
                      "i": np.arange(len(df))}).sort_values("t")
    m = pd.merge_asof(q, ev, left_on="t", right_on="date", by="h3", allow_exact_matches=False)
    return m.sort_values("i").n.fillna(0).to_numpy()


def window_count(df, events, offset) -> np.ndarray:
    """Events per cell in [week - offset, week)."""
    return count_before(df, events, df.week) - count_before(df, events, df.week - offset)


def asof(base: pd.DataFrame, right: pd.DataFrame, right_on: str, cols: list) -> pd.DataFrame:
    """Latest `right` row per cell with right_on strictly before the week."""
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")}).sort_values(right_on)
    base = base.assign(week=base.week.astype("datetime64[ns]"))
    out = pd.merge_asof(base.sort_values("week"), right[["h3", right_on] + cols],
                        left_on="week", right_on=right_on, by="h3", allow_exact_matches=False)
    return out.drop(columns=right_on)


def assemble(base: pd.DataFrame, weather: pd.DataFrame, static: pd.DataFrame, ndvi: pd.DataFrame,
             burns: pd.DataFrame, perim_near: pd.DataFrame, ign: pd.DataFrame,
             cells: gpd.GeoDataFrame) -> pd.DataFrame:
    """base: [h3, week]. Returns base + FEATURES. Every history feature uses only events
    strictly before the week."""
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
    df = df.drop(columns="last_burn").reset_index(drop=True)

    y10, y20 = pd.DateOffset(years=10), pd.DateOffset(years=20)
    df["perim_10km_20y"] = window_count(df, perim_near, y20)
    near = near_pairs(ignition_points(ign), cells)
    near_large = near_pairs(ignition_points(ign, LARGE_ACRES), cells)
    df["ign_cell_10y"] = window_count(df, ign[["h3", "date"]], y10)
    df["ign_10km_30d"] = window_count(df, near, pd.Timedelta(days=30))
    df["ign_10km_365d"] = window_count(df, near, pd.Timedelta(days=365))
    df["ign_10km_10y"] = window_count(df, near, y10)
    df["large_10km_10y"] = window_count(df, near_large, y10)

    woy = df.week.dt.dayofyear / 365.25 * 2 * np.pi
    df["woy_sin"], df["woy_cos"] = np.sin(woy), np.cos(woy)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df.sort_values(["week", "h3"]).reset_index(drop=True)


def load_static() -> pd.DataFrame:
    static = pd.read_parquet(PROC / "static.parquet")
    pop = pd.read_parquet(PROC / "cell_pop.parquet")
    static = static.merge(pop[["h3", "pop"]], on="h3", how="left")
    static["log_pop_density"] = np.log1p(static.pop("pop").fillna(0) / CELL_KM2)
    return static.merge(pd.read_parquet(PROC / "human.parquet"), on="h3", how="left")


def cell_weather(weekly_px: pd.DataFrame) -> pd.DataFrame:
    cp = pd.read_parquet(PROC / "cell_pixel.parquet")
    return cp.merge(weekly_px, on="pixel").drop(columns="pixel")


def labels(burns: pd.DataFrame, ign: pd.DataFrame) -> pd.DataFrame:
    """Targets per (h3, week): ign, large, fire (+ event size for burn weighting)."""
    size = burns.groupby(["FIRE_NAME", "alarm_date"]).h3.transform("size")
    fire = burns.assign(event_cells=size).groupby(["h3", "week"]).event_cells.max().reset_index()
    fire["fire"] = 1
    any_ign = ign.groupby(["h3", "week"]).size().rename("ign").clip(upper=1).reset_index()
    large = ign[ign.acres >= LARGE_ACRES].groupby(["h3", "week"]).size().rename("large") \
        .clip(upper=1).reset_index()
    out = fire.merge(any_ign, on=["h3", "week"], how="outer").merge(large, on=["h3", "week"], how="outer")
    return out.astype({"week": "datetime64[ns]"})


if __name__ == "__main__":
    cells = gpd.read_parquet(PROC / "cells.parquet")
    burns = pd.read_parquet(PROC / "burns.parquet")
    ign = pd.read_parquet(PROC / "ignitions.parquet")
    weather = cell_weather(pd.read_parquet(PROC / "weather_weekly.parquet"))
    weeks = weather.week.unique()
    weeks = weeks[weeks <= pd.Timestamp(f"{burns.alarm_date.dt.year.max()}-12-31")]

    base = pd.MultiIndex.from_product([cells.h3, weeks], names=["h3", "week"]).to_frame(index=False)
    base["week"] = base.week.astype("datetime64[ns]")
    perim_near = near_pairs(perimeter_points(gpd.read_parquet(PROC / "fires.parquet")), cells)
    panel = assemble(base, weather, load_static(), pd.read_parquet(PROC / "ndvi.parquet"),
                     burns, perim_near, ign, cells)
    panel = panel.merge(labels(burns, ign), on=["h3", "week"], how="left")
    panel[list(TARGETS)] = panel[list(TARGETS)].fillna(0).astype("int8")
    panel["event_cells"] = panel.event_cells.fillna(0).astype("float32")
    panel = panel.merge(cells[["h3", "block"]], on="h3")
    panel.to_parquet(PROC / "panel.parquet")
    print(f"{len(panel):,} rows, {len(FEATURES)} features, "
          f"{panel.week.min():%Y-%m-%d} to {panel.week.max():%Y-%m-%d}")
    for t, name in TARGETS.items():
        print(f"  {t:5s} {name:28s} {panel[t].sum():6,} positives ({panel[t].mean():.4%})")
    miss = panel[FEATURES].isna().mean()
    print("missing share:\n", miss[miss > 0].round(4).to_string())
