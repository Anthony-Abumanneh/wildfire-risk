"""Leakage, metric, validation, and training/forecast parity tests.  Run: pytest -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from config import PROC
from src.features import FEATURES, count_before, window_count
from src.modeling import capture
from src.weather import VARS, weekly_features


def synthetic_weather(n_days=420, n_px=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    arrs = {v: rng.uniform(0, 100, (n_days, n_px)) for v in VARS.values()}
    arrs["wdir"] = rng.uniform(0, 360, (n_days, n_px))
    ref = np.sort(arrs["erc"][:365], axis=0)
    return dates, arrs, ref


# ---------- leakage ----------

def test_count_before_excludes_same_day_and_future():
    events = pd.DataFrame({"h3": ["a", "a", "a"], "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-02-01"])})
    df = pd.DataFrame({"h3": ["a"], "week": pd.to_datetime(["2020-01-08"])})
    assert count_before(df, events, df.week)[0] == 1  # only Jan 1; Jan 8 is the same day, Feb is future
    assert window_count(df, events, pd.Timedelta(days=3))[0] == 0


def test_weather_features_ignore_days_after_window():
    dates, arrs, ref = synthetic_weather()
    anchor = dates[380]
    before = weekly_features(dates, arrs, [anchor], ref)
    future = {v: a.copy() for v, a in arrs.items()}
    for a in future.values():
        a[387:] = -999  # everything after the 7-day window
    after = weekly_features(dates, future, [anchor], ref)
    pd.testing.assert_frame_equal(before, after)


def test_fuel_features_use_only_days_before_anchor():
    dates, arrs, ref = synthetic_weather()
    anchor = dates[380]
    before = weekly_features(dates, arrs, [anchor], ref)
    changed = {v: a.copy() for v, a in arrs.items()}
    for v in ("erc", "bi", "fm100", "fm1000", "pr"):
        changed[v][380:] = -999  # anchor day onward
    after = weekly_features(dates, changed, [anchor], ref)
    for col in ["erc_prev", "bi_prev", "fm100_prev", "fm1000_prev", "erc_7d_prev", "pr_30d", "pr_365d"]:
        np.testing.assert_allclose(before[col], after[col])


# ---------- metrics ----------

def test_capture_perfect_and_random():
    weeks = np.repeat(pd.date_range("2020-01-06", periods=10, freq="W-MON"), 100)
    df = pd.DataFrame({"week": weeks, "y": 0})
    df.loc[df.groupby("week").head(1).index, "y"] = 1  # one event per week
    perfect = df.y.to_numpy().astype(float)
    assert capture(df, "y", perfect, budgets=(0.01,))["recall@1%"] == 1.0
    worst = -perfect
    assert capture(df, "y", worst, budgets=(0.01,))["recall@1%"] == 0.0


# ---------- validation ----------

def test_validate_raises_on_missing_features():
    from src.forecast import DataError, validate
    X = pd.DataFrame({f: [0.5] for f in FEATURES})
    X["tmax_max"], X["rhmin_min"], X["wind_max"], X["pr_week"] = 30, 20, 5, 0
    X.loc[0, "ndvi"] = np.nan
    ndvi = pd.DataFrame({"available": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(DataError):
        validate(X, pd.Timestamp("2026-01-10"), ndvi, pd.Timestamp("2026-01-11"))


# ---------- training/forecast parity ----------

@pytest.mark.skipif(not (PROC / "panel.parquet").exists() or not (PROC / "gridmet_daily.parquet").exists(),
                    reason="needs built data")
def test_forecast_path_reproduces_training_features():
    """Rebuild one historical week through the forecast-time code path and compare to the panel."""
    import geopandas as gpd
    from src.features import assemble, cell_weather, load_static, near_pairs, perimeter_points
    from src.weather import load_erc_reference, to_arrays

    week = pd.Timestamp("2018-07-02")
    cells = gpd.read_parquet(PROC / "cells.parquet")
    daily = pd.read_parquet(PROC / "gridmet_daily.parquet")
    px = pd.read_parquet(PROC / "gridmet_pixels.parquet")
    dates, arrs = to_arrays(daily, px)
    wk = weekly_features(dates, arrs, [week], load_erc_reference())
    burns = pd.read_parquet(PROC / "burns.parquet")
    fires = gpd.read_parquet(PROC / "fires.parquet")
    ign = pd.read_parquet(PROC / "ignitions.parquet")
    base = cells[["h3"]].assign(week=week.to_datetime64())
    X = assemble(base, cell_weather(wk), load_static(), pd.read_parquet(PROC / "ndvi.parquet"), burns,
                 near_pairs(perimeter_points(fires), cells), ign, cells).set_index("h3")
    panel = pd.read_parquet(PROC / "panel.parquet", filters=[("week", "==", week)]).set_index("h3")
    np.testing.assert_allclose(X.loc[panel.index, FEATURES].to_numpy(), panel[FEATURES].to_numpy(),
                               rtol=1e-5, atol=1e-5)
