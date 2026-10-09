"""Experiment: does a DAILY model beat the weekly one?

Daily rows = same-day weather/fuel/Santa Ana + day-of-week, plus the slow features (terrain,
human, NDVI, history) taken from the weekly panel row the day falls in (all as-of week start,
so no leakage). Built one year at a time to fit in 8 GB RAM; training keeps all positives and
a sample of negatives. Daily OOF probabilities are aggregated to weeks with
P(week) = 1 - prod(1 - p_day) and compared with the weekly model on identical year folds.

usage: python src/daily.py [targets...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import PROC, SEED, TEST_START
from src.experiments import load_train, log
from src.features import (FEATURES, HISTORY, HUMAN, IGNITION_HISTORY, STATIC, TARGET_FEATURES,
                          VEG, cell_weather)
from src.modeling import DEFAULT_PARAMS, Ensemble, metrics, year_folds
from src.weather import load_erc_reference, to_arrays, weekly_features

SLOW = STATIC + HUMAN + VEG + HISTORY + IGNITION_HISTORY
DAILY_WX = ["tmax_max", "rhmin_min", "wind_max", "vpd_max", "pr_week", "pr_30d", "pr_90d",
            "pr_365d", "days_since_rain", "fm100_prev", "fm1000_prev", "erc_prev", "bi_prev",
            "erc_7d_prev", "erc_pct_prev", "offshore_days", "santa_ana_days", "santa_ana_wind_max"]
NEG_RATE = 0.03


def daily_labels() -> dict:
    ign = pd.read_parquet(PROC / "ignitions.parquet")
    burns = pd.read_parquet(PROC / "burns.parquet")
    return {
        "ign": set(zip(ign.h3, ign.date)),
        "large": set(zip(ign[ign.acres >= 100].h3, ign[ign.acres >= 100].date)),
        "fire": set(zip(burns.h3, burns.alarm_date)),
    }


class DailyBuilder:
    def __init__(self):
        daily = pd.read_parquet(PROC / "gridmet_daily.parquet")
        px = pd.read_parquet(PROC / "gridmet_pixels.parquet")
        self.dates, self.arrs = to_arrays(daily, px)
        self.erc_ref = load_erc_reference()
        self.slow = load_train(SLOW)[["h3", "week", *SLOW]]
        self.labels = daily_labels()

    def year(self, year: int, neg_rate=None, rng=None) -> pd.DataFrame:
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        wx = weekly_features(self.dates, self.arrs, days, self.erc_ref, window=1)
        df = cell_weather(wx).rename(columns={"week": "date"})
        df["week"] = df.date.dt.to_period("W-SUN").dt.start_time.astype("datetime64[ns]")
        df = df.merge(self.slow, on=["h3", "week"], how="inner")
        for t, s in self.labels.items():
            df[t] = np.fromiter(((h, d) in s for h, d in zip(df.h3, df.date)), bool, len(df)).astype("int8")
        if neg_rate is not None:
            keep = (df[list(self.labels)].max(1) == 1) | (rng.random(len(df)) < neg_rate)
            df = df[keep.to_numpy()]
        doy = df.date.dt.dayofyear / 365.25 * 2 * np.pi
        df["woy_sin"], df["woy_cos"] = np.sin(doy), np.cos(doy)
        df["dow"] = df.date.dt.dayofweek
        df["weekend"] = (df.dow >= 5).astype("int8")
        return df.reset_index(drop=True)


def daily_features(target):
    slow = [f for f in TARGET_FEATURES[target] if f in SLOW]
    return DAILY_WX + slow + ["woy_sin", "woy_cos", "dow", "weekend"]


def run(targets):
    weekly = load_train(FEATURES)
    years = sorted(weekly.week.dt.year.unique())
    folds = [sorted(weekly.week.dt.year.iloc[va].unique()) for _, va in year_folds(weekly)]
    b = DailyBuilder()
    rng = np.random.default_rng(SEED)
    sampled = pd.concat([b.year(y, NEG_RATE, rng) for y in years], ignore_index=True)
    print(f"daily training sample: {len(sampled):,} rows", flush=True)

    for target in targets:
        feats = daily_features(target)
        weekly_oof = Ensemble(TARGET_FEATURES[target], target, neg_rate=0.1).fit(weekly)
        agg = pd.Series(0.0, index=pd.MultiIndex.from_frame(weekly[["h3", "week"]]))
        for held in folds:
            tr = sampled[~sampled.date.dt.year.isin(held)]
            y = tr[target].to_numpy()
            w = np.where(y == 1, 1.0, 1 / NEG_RATE)
            m = lgb.LGBMClassifier(**DEFAULT_PARAMS).fit(tr[feats], y, sample_weight=w)
            for yr in held:
                full = b.year(yr)
                p = m.predict_proba(full[feats])[:, 1]
                wk = pd.DataFrame({"h3": full.h3, "week": full.week, "lq": np.log1p(-np.clip(p, 0, 0.999))}) \
                    .groupby(["h3", "week"]).lq.sum()
                agg.loc[wk.index] = 1 - np.exp(wk.to_numpy())
            print(f"  {target}: fold {held} done", flush=True)
        daily_oof = agg.to_numpy()
        log([{"experiment": "weekly model (default params, same folds)", "target": target,
              **metrics(weekly, target, weekly_oof)},
             {"experiment": "daily model aggregated to week", "target": target,
              **metrics(weekly, target, daily_oof)},
             {"experiment": "mean of weekly + daily", "target": target,
              **metrics(weekly, target, 0.5 * (weekly_oof / weekly_oof.mean() + daily_oof / daily_oof.mean()))}])
        print(pd.read_csv(PROC.parent.parent / "reports" / "experiments.csv").tail(3)
              [["experiment", "target", "pr_auc_lift", "roc_auc", "recall@5%", "recall@10%"]].to_string(), flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or ["ign", "large", "fire"])
