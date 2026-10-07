"""Train + evaluate the wildfire model.

Evaluation:
  1. Leave-years-out CV (2001-2018): folds hold out whole years. (Spatial-block CV leaked:
     one large fire burns many blocks in the same week, so held-out blocks were "seen".)
  2. Temporal holdout (2019+): the CV ensemble trained on 2001-2018 forecasts later years.
Production: the same recipe refit on all labeled years -> models/ensemble.joblib.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import MODELS, PROC, REPORTS, SEED, TEST_START
from src.features import FEATURES, WEATHER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_FOLDS = 5
NEG_RATE = 0.2  # keep 20% of negatives during training, reweighted x5
PARAMS = dict(objective="binary", learning_rate=0.03, n_estimators=500, num_leaves=31,
              min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=SEED, verbose=-1)


class Ensemble:
    """Year-blocked CV fold models + isotonic calibration fit on their out-of-fold predictions."""

    def __init__(self, features):
        self.features = features
        self.models, self.calibrator = [], None

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        X, y = df[self.features], df.fire.to_numpy()
        oof = np.zeros(len(df))
        for tr, va in GroupKFold(N_FOLDS).split(X, y, df.week.dt.year):
            keep = tr[(y[tr] == 1) | (rng.random(len(tr)) < NEG_RATE)]
            # each fire's cells share weight ~sqrt(n), so megafires don't dominate
            w = np.where(y[keep] == 1, 1 / np.sqrt(df.event_cells.to_numpy()[keep].clip(1)), 1 / NEG_RATE)
            m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[keep], y[keep], sample_weight=w)
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            self.models.append(m)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
        return oof

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated fold-model probabilities, shape (rows, folds)."""
        return np.column_stack([m.predict_proba(df[self.features])[:, 1] for m in self.models])

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrated probability, spread across fold models (uncertainty), and the raw
        ensemble score. Isotonic calibration is a step function (ties), so rank on `score`."""
        R = self.raw(df)
        P = np.column_stack([self.calibrator.predict(r) for r in R.T])
        return self.calibrator.predict(R.mean(1)), P.std(1), R.mean(1)

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Mean SHAP (log-odds) across fold models."""
        vals = [shap.TreeExplainer(m).shap_values(df[self.features]) for m in self.models]
        vals = [v[1] if isinstance(v, list) else v for v in vals]
        return np.mean(vals, axis=0)


def capture_at(df: pd.DataFrame, score: np.ndarray, top: float) -> float:
    """Share of burned cell-weeks that fell in the top `top` share of cells that week."""
    rank = pd.Series(score, index=df.index).groupby(df.week).rank(ascending=False, pct=True)
    return float((rank[df.fire == 1] <= top).mean())


def metrics(df: pd.DataFrame, score: np.ndarray, prob: np.ndarray | None = None) -> dict:
    """Ranking metrics on `score`; Brier on calibrated `prob` (defaults to score)."""
    y = df.fire.to_numpy()
    prob = score if prob is None else prob
    return {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            "brier": brier_score_loss(y, prob), "base_rate": y.mean(),
            "pr_auc_lift": average_precision_score(y, score) / y.mean(),
            "capture_top5": capture_at(df, score, 0.05), "capture_top10": capture_at(df, score, 0.10)}


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: cell burn frequency x seasonal (month) rate, both from training years."""
    cell = train.groupby("h3").fire.mean()
    month = train.groupby(train.week.dt.month).fire.mean()
    p = test.h3.map(cell).fillna(cell.mean()).to_numpy() * test.week.dt.month.map(month).to_numpy()
    return p / train.fire.mean()


def reliability_plot(y, p, path):
    bins = np.quantile(p, np.linspace(0.9, 1, 11))
    bins = np.unique(np.concatenate([[0], bins]))
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    d = pd.DataFrame({"y": y, "p": p, "b": idx}).groupby("b").mean()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lim = max(d.p.max(), d.y.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#999", lw=1, ls="--")
    ax.plot(d.p, d.y, "o-", color="#d9480f")
    ax.set(xlabel="Predicted probability", ylabel="Observed fire rate",
           title="Calibration (2019+ holdout)", xlim=(0, lim), ylim=(0, lim))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_plot(ens, sample, path):
    sv = ens.shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=ens.features).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color="#d9480f")
    ax.set(xlabel="mean |SHAP| (log-odds)", title="Top drivers of predicted fire risk")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def backtest_map(test, p, fire_name, path):
    """Predicted risk the week a real fire started vs. its actual perimeter."""
    cells = gpd.read_parquet(PROC / "cells.parquet")
    county = gpd.read_parquet(PROC / "county.parquet")
    fires = gpd.read_parquet(PROC / "fires.parquet")
    fire = fires[fires.FIRE_NAME.str.upper() == fire_name.upper()]
    fire = fire[fire.alarm_date >= TEST_START].sort_values("GIS_ACRES").tail(1)
    week = fire.alarm_date.dt.to_period("W-SUN").dt.start_time.iloc[0]
    mask = (test.week == week).to_numpy()
    wk = cells.merge(pd.DataFrame({"h3": test.h3[mask], "p": p[mask]}), on="h3")
    wk["pct"] = wk.p.rank(pct=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    wk.plot(column="pct", cmap="YlOrRd", ax=ax, linewidth=0, legend=True,
            legend_kwds={"label": "Risk percentile that week", "shrink": 0.7})
    county.boundary.plot(ax=ax, color="#555", lw=0.6)
    fire.boundary.plot(ax=ax, color="#1c7ed6", lw=2)
    burned = wk[wk.intersects(fire.geometry.iloc[0])]
    title = (f"{fire_name.title()} Fire ({fire.alarm_date.iloc[0]:%b %d, %Y}): perimeter in blue\n"
             f"burned cells' median risk percentile = {burned.pct.median():.0%}")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(burned.pct.median())


if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")
    train = panel[panel.week < TEST_START].reset_index(drop=True)
    test = panel[panel.week >= TEST_START].reset_index(drop=True)
    print(f"train {len(train):,} rows / {train.fire.sum()} fires; test {len(test):,} / {test.fire.sum()}")

    results = {}
    full = Ensemble(FEATURES)
    oof = full.fit(train)
    results["Full model - leave-years-out CV (2001-2018)"] = metrics(train, oof, full.calibrator.predict(oof))
    p_test, sd_test, s_test = full.predict(test)
    results["Full model - temporal holdout (2019+)"] = metrics(test, s_test, p_test)

    wx = Ensemble(WEATHER + ["woy_sin", "woy_cos"])
    wx.fit(train)
    _, _, s_wx = wx.predict(test)
    results["Weather-only model - temporal holdout"] = metrics(test, s_wx, wx.calibrator.predict(s_wx))
    results["Climatology baseline - temporal holdout"] = metrics(test, climatology(train, test))

    table = pd.DataFrame(results).T
    print(table.round(4).to_string())

    reliability_plot(test.fire.to_numpy(), p_test, REPORTS / "calibration.png")
    sample = test.sample(20_000, random_state=SEED)
    drivers = shap_plot(full, sample, REPORTS / "shap_importance.png")
    backtest = {name: backtest_map(test, s_test, name, REPORTS / f"backtest_{name.lower().replace(' ', '_')}.png")
                for name in ["VALLEY", "BORDER 2"]}
    unc_corr = float(np.corrcoef(sd_test, p_test)[0, 1])

    json.dump({"results": table.to_dict(orient="index"),
               "top_drivers": drivers.round(4).to_dict(),
               "backtest_median_percentile": backtest,
               "uncertainty_vs_risk_corr": unc_corr,
               "n_train": len(train), "n_test": len(test)},
              open(REPORTS / "metrics.json", "w"), indent=2, default=float)

    prod = Ensemble(FEATURES)
    prod.fit(panel)
    joblib.dump(prod, MODELS / "ensemble.joblib")
    print("saved production ensemble trained on", panel.week.min().date(), "to", panel.week.max().date())
