"""Shared modeling pieces: fold ensemble, benchmarks, and metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import SEED

N_FOLDS = 5
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.03, n_estimators=500, num_leaves=31,
                      min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=8)


def year_folds(df: pd.DataFrame, n: int = N_FOLDS):
    """Leave-years-out folds: whole years are held out together."""
    return GroupKFold(n).split(df, groups=df.week.dt.year)


def sample_weights(df: pd.DataFrame, target: str, idx: np.ndarray, neg_rate: float) -> np.ndarray:
    y = df[target].to_numpy()[idx]
    pos_w = np.ones(len(idx))
    if target == "fire":  # each fire's cells share weight ~sqrt(n), so megafires don't dominate
        pos_w = 1 / np.sqrt(df.event_cells.to_numpy()[idx].clip(1))
    return np.where(y == 1, pos_w, 1 / neg_rate)


class Ensemble:
    """Leave-years-out fold models + isotonic calibration fit on their out-of-fold scores."""

    def __init__(self, features, target, params=None, neg_rate=0.2, calib_years=None):
        self.features, self.target = list(features), target
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.neg_rate, self.calib_years = neg_rate, calib_years
        self.models, self.calibrator = [], None

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        X, y = df[self.features], df[self.target].to_numpy()
        oof = np.zeros(len(df))
        for tr, va in year_folds(df):
            keep = tr[(y[tr] == 1) | (rng.random(len(tr)) < self.neg_rate)]
            m = lgb.LGBMClassifier(**self.params).fit(
                X.iloc[keep], y[keep], sample_weight=sample_weights(df, self.target, keep, self.neg_rate))
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            self.models.append(m)
        self.fit_calibrator(df, oof)
        return oof

    def fit_calibrator(self, df, score):
        """Isotonic on OOF scores; optionally only the most recent `calib_years` years,
        so absolute probabilities track the recent fire regime."""
        mask = np.ones(len(df), bool)
        if self.calib_years:
            mask = (df.week.dt.year > df.week.dt.year.max() - self.calib_years).to_numpy()
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1) \
            .fit(score[mask], df[self.target].to_numpy()[mask])

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict_proba(df[self.features])[:, 1] for m in self.models])

    def predict(self, df: pd.DataFrame):
        """Calibrated probability, spread across fold models, raw ensemble score (for ranking)."""
        R = self.raw(df)
        P = np.column_stack([self.calibrator.predict(r) for r in R.T])
        return self.calibrator.predict(R.mean(1)), P.std(1), R.mean(1)

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        vals = [shap.TreeExplainer(m).shap_values(df[self.features]) for m in self.models]
        vals = [v[1] if isinstance(v, list) else v for v in vals]
        return np.mean(vals, axis=0)


# ---------- benchmarks (all leakage-free: fit on training years only) ----------

def history_rate(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Fire-history baseline: cell rate x seasonal (month) rate, smoothed with a prior."""
    k = 50  # pseudo-weeks shrinking sparse cells toward the county rate
    base = train[target].mean()
    g = train.groupby("h3")[target].agg(["sum", "count"])
    cell = (g["sum"] + k * base) / (g["count"] + k)
    month = train.groupby(train.week.dt.month)[target].mean() / base
    return test.h3.map(cell).fillna(base).to_numpy() * test.week.dt.month.map(month).fillna(1).to_numpy()


def oof_benchmark(df: pd.DataFrame, target: str, fn) -> np.ndarray:
    oof = np.zeros(len(df))
    for tr, va in year_folds(df):
        oof[va] = fn(df.iloc[tr], df.iloc[va], target)
    return oof


def erc_score(train, test, target):
    """What agencies use: ERC percentile (fuel dryness), no spatial model."""
    return test.erc_pct_prev.to_numpy() + 1e-6 * test.santa_ana_days.to_numpy()


def erc_x_history(train, test, target):
    return (0.05 + test.erc_pct_prev.to_numpy()) * history_rate(train, test, target)


# ---------- metrics ----------

def capture(df: pd.DataFrame, target: str, score: np.ndarray, budgets=BUDGETS) -> dict:
    """Alert-budget metrics: flag the top k% cells each week.
    recall = share of events caught; precision = share of flags that had an event."""
    rank = pd.Series(score + 1e-12 * np.random.default_rng(0).random(len(score)), index=df.index) \
        .groupby(df.week).rank(ascending=False, pct=True)
    y = df[target].to_numpy() == 1
    out = {}
    for k in budgets:
        flag = (rank <= k).to_numpy()
        out[f"recall@{k:.0%}"] = y[flag].sum() / max(y.sum(), 1)
        out[f"precision@{k:.0%}"] = y[flag].mean()
    return out


def metrics(df: pd.DataFrame, target: str, score: np.ndarray, prob=None) -> dict:
    y = df[target].to_numpy()
    prob = score if prob is None else prob
    ap = average_precision_score(y, score)
    return {"pr_auc": ap, "pr_auc_lift": ap / y.mean(), "roc_auc": roc_auc_score(y, score),
            "brier": brier_score_loss(y, prob), "base_rate": y.mean(), **capture(df, target, score)}
