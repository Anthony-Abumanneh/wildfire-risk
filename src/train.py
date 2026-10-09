"""Final evaluation + production models for all targets.

Model choices (features, params, stacking, calibration window) were made with leave-years-out
CV on 2001-2018 (src/experiments.py, reports/experiments.csv) and are frozen in
models/decisions.json BEFORE this script scores the 2019+ holdout. Run it once.

Outputs: reports/metrics.json, reports/*.png, models/ensemble_<target>.joblib,
models/model_card_<target>.json
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import MODELS, PROC, REPORTS, SEED, TEST_START
from src.experiments import best_params, stack_features
from src.features import TARGET_FEATURES, TARGETS
from src.modeling import (Ensemble, capture, erc_score, erc_x_history, history_rate, metrics,
                          year_folds)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"Model": "#d9480f", "Fire history": "#1c7ed6", "ERC percentile": "#868e96",
          "ERC x history": "#37b24d"}


class Stacked:
    """Ensemble + logistic stacker over [model logit, fire-history rate, ERC percentile].
    `hist_train` holds the labeled weeks the history rate is computed from."""

    def __init__(self, ens: Ensemble, hist_train: pd.DataFrame, stacker):
        self.ens, self.hist_train, self.stacker = ens, hist_train, stacker
        self.target = ens.target

    def raw_score(self, df):
        Z = stack_features(self.ens.raw(df).mean(1), history_rate(self.hist_train, df, self.target),
                           df.erc_pct_prev.to_numpy())
        return self.stacker.predict_proba(Z)[:, 1]

    def predict(self, df):
        s = self.raw_score(df)
        return self.ens.calibrator.predict(s), self.ens.predict(df)[1], s

    def shap_values(self, df):
        return self.ens.shap_values(df)

    @property
    def features(self):
        return self.ens.features


def decisions() -> dict:
    return json.loads((MODELS / "decisions.json").read_text())


def fit_target(df: pd.DataFrame, target: str, dec: dict):
    """Fit ensemble (+ optional stacker) on df -> model with .predict() -> (prob, sd, score)."""
    ens = Ensemble(TARGET_FEATURES[target], target, best_params(target), neg_rate=0.2,
                   calib_years=dec.get("calib_years"))
    oof = ens.fit(df)
    if not dec.get("stack"):
        return ens
    hist_oof = np.zeros(len(df))  # out-of-fold history rates, so the stacker sees no leakage
    for tr, va in year_folds(df):
        hist_oof[va] = history_rate(df.iloc[tr], df.iloc[va], target)
    Z = stack_features(oof, hist_oof, df.erc_pct_prev.to_numpy())
    stacker = LogisticRegression(max_iter=1000).fit(Z, df[target])
    ens.fit_calibrator(df, stacker.predict_proba(Z)[:, 1])  # calibrate the final (stacked) score
    return Stacked(ens, df[["h3", "week", target]].copy(), stacker)


def recall_curve(test, target, scores: dict, path):
    ks = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, s in scores.items():
        c = capture(test, target, s, budgets=ks)
        ax.plot(np.array(ks) * 100, [c[f"recall@{k:.0%}"] for k in ks], "o-", label=name,
                color=COLORS.get(name), lw=2.2 if name == "Model" else 1.2)
    ax.plot(np.array(ks) * 100, ks, ls="--", color="#bbb", label="Random")
    ax.set(xlabel="Alert budget: % of cells flagged each week", ylabel="Share of events caught",
           title=f"{TARGETS[target]} (2019+ holdout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability_plot(y, p, path, title):
    q = np.unique(np.concatenate([[0], np.quantile(p, np.linspace(0.8, 1, 11))]))
    idx = np.clip(np.digitize(p, q) - 1, 0, len(q) - 2)
    d = pd.DataFrame({"y": y, "p": p, "b": idx}).groupby("b").mean()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    lim = max(d.p.max(), d.y.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#999", lw=1, ls="--")
    ax.plot(d.p, d.y, "o-", color="#d9480f")
    ax.set(xlabel="Predicted probability", ylabel="Observed rate", title=title, xlim=(0, lim), ylim=(0, lim))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_plot(model, sample, path, title):
    sv = model.shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=model.features).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color="#d9480f")
    ax.set(xlabel="mean |SHAP| (log-odds)", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def backtest_map(test, s, fire_name, path):
    cells = gpd.read_parquet(PROC / "cells.parquet")
    county = gpd.read_parquet(PROC / "county.parquet")
    fires = gpd.read_parquet(PROC / "fires.parquet")
    fire = fires[(fires.FIRE_NAME.str.upper() == fire_name) & (fires.alarm_date >= TEST_START)] \
        .sort_values("GIS_ACRES").tail(1)
    week = fire.alarm_date.dt.to_period("W-SUN").dt.start_time.iloc[0]
    mask = (test.week == week).to_numpy()
    wk = cells.merge(pd.DataFrame({"h3": test.h3[mask], "s": s[mask]}), on="h3")
    wk["pct"] = wk.s.rank(pct=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    wk.plot(column="pct", cmap="YlOrRd", ax=ax, linewidth=0, legend=True,
            legend_kwds={"label": "Burn-risk percentile that week", "shrink": 0.7})
    county.boundary.plot(ax=ax, color="#555", lw=0.6)
    fire.boundary.plot(ax=ax, color="#1c7ed6", lw=2)
    burned = wk[wk.intersects(fire.geometry.iloc[0])]
    ax.set_title(f"{fire_name.title()} Fire ({fire.alarm_date.iloc[0]:%b %d, %Y}): perimeter in blue\n"
                 f"burned cells' median risk percentile = {burned.pct.median():.0%}", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(burned.pct.median())


def redflag_comparison(test, target, s) -> dict | None:
    """NWS Red Flag Warnings vs. the model at the same alert volume (cell-weeks flagged).
    Note: warnings are issued hours-to-days ahead, so this favors the warnings."""
    p = PROC / "redflag.parquet"
    if not p.exists():
        return None
    rfw = pd.read_parquet(p).astype({"week": "datetime64[ns]"})
    flag = test[["h3", "week"]].merge(rfw, on=["h3", "week"], how="left").rfw.fillna(0).to_numpy() == 1
    y = test[target].to_numpy() == 1
    top = np.zeros(len(s), bool)
    top[np.argsort(-s)[:flag.sum()]] = True
    return {"cell_weeks_flagged": int(flag.sum()), "share_flagged": float(flag.mean()),
            "rfw_recall": float(y[flag].sum() / y.sum()), "rfw_precision": float(y[flag].mean()),
            "model_recall_same_volume": float(y[top].sum() / y.sum()),
            "model_precision_same_volume": float(y[top].mean())}


def git_hash():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    return r.stdout.strip() or None


if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")
    train = panel[panel.week < TEST_START].reset_index(drop=True)
    test = panel[panel.week >= TEST_START].reset_index(drop=True)
    dec_all = decisions()
    out = {"holdout": f"{TEST_START} to {test.week.max():%Y-%m-%d}", "targets": {}}
    rng = np.random.default_rng(SEED)

    for target in TARGETS:
        dec = dec_all[target]
        print(f"== {target}: {TARGETS[target]} ({test[target].sum()} test events)", flush=True)
        model = fit_target(train, target, dec)
        p_test, sd_test, s_test = model.predict(test)
        bench = {"Fire history": history_rate(train, test, target),
                 "ERC percentile": erc_score(train, test, target),
                 "ERC x history": erc_x_history(train, test, target)}
        res = {"Model": metrics(test, target, s_test, p_test),
               **{k: metrics(test, target, v) for k, v in bench.items()}}
        res["Model"]["pred_to_obs_ratio"] = float(p_test.sum() / test[target].sum())
        print(pd.DataFrame(res).T[["pr_auc_lift", "roc_auc", "recall@5%", "recall@10%"]].round(3).to_string(),
              flush=True)

        recall_curve(test, target, {"Model": s_test, **bench}, REPORTS / f"recall_{target}.png")
        reliability_plot(test[target].to_numpy(), p_test, REPORTS / f"calibration_{target}.png",
                         f"Calibration: {TARGETS[target]}")
        sample = test.iloc[rng.choice(len(test), 20_000, replace=False)]
        drivers = shap_plot(model, sample, REPORTS / f"shap_{target}.png", f"Top drivers: {TARGETS[target]}")
        entry = {"results": res, "decision": dec, "top_drivers": drivers.round(4).to_dict(),
                 "red_flag": redflag_comparison(test, target, s_test),
                 "uncertainty_vs_risk_corr": float(np.corrcoef(sd_test, p_test)[0, 1])}
        if target == "fire":
            entry["backtests"] = {n: backtest_map(test, s_test, n,
                                                  REPORTS / f"backtest_{n.lower().replace(' ', '_')}.png")
                                  for n in ["VALLEY", "BORDER 2"]}
        out["targets"][target] = entry
        json.dump(out, open(REPORTS / "metrics.json", "w"), indent=2, default=float)

    # production: same recipe refit on every labeled year
    for target in TARGETS:
        model = fit_target(panel, target, dec_all[target])
        joblib.dump(model, MODELS / f"ensemble_{target}.joblib")
        card = {"target": target, "description": TARGETS[target],
                "version": pd.Timestamp.now().strftime("%Y%m%d"), "git": git_hash(),
                "trained_on": f"{panel.week.min():%Y-%m-%d} to {panel.week.max():%Y-%m-%d}",
                "features": model.features, "params": (model.ens if isinstance(model, Stacked) else model).params,
                "decision": dec_all[target], "holdout_metrics": out["targets"][target]["results"]["Model"],
                "base_rate_all_years": float(panel[target].mean())}
        (MODELS / f"model_card_{target}.json").write_text(json.dumps(card, indent=2, default=float))
        print(f"saved production model: {target}", flush=True)
