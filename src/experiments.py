"""Model-selection experiments. Leave-years-out CV on 2001-2018 ONLY — the 2019+ holdout
is never touched here. Results append to reports/experiments.csv.

usage: python src/experiments.py ablation [targets...]
       python src/experiments.py tune <target> [n_trials]
       python src/experiments.py stack [targets...]
       python src/experiments.py calib [targets...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from config import MODELS, PROC, REPORTS, SEED, TEST_START
from src.features import (FEATURES, FUELS, HISTORY, HUMAN, IGNITION_HISTORY, SEASON, STATIC,
                          TARGET_FEATURES, TARGETS, VEG, WEATHER)
from src.modeling import (Ensemble, erc_score, erc_x_history, history_rate, metrics, oof_benchmark,
                          year_folds)

LOG = REPORTS / "experiments.csv"
OLD = WEATHER + STATIC + VEG + HISTORY + SEASON
ABLATION = {
    "v1 features (shipped model)": OLD,
    "+ fuels/ERC/Santa Ana": OLD + FUELS,
    "+ human drivers/WUI": OLD + FUELS + HUMAN,
    "+ ignition history (all)": FEATURES,
}


def load_train(columns=None) -> pd.DataFrame:
    cols = None if columns is None else list(dict.fromkeys(
        ["h3", "week", "block", "event_cells", *TARGETS, *columns]))
    df = pd.read_parquet(PROC / "panel.parquet", columns=cols)
    return df[df.week < TEST_START].reset_index(drop=True)


def log(rows: list[dict]):
    out = pd.DataFrame(rows)
    out.insert(0, "run_at", pd.Timestamp.now().isoformat(timespec="seconds"))
    if LOG.exists():  # rows from different modes have different columns -> align, then rewrite
        out = pd.concat([pd.read_csv(LOG), out], ignore_index=True)
    out.to_csv(LOG, index=False)


def run_ablation(targets):
    df = load_train(FEATURES)
    rows = []
    for target in targets:
        for name, fn in [("Benchmark: fire history", history_rate),
                         ("Benchmark: ERC percentile", erc_score),
                         ("Benchmark: ERC x history", erc_x_history)]:
            rows.append({"experiment": name, "target": target, **metrics(df, target, oof_benchmark(df, target, fn))})
        for name, feats in ABLATION.items():
            t0 = time.time()
            ens = Ensemble(feats, target, neg_rate=0.1)
            oof = ens.fit(df)
            rows.append({"experiment": name, "target": target, "n_features": len(feats),
                         **metrics(df, target, oof, ens.calibrator.predict(oof)),
                         "seconds": round(time.time() - t0)})
            print(f"{target:5s} {name:30s} lift {rows[-1]['pr_auc_lift']:.2f}  "
                  f"roc {rows[-1]['roc_auc']:.3f}  recall@5% {rows[-1]['recall@5%']:.3f}", flush=True)
        log(rows)
        rows = []


def params_path(target):
    return MODELS / f"params_{target}.json"


def best_params(target) -> dict:
    p = params_path(target)
    return json.loads(p.read_text()) if p.exists() else {}


def run_tune(target, n_trials=25):
    """Optuna search over LightGBM params, objective = leave-years-out CV PR-AUC."""
    df = load_train(FEATURES)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 3000, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
        oof = Ensemble(TARGET_FEATURES[target], target, params, neg_rate=0.1).fit(df)
        return average_precision_score(df[target], oof)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({})  # default params as the first trial, so tuning can't do worse
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"  trial {t.number}: {t.value:.5f} (best {s.best_value:.5f})",
                                                 flush=True)])
    params_path(target).write_text(json.dumps(study.best_params, indent=2))
    oof = Ensemble(TARGET_FEATURES[target], target, study.best_params, neg_rate=0.1).fit(df)
    log([{"experiment": "tuned (Optuna)", "target": target, "n_features": len(TARGET_FEATURES[target]),
          **metrics(df, target, oof)}])


def stack_features(model_oof, hist, erc):
    return np.column_stack([logit(np.clip(model_oof, 1e-6, 1 - 1e-6)), np.log(hist + 1e-6), erc])


def run_stack(targets):
    """Blend model with the fire-history baseline + ERC via logistic stacking, evaluated with
    nested leave-years-out CV (the stacker never sees the year it scores)."""
    df = load_train(FEATURES)
    for target in targets:
        oof = Ensemble(TARGET_FEATURES[target], target, best_params(target), neg_rate=0.1).fit(df)
        hist = oof_benchmark(df, target, history_rate)
        Z, y = stack_features(oof, hist, df.erc_pct_prev.to_numpy()), df[target].to_numpy()
        blend = np.zeros(len(df))
        for tr, va in year_folds(df):
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[tr], y[tr])
            blend[va] = lr.predict_proba(Z[va])[:, 1]
        log([{"experiment": "tuned model (stack input)", "target": target, **metrics(df, target, oof)},
             {"experiment": "stacked: model + history + ERC", "target": target, **metrics(df, target, blend)}])
        print(target, "stack coef:", LogisticRegression(max_iter=1000).fit(Z, y).coef_.round(3))


def run_calib(targets, years=range(2012, 2019)):
    """Forward-chaining calibration check: calibrate on years < y, score year y.
    Compares using all past OOF years vs. only the most recent K."""
    from sklearn.isotonic import IsotonicRegression
    df = load_train(FEATURES)
    yr = df.week.dt.year.to_numpy()
    for target in targets:
        oof = Ensemble(TARGET_FEATURES[target], target, best_params(target), neg_rate=0.1).fit(df)
        y = df[target].to_numpy()
        rows = []
        for K in [None, 10, 7, 5]:
            pred_sum = obs_sum = brier = n = 0
            for t in years:
                tr = (yr < t) if K is None else (yr < t) & (yr >= t - K)
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[tr], y[tr])
                p = iso.predict(oof[yr == t])
                pred_sum += p.sum(); obs_sum += y[yr == t].sum()
                brier += ((p - y[yr == t]) ** 2).sum(); n += (yr == t).sum()
            rows.append({"experiment": f"calibration window: {'all past' if K is None else f'last {K}y'}",
                         "target": target, "pred_to_obs_ratio": pred_sum / obs_sum, "brier": brier / n})
            print(rows[-1], flush=True)
        log(rows)


if __name__ == "__main__":
    mode, *targets = sys.argv[1:] or ["ablation"]
    if mode == "ablation":
        run_ablation(targets or list(TARGETS))
    elif mode == "tune":
        run_tune(targets[0], int(targets[1]) if len(targets) > 1 else 25)
    elif mode == "stack":
        run_stack(targets or list(TARGETS))
    elif mode == "calib":
        run_calib(targets or list(TARGETS))
    show = pd.read_csv(LOG)
    print(show.groupby(["target", "experiment"], sort=False)[
        ["pr_auc_lift", "roc_auc", "recall@5%", "recall@10%", "precision@5%"]].last().round(3).to_string())
