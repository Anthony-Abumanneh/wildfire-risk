"""Live performance monitoring: score archived forecasts against ignitions that were later
reported (WFIGS), once each forecast window is closed plus a reporting lag.

Writes reports/monitoring.json: per-forecast and pooled recall@k, plus a drift check comparing
the live ignition rate with the training-period rate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import MODELS, PROC, REPORTS, ROOT

HISTORY_DIR = ROOT / "data" / "forecast" / "history"
REPORTING_LAG_DAYS = 14
BUDGETS = (0.05, 0.10)


def score_forecast(fc: pd.DataFrame, ign: pd.DataFrame, start: pd.Timestamp) -> dict:
    window = ign[(ign.date >= start) & (ign.date < start + pd.Timedelta(days=7))]
    hit = fc.h3.isin(set(window.h3)).to_numpy()
    large = fc.h3.isin(set(window[window.acres >= 100].h3)).to_numpy()
    out = {"valid_from": f"{start:%Y-%m-%d}", "ignition_cells": int(hit.sum()), "large_cells": int(large.sum())}
    for k in BUDGETS:
        top = fc.score_ign.rank(ascending=False, pct=True).to_numpy() <= k
        out[f"ign_hits@{k:.0%}"] = int((hit & top).sum())
    return out


def run(ign: pd.DataFrame | None = None):
    ign = pd.read_parquet(PROC / "ignitions.parquet") if ign is None else ign
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=7 + REPORTING_LAG_DAYS)
    rows = []
    for f in sorted(HISTORY_DIR.glob("*.parquet")):
        start = pd.Timestamp(f.stem)
        if start <= cutoff:
            rows.append(score_forecast(pd.read_parquet(f), ign, start))
    df = pd.DataFrame(rows)
    report = {"scored_forecasts": len(df), "reporting_lag_days": REPORTING_LAG_DAYS}
    if len(df):
        recent = df.tail(90)
        report["pooled_last_90"] = {
            f"ign_recall@{k:.0%}": float(recent[f"ign_hits@{k:.0%}"].sum() / max(recent.ignition_cells.sum(), 1))
            for k in BUDGETS}
        report["per_forecast"] = df.tail(30).to_dict(orient="records")
    # drift: weekly ignition-cell rate in the last 365 days vs. the training period
    card = json.loads((MODELS / "model_card_ign.json").read_text())
    panel_rate = card["base_rate_all_years"]
    last = ign[ign.date >= pd.Timestamp.today() - pd.Timedelta(days=365)]
    n_cells = pd.read_parquet(PROC / "cells.parquet", columns=["h3"]).shape[0]
    live_rate = last.groupby("week").h3.nunique().sum() / (52 * n_cells)
    report["drift"] = {"training_weekly_ignition_rate": float(panel_rate), "last_365d_rate": float(live_rate),
                       "ratio": float(live_rate / panel_rate) if panel_rate else np.nan}
    (REPORTS / "monitoring.json").write_text(json.dumps(report, indent=2, default=float))
    print("monitoring:", {k: v for k, v in report.items() if k != "per_forecast"})


if __name__ == "__main__":
    run()
