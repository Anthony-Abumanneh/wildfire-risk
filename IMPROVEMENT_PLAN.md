# Improvement plan — progress log (WIP, branch `improvements`)

Rule: all model selection uses leave-years-out CV on 2001–2018. The 2019–2025 holdout is scored ONCE at the end.

## Status (paused 2026-10-08)
- [x] weather.py: added gridMET fm100, fm1000, ERC, BI, wind direction -> fuel `_prev` features,
      ERC percentile vs 2001–2018, Santa Ana / offshore-day counts. Rebuild was running (re-run `src/weather.py` if
      `data/processed/erc_reference.npy` is missing).
- [x] FPA-FOD downloaded to data/raw/fpa_fod.zip (221 MB zip, 928 MB unzipped — disk has ~2.9 GB free).
- [x] ignitions.py written (FPA-FOD 1992–2020 + WFIGS 2021+), not yet run. Check overlap-year counts it prints.
- [x] human.py written (OSM roads/power/campgrounds, border/coast distance, SILVIS-style WUI, ring-1 context).
      First run failed: overpass-api.de refused connections -> added mirror fallback, not yet re-run.

## Next steps
1. Run `src/ignitions.py`, then delete data/raw/fpa_fod* to free disk.
2. Run `src/human.py`.
3. features.py: merge human.parquet into static; add recent-ignition features (10 km, past 30/365 d, 10 y)
   from real ignition points; build 3 targets: ignition (any), large fire (>=100 ac), burn (perimeters).
4. experiments.py: CV harness on 2001–2018 logging to reports/experiments.csv (baseline -> +fuels -> +human -> ...).
5. Optuna tuning (CV), blend with fire-history baseline, recency-weighted recalibration.
6. Daily-model experiment (8 GB RAM: build per-year chunks, subsample negatives for training).
7. Benchmarks: ERC percentile, ERC x history, NWS Red Flag Warnings (IEM archive, WFO SGX).
8. Alert-budget metrics (recall/precision at 1,2,5,10,20% of cells) -> final single holdout run.
9. Deployment: forecast parity test + pytest suite, data validation checks, model versioning (models/*.json),
   monitor.py (score past forecasts vs. new fires), Dockerfile/render.yaml (user must connect hosting account).
10. Update app (ignition / large-fire / burn layers), README, resume bullets; merge to main and push.
