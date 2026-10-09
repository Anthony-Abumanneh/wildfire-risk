# San Diego Wildfire Risk

A 7-day **wildfire outlook for every ~5 km² cell in San Diego County**, with three forecasts per cell: **any fire starts**, a **large fire (100+ acres) starts**, and **the cell burns**. Risk is then translated into **who is exposed** and **how hard it is for them to get out**. The models train on 25 years (2001–2025) of 13.8k ignition points and CAL FIRE perimeters, joined with daily weather, NFDRS fuel and fire-danger indices, Santa Ana winds, satellite vegetation, terrain, land cover, human ignition sources, census population and the OpenStreetMap road network. A Dash dashboard refreshes the outlook daily from live forecasts and tracks its own live accuracy.

## Results

Scored **once** on a 2019–2025 holdout. Every modeling choice was frozen beforehand using cross-validation on 2001–2018.

| Forecast | Test events | Method | PR-AUC lift | ROC-AUC | Recall@5% | Recall@10% |
|---|---|---|---|---|---|---|
| **Any fire starts** | 1,943 | **Model** | **7.9×** | **0.848** | **30.4%** | **43.7%** |
| | | Fire history | 5.8× | 0.816 | 26.3% | 41.3% |
| | | ERC percentile (agency fire danger) | 1.2× | 0.569 | 3.5% | 6.5% |
| **Cell burns** | 295 | **Model** | **7.0×** | **0.865** | **26.4%** | **38.3%** |
| | | Fire history | 2.4× | 0.729 | 24.4% | 34.9% |
| | | ERC × fire history | 2.5× | 0.760 | 22.0% | 32.5% |
| **Large fire starts** | 23 | **Model** | (45×)* | **0.893** | 17.4% | **34.8%** |
| | | ERC × fire history | 4.9× | 0.775 | 13.0% | 26.1% |

*PR-AUC lift = PR-AUC ÷ base rate. Recall@k = share of events that fell in that week's top k% riskiest cells.*
*\*With only 23 large fires, PR-AUC is unstable (a few top-ranked hits dominate it); ROC-AUC and recall are the reliable numbers.*

- **vs. NWS Red Flag Warnings:** warnings covered 3.0% of cell-weeks and caught **6%** of ignitions and **21%** of burned cells. The model, flagging the same number of cell-weeks, caught **27%** and **24%**. The warnings are issued hours to days ahead, so this comparison favours them.
- **Calibration:** predicted/observed event counts are 0.92 for ignitions and burns, down from about 2× in v1, after calibrating on the most recent 5 years. Large-fire probabilities still run about 2× high.
- **vs. v1** (burn target, same holdout): lift went from 4.6× to 7.0×, ROC-AUC from 0.842 to 0.865, and recall@5% from 19% to 26%. v1 lost to fire history at the top 5%; this version beats it.
- **Evacuation:** 6 communities have ≤3 edge-disjoint escape routes. Borrego Springs (pop. 3,073) has 2 routes and a 65-minute median drive to a highway.

## Approach

**Data (all free, no API keys)**

- **Labels:**
  - USFS FPA-FOD ignition points (1992–2020) plus NIFC WFIGS incidents (2021+): 13,773 fires with origin, date and size.
  - CAL FIRE FRAP perimeters for "cell burns".
- **Weather:** gridMET daily temperature, humidity, wind, VPD and precipitation, aggregated over the forecast week, plus antecedent rain and days since rain.
- **Fuels and fire danger:** gridMET 100-hr/1000-hr dead fuel moisture, ERC, burning index, and ERC percentile vs. 2001–2018. All use the last day observed *before* the forecast window.
- **Santa Ana winds:** count of days with offshore (NE–E) wind ≥3 m/s and humidity ≤25%.
- **Human ignition sources:** OSM road density, power lines and campgrounds; distance to the US–Mexico border and to the coast; SILVIS-style WUI (intermix/interface) from 2020 Census housing and land cover; neighbouring-cell context.
- **Vegetation and terrain:** MODIS NDVI and its anomaly, Copernicus DEM, ESA WorldCover.
- **History, all as-of each week:** fires started in the cell (past 10 years), fires within 10 km (past 30 days, 1 year, 10 years), large fires nearby, years since the cell last burned.

**Model selection (leave-years-out CV on 2001–2018 only; log in `reports/experiments.csv`)**

| Step | Any fire starts | Large fire | Cell burns |
|---|---|---|---|
| v1 features | 5.8× | 4.0× | 4.0× |
| + fuels / ERC / Santa Ana | 5.8× | 4.5× | 5.9× |
| + human drivers / WUI | 5.9× | 4.6× | **6.7×** |
| + ignition history | **7.4×** | **4.9×** | 5.7× (dropped) |
| + Optuna tuning (20 trials) | **7.8×** | **6.3×** | **7.6×** |

Tried and rejected:
- **Stacking with fire history + ERC:** no gain (7.80× vs 7.84×).
- **A daily model aggregated to weeks:** equal or worse PR-AUC for every target.

Adopted:
- **Calibrating on the last 5 years:** cut over-prediction from 2.3× to 1.2× on burns in forward-chaining tests.

**Model**

- LightGBM ensemble of 5 leave-years-out fold models with isotonic calibration. Spread across fold models is reported as uncertainty.
- Negatives are down-sampled 5× and reweighted. Burn positives are weighted 1/√(cells burned by that fire) so megafires don't dominate.
- Rankings use the raw ensemble score; isotonic steps create ties.
- SHAP explains every forecast. The top drivers are:
  - **Ignitions:** local ignition history and road density.
  - **Large fires:** heat, VPD and dryness.
  - **Burns:** surrounding vegetation, season and heat.

**Exposure & evacuation**

- **Expected people exposed** = Σ P(burn) × population.
- **Escape routes** = max-flow / min-cut on the OSM drive network: edge-disjoint paths from a community to a highway at least 2 km outside it.
- **Minutes to highway** = multi-source Dijkstra drive time.

**Operations**

- **Daily GitHub Action:** runs tests, then the forecast, which combines observed gridMET weather with the Open-Meteo 7-day forecast.
- **Validation:** missing features, out-of-range weather or stale observations fail the job, so GitHub emails the owner.
- **Monitoring:** each forecast is archived and scored against later-reported fires (`reports/monitoring.json`), and the live ignition rate is checked against training to detect drift.
- **Versioning:** every model ships with a model card (`models/model_card_*.json`) recording version, git hash, training range, features, parameters and holdout metrics.
- **Tests:** leakage tests, plus a parity test that rebuilds a historical week through the live-forecast code path and matches the training panel exactly.

**Limitations**

- Large fires are rare (23 in the holdout), so their metrics have wide uncertainty and their probabilities run about 2× high.
- WFIGS (2021+) records about 20% fewer small fires than FPA-FOD, so recent ignition labels are slightly incomplete.
- Single-event backtests got slightly worse vs. v1. Burned cells sat at the 74th percentile for the Valley Fire (v1: 91st) and the 66th for Border 2 (v1: 73rd), even though aggregate metrics improved.
- MODIS NDVI composites on Planetary Computer can lag 2+ months; the dashboard shows a warning when that happens.
- Escape routes are counted per Census place, so large places can overstate options for individual neighbourhoods.

## Hardware & Performance

- **Hardware:** Apple Silicon laptop, 8 GB RAM, CPU only (no GPU needed).
- **Build time:** about 3.5 hours end to end. Optuna tuning takes ~1.5 h, ablations ~20 min, the daily-model experiment ~45 min, the final holdout plus production fit ~30 min, and data downloads ~1 h.
- **Daily forecast:** about 1.5 minutes.

## Setup Instructions

Python 3.12:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # http://localhost:8050
```

The repo ships with the trained models, processed features and the latest forecast. Large intermediates (raw downloads, daily gridMET, the 2.4M-row panel) are git-ignored and rebuilt by the pipeline.

**Deploy:** `Dockerfile` and `render.yaml` are included. Connect the repo in Render (free plan) and it redeploys whenever the daily forecast commit lands.

## How to Reproduce Results

```
python src/grid.py        # H3 grid
python src/fires.py       # CAL FIRE perimeters
python src/ignitions.py   # FPA-FOD + WFIGS ignition points
python src/weather.py     # gridMET weather, fuels, ERC, Santa Ana
python src/static.py      # terrain + land cover
python src/ndvi.py        # MODIS NDVI
python src/exposure.py    # population, communities, escape routes
python src/human.py       # roads, power lines, WUI, border/coast
python src/redflag.py     # NWS Red Flag Warnings (benchmark)
python src/features.py    # cell-week panel, 59 features, 3 targets
python src/experiments.py ablation            # model selection (CV 2001-2018)
python src/experiments.py tune ign 20         # likewise for fire, large
python src/experiments.py stack && python src/experiments.py calib
python src/daily.py                           # daily vs weekly experiment
python src/train.py       # one-time holdout + production models
python src/forecast.py    # today's outlook + monitoring
pytest -q
```

## Repository Structure

```
config.py                  region, grid, train/test split
app.py                     Dash dashboard (3 forecasts, drivers, communities, performance)
src/
  grid.py fires.py ignitions.py weather.py static.py ndvi.py human.py exposure.py redflag.py
                           data ingestion + feature sources
  features.py              panel assembly + targets (shared with the live forecast)
  modeling.py              fold ensemble, benchmarks, alert-budget metrics
  experiments.py           ablation, Optuna, stacking, calibration-window experiments
  daily.py                 daily-model experiment
  train.py                 frozen decisions -> one-time holdout -> production models
  forecast.py              daily outlook + data validation + archive
  monitor.py               live accuracy + drift
models/                    ensembles, Optuna params, decisions.json, model cards
reports/                   metrics.json, experiments.csv, monitoring.json, figures
tests/                     leakage, metric, validation, parity tests
.github/workflows/         daily test + forecast job
Dockerfile, render.yaml    deployment
```

Data sources: USFS FPA-FOD (Short 2022), NIFC WFIGS, CAL FIRE FRAP, gridMET (Abatzoglou 2013), Open-Meteo, NASA MODIS, Copernicus DEM and ESA WorldCover via Microsoft Planetary Computer, U.S. Census TIGER/Line 2020, OpenStreetMap contributors, Natural Earth, NWS warnings via the Iowa Environmental Mesonet.
