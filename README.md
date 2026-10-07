# San Diego Wildfire Risk

Forecasting the **7-day probability of wildfire** for every ~5 km² cell in San Diego County, then translating that risk into **who is exposed** and **how hard it is for them to get out**. The model is trained on 25 years of CAL FIRE perimeters (2001–2025; 2.4M cell-weeks, 1,777 burned) joined with daily weather, satellite vegetation, terrain, land cover, census population, and the OpenStreetMap road network. A Dash dashboard refreshes the outlook daily from live weather forecasts.

## Results

| Model (scored on 2019–2025, trained on 2001–2018) | PR-AUC lift | ROC-AUC | Top-5% capture | Top-10% capture |
|---|---|---|---|---|
| Climatology baseline (cell history × season) | 2.3× | 0.711 | **29.2%** | 29.2% |
| Weather-only LightGBM | 2.6× | 0.753 | 5.8% | 12.9% |
| **Full model** | **4.6×** | **0.842** | 19.0% | **33.6%** |

*PR-AUC lift = PR-AUC ÷ base fire rate (0.04% of cell-weeks). Top-k% capture = share of burned cells that were in that week's top k% riskiest cells.*

- **Backtests:** cells burned by the 2020 Valley Fire sat at the **91st** risk percentile the week it ignited; Border 2 (2025) at the **73rd**.
- **Evacuation:** 6 communities have ≤3 edge-disjoint escape routes. Borrego Springs (pop. 3,073) has 2 routes and a 65-minute median drive to a highway.
- Climatology still wins at the very top 5%. Past burn scars are a strong signal for *where*, and the full model trades some of that for better *when*.

## Approach

**Data (all free, no API keys)**

- **Labels:** CAL FIRE FRAP fire perimeters, intersected with an H3 (res-7) hexagon grid. A cell-week is positive if a fire that started that week touched the cell.
- **Weather:** gridMET daily max temperature, min humidity, wind, vapor pressure deficit (VPD), and precipitation, aggregated to weekly features plus 30/90/365-day antecedent rain and days since rain.
- **Vegetation:** MODIS 16-day NDVI and its anomaly vs. the cell's seasonal normal, using only composites published *before* the forecast week.
- **Static:** Copernicus 30 m DEM (elevation, slope, aspect, ruggedness), ESA WorldCover land-cover fractions, 2020 Census block population.
- **History:** years since the cell last burned, prior burns since 1950, and fires that started within 10 km in the previous 20 years. All are computed as-of each week, with no future leakage.

**Model**

- LightGBM ensemble of 5 fold models, isotonic-calibrated on out-of-fold predictions. Spread across the fold models is reported as uncertainty.
- **Leave-years-out CV**, not spatial-block CV. Spatial blocks leaked: one large fire burns many blocks in the same week, which inflated CV PR-AUC to 0.36 against 0.002 on true future years. Holding out whole years makes CV agree with the temporal holdout (4.3× vs 4.6× lift).
- Positives are weighted by 1/√(cells burned by that fire), so Cedar (2003) and Witch (2007) don't drown out the hundreds of smaller fires.
- Ranking metrics use the raw ensemble score. Isotonic calibration is a step function, so its ties distort within-week ranks.
- SHAP values per cell explain each forecast. The top global drivers are NDVI, seasonality, 90-day rainfall, local ignition history, and peak VPD.

**Exposure & evacuation**

- **Expected people exposed** = Σ P(fire) × population, summed per cell and per community (Census places).
- **Escape routes** = max-flow / min-cut on the OSM drive network (87k nodes): the number of edge-disjoint paths from inside a community to a highway at least 2 km outside it, capped at 10.
- **Minutes to highway** = median drive time from a community's road nodes to the nearest motorway or trunk road (multi-source Dijkstra).

**Live forecast**

- Each day, `forecast.py` combines the latest observed gridMET weather with the Open-Meteo 7-day forecast (same variables and units), refreshes perimeters and NDVI, and writes a new outlook. A GitHub Action runs it every morning.

**Limitations**

- Calibrated probabilities run about 2× high on 2019–2025, because those years burned half as much as the 2001–2018 training years. Rankings hold up; absolute probabilities should be read as relative risk.
- Perimeters miss many small (<10 acre) fires. Ignition locations are approximated by perimeter centroids. Escape routes are counted at the Census-place level, so large place polygons can overstate the routes available to individual neighborhoods.

## Hardware & Performance

- **Hardware:** Apple Silicon laptop, CPU only (no GPU needed).
- **Build time:** about 35 minutes end to end. NDVI download takes ~17 min, the road network and evacuation analysis ~7 min, training and evaluation ~4.5 min, gridMET ~2 min.
- **Daily forecast:** about 1 minute.

## Setup Instructions

Python 3.12. Create an environment and install dependencies:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The repo ships with the trained model, processed features, and the latest forecast, so you can run `python app.py` right away and open http://localhost:8050. Large intermediates (raw downloads, daily gridMET, the 2.4M-row panel) are git-ignored and rebuilt by the pipeline.

## How to Reproduce Results

Run the pipeline in order from the repo root:

```
python src/grid.py       # H3 grid over the county
python src/fires.py      # CAL FIRE perimeters -> weekly cell labels
python src/weather.py    # gridMET -> weekly fire-weather features
python src/static.py     # terrain + land cover
python src/ndvi.py       # MODIS NDVI
python src/exposure.py   # population, communities, evacuation routes
python src/features.py   # assemble the cell-week panel
python src/train.py      # CV, holdout, baselines, figures, production model
python src/forecast.py   # today's 7-day outlook
python app.py            # dashboard
```

`train.py` writes all metrics to `reports/metrics.json`, plus `reports/shap_importance.png`, `calibration.png`, and backtest maps for the Valley and Border 2 fires.

## Repository Structure

```
config.py                   region, grid resolution, train/test split
app.py                      Dash dashboard (map, cell drivers, community table, model tab)
src/
  grid.py                   county boundary + H3 grid
  fires.py                  perimeter download + cell labels
  weather.py                gridMET download + weekly features (shared with forecast)
  static.py                 DEM terrain + WorldCover land cover
  ndvi.py                   MODIS NDVI per cell
  exposure.py               census population, communities, max-flow escape routes
  features.py               panel assembly (shared with forecast)
  train.py                  ensemble, CV, baselines, calibration, SHAP, backtests
  forecast.py               daily 7-day outlook from gridMET + Open-Meteo
models/ensemble.joblib      production model (trained on 2001–2025)
reports/                    metrics.json + figures
data/forecast/              latest outlook used by the dashboard
.github/workflows/          daily forecast refresh
```

Data sources: CAL FIRE FRAP, gridMET (Abatzoglou 2013), Open-Meteo, NASA MODIS MOD13Q1 and Copernicus DEM / ESA WorldCover via Microsoft Planetary Computer, U.S. Census TIGER/Line 2020, and OpenStreetMap contributors.
