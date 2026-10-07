from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for p in (RAW, PROC, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

STATE_FIPS, COUNTY_FIPS = "06", "073"  # San Diego County
BBOX = (-117.65, 32.50, -116.05, 33.55)  # lon_min, lat_min, lon_max, lat_max

H3_RES = 7     # ~5 km² prediction cells
BLOCK_RES = 5  # ~250 km² blocks for spatial cross-validation

TRAIN_START = "2001-01-01"
TEST_START = "2019-01-01"  # temporal holdout: train < 2019, test >= 2019
CRS_M = "EPSG:3310"        # California Albers (meters)

SEED = 42
