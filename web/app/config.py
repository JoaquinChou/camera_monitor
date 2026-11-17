"""Runtime configuration for the web demo server.

Every path and the listen address can be overridden through environment
variables, so the same code runs unchanged on any host (bare metal, Docker,
CI, ...). This module is the single source of truth for where outputs and
static assets live.
"""
import os
from pathlib import Path

# Repository root: <repo>/web/app/config.py  ->  <repo>
REPO_DIR = Path(__file__).resolve().parent.parent.parent

# Listen address / port.
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8001"))

# Pipeline output directory (videos / fishing_res / evidence / img_res).
OUTPUT_DIR = Path(os.environ.get("WEB_OUTPUT_DIR", str(REPO_DIR / "output")))

# Fishing image datasets (source images for the image-case gallery). The demo
# does not need to know where the data physically lives: override WEB_FISHING_DATA.
FISHING_DATA_DIR = Path(
    os.environ.get("WEB_FISHING_DATA", str(REPO_DIR / "data" / "fishing"))
)

# Frontend static assets served under /static/.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# AIS spatio-temporal analysis results. Bundled with the web app so the demo is
# self-contained / offline-friendly; override WEB_AIS_DATA_DIR to point at a
# freshly reproduced run (see the AIS project README for regeneration).
AIS_DATA_DIR = Path(
    os.environ.get("WEB_AIS_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
)

# Derived output locations ---------------------------------------------------
VIDEOS_DIR = OUTPUT_DIR / "videos"
RESULTS_DIR = OUTPUT_DIR / "fishing_res"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
IMG_RESULTS_FILE = OUTPUT_DIR / "img_res" / "batch_images_results.json"

# How many image cases the gallery surfaces per category / as error examples.
IMAGE_CASES_PER_CATEGORY = int(os.environ.get("WEB_IMAGE_CASES_PER_CATEGORY", "6"))
IMAGE_ERROR_CASES = int(os.environ.get("WEB_IMAGE_ERROR_CASES", "3"))

# Derived AIS data file locations ---------------------------------------------
AIS_HISTORY_FILE = AIS_DATA_DIR / "ais_history.json"
AIS_METRICS_FILE = AIS_DATA_DIR / "ais_metrics.json"
AIS_FORECAST_FILE = AIS_DATA_DIR / "ais_forecast.csv"
