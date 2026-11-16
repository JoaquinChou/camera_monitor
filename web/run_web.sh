#!/usr/bin/env bash
# =============================================================================
# run_web.sh - start the web demo server (frontend + pipeline outputs + APIs)
#
# Serves:
#   /             demo frontend (two buttons: fishing cases / AIS stats)
#   /api/*        JSON APIs (summary / video-cases / image-cases / ais)
#   /media/*      videos, evidence montages, results, image cases (Range-enabled)
#
# Overridable env vars:
#   WEB_HOST             listen address,  default 0.0.0.0
#   WEB_PORT             listen port,     default 8001
#   WEB_OUTPUT_DIR       pipeline output, default <repo>/output
#   WEB_FISHING_DATA     image datasets,  default /home/jcx/code/camera_monitor/data/fishing
# =============================================================================
set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WEB_DIR}"

exec python3 server.py "$@"
