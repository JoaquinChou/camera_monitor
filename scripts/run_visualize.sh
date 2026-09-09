#!/usr/bin/env bash
# =============================================================================
# run_visualize.sh - draw detection/VLM results onto the video
#
# CPU-only drawing: loads no model and calls no VLM. Container is removed on exit (--rm).
#
# Overridable env vars (all have defaults):
#   IMAGE        image name, default camera_monitor:cuda12
#   HOST_DATA    host data dir, default <repo>/data
#   HOST_OUTPUT  host output dir, default <repo>/output
#   INPUT_VIDEO  input video (container path), default /app/data/videos/test_data_part_2.mp4
#   RESULT_JSON  recognition result (container path), default /app/output/fishing_res/test_data_part_2.json
#   OUTPUT_DIR   output dir (container path), default /app/output/videos
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-camera_monitor:cuda12}"
HOST_DATA="${HOST_DATA:-${REPO_DIR}/data}"
HOST_OUTPUT="${HOST_OUTPUT:-${REPO_DIR}/output}"

INPUT_VIDEO="${INPUT_VIDEO:-/app/data/videos/test_data_part_2.mp4}"
RESULT_JSON="${RESULT_JSON:-/app/output/fishing_res/test_data_part_2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output/videos}"

docker run --rm \
    -v "${HOST_DATA}:/app/data" \
    -v "${HOST_OUTPUT}:/app/output" \
    -e INPUT_VIDEO="${INPUT_VIDEO}" \
    -e RESULT_JSON="${RESULT_JSON}" \
    -e OUTPUT_DIR="${OUTPUT_DIR}" \
    "${IMAGE}" python3 vis/video_vis.py
