#!/usr/bin/env bash
# =============================================================================
# run_batch_images.sh - batch evaluation of fishing recognition on image datasets
#
# Mounts the fishing image datasets (read-only) and runs the batch evaluation
# inside the container. Container is removed on exit (--rm).
#
# Overridable env vars (all have defaults):
#   IMAGE         image name, default camera_monitor:cuda12
#   FISHING_DATA  host dir of the fishing datasets,
#                 default /home/jcx/code/camera_monitor/data/fishing
#   BULAO_DIR     bulao (捕捞) image folder, container path
#                 default /app/fishing_data/bulao-img/images
#   CHUIDIAO_DIR  chuidiao (垂钓) image folder, container path
#                 default /app/fishing_data/chuidiao-img/images
#   OUTPUT_JSON   result JSON path, container path
#                 default /app/output/img_res/batch_images_results.json
#   VLLM_BASE_URL vLLM endpoint, default http://localhost:7864
#   VLLM_MODEL    model name, default Qwen3-VL-8B-Instruct
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-camera_monitor:cuda12}"
FISHING_DATA="${FISHING_DATA:-/home/jcx/code/camera_monitor/data/fishing}"
BULAO_DIR="${BULAO_DIR:-/app/fishing_data/bulao-img/images}"
CHUIDIAO_DIR="${CHUIDIAO_DIR:-/app/fishing_data/chuidiao-img/images}"
OUTPUT_JSON="${OUTPUT_JSON:-/app/output/img_res/batch_images_results.json}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:7864}"
VLLM_MODEL="${VLLM_MODEL:-Qwen3-VL-8B-Instruct}"

docker run --rm --user "$(id -u):$(id -g)" --network host \
    -v "${REPO_DIR}:/app" \
    -v "${FISHING_DATA}:/app/fishing_data:ro" \
    -e BULAO_DIR="${BULAO_DIR}" \
    -e CHUIDIAO_DIR="${CHUIDIAO_DIR}" \
    -e OUTPUT_JSON="${OUTPUT_JSON}" \
    -e VLLM_BASE_URL="${VLLM_BASE_URL}" \
    -e VLLM_MODEL="${VLLM_MODEL}" \
    "${IMAGE}" python3 test/test_batch_images.py
