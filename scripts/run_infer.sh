#!/usr/bin/env bash
# =============================================================================
# run_infer.sh - end-to-end inference: video -> YOLO detect(GPU) -> Qwen3-VL -> JSON
#
# The container is removed automatically when it exits (--rm).
#
# Overridable env vars (all have defaults):
#   IMAGE         image name, default camera_monitor:cuda12
#   HOST_DATA     host data dir (videos/images), default <repo>/data
#   HOST_MODELS   host ONNX model dir, default <repo>/onnx
#   HOST_OUTPUT   host output dir, default <repo>/output
#   GPU_ID        GPU index inside the container, default 0
#   VLLM_BASE_URL vLLM endpoint, default http://localhost:7864 (with --network host)
#   VLLM_MODEL    model name, default Qwen3-VL-8B-Instruct
#   DATA_CONFIG   video list config (container path), default /app/configs/test_data.json
#   APP_CONFIG    main config (container path), default /app/configs/fishing_app.json
#   RESULT_DIR    result output dir (container path), default /app/output/fishing_res
#   EVIDENCE_DIR  evidence output dir (container path), default /app/output/evidence
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-camera_monitor:cuda12}"
HOST_DATA="${HOST_DATA:-${REPO_DIR}/data}"
HOST_MODELS="${HOST_MODELS:-${REPO_DIR}/onnx}"
HOST_OUTPUT="${HOST_OUTPUT:-${REPO_DIR}/output}"

GPU_ID="${GPU_ID:-0}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:7864}"
VLLM_MODEL="${VLLM_MODEL:-Qwen3-VL-8B-Instruct}"
DATA_CONFIG="${DATA_CONFIG:-/app/configs/test_data.json}"
APP_CONFIG="${APP_CONFIG:-/app/configs/fishing_app.json}"
RESULT_DIR="${RESULT_DIR:-/app/output/fishing_res}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/app/output/evidence}"

docker run --rm --user "$(id -u):$(id -g)" --gpus all --network host \
    -v "${REPO_DIR}:/app" \
    -v "${HOST_DATA}:/app/data" \
    -v "${HOST_MODELS}:/app/onnx:ro" \
    -v "${HOST_OUTPUT}:/app/output" \
    -e GPU_ID="${GPU_ID}" \
    -e VLLM_BASE_URL="${VLLM_BASE_URL}" \
    -e VLLM_MODEL="${VLLM_MODEL}" \
    -e DATA_CONFIG="${DATA_CONFIG}" \
    -e APP_CONFIG="${APP_CONFIG}" \
    -e RESULT_DIR="${RESULT_DIR}" \
    -e EVIDENCE_DIR="${EVIDENCE_DIR}" \
    "${IMAGE}" python3 app/fishing_application.py
