#!/usr/bin/env bash
# =============================================================================
# run_test.sh - cover all test/ examples in one script
#
# Usage: ./run_test.sh {yolo26|multi|vlm|fishing|all}   (default: all)
#   yolo26  : single-image YOLO detection (GPU)
#   multi   : single-image multi-object detection person/boat (GPU)
#   vlm     : Qwen3-VL client test (text + optional image)
#   fishing : fishing recognition task test (VLM image sequence)
#   all     : run all of the above in order
#
# Each container is removed automatically when it exits (--rm).
#
# Overridable env vars (all have defaults):
#   IMAGE        image name, default camera_monitor:cuda12
#   HOST_DATA    host data dir, default <repo>/data
#   HOST_MODELS  host ONNX model dir, default <repo>/onnx
#   HOST_OUTPUT  host output dir, default <repo>/output
#   GPU_ID       GPU index inside the container, default 0
#   VLLM_BASE_URL / VLLM_MODEL / MODEL_CONFIG etc. see each test script
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
MODEL_CONFIG="${MODEL_CONFIG:-/app/configs/yolo26_onnx.json}"

usage() {
    echo "Usage: $0 {yolo26|multi|vlm|fishing|all}" >&2
    echo "  yolo26  : single-image YOLO detection (GPU)" >&2
    echo "  multi   : single-image multi-object detection person/boat (GPU)" >&2
    echo "  vlm     : Qwen3-VL client test (text + optional image)" >&2
    echo "  fishing : fishing recognition task test (VLM image sequence)" >&2
    echo "  all     : run all of the above in order (default)" >&2
}

test_yolo26() {
    echo "===== [test_yolo26] single-image YOLO detection ====="
    docker run --rm --gpus all \
        -v "${HOST_DATA}:/app/data" \
        -v "${HOST_MODELS}:/app/onnx:ro" \
        -v "${HOST_OUTPUT}:/app/output" \
        -e GPU_ID="${GPU_ID}" \
        -e MODEL_CONFIG="${MODEL_CONFIG}" \
        "${IMAGE}" python3 test/test_yolo26.py
}

test_multi() {
    echo "===== [test_multi_detection] single-image multi-object detection ====="
    docker run --rm --gpus all \
        -v "${HOST_DATA}:/app/data" \
        -v "${HOST_MODELS}:/app/onnx:ro" \
        -v "${HOST_OUTPUT}:/app/output" \
        -e GPU_ID="${GPU_ID}" \
        -e MODEL_CONFIG="${MODEL_CONFIG}" \
        "${IMAGE}" python3 test/test_multi_detection.py
}

test_vlm() {
    echo "===== [test_qwen3_vl_vlm] Qwen3-VL client test ====="
    docker run --rm --network host \
        -v "${HOST_DATA}:/app/data" \
        -e VLLM_BASE_URL="${VLLM_BASE_URL}" \
        -e VLLM_MODEL="${VLLM_MODEL}" \
        "${IMAGE}" python3 test/test_qwen3_vl_vlm.py
}

test_fishing() {
    echo "===== [test_fishing_recognition] fishing recognition task test ====="
    docker run --rm --network host \
        -v "${HOST_DATA}:/app/data" \
        -e VLLM_BASE_URL="${VLLM_BASE_URL}" \
        -e VLLM_MODEL="${VLLM_MODEL}" \
        "${IMAGE}" python3 test/test_fishing_recognition.py
}

case "${1:-all}" in
    yolo26)  test_yolo26 ;;
    multi)   test_multi ;;
    vlm)     test_vlm ;;
    fishing) test_fishing ;;
    all)     test_yolo26; test_multi; test_vlm; test_fishing ;;
    *)       usage; exit 1 ;;
esac
