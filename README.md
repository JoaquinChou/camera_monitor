# camera_monitor

AI-powered illegal-fishing surveillance for the Yangtze River fishing-ban enforcement.
Given a video stream, it detects **persons/boats** with a YOLO26 ONNX model, then asks a
**Qwen3-VL** vision-language model (served by vLLM) whether illegal fishing is happening,
and finally writes per-frame results to JSON and an annotated video.

- Detection runs on **GPU via onnxruntime (CUDA 12)**.
- VLM runs on a **shared HTTP vLLM service** (`Qwen3-VL-8B-Instruct`).
- Support for **single-process** and **multi-process (multi-GPU)** inference.

> **Convention**: `<repo>` below means the repository root — the directory you get after
> `git clone` (it contains `app/`, `scripts/`, `Dockerfile`, ...). All host paths are
> relative to it, so the project works from any clone location.

---

## 1. Pipeline

```
video
  └─ Camera.read_frame_by_video()          # decode frames one by one
      └─ SystemApp.process()               # YOLO26 detect person / boat
          └─ Fishing.fishing_recog_infer() # collect key frames (8 per batch)
              └─ FishingRecognitionTask()  # Qwen3-VL → {reasoning, answer: Yes/No}
                  └─ save JSON + VideoVisualizer → annotated .mp4
```

---

## 2. Repository layout

```
camera_monitor/
├── app/
│   ├── fishing_application.py   # Fishing: single-video pipeline + single-process entry
│   └── parallel_runner.py       # multi-process entry (round-robin split + GPU modulo)
├── models/
│   ├── __init__.py              # exports Yolo26Detection, Qwen3VLClient
│   ├── detection/yolo26.py      # YOLO26 ONNX detector (onnxruntime, CUDA/CPU)
│   └── vlm/qwen3_vl_vlm.py      # Qwen3-VL HTTP client (OpenAI-compatible vLLM)
├── tasks/
│   ├── multi_detection.py       # MultiDetectionTask: filter person/boat from YOLO output
│   └── fishing_recognition.py   # FishingRecognitionTask: VLM prompt + JSON parse
├── system_app/
│   └── system_app.py            # SystemApp: per-frame detection pipeline
├── utils/
│   ├── data_types.py            # ObjInfo / FrameInfo / FrameResult / VideoCache / Camera
│   ├── common_utils.py          # image / json / video helpers
│   ├── detection_utils.py       # letterbox, sigmoid, bbox ops, drawing
│   └── logger.py                # loguru setup
├── vis/
│   └── video_vis.py             # VideoVisualizer: draw boxes + VLM result onto video
├── configs/
│   ├── fishing_app.json         # main app config (VLM endpoint, multi_detect path)
│   ├── yolo26_onnx.json         # YOLO model config (model_path, thresholds, classes)
│   ├── test_data.json           # video list (single-process)
│   └── test_data_mp.json        # video list for multi-process test (8 copies)
├── test/
│   ├── test_yolo26.py           # single-image YOLO detection
│   ├── test_multi_detection.py  # single-image person/boat detection
│   ├── test_qwen3_vl_vlm.py     # VLM client test (text + optional image)
│   └── test_fishing_recognition.py # fishing recognition task test
├── scripts/
│   ├── run_infer.sh             # single-process end-to-end inference
│   ├── run_infer_mp.sh          # multi-process end-to-end inference
│   ├── run_visualize.sh         # draw results onto video
│   └── run_test.sh              # run all test examples
├── Dockerfile                   # CUDA 12.4 image, onnxruntime-gpu 1.18 (CUDA 12)
├── requirements.txt
├── pyproject.toml
├── data/                        # videos + images   (mounted, gitignored)
├── output/                      # results           (mounted, gitignored)
└── logs/                        # logs              (gitignored)
```

---

## 3. Prerequisites

- **Docker** + **nvidia-container-toolkit** (GPU passthrough; `Default Runtime: nvidia`).
- **vLLM** serving `Qwen3-VL-8B-Instruct` at `http://localhost:7864` (default; set `VLLM_BASE_URL` otherwise).
- **ONNX model** file (e.g. `modified_yolo26x_ultralytics.onnx`) placed in `<repo>/onnx/`
  — or set `HOST_MODELS` to its actual directory. It is mounted read-only.
- **GPUs**: 8x RTX 3090 in the reference setup. GPU 0/1 are used by vLLM; detection uses GPU 2-5 (see notes).

---

## 4. Build the Docker image

```bash
cd <repo>          # the cloned repository root

# Skip if the image already exists
if ! docker image inspect camera_monitor:cuda12 >/dev/null 2>&1; then
    docker build -t camera_monitor:cuda12 .
else
    echo "Image camera_monitor:cuda12 already exists, skip build"
fi
```

The image bakes the code + configs into `/app`. `data/`, `output/` and the ONNX models
are mounted at runtime (not baked in). It is based on `nvidia/cuda:12.4.1-base-ubuntu22.04`
and installs `cuda-libraries-12-4` + `libcudnn8` + `onnxruntime-gpu==1.18.0` (CUDA 12 build),
so ONNX runs on GPU out of the box.

---

## 5. Run

All scripts use `docker run --rm` (container auto-removed on exit) and read config via
environment variables. See [Section 6](#6-environment-variables) for the full list.

```bash
cd <repo>/scripts
```

### 5.1 Single-process inference

```bash
./run_infer.sh                # one GPU (default GPU_ID=0), videos from test_data.json
GPU_ID=2 ./run_infer.sh       # use physical GPU 2 (vLLM occupies GPU 0/1)
```

### 5.2 Multi-process inference

Splits the video list across `NUM_PROCESSES` workers (round-robin); worker `i` runs on
`GPU_IDS[i % len(GPU_IDS)]`. Defaults: `NUM_PROCESSES=8`, `GPU_IDS=2,3,4,5`.

```bash
./run_infer_mp.sh                                          # defaults (8 procs, GPU 2-5)
NUM_PROCESSES=4 GPU_IDS=2,3,4,5 ./run_infer_mp.sh          # 4 procs on GPU 2-5
```

### 5.3 Visualization

```bash
./run_visualize.sh
```

### 5.4 Tests (one script covers all examples)

```bash
./run_test.sh                # run all tests
./run_test.sh yolo26         # single-image YOLO detection
./run_test.sh multi          # single-image person/boat detection
./run_test.sh vlm            # VLM client test
./run_test.sh fishing        # fishing recognition task test
```

---

## 6. Environment variables

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `IMAGE` | `camera_monitor:cuda12` | all scripts | docker image name |
| `HOST_DATA` | `<repo>/data` | all scripts | host data dir (mounted to `/app/data`) |
| `HOST_MODELS` | `<repo>/onnx` | infer/test scripts | host ONNX dir (mounted ro to `/app/onnx`) |
| `HOST_OUTPUT` | `<repo>/output` | all scripts | host output dir (mounted to `/app/output`) |
| `GPU_ID` | `0` | single-process / detection tests | detection GPU index |
| `GPU_IDS` | `2,3,4,5` | multi-process | comma-separated GPUs to use |
| `NUM_PROCESSES` | `8` | multi-process | number of worker processes |
| `VLLM_BASE_URL` | `http://localhost:7864` | infer / vlm tests | vLLM endpoint |
| `VLLM_MODEL` | `Qwen3-VL-8B-Instruct` | infer / vlm tests | served model name |
| `DATA_CONFIG` | `/app/configs/test_data.json` | infer | video list config (container path) |
| `APP_CONFIG` | `/app/configs/fishing_app.json` | infer | main app config (container path) |
| `RESULT_DIR` | `/app/output/fishing_res` | infer | result output dir (container path) |
| `INPUT_VIDEO` | `/app/data/videos/test_data_part_2.mp4` | visualize | input video (container path) |
| `RESULT_JSON` | `/app/output/fishing_res/test_data_part_2.json` | visualize | result JSON (container path) |
| `OUTPUT_DIR` | `/app/output/videos` | visualize | output dir (container path) |
| `MODEL_CONFIG` | `/app/configs/yolo26_onnx.json` | detection tests | YOLO config (container path) |
| `INPUT_IMAGE` | `/app/data/images/...` | detection/vlm tests | input image (container path) |
| `OUTPUT_IMAGE` | `/app/output/images/...` | detection tests | output image (container path) |
| `INPUT_IMAGES` | `/app/data/images/...` | fishing test | comma-separated image list |

> `<repo>` = repository root; `HOST_DATA`/`HOST_MODELS`/`HOST_OUTPUT` are resolved relative
> to it by the scripts, so no absolute paths are required. Container paths (`/app/...`) are
> fixed inside the image and should not be changed.

---

## 7. Multi-process design

- **Round-robin split**: worker `i` handles `video_list[i::NUM_PROCESSES]`.
- **GPU assignment**: `gpu_id = GPU_IDS[worker_id % len(GPU_IDS)]`.
- **`spawn` context** (not `fork`): CUDA contexts are not fork-safe, so each worker
  creates its own `Fishing` instance and onnxruntime session inside the child process.
- **Memory**: `cudnn_conv_algo_search=HEURISTIC` + `arena_extend_strategy=kSameAsRequested`
  keep the workspace small so several sessions can share one GPU without OOM.

---

## 8. Notes / caveats

- **GPU layout**: GPU 0/1 are occupied by the shared vLLM service; use GPU 2-5 for detection.
- **VLM is the shared bottleneck**: detection scales with process count, but all VLM
  requests hit the single vLLM (`--max-num-seqs 8`). A small oversubscription
  (`NUM_PROCESSES` slightly above `len(GPU_IDS)`) helps hide VLM latency.
- **Duplicate video entries** (as in `test_data_mp.json`) make every worker write the
  same output JSON (last writer wins); real multi-video use produces one JSON per video.
- **Import mechanism**: the image sets `PYTHONPATH=/app`, so the top-level packages
  (`models/utils/tasks/system_app`) are importable with no install step (equivalent to
  `pip install -e .`).
