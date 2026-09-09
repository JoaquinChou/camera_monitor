# =============================================================================
# camera_monitor - ONNX(GPU) detection + Qwen3-VL(HTTP) recognition image
#
# Key points for GPU inference:
#   1. Base image nvidia/cuda:12.4.1-base-ubuntu22.04 only ships cudart; add:
#        - cuda-libraries-12-4  -> cublas / cublasLt / cufft / curand / cusolver ...
#        - libcudnn8 (8.9.7)    -> cuDNN required by the onnxruntime-gpu CUDA12 build
#   2. onnxruntime-gpu must be the CUDA 12 build (1.18.0, from the onnxruntime-cuda-12
#      feed), NOT the PyPI default 1.16.x/1.18.x (CUDA 11 build, missing libcublasLt.so.11).
#   3. Install only the runtime deps the app actually imports; drop training/serving-only
#      torch / transformers / flash-attn / tensorrt_llm / nvidia-ammo / mpi4py.
#
# Run example (GPU passthrough + data/model/output mounts; GPU index starts at 0):
#   docker run --rm --gpus '"device=0"' \
#       -v $PWD/data:/app/data -v $PWD/output:/app/output \
#       -v /home/jcx/data/onnx:/app/onnx \
#       camera_monitor:cuda12
# =============================================================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# PYTHONPATH=/app makes the top-level packages (models/utils/tasks/system_app)
# importable inside the container -- equivalent to `pip install -e .` but with no
# install step.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps + CUDA math libs + cuDNN (NVIDIA apt repo ships with the base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
        cuda-libraries-12-4 \
        libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# All runtime PyPI deps in one shot, pinned to the validated set.
# Note: opencv-python 5.x forces numpy>=2 (and needs Qt5) on Python>=3.9, which
# conflicts with onnxruntime 1.18 (numpy 1.x); use headless 4.x instead.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        numpy==1.26.4 \
        scipy==1.13.1 \
        opencv-python-headless==4.10.0.84 \
        requests==2.34.2 \
        loguru==0.7.3 \
        flatbuffers==25.12.19 \
        protobuf==7.36.1 \
        sympy==1.14.0 \
        coloredlogs==15.0.1 \
        packaging==26.3

# onnxruntime-gpu: CUDA 12 build.
# --index-url targets the CUDA12 feed only; --no-deps avoids the PyPI CUDA11 1.18.0.
RUN python3 -m pip install --no-deps \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
        onnxruntime-gpu==1.18.0

WORKDIR /app

# Copy app code (data/models/output are mounted as volumes, not baked in)
COPY app         /app/app
COPY models      /app/models
COPY tasks       /app/tasks
COPY utils       /app/utils
COPY system_app  /app/system_app
COPY vis         /app/vis
COPY configs     /app/configs
COPY test        /app/test

CMD ["python3", "app/fishing_application.py"]
