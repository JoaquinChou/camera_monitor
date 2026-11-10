# camera_monitor

面向长江禁渔执法的 AI 违规钓鱼监控系统。输入视频流，先用 YOLO26 ONNX 模型检测**人员/船只**，
再调用 **Qwen3-VL** 视觉语言模型（由 vLLM 服务）判断是否存在违规钓鱼行为，最终把逐帧结果写入
JSON 并生成标注视频。

- 检测在 **GPU 上通过 onnxruntime（CUDA 12）** 运行。
- VLM 走**共享的 HTTP vLLM 服务**（`Qwen3-VL-8B-Instruct`）。
- 支持**单进程**与**多进程（多卡）**推理。

> **约定**：下文的 `<repo>` 指仓库根目录——即你 `git clone` 后得到的目录（里面包含
> `app/`、`scripts/`、`Dockerfile` 等）。所有宿主机路径都相对它解析，因此项目在任何
> clone 位置都能直接运行。

---

## 1. 处理流程

```
视频
  └─ Camera.read_frame_by_video()          # 逐帧解码
      └─ SystemApp.process()               # YOLO26 检测 person / boat
          └─ Fishing.fishing_recog_infer() # 收集关键帧（每批 8 帧）
              └─ FishingRecognitionTask()  # Qwen3-VL → {reasoning, answer: Yes/No}
                  └─ 保存 JSON + VideoVisualizer → 标注视频 .mp4
```

---

## 2. 目录结构

```
camera_monitor/
├── app/
│   ├── fishing_application.py   # Fishing：单视频流水线 + 单进程入口
│   └── parallel_runner.py       # 多进程入口（轮询分片 + GPU 取余）
├── models/
│   ├── __init__.py              # 导出 Yolo26Detection、Qwen3VLClient
│   ├── detection/yolo26.py      # YOLO26 ONNX 检测器（onnxruntime，CUDA/CPU）
│   └── vlm/qwen3_vl_vlm.py      # Qwen3-VL HTTP 客户端（OpenAI 兼容 vLLM）
├── tasks/
│   ├── multi_detection.py       # MultiDetectionTask：从 YOLO 输出筛选 person/boat
│   └── fishing_recognition.py   # FishingRecognitionTask：VLM 提示词 + JSON 解析
├── system_app/
│   └── system_app.py            # SystemApp：逐帧检测流水线
├── utils/
│   ├── data_types.py            # ObjInfo / FrameInfo / FrameResult / VideoCache / Camera
│   ├── common_utils.py          # 图像 / JSON / 视频工具
│   ├── detection_utils.py       # letterbox、sigmoid、bbox 运算、绘制
│   └── logger.py                # loguru 配置
├── vis/
│   └── video_vis.py             # VideoVisualizer：把检测框 + VLM 结果画到视频上
├── configs/
│   ├── fishing_app.json         # 主应用配置（VLM 地址、multi_detect 路径）
│   ├── yolo26_onnx.json         # YOLO 模型配置（model_path、阈值、类别）
│   ├── test_data.json           # 视频列表（单进程）
│   └── test_data_mp.json        # 多进程测试视频列表（8 份）
├── test/
│   ├── test_yolo26.py           # 单图 YOLO 检测
│   ├── test_multi_detection.py  # 单图 person/boat 检测
│   ├── test_qwen3_vl_vlm.py     # VLM 客户端测试（文本 + 可选图像）
│   └── test_fishing_recognition.py # 钓鱼识别任务测试
├── scripts/
│   ├── run_infer.sh             # 单进程端到端推理
│   ├── run_infer_mp.sh          # 多进程端到端推理
│   ├── run_visualize.sh         # 结果可视化出片
│   └── run_test.sh              # 运行全部测试样例
├── Dockerfile                   # CUDA 12.4 镜像，onnxruntime-gpu 1.18（CUDA 12）
├── requirements.txt
├── pyproject.toml
├── data/                        # 视频 + 图片（挂载，gitignore）
├── output/                      # 结果（挂载，gitignore）
└── logs/                        # 日志（gitignore）
```

---

## 3. 环境依赖

- **Docker** + **nvidia-container-toolkit**（GPU 透传；`Default Runtime: nvidia`）。
- **vLLM** 服务，`Qwen3-VL-8B-Instruct` 运行在 `http://localhost:7864`（默认；否则设 `VLLM_BASE_URL`）。
- **ONNX 模型**文件（如 `modified_yolo26x_ultralytics.onnx`）放到 `<repo>/onnx/`——
  或把 `HOST_MODELS` 指向它的实际目录。它以只读方式挂载。
- **GPU**：参考环境为 8 张 RTX 3090。GPU 0/1 被 vLLM 占用；检测用 GPU 2-5（见注意事项）。

---

## 4. 构建 Docker 镜像

```bash
cd <repo>          # clone 得到的仓库根目录

# 若镜像已存在则跳过
if ! docker image inspect camera_monitor:cuda12 >/dev/null 2>&1; then
    docker build -t camera_monitor:cuda12 .
else
    echo "镜像 camera_monitor:cuda12 已存在，跳过构建"
fi
```

镜像提供运行时环境（CUDA / cuDNN / onnxruntime / Python 依赖），并把代码固化到 `/app` 作为兜底。
运行时脚本还会把仓库根目录挂载到 `/app`，因此**改代码/配置无需重建镜像**——只有 Dockerfile 或
依赖变化时才需要重建。基于 `nvidia/cuda:12.4.1-base-ubuntu22.04`，安装 `cuda-libraries-12-4` +
`libcudnn8` + `onnxruntime-gpu==1.18.0`（CUDA 12 构建），ONNX 开箱即用 GPU 推理。

---

## 5. 运行

所有脚本都用 `docker run --rm`（容器跑完自动销毁），挂载仓库根目录到 `/app`（代码实时生效、
免重建），以宿主机用户身份运行（`--user`），并通过环境变量读取配置。
完整变量见 [第 6 节](#6-环境变量)。

```bash
cd <repo>/scripts
```

### 5.1 单进程推理

```bash
./run_infer.sh                # 单卡（默认 GPU_ID=0），视频来自 test_data.json
GPU_ID=2 ./run_infer.sh       # 用物理 GPU 2（GPU 0/1 被 vLLM 占用）
```

### 5.2 多进程推理

把视频列表按 `NUM_PROCESSES` 切分成多个 worker（轮询）；worker `i` 跑在
`GPU_IDS[i % len(GPU_IDS)]` 上。默认：`NUM_PROCESSES=8`、`GPU_IDS=2,3,4,5`。

```bash
./run_infer_mp.sh                                    # 默认（8 进程，GPU 2-5）
NUM_PROCESSES=4 GPU_IDS=2,3,4,5 ./run_infer_mp.sh    # 4 进程，GPU 2-5
```

### 5.3 可视化

```bash
./run_visualize.sh
```

### 5.4 测试（一个脚本覆盖全部样例）

```bash
./run_test.sh                # 运行全部测试
./run_test.sh yolo26         # 单图 YOLO 检测
./run_test.sh multi          # 单图 person/boat 检测
./run_test.sh vlm            # VLM 客户端测试
./run_test.sh fishing        # 钓鱼识别任务测试
```

---

## 6. 环境变量

| 变量 | 默认值 | 用于 | 含义 |
|---|---|---|---|
| `IMAGE` | `camera_monitor:cuda12` | 全部脚本 | 镜像名 |
| `HOST_DATA` | `<repo>/data` | 全部脚本 | 宿主机数据目录（挂载到 `/app/data`）|
| `HOST_MODELS` | `<repo>/onnx` | 推理/检测脚本 | 宿主机 ONNX 目录（只读挂载到 `/app/onnx`）|
| `HOST_OUTPUT` | `<repo>/output` | 全部脚本 | 宿主机输出目录（挂载到 `/app/output`）|
| `GPU_ID` | `0` | 单进程/检测测试 | 检测 GPU 序号 |
| `GPU_IDS` | `2,3,4,5` | 多进程 | 逗号分隔的要使用的 GPU |
| `NUM_PROCESSES` | `8` | 多进程 | worker 进程数 |
| `VLLM_BASE_URL` | `http://localhost:7864` | 推理/VLM 测试 | vLLM 服务地址 |
| `VLLM_MODEL` | `Qwen3-VL-8B-Instruct` | 推理/VLM 测试 | 服务模型名 |
| `DATA_CONFIG` | `/app/configs/test_data.json` | 推理 | 视频列表配置（容器内路径）|
| `APP_CONFIG` | `/app/configs/fishing_app.json` | 推理 | 主应用配置（容器内路径）|
| `RESULT_DIR` | `/app/output/fishing_res` | 推理 | 结果输出目录（容器内路径）|
| `EVIDENCE_DIR` | `/app/output/evidence` | 推理 | 证据蒙太奇输出目录（容器内路径）|
| `INPUT_VIDEO` | `/app/data/videos/test_data_part_2.mp4` | 可视化 | 输入视频（容器内路径）|
| `RESULT_JSON` | `/app/output/fishing_res/test_data_part_2.json` | 可视化 | 结果 JSON（容器内路径）|
| `OUTPUT_DIR` | `/app/output/videos` | 可视化 | 输出目录（容器内路径）|
| `MODEL_CONFIG` | `/app/configs/yolo26_onnx.json` | 检测测试 | YOLO 配置（容器内路径）|
| `INPUT_IMAGE` | `/app/data/images/...` | 检测/VLM 测试 | 输入图片（容器内路径）|
| `OUTPUT_IMAGE` | `/app/output/images/...` | 检测测试 | 输出图片（容器内路径）|
| `INPUT_IMAGES` | `/app/data/images/...` | 钓鱼测试 | 逗号分隔的图片列表 |

> `<repo>` = 仓库根目录；`HOST_DATA`/`HOST_MODELS`/`HOST_OUTPUT` 由脚本相对它自动解析，
> 无需写绝对路径。容器内路径（`/app/...`）是镜像里固定的，不要改动。

---

## 7. 多进程设计

- **轮询分片**：worker `i` 处理 `video_list[i::NUM_PROCESSES]`。
- **GPU 分配**：`gpu_id = GPU_IDS[worker_id % len(GPU_IDS)]`。
- **`spawn` 上下文**（不用 `fork`）：CUDA 上下文不是 fork 安全的，因此每个 worker 都在
  子进程内创建自己的 `Fishing` 实例和 onnxruntime 会话。
- **显存**：`cudnn_conv_algo_search=HEURISTIC` + `arena_extend_strategy=kSameAsRequested`
  让 workspace 保持较小，使单卡可承载多个会话而不 OOM。

---

## 8. 注意事项

- **GPU 布局**：GPU 0/1 被共享的 vLLM 服务占用；检测请用 GPU 2-5。
- **VLM 是共享瓶颈**：检测可随进程数并行扩展，但所有 VLM 请求都打到同一个 vLLM
  （`--max-num-seqs 8`）。略微超配（`NUM_PROCESSES` 略大于 `len(GPU_IDS)`）有助于掩盖 VLM 延迟。
- **重复视频条目**（如 `test_data_mp.json`）会让每个 worker 写同一个输出 JSON（后写者覆盖）；
  真实多视频场景下每个视频生成独立 JSON，无冲突。
- **证据蒙太奇**：VLM 判定为 "Yes" 时，该批关键帧会被拼成带帧号标注的网格大图（默认 4x2），
  保存到 `<EVIDENCE_DIR>/{video}/evidence_NNN.jpg`，每个视频最多存 `evidence_num` 张。
  `evidence_num`/`evidence_cols`/`evidence_rows` 在 `fishing_app.json` 中配置。
- **导入机制**：镜像里通过 `PYTHONPATH=/app` 让顶层包（`models/utils/tasks/system_app`）
  可直接 import，等价于 `pip install -e .`，无需额外安装步骤。
