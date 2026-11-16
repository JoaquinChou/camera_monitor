# Web 演示平台

面向验收汇报的轻量 Web 展示页，用于演示「钓鱼捕捞行为案例」与「AIS 统计分析（接口预留）」。

## 功能

- **首页**：两个入口按钮 —— ① 钓鱼捕捞行为案例展示；② AIS 统计分析（预留）。
- **钓鱼行为案例**：
  - 顶部指标卡（总体 / 捕捞 / 垂钓检出率、视频案例数、证据蒙太奇数）；
  - 视频行为案例（标注视频播放 + 证据蒙太奇 + 行为判定记录）；
  - 图像识别案例（图片 + 模型判定 Yes/No + 判定理由）。
- **AIS 统计分析**：`/api/ais` 接口已预留，接入数据源后即可扩展。

## 目录结构（高内聚 / 低耦合）

```
web/
├── server.py           # 入口：启动线程化 HTTP 服务
├── app/
│   ├── config.py       # 配置层：路径 / 端口（环境变量可覆盖）
│   ├── data.py         # 数据访问层：读 output 产物 → 结构化 case（无 HTTP 依赖）
│   ├── media.py        # 媒体层：安全路径解析 + HTTP Range 支持（视频可拖动）
│   └── handler.py      # HTTP 层：路由分发（/api、/media、/static）
├── static/             # 前端（index.html + css + js）
├── run_web.sh          # 一键启动脚本
└── README.md
```

分层说明：`data.py` 只读文件系统、返回纯 dict/list；`media.py` 只负责文件流式输出与 Range；
`handler.py` 只做 HTTP 路由，不解析业务数据；`config.py` 统一管理路径。新增一个数据源只需
在 `data.py` 增加一个加载函数并在 `handler.py` 注册一个路由，无需改动前端结构。

## 运行

```bash
cd /home/jcx/zgx_code/camera_monitor/web
bash run_web.sh                # 默认 0.0.0.0:8001
# 或
python3 server.py
```

浏览器访问 `http://<主机IP>:8001`。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEB_HOST` | `0.0.0.0` | 监听地址 |
| `WEB_PORT` | `8001` | 监听端口 |
| `WEB_OUTPUT_DIR` | `<repo>/output` | 推理产物目录（videos / fishing_res / evidence / img_res） |
| `WEB_FISHING_DATA` | `/home/jcx/code/camera_monitor/data/fishing` | 图像案例源数据集目录 |

## API

| 接口 | 说明 |
|---|---|
| `GET /api/summary` | 总体统计（视频数、证据数、图像指标） |
| `GET /api/video-cases` | 视频案例列表（视频 + 证据 + 判定记录） |
| `GET /api/image-metrics` | 图像批量评测指标 |
| `GET /api/image-cases` | 图像案例（正确样例 + 误判样例） |
| `GET /api/ais` | AIS 统计（预留，返回 `status: reserved`） |

## 说明

- 后端使用 Python 标准库 `http.server`，零新增依赖；媒体输出支持 HTTP Range，视频可拖进度条播放。
- 前端使用原生 HTML/CSS/JS，无框架依赖，便于二次开发与部署。
