# Web 演示平台

面向验收汇报的轻量 Web 展示页，用于演示「钓鱼捕捞行为案例」与「AIS 时空统计分析」。
下文 `<repo>` 均指仓库根目录。

## 功能

- **首页**：两个入口 —— ① 钓鱼捕捞行为案例展示；② AIS 时空统计分析。
- **钓鱼行为案例**：
  - 顶部指标卡（总体 / 捕捞 / 垂钓检出率、视频案例数、证据蒙太奇数）；
  - 视频行为案例（标注视频播放 + 证据蒙太奇 + 行为判定记录）；
  - 图像识别案例（图片 + 模型判定 Yes/No + 判定理由）。
- **AIS 时空统计分析**（数据驱动、离线内嵌）：
  - 摘要卡片：统计时空分布规律 / 热力图与时序图 / 建模预测与回测评估；
  - 历史活动分布：网格热力图 + 日内时段、逐日时序图（日期范围、时段、网格三向联动筛选）；
  - 回测对比：沿用昨天 / 7 日同期均值 / 泊松梯度提升树，按 MAE、RMSE、Top10 覆盖率、峰值命中评估；
  - 预测结果：次日 748 网格 × 6 时段的预测热点图、全区柱状图与 Top5，支持 CSV / JSON 下载。

## 目录结构（高内聚 / 低耦合）

```
web/
├── server.py           # 入口：启动线程化 HTTP 服务
├── app/
│   ├── config.py       # 配置层：路径 / 端口（环境变量可覆盖，默认走仓库相对路径）
│   ├── data.py         # 数据访问层：读 output 产物 → 结构化 case（无 HTTP 依赖）
│   ├── ais_data.py     # AIS 数据层：读内嵌结果 → 摘要 / 历史 / 回测 / 预测（无 HTTP 依赖）
│   ├── media.py        # 媒体层：安全路径解析 + HTTP Range 支持（视频可拖动）
│   ├── handler.py      # HTTP 层：路由分发（/api、/media、/static、下载）
│   └── data/           # 内嵌的 AIS 结果（history / metrics / forecast，离线可用）
├── static/             # 前端（index.html + css + js）
├── run_web.sh          # 一键启动脚本
└── README.md
```

分层说明：`data.py` / `ais_data.py` 只读文件系统、返回纯 dict/list；`media.py` 只负责文件流式输出与 Range；
`handler.py` 只做 HTTP 路由，不解析业务数据；`config.py` 统一管理路径。新增一个数据源只需在数据层
增加一个加载函数并在 `handler.py` 注册一个路由，无需改动前端结构。

## 运行

```bash
cd <repo>/web
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
| `WEB_FISHING_DATA` | `<repo>/data/fishing` | 图像案例源数据集目录（数据在仓库外时用绝对路径覆盖） |
| `WEB_AIS_DATA_DIR` | `<repo>/web/app/data` | AIS 结果目录（ais_history.json / ais_metrics.json / ais_forecast.csv） |

## API

| 接口 | 说明 |
|---|---|
| `GET /api/summary` | 总体统计（视频数、证据数、图像指标） |
| `GET /api/video-cases` | 视频案例列表（视频 + 证据 + 判定记录） |
| `GET /api/image-metrics` | 图像批量评测指标 |
| `GET /api/image-cases` | 图像案例（正确样例 + 误判样例） |
| `GET /api/ais` | AIS 统计 bundle（summary / metrics / history / forecast） |
| `GET /api/ais/forecast.csv` | 预测结果 CSV 下载 |
| `GET /api/ais/metrics.json` | 回测指标 JSON 下载 |

## 说明

- 后端使用 Python 标准库 `http.server`，零新增依赖；媒体输出支持 HTTP Range，视频可拖进度条播放。
- 前端使用原生 HTML/CSS/JS，无框架依赖；AIS 视图数据内嵌、可离线，便于二次开发与部署。
- 路径默认采用仓库相对位置，部署到其他机器时用上面的环境变量覆盖即可，无需改动代码。
