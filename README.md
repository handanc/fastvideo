# FastVideo3 🎬

**基于 LangGraph 多智能体的全 API 化短剧视频生成教学平台。**

输入一个主题，输出一部有人物对话、有人物动作、有转场动画的完整短剧视频。
所有生成推理走 API KEY（DeepSeek + 阿里云百炼），**无需 GPU**；本地仅保留 CPU 级组件（YOLOv8-seg + rembg 双人同屏、FFmpeg 剪辑）。

- 需求与设计：[PRD.md](PRD.md) · [TDD.md](TDD.md)
- 前代项目：FastVideo2（本地推理单体版），本项目为其架构现代化重构

## 快速开始（3 步跑通）

```bash
# 1) 环境（conda，详见 scripts/setup.sh / setup.ps1）
conda env create -f environment.yaml && conda activate fastvideo3
pip install -e .

# 2) 零成本演示（Fake 模式，无需任何 KEY）
fastvideo3 run --topic "校园喜剧" --duration 30 --fake

# 3) 真实生成：复制 .env.example → .env，填入两把 KEY 后
fastvideo3 run --topic "职场新人的第一天" --duration 60
```

**一把 `DASHSCOPE_API_KEY` 跑通全系统**（2026-08 起默认全百炼栈，性价比选型）：

| 能力 | 模型 | 参考价 |
|---|---|---|
| 编剧 | `qwen3.7-plus`（百炼模型市场专属端点，见 `.env`） | 按工作空间计费 |
| 文生图 | `wanx2.1-t2i-turbo` | ~¥0.18/张 |
| 图生视频 | `wan2.2-i2v-flash`（480P） | ¥0.11/秒 |
| 口型 | `wan2.2-s2v`（480P，新账号送 100 秒） | ¥0.5/秒 |
| 配音 | `cosyvoice-v2`（失败自动降级免费 edge-tts） | ¥2/万字符 |
| 视觉评审 | `qwen-vl-plus` | ¥1.5/百万 token |

## 常用命令

| 命令 | 说明 |
|---|---|
| `fastvideo3 run --topic ... [--fake] [--review gated] [--talk-mode audio_overlay]` | 端到端生成 |
| `fastvideo3 stage script/assets/storyboard/shots/audio/talk/edit` | 单阶段运行（教学） |
| `fastvideo3 resume <run_id> --decision approve` | 续跑 / 人工审片回复 |
| `fastvideo3 doctor` | 环境自检 |
| `fastvideo3 cost <run_id>` / `fastvideo3 ls-runs` | 成本台账 / 历史运行 |
| `fastvideo3 serve` | **Vue Web 面板**（FastAPI + Vue3，见下节） |
| `streamlit run fastvideo3/app/streamlit_app.py` | Streamlit 面板（教学观测用） |

## Vue Web 面板

参考 FastVideo 的 "FastAPI 服务 + Web UI" 架构：`fastvideo3/app/api.py` 提供 REST 接口，`web/`（Vite + Vue 3）提供前端，构建产物由 FastAPI 同源托管。

```bash
# 1) 安装后端 Web 依赖 + 构建前端（一次性）
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..

# 2) 启动（打开 http://127.0.0.1:8000）
fastvideo3 serve
```

前端开发模式（热更新）：`fastvideo3 serve` 起后端后，在 `web/` 里 `npm run dev`，访问 http://127.0.0.1:5173（`/api` 自动代理到 8000）。

页面功能（4 个视图，无路由依赖）：

| 视图 | 功能 |
|---|---|
| 新建任务 | 主题/风格/时长/审片模式/口型模式/画质/画幅/熔断/Fake 开关 → 后台启动生成 |
| 运行历史 | 状态徽章（完成/运行中/待审片/失败/停摆）+ 成本摘要，5s 自动刷新 |
| 运行详情 | 事件流时间线（3s 轮询）· 成本明细 · 资产/分镜画廊 · 成片与镜头片段播放 · 审片通过/驳回续跑 |
| 环境自检 | JSON 版 doctor（依赖/FFmpeg/KEY/权重/数据库） |

REST API（Swagger 文档：`/docs`）：`GET/POST /api/runs` · `GET /api/runs/{id}` · `POST /api/runs/{id}/resume` · `GET /api/files/{id}/{stage}/{name}` · `GET /api/doctor` · `GET /api/health`

端到端冒烟（`fastvideo3 serve` 启动后，另开终端）：`python scripts/smoke_web_api.py`——Fake 模式零成本验证「新建任务 → 后台生成 → 成片下载 → 越权防护」全链路。

## 架构一图

```
主题 → [编剧] → 结构化剧本 → [人工审片*]
     → [美术] 角色/场景资产库（一致性锚点）
     → [分镜师] 每镜头首帧 + VLM 抽查
     → 并行: [I2V 动作/风景镜头]  [TTS 对白] → [S2V 对话镜头直出]
     → [剪辑] 规格统一 → xfade 转场 → 字幕 → BGM → 成片
     → [终审 VLM] → [人工看片*] → final.mp4 + 成本报告
```

编排：LangGraph 状态图 + SqliteSaver 检查点（进阶 `docker compose up -d` 切 PostgreSQL）。
口型三档 `talk_mode`：`s2v_direct`（默认）/ `v2v_overlay` / `audio_overlay`（低成本）。

## 项目结构

```
fastvideo3/
├── core/       # 配置/schema/存储/成本/任务台账（第1讲素材）
├── providers/  # DeepSeek/百炼/SiliconFlow/可灵/edge-tts + Fake 全家桶
├── modules/    # m1剧本 … m8评审（教学章节 1:1 映射）
├── graph/      # LangGraph 状态/节点/图构建（第6讲素材）
├── app/        # Typer CLI + FastAPI Web API + Streamlit 面板
└── utils/      # ffmpeg/音频/图像/YOLO/rembg（懒加载）
web/            # Vue 3 前端（Vite 构建，dist 由 FastAPI 托管）
```

## 开发者（CI 自检）

```bash
pip install -e ".[dev]"
ruff check .          # 风格
mypy fastvideo3       # 类型
pytest -m "not api"   # 离线单测 + Fake 全链路冒烟 + Web API 单测（无需 KEY）
pytest -m api         # 真实 API 冒烟（需 .env，nightly）
cd web && npm ci && npm run build   # 前端构建门禁（CI job: web-build）
```

## 与 TDD 的实现差异说明

1. Provider 层用 **httpx 直连**（更薄、更教学友好），未强依赖 LangChain 的 LLM 封装；LangGraph 仍是编排骨架。
2. 并行采用"图级扇出（shots/audio 双分支）+ 节点内信号量并发"；逐镜头 `Send()` map-reduce 留作进阶练习（见 TDD §5.4）。
3. 双人同屏在缺少 ultralytics/rembg 时自动降级为左右拼接，保证任何环境可交付。
