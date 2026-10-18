# FastVideo3 🎬

输入一个主题，产出一部带对白、带口型、带转场的竖屏短剧 —— **全程走 API，不需要 GPU**。

- **五步一条路径**：`剧本 → 角色图 → 场景图 → 场景视频 → 成片`。点一次补一步，每步之间人都能介入。
- **多模态**：一次 wan3.0 调用吃「角色立绘 + 场景图 + 音色样本 + 多镜头提示词」，直接吐出带口型的整段戏。
- **分辨率是唯一旋钮**（480P / 720P / 1080P）：尺寸、一致性深度、单价量级、是否自动重拍都由它带出来。
- **一把 `DASHSCOPE_API_KEY` 跑通全系统**；钱到上限就停下问人，不静默少出片；产物、血缘、成本都在盘上，中断不需要"恢复"功能。

设计文档：[PRD.md](PRD.md) · [TDD.md](TDD.md) · 踩坑复盘：[INTEGRATION.md](INTEGRATION.md)

## 快速开始

```bash
./scripts/setup.sh                                    # .venv → 按 lock 装依赖 → pip install -e .
                                                      # → provision（ffmpeg/权重/模板 .env）→ web 构建 → doctor
fastvideo3 new --topic "校园喜剧" --duration 30 --fake  # Fake 模式：零成本、无需任何 KEY
fastvideo3 step <run_id>                              # 点一次补一步，共 5 次

fastvideo3 doctor --probe                             # 真跑前先探活：欠费/未开通别等烧钱才暴露
fastvideo3 new --topic "职场新人的第一天" --resolution 720P --budget 150
fastvideo3 step <run_id>                              # 到成本上限会停下问人
fastvideo3 budget-continue <run_id> 200               #   抬高上限，从停下处接着跑

pip install -e ".[web]" && fastvideo3 serve           # Web 面板 http://127.0.0.1:8000
```

面板是推荐用法（五步之间能改剧本稿、换定妆照、删图重出、只重跑某一组）；`web/` 里 `npm run dev` 可热更新。其余常用命令：

| 想做的事 | 命令 |
| --- | --- |
| 按意见改剧本（只作废真被改到的镜头） | `fastvideo3 script-regen <run> --prompt "…" --mode revise` |
| 只重出一张图 / 一组视频 | 面板单卡「生成 / 恢复」，或 `fastvideo3 asset-gen` · `fastvideo3 scene-gen` |
| 进度 / 成本 / 降级 | `fastvideo3 cost <run>` · `output/<run>/run_summary.json` |
| 一致性报告 / 消融四臂 / 只重跑剪辑（零 API 费） | `fastvideo3 consistency <run>` · `python scripts/run_ablation.py --fake` · `python scripts/recompose_final.py <run>` |

## 多模态 Agent 编排

同一个 run 有**两种编排、一份执行器**：分步模式（面板「开始生成」/ `fastvideo3 step`）与图模式（`fastvideo3 graph`）。缺口由 `core/progress.py` 从盘上算，执行只走 `core/stages.py` 这一份装配。

```mermaid
flowchart LR
    S["剧本"] --> A["角色图 ‖ 场景图"]
    A --> V["场景视频<br/>立绘+场景图+音色样本+提示词<br/>→ 一次调用出画面·语音·口型"]
    V --> C["成片<br/>剪辑 → 一致性度量 → 闸门 → 终审"]
    C --> G{"人审"}
    G -- 点名重出某几组 --> V
    G -- 收片 --> F(["final.mp4"])
```

链上任何一步之前，**实时花费到上限就停下问人**（`core/pause.py`）：停止提交新的付费任务、已出产物一个不动，确认后从停下处接着跑。上游产物一变，下游按血缘精确标 stale（`lineage.json`）；在盘且未标 stale 的一律复用，在途异步任务接着轮询而不是重新提交（重新提交就是为同一秒画面付第二次钱）。

自动重做 ≤2 轮只在 `compose` 那一步里，**图上的回边只承载"人"的决定**；节点内部不复制任何业务判断（同一事实写两份实现，是本项目历史上所有"看起来通电、实际无效"退化的共同根源）。

| 步骤 | 产出 | 模型 | 人能做什么 |
| --- | --- | --- | --- |
| ① 剧本 | `script.json` | `qwen3.7-flash` | 编辑剧本稿（**任意排版都能存**：本地确定性整理，整不动才调一次 TEXT 只改排版并逐字校验台词没被改）· 按意见重写 · 删除重生成 |
| ② 角色图 | `assets/char_*.png` | `wan2.7-image` | 生成 / 恢复 / 上传替换 / 改外貌描述（立绘是全片身份锚点） |
| ③ 场景图 | `assets/scene_*.png` | `wan2.7-image` | 同上；超 25s 拆出的多段共用同一张场景图 |
| ④ 场景视频 | `clips/scene_*.mp4` + 同名 `.json` | `wan3.0-video`（画面+语音+口型一次出）· `cosyvoice-v2`（出音色样本） | 逐组改提示词、增删参考图、改时长、只重出某一组 |
| ⑤ 成片 | `final/final.mp4` + 字幕 + 度量/QC 报告 | 本地 FFmpeg + `qwen-vl-plus` | 看片、点名重出、改字幕后只重跑剪辑 |

第 ④ 步的输入合同（官方硬约束，超了直接 400）：`reference_image` ≤10 张（编号即提示词里的图N）、`reference_audio` ≤5 段且总 ≤15s、时长 2–30s，且参考素材与 `first_frame/last_frame` **互斥**；每段软顶 25s，超了按镜头边界拆分。素材计划只有一份（`m4_scene.plan_inputs`），面板预览与实发请求同源。

恢复不需要特殊代码：产物在盘且未被标 stale 就跳过，在途异步任务接着轮询而不重新提交。图模式：

```bash
fastvideo3 graph <run>                        # 跑到人审门停下（成片已产出）
fastvideo3 graph <run> --resume --redo scene_2   # 只重出这一组，再交人审
fastvideo3 graph <run> --resume --finish         # 收片
```

每条质量说法都要能对着一条 run 核，盘上证据是：`events.jsonl` 的 `keyframe_gen{ref_count,ref_ids}`（首帧带没带那张角色图）· `clips/scene_*.json` 的 `ref_images`/`dropped`（真发了什么、丢了谁）· `cost.json`（钱打在哪个模型上）· `consistency_report.json` + `degradations.json`（度量与降级）。

## 项目结构

```
fastvideo3/
├── core/       # 判据层：config(档位) progress(缺口) stages(唯一装配) storage(产物+血缘+事件)
│               #   scenegroup/voiceref/voice_budget draft_norm+script_text(剧本稿) cost/pricing
│               #   pause(成本闸门) tasklock model_catalog(面板可选模型) envfile(.env 定点写回)
├── modules/    # 执行层 m1 剧本 → m9 度量（与教学章节 1:1；m4_scene 是当前主干）
├── providers/  # base(能力位/协议) factory(装配) dashscope/deepseek/siliconflow/kling/fal/edge_tts + fake
├── graph/      # LangGraph：builder(节点与回边) nodes(只调执行器) state checkpointer
├── app/        # cli.py(Typer) · api.py(FastAPI，42 路由) · streamlit_app.py(教学观测)
└── utils/      # ffmpeg · audio · image · faceid(SFace) · lipsync · matting · emotion
web/            # Vue 3 面板：NewRunForm · RunList · RunDetail(+ProductionBoard+StepBar) · DoctorView
configs/        # default.yaml(=480P 本体) + profile_720P.yaml + profile_1080P.yaml + ablations/
prompts/ scripts/ tests/
output/<run_id>/# script.json assets/ storyboard/ audio/ clips/ final/ qc/
                # cost.json events.jsonl lineage.json degradations.json run_summary.json …
```

## 技术栈

| 层 | 选型 |
| --- | --- |
| 语言 / 并发 | Python ≥ 3.11，`asyncio` 全程异步（模块内信号量并发） |
| 数据模型 | Pydantic v2 —— `core/schema.py` 是剧本/资产/成本/事件的唯一校验 |
| 编排 | LangGraph `StateGraph` + `interrupt()`；检查点 SQLite（每 run 一个 `graph.sqlite`）或 PostgreSQL（`[postgres]` + `docker compose up -d`） |
| 服务 / 前端 | FastAPI + Uvicorn（`[web]`）· Vue 3.5 + Vite 6；`web/dist/` 由后端同源托管（`index.html` 走 `no-cache`，`/assets/*` 内容寻址长缓存） |
| 命令行 | Typer + Rich，与面板落到同一份执行器 |
| 生成模型 | 文本 `qwen3.7-flash`（plus/max/3.8 可切，DeepSeek 兜底）· 文生图 `wan2.7-image`（支持参考图 → 三档都靠它锁脸）· 图生视频与口型 `wan3.0-video` 场景级生成 · 配音 `cosyvoice-v2`（降级免费 edge-tts 并留痕）· 评审 `qwen-vl-plus` |
| 本地组件 | FFmpeg（`bin/` 内置优先于系统 PATH）· Pillow · SFace 人脸 embedding（缺权重/缺能力自动退回并如实标注）· 可选 YOLO+rembg |
| 台账 | SQLite JobStore（可切 PG）+ 纯文件 JSON：`cost.json` 逐条计费、`events.jsonl` 全事件、`lineage.json` 血缘、`degradations.json` 降级 |
| HTTP | 直接 httpx + tenacity，不套 LangChain 的 LLM 封装：请求形状、错误文本、计费口径都要能直接读 |

**档位差异**（`GET /api/resolutions` 就是这张表；单价来自 `pricing.yaml`，`fastvideo3 refresh-prices` 从官网价目页回写）：

| 档位 | 成片尺寸 | 30 秒预估 | 锁脸 | 度量 | 自动重拍 | 立绘抽卡 |
| --- | --- | --- | --- | --- | --- | --- |
| 480P 教学/试跑 | 540×960 | ≈¥10–16 | ✔ | 抽样 | ✘（只报告） | ✘ |
| 720P 交付 | 720×1280 | ≈¥23–35 | ✔ | 抽样 | ✔ ≤2 轮 | ✘ |
| 1080P 商业级 | 1080×1920 | ≈¥58–88 | ✔ | 全检 | ✔ ≤2 轮 | ✔ 2 候选 |

场景片段按输出秒计价：480P ¥0.3/秒 · 720P ¥0.6/秒 · 1080P ¥1.2/秒。锁脸三档都开，**重拍是唯一按档花钱的差别**。

## 开发者

```bash
pip install -e ".[dev]" && ruff check . && mypy fastvideo3
pytest            # 739 条：单元 + Fake 全链路 + Web API，全部离线（无需 KEY）
pytest -m api     # 真实 API 冒烟（需 .env）
cd web && npm ci && npm run build
```

三条硬规矩（都有测试钉着）：**一份判据**（同一事实写两份实现就会漂一次、静默退化一次）· **降级必须响亮**（进 `events.jsonl` + `degradations.json` 并出现在看板上，只写日志等于没写）· **不覆盖人工上传的资产**（`uploads.json` 是硬边界，血缘作废与重生成都跳过它）。
