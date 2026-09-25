# FastVideo3 技术设计文档（TDD）

| 项 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 日期 | 2026-08-22 |
| 对应需求 | `PRD.md` v1.0 |
| 状态 | 待评审 |
| 构建流程 | PRD ✅ → **TDD（本文档）** → 生成各模块 → CI 自检 → 系统联调 |

---

## 1. 概述与设计原则

### 1.1 技术目标

实现 PRD 的两大约束：**① 所有生成类推理全部 API KEY 化（零 GPU）；② LangGraph 多智能体编排，架构可教学、可演进。**

### 1.2 设计原则

| 原则 | 落地方式 |
|---|---|
| **API-first** | 一切生成能力（文本/图像/视频/口型/语音/视觉评审）均通过 Provider 抽象调用云端 API；本地只保留 CPU 级能力（YOLOv8-seg、rembg、FFmpeg） |
| **Schema-first** | Pydantic v2 定义全链路数据契约（剧本→资产→分镜→镜头→成片），LLM 输出强制 JSON Schema 校验 |
| **Provider 模式** | 每类能力一个抽象接口，供应商是可插拔实现；新增供应商零业务代码改动 |
| **图即流程** | 生产流程 = LangGraph 状态图；评审循环、并行镜头、人工审片、断点续跑全部由图机制承载 |
| **可离线教学** | Fake Provider 用本地占位素材替代全部 API，零成本跑通全图（教学与单测默认模式） |
| **失败可恢复** | 所有外部调用幂等化 + 检查点持久化 + 任务台账，任何位置中断可续跑 |

### 1.3 总体技术选型结论

| 关注点 | 选型 | 理由 |
|---|---|---|
| 多智能体编排 | **LangGraph ≥1.0**（Python） | 生产级事实标准：状态图、循环、检查点、human-in-the-loop、Send 并行 |
| LLM 接入层 | LangChain + OpenAI 兼容 SDK | DeepSeek/百炼均兼容 OpenAI 协议，一套代码双供应商 |
| 数据契约 | Pydantic v2 + pydantic-settings | 类型安全 + 配置管理一体 |
| 异步 | asyncio + httpx | 视频/图像 API 均为异步任务，天然适配 |
| 数据库 | 默认 SQLite（零配置）；进阶 PostgreSQL 16（Docker Compose 一键） | 教学零门槛 / 生产可并发，两档配置 |
| 本地 CPU 组件 | ultralytics（YOLOv8n-seg）+ rembg + FFmpeg | 继承 FastVideo2 的双人同屏能力 |
| 界面 | Typer CLI + Streamlit Web | CLI 服务教学脚本化，Web 服务可视化 |
| 质量 | pytest + ruff + mypy + GitHub Actions | 每模块独立 CI 自检 |

---

## 2. 系统架构

### 2.1 分层架构

```
┌────────────────────────────────────────────────────────────────┐
│ 接入层    Streamlit Web 面板  │  Typer CLI  │  (扩展期) A2A 网关  │
├────────────────────────────────────────────────────────────────┤
│ 编排层    LangGraph StateGraph                                  │
│           Supervisor(导演) + 7 专职 Agent + Critic 回路          │
│           Checkpointer: SqliteSaver / PostgresSaver             │
├────────────────────────────────────────────────────────────────┤
│ 模块层    m1 剧本 │ m2 资产 │ m3 分镜 │ m4 镜头 │ m5 配音        │
│           m6 口型/双人同屏 │ m7 剪辑合成 │ m8 评审                │
├────────────────────────────────────────────────────────────────┤
│ Provider 层  Text│T2I│I2V│Talk│TTS│VLM  统一异步契约+重试+计费    │
│   实现: DeepSeek / 百炼DashScope / SiliconFlow / 可灵 / edge-tts │
│   Fake: 本地占位素材（离线教学/测试）                              │
├────────────────────────────────────────────────────────────────┤
│ 本地 CPU 组件  YOLOv8n-seg（人物分割） │ rembg（抠图）            │
│               FFmpeg（转场/拼接/混音/字幕） │ librosa（音频）      │
├────────────────────────────────────────────────────────────────┤
│ 基础设施    SQLite/PostgreSQL │ 文件系统 output/<run_id>/ │ .env  │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 一次完整生成的数据流

```
topic ─→ Script(JSON) ─→ AssetRegistry(角色/场景图)
      ─→ Storyboard(每镜头首帧) ─┬→ I2V jobs(动作/风景镜头) → clips/*.mp4
                                 └→ TTS → wavs ─→ S2V jobs(对话镜头: 首帧+音频直出) → talk/*.mp4
      ─→ Editor(规格统一 → xfade 转场 → 字幕 → BGM 混音) ─→ final.mp4
      ─→ Reviewer(VLM 评审记录) + cost.json（成本台账）
```

---

## 3. 运行环境与部署（★ 环境配置专章）

> 对应需求：conda 虚拟环境 + requirements.txt；数据库走 Docker 镜像、docker compose 一键拉起。

### 3.1 环境总览

| 组件 | 要求 | 必需性 |
|---|---|---|
| 操作系统 | Windows 10/11、Linux、macOS | — |
| Miniconda/Anaconda | Python 3.11 环境 | **必需** |
| GPU | **不需要**（推理全 API 化；YOLO/rembg 走 CPU） | — |
| FFmpeg ≥ 6.0 | 转场/合成引擎 | **必需** |
| Docker + Docker Compose | 仅进阶数据库模式需要 | 可选 |
| API KEY | DeepSeek + 阿里云百炼（最小集，见 §10） | **必需**（Fake 模式除外） |

### 3.2 一键环境搭建流程（将固化为 `scripts/setup.sh` / `setup.ps1`）

```bash
# ── 步骤 1：创建 conda 虚拟环境（Python 3.11）─────────────
conda env create -f environment.yaml        # 或手动:
# conda create -n fastvideo3 python=3.11 -y
conda activate fastvideo3

# ── 步骤 2：安装 Python 依赖 ─────────────────────────────
# ⚠ 先装 CPU 版 PyTorch（仅 YOLO 本地推理用），否则会默认拉 2.5GB+ 的 CUDA 版:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                            # 以可编辑模式安装本项目, 获得 fastvideo3 CLI

# ── 步骤 3：安装 FFmpeg ──────────────────────────────────
# Windows:  winget install --id Gyan.FFmpeg        (或 conda install -c conda-forge ffmpeg)
# Linux:    sudo apt install ffmpeg
# macOS:    brew install ffmpeg
ffmpeg -version                             # 自检: 需 >= 6.0

# ── 步骤 4：配置 API KEY ─────────────────────────────────
cp .env.example .env                        # 然后填入真实 KEY (见 §10)

# ── 步骤 5：本地小模型下载（CPU 组件）────────────────────
python -m fastvideo3.utils.fetch_models     # 下载 yolov8n-seg.pt (~6MB) 到 models/

# ── 步骤 6（可选·进阶）：数据库 ──────────────────────────
docker compose up -d                        # 一键拉起 PostgreSQL 16
fastvideo3 doctor                           # 环境自检: 依赖/ffmpeg/KEY/DB 全项体检
```

### 3.3 `environment.yaml`（conda 环境定义）

```yaml
name: fastvideo3
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - pip
  - ffmpeg>=6.0          # conda-forge 渠道, 跨平台一致
  - pip:
      - -r requirements.txt
```

### 3.4 `requirements.txt` 依赖清单（锁定版本于 M1 开工时生成锁文件）

| 分组 | 依赖 | 用途 |
|---|---|---|
| 编排 | `langgraph>=1.0`、`langchain`、`langchain-openai`、`langchain-community` | 多智能体图与 LLM 接入 |
| LLM/SDK | `openai>=1.x`（DeepSeek 及一切 OpenAI 兼容端点）、`dashscope>=1.24` | 文本/图像/视频/口型/语音/视觉 |
| 数据 | `pydantic>=2.7`、`pydantic-settings`、`python-dotenv`、`pyyaml` | 契约与配置 |
| 网络 | `httpx`、`tenacity` | 异步 HTTP 与重试 |
| 语音 | `edge-tts`（免费默认 TTS）、`librosa`、`soundfile` | 对白合成与音频处理 |
| 视觉本地 | `torch`（**仅 CPU 版**，经 cpu index 安装）、`ultralytics`（YOLOv8n-seg）、`rembg[cpu]`、`onnxruntime`、`opencv-python`、`pillow`、`numpy` | 双人分割/抠图/合成（无 GPU 依赖） |
| 视频 | `ffmpeg-python`（薄封装，复杂滤镜仍走原生命令） | 转场/拼接/混音/字幕 |
| 界面 | `streamlit`、`typer`、`rich` | Web 面板与 CLI |
| 数据库 | `SQLAlchemy>=2`、`langgraph-checkpoint-sqlite`（默认）；可选组 `[postgres]`: `psycopg[binary]`、`langgraph-checkpoint-postgres` | 检查点与任务台账 |
| 开发 | `pytest`、`pytest-asyncio`、`pytest-cov`、`ruff`、`mypy`、`respx`（HTTP mock） | 测试与 CI |

### 3.5 数据库设计：两档模式

**为什么需要数据库**：① LangGraph Checkpointer 持久化图状态（断点续跑的根基）；② 异步生成任务台账（I2V/S2V 任务的 job_id、状态、产物路径）；③ 成本流水与评审记录。

**模式 A：`edu-sqlite`（教学默认，零配置）**
- SQLite 文件 `output/fastvideo3.db`，随项目目录走，无需任何安装。
- LangGraph 使用 `SqliteSaver`；任务台账用 SQLAlchemy 同一文件。

**模式 B：`prod-postgres`（进阶/多人并发，Docker 一键）**

`docker-compose.yml` 设计（M1 随脚手架交付）：

```yaml
services:
  postgres:
    image: postgres:16-alpine          # 官方镜像, 拉取即用
    container_name: fv3-postgres
    environment:
      POSTGRES_DB: ${FV3_PG_DB:-fastvideo3}
      POSTGRES_USER: ${FV3_PG_USER:-fv3}
      POSTGRES_PASSWORD: ${FV3_PG_PASSWORD:-fv3_change_me}
    ports: ["5432:5432"]
    volumes: [fv3_pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${FV3_PG_USER:-fv3}"]
      interval: 5s
      retries: 10
  pgadmin:                              # 可选: 数据库可视化(教学演示用)
    image: dpage/pgadmin4:latest
    profiles: ["tools"]                 # docker compose --profile tools up -d 才启动
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@fastvideo3.local
      PGADMIN_DEFAULT_PASSWORD: ${FV3_PGADMIN_PASSWORD:-admin}
    ports: ["5050:80"]
    depends_on: [postgres]
volumes:
  fv3_pgdata:
```

- 一键命令：`docker compose up -d`（拉镜像 + 起服务 + 健康检查）；`docker compose down` 停止；数据卷持久化。
- 应用侧通过配置切换：`config.profile: edu-sqlite | prod-postgres`，Checkpointer 工厂据此选择 `SqliteSaver` 或 `PostgresSaver`，**业务代码零感知**。
- 无 Docker 的学生永远不需要碰这一节——这是教学设计，不是妥协。

### 3.6 环境自检 `fastvideo3 doctor`

逐项检查并输出表格：conda/Python 版本 → 依赖可导入 → `ffmpeg -version` → `.env` KEY 存在性（不打印明文）→ 各 API `health_check`（可选）→ 数据库连通 → `models/yolov8n-seg.pt` 就位。任一失败给出修复指引。

---

## 4. 数据模型设计（Pydantic v2）

### 4.1 核心实体关系

```
ProjectConfig 1─1 Script 1─N Scene 1─N Shot
Script 1─N Character 1─1 Asset(立绘)
Scene 1─1 Asset(场景概念图)
Shot 1─1 StoryboardFrame(首帧) 1─0..1 AudioClip(台词) 1─0..1 ShotClip(视频)
RunJob 记录每个异步 API 任务; ReviewResult 记录每次评审; CostEntry 记录每笔花费
```

### 4.2 关键模型定义（设计签名）

```python
class BaseInfo(BaseModel):
    topic: str
    style: str
    culture_background: str
    target_duration_sec: int = 180
    episode: int = 1
    aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"  # 竖屏短剧为默认形态


class ShotType(str, Enum):
    DIALOGUE = "dialogue"  # 对话镜头 → TTS + S2V 直出(默认模式)
    ACTION = "action"  # 动作镜头 → 仅 I2V
    SCENERY = "scenery"  # 风景转场镜头 → I2V, 作转场素材


class TransitionType(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE = "fade"
    SCENERY = "scenery"


class DialogueLine(BaseModel):
    character_id: str
    text: str
    emotion: str = "neutral"


class Shot(BaseModel):
    shot_id: str
    scene_id: str
    type: ShotType
    duration_sec: float = 5.0  # 镜头时长(I2V/S2V 定长输出)
    characters: list[str]  # character_id 列表
    action: str  # 动作/画面描述
    camera: str = ""  # 镜头语言(推/拉/摇/移/特写)
    dialogues: list[DialogueLine] = []
    transition_out: TransitionType = TransitionType.CUT
    visual_prompt: str = ""  # 供 T2I/I2V 的最终提示词


class Scene(BaseModel):
    scene_id: str
    title: str
    location: str
    time_of_day: str
    summary: str
    shots: list[Shot]


class Character(BaseModel):
    character_id: str
    name: str
    appearance: str  # 外貌固化描述
    voice_id: str = ""  # TTS 音色


class Script(BaseModel):
    base_info: BaseInfo
    characters: list[Character]
    scenes: list[Scene]
    version: int = 1


class ProjectConfig(BaseModel):
    """一次生成任务的全部运行配置（写入 run_meta.json，续跑时原样恢复）"""

    run_id: str
    base_info: BaseInfo
    profile: Literal["edu-sqlite", "prod-postgres"] = "edu-sqlite"
    review_mode: Literal["gated", "auto"] = "gated"
    talk_mode: Literal["s2v_direct", "v2v_overlay", "audio_overlay"] = "s2v_direct"
    budget_limit_cny: float = 150.0
    max_parallel_jobs: int = 4


class Asset(BaseModel):
    asset_id: str
    kind: Literal["character", "scene"]
    owner_id: str
    uri: str
    meta: dict = {}


class RunJob(BaseModel):
    job_id: str
    capability: str
    provider: str
    params: dict
    status: Literal["pending", "running", "succeeded", "failed"]
    output_uri: str | None
    error: str | None
    submitted_at: datetime
    finished_at: datetime | None


class ReviewResult(BaseModel):
    target: str
    passed: bool
    score: float
    feedback: str
    rubric: str


class CostEntry(BaseModel):
    ts: datetime
    capability: str
    provider: str
    model: str
    amount_cny: float
    note: str = ""
```

### 4.3 输出目录契约（`output/<run_id>/`）

```
output/<run_id>/
├── script.json            # 结构化剧本(含迭代版本)
├── assets/                # 角色立绘/场景概念图 + registry.json
├── storyboard/            # shot_<id>.png 首帧
├── audio/                 # line_<id>.wav 对白
├── clips/                 # shot_<id>.mp4 I2V 产物
├── talk/                  # shot_<id>_talk.mp4 口型产物
├── review/                # 评审记录 JSON + 截图
├── final/                 # final.mp4 + srt 字幕
├── cost.json              # 成本台账
└── run_meta.json          # 配置快照、进度、断点信息
```

---

## 5. 多智能体设计（LangGraph）

### 5.1 Agent 角色清单

| Agent | 职责 | 主要能力依赖 | 关键提示词要点 |
|---|---|---|---|
| **director（导演/Supervisor）** | 总调度：解析输入、推进阶段、分配重试预算、熔断决策 | LLM | 只调度不生产；输出结构化调度指令 |
| **screenwriter（编剧）** | 主题→结构化剧本；按评审意见迭代修订 | LLM（JSON 输出） | 三幕结构；每场景含对话/动作/转场；台词口语化 |
| **art_director（美术）** | 角色立绘、场景概念图的提示词工程与生成 | T2I | 外貌描述固化（同一角色逐字复用）；风格锚定 |
| **storyboarder（分镜师）** | 每镜头首帧生成（含参考图注入） | T2I + 资产图 | 构图/景别与 camera 字段一致 |
| **shot_worker（镜头生成，×N 并行）** | **动作/风景镜头**的 I2V 提交与轮询（对话镜头仅在 v2v_overlay/audio_overlay 模式进入） | I2V | 提示词 = action + camera + 风格后缀 |
| **sound_engineer（配音）** | 对白 TTS、音色管理、时长对齐 | TTS | 情感标签注入 |
| **talk_worker（口型，×N 并行）** | 对话镜头：默认模式 S2V 直出（首帧+音频 → 说话视频，不依赖 I2V）；双人对话每角色各生成一条说话视频后合成同屏 | Talk + YOLO/rembg | — |
| **editor（剪辑）** | 转场策略执行、字幕、BGM、混音、成片 | FFmpeg | 转场类型来自剧本 `transition_out` |
| **critic（评审）** | 剧本评审、分镜一致性抽查、镜头抽查、成片终审 | LLM + VLM | 统一 Rubric 量表；输出结构化评审结论 |

### 5.2 状态定义

```python
class ProductionState(TypedDict):
    project: ProjectConfig
    phase: str  # 当前阶段
    script: Script | None
    assets: dict[str, Asset]  # asset_id -> Asset
    storyboard: dict[str, str]  # shot_id -> 首帧路径
    jobs: Annotated[dict[str, RunJob], merge_jobs]  # 异步任务台账
    clips: dict[str, str]  # shot_id -> 片段路径
    audio: dict[str, str]  # line_id -> wav 路径
    talks: dict[str, str]  # shot_id -> 口型片段路径
    reviews: list[ReviewResult]
    final_video: str | None
    cost: list[CostEntry]
    retry_budget: dict[str, int]  # target -> 剩余重试次数
    messages: Annotated[list, add_messages]  # Agent 间消息轨迹
```

### 5.3 图拓扑（含循环、并行、人工介入）

```mermaid
graph TD
    START --> intake[intake 解析输入/校验KEY/建run目录]
    intake --> screenwriter[编剧 Agent]
    screenwriter --> critic_script{critic 剧本评审}
    critic_script -- "驳回且预算>0" --> screenwriter
    critic_script -- 通过 --> gate_script["人工审片 interrupt()<br/>(可配置跳过)"]
    gate_script --> art[美术 Agent 资产库]
    art --> storyboard[分镜师 Agent]
    storyboard --> critic_board{critic VLM 分镜抽查}
    critic_board -- 驳回个别镜头 --> storyboard
    critic_board -- 通过 --> fanout
    fanout{{Send 并行分发}} --> sound[TTS 对白 xN]
    fanout --> shots[I2V 动作/风景镜头 xN]
    sound --> talk[S2V 对话镜头直出 xN<br/>含双人同屏合成]
    shots --> collect[收集与规格统一]
    talk --> collect
    collect --> critic_shots{critic 镜头抽查}
    critic_shots -- "动作/风景镜头重试" --> shots
    critic_shots -- "对话镜头重试" --> talk
    critic_shots -- 通过 --> editor[剪辑 Agent 转场/合成]
    editor --> critic_final{critic 成片终审}
    critic_final -- 驳回定位到环节 --> fanout
    critic_final -- 通过 --> gate_final["人工看片 interrupt()<br/>(可配置跳过)"]
    gate_final --> END
```

### 5.4 关键机制

| 机制 | 实现 |
|---|---|
| 并行镜头生成 | `Send("shot_node", shot_ctx)` map-reduce 模式；并发上限由配置 `max_parallel_jobs` 约束（防限流），**且不得超过供应商账号的异步任务并发配额**（百炼新账号视频任务并发常仅 2–5，按供应商单独配置） |
| 检查点续跑 | `SqliteSaver`（默认）/ `PostgresSaver`（进阶）；`run_id` 即 `thread_id`；`fastvideo3 resume <run_id>` 恢复 |
| 人工介入 | `interrupt()` 于两处审片门；配置 `review_mode: auto \| gated`，教学全自动演示用 `auto` |
| 评审循环 | 条件边读 `ReviewResult.passed` 与 `retry_budget`；预算耗尽 → 降级放行并记录（保证流程不卡死） |
| 成本熔断 | 每笔 `CostEntry` 后检查 `budget_limit_cny`，超限进 `HALT` 态等待人工决策 |
| 可观测 | 每节点进出写事件流（时间/耗时/花费/输入输出摘要）→ Streamlit 时间线与 `run_meta.json` |

---

## 6. 模块详细设计

> 每个模块：单一职责、独立可运行（`fastvideo3 stage <name>`）、接口先行、Fake 可替换。接口签名为设计稿，实现阶段细化。

### 6.0 基础层 `core/`

| 模块 | 职责 | 关键接口 |
|---|---|---|
| `core.config` | 配置加载：`.env` + `configs/*.yaml`，provider 路由 | `get_settings() -> Settings` |
| `core.schema` | §4 全部 Pydantic 模型 | — |
| `core.storage` | run 目录管理、产物路径契约 | `RunStorage(run_id).path(stage, name)` |
| `core.cost` | 成本台账：记账、汇总、熔断检查 | `ledger.record(entry)`、`ledger.exceeded()` |
| `core.jobs` | 异步任务台账（RunJob 持久化与查询） | `jobs.submit/get/update` |

### 6.1 Provider 层 `providers/`

```python
class Provider(ABC):
    capability: Capability        # TEXT/T2I/I2V/TALK/TTS/VLM
    name: str; model: str
    async def health_check(self) -> bool

class TextProvider(Provider):     # 剧本/提示词/评审意见
    async def chat(self, messages, *, json_schema: type[BaseModel] | None) -> str
class T2IProvider(Provider):      # 立绘/场景/首帧
    async def generate(self, prompt, *, size, ref_image: Path | None) -> ImageResult
class I2VProvider(Provider):      # 异步任务模式
    async def submit(self, image: Path, prompt: str, **kw) -> str        # job_id
    async def poll(self, job_id: str) -> JobStatus                       # 含产物下载
class TalkProvider(Provider):     # 音频驱动说话视频 (S2V)
    async def submit(self, visual: Path, audio: Path) -> str
    async def poll(self, job_id: str) -> JobStatus
class TTSProvider(Provider):
    async def synthesize(self, text, *, voice, emotion) -> Path          # wav
class VLMProvider(Provider):      # 视觉评审
    async def review(self, media: list[Path], *, rubric: str) -> ReviewResult
```

统一横切能力（`providers/base.py` 实现，各子类零重复）：
- **重试**：tenacity，3 次 + 指数退避 + 抖动；限流码（429）特殊退避
- **计费**：每次成功调用按 `pricing.yaml` 单价写入成本台账
- **降级**：`fallback_provider` 配置，主供应商连续失败自动切换
- **能力声明**：Provider 自报能力集（如 `supports_reference_image`、`supports_video_input`、`max_duration`），模块层按能力路由而非写死供应商——这是教学上"面向接口编程"的直观案例

### 6.2 业务模块层 `modules/`（开发顺序 = CI 顺序）

| # | 模块 | 输入 → 输出 | 依赖 | 失败策略 |
|---|---|---|---|---|
| m1 | `m1_scripting` | BaseInfo → Script(JSON)；已有剧本走**解析模式** | Text | Schema 校验失败/评审驳回 → 重写（≤3 轮）；**审核拒绝 → 改写措辞重试（≤2 次）**；台词句长校验（≤30 字/句） |
| m2 | `m2_assets` | Script → AssetRegistry | T2I | 单资产重试 ≤2，失败阻断（资产是后续锚点） |
| m3 | `m3_storyboard` | Script+Assets → 首帧集 | T2I | 单镜头重试 ≤2，超限标记人工 |
| m4 | `m4_shots` | 首帧+提示词 → 片段（**仅动作/风景镜头；对话镜头在 v2v_overlay/audio_overlay 模式才进入**） | I2V | 并行提交；轮询超时 30min；失败重试 ≤2 |
| m5 | `m5_audio` | 台词 → wavs + 时长对齐表 | TTS | 重试 ≤2；**时长对齐：音频超镜头时长先 ≤1.2× 提速，仍超则交 m7 补静帧** |
| m6 | `m6_talk` | **默认模式（s2v_direct）**：对话镜头首帧+音频 → 说话片段（S2V 直出，不依赖 m4 产物）；双人对话每角色各生成一条说话视频后合成同屏 | Talk + YOLO/rembg | S2V 失败 → 降级 audio_overlay（I2V+音频直叠）保交付 |
| m7 | `m7_edit` | 全部片段+音轨 → final.mp4 | FFmpeg | **先做规格统一：全部片段归一化分辨率/帧率/像素格式（I2V 与 S2V 来源混杂）**；转场参数错误回退硬切；合成失败阻断；SRT 字幕时间轴由 m5 时长对齐表导出 |
| m8 | `m8_review` | 各阶段产物 → ReviewResult | LLM+VLM | 评审自身失败 → 放行并告警（评审不是硬门槛） |

**双人同屏合成流程（m6 内，纯 CPU）**：
```
talkA.mp4 ─┐                     ┌→ 画中画/左右布局合成(FFmpeg)
           ├→ YOLOv8n-seg 人物框  │   + 双音轨混音
talkB.mp4 ─┘  + rembg 抠图前景  ──┘
```

### 6.3 编排层 `graph/`

| 文件 | 职责 |
|---|---|
| `state.py` | ProductionState 定义与 reducer |
| `nodes.py` | 每个图节点 = 对模块层的一次编排调用（含 Send 并行节点） |
| `edges.py` | 条件边：评审路由、预算检查、熔断路由 |
| `builder.py` | `build_graph(profile) -> CompiledGraph`，注入 Checkpointer 与配置 |

### 6.4 接入层 `app/`

- `app/cli.py`（Typer）：`run / stage / resume / doctor / cost / ls-runs`
- `app/streamlit_app.py`：新建任务（主题表单）→ 进度时间线 → Agent 事件流 → 产物画廊（图片网格/视频播放）→ 成本看板 → 审片交互（通过/驳回+意见）

---

## 7. API 供应商选型与契约

### 7.1 能力 × 供应商矩阵（2026-08 调研结论）

| 能力 | **默认供应商/模型** | 备选 | 调用模式 |
|---|---|---|---|
| 文本/编剧 | **DeepSeek** `deepseek-chat` | 百炼 `qwen-plus`（单 KEY 极简模式） | OpenAI 兼容，同步 |
| VLM 评审 | **百炼** `qwen-vl-max` | `qwen-vl-plus`（低价档） | OpenAI 兼容，同步 |
| T2I | **百炼** `wan2.6-t2i`（1024²，最大 1440²） | SiliconFlow `FLUX.1-dev`、百炼 `wan2.5-i2i`（参考图编辑） | 异步任务 |
| I2V | **百炼** `wan2.2-i2v`（480P 教学默认/720P） | `wan2.7-i2v`（高质量）、SiliconFlow Wan2.2、可灵 I2V | 异步任务（`X-DashScope-Async`） |
| 口型/Talk | **百炼** `wan2.2-s2v`（图+音频→说话视频，替代本地 MultiTalk）；更先进档 `videoretalk`（视频+音频→口型替换，v2v_overlay 模式） | 可灵 Lip Sync、HeyGen Avatar API、InfiniteTalk | 异步任务；输入需临时 OSS URL（`utils/dashscope_upload.py`） |
| TTS | **edge-tts**（免费、无需 KEY，教学默认） | 百炼 `cosyvoice-v2`、MiniMax | 同步 |
| BGM | 本地免版权曲库 `assets/bgm/`（零成本） | 扩展期接音乐生成 API | 本地 |

> 说明：① **edge-tts 免费但依赖微软在线服务**，存在限流与可用性波动，正式课堂演示建议切 `cosyvoice`；② wan2.2-s2v 为**单图+音频**生成说话视频，故默认模式下对话镜头无需等待 I2V——关键路径更短、且省下对话镜头的 I2V 费用。

> **最小 KEY 集**：`DEEPSEEK_API_KEY` + `DASHSCOPE_API_KEY` 两把即可跑通全功能。
> **极简模式**：LLM 也切到百炼 `qwen`，则**仅需一把** `DASHSCOPE_API_KEY`。
> **MultiTalk 说明**：MeiGen MultiTalk 无公开云 API，FastVideo3 将其职责（多人说话视频）由 `TalkProvider` 承接，默认走百炼 wan2.2-s2v；本地 MultiTalk 部署列为扩展期课题（需 GPU）。

### 7.2 异步任务统一契约

```
submit(params) ──→ job_id
poll(job_id)   ──→ {status: pending|running|succeeded|failed, progress, output_url?}
                   succeeded → 下载产物至 output/<run_id>/<stage>/
轮询策略: 前 2min 每 5s, 之后每 15s, 总超时 30min, 超时判 failed 进重试
```

### 7.3 成本估算（3 分钟短剧 ≈ 36 镜头，默认配置，实现阶段以 `pricing.yaml` 校准）

假设对话镜头 ≈16、动作/风景镜头 ≈20（占比由编剧 prompt 约束，可配置）：

| 项 | 数量 | 估算 |
|---|---|---|
| LLM 剧本+提示词 | ~3 万 tokens | ≈ ¥0.5 |
| T2I（立绘+场景+分镜） | ~50 张 | ≈ ¥5–15 |
| I2V 480P（动作/风景；降级模式含对话镜头） | ~20 段 | ≈ ¥12–25 |
| S2V（对话镜头，默认模式，按时长计费） | ~16 段 × 5s | ≈ ¥30–80，**主要变量** |
| TTS | 全片 | ¥0（edge-tts） |
| **合计** | | **默认模式（S2V）≤ ¥150/片；audio_overlay 降级模式 ≤ ¥50/片** |

> S2V 单价显著高于 I2V 且定价变动快。**P1 完成 Provider 实现前，必须先实测校准 `pricing.yaml` 并做成本模拟**，预算熔断阈值据此设定；这也是教学点之一（"先算账，再开工"）。

---

## 8. 本地 CPU 组件（继承自 FastVideo2）

| 组件 | 模型/工具 | 用途 | 性能要求 |
|---|---|---|---|
| `utils/yolo.py` | `models/yolov8n-seg.pt`（~6MB） | 说话视频中人物检测/分割框 | CPU 单帧 < 200ms |
| `utils/matting.py` | rembg（onnxruntime CPU） | 人物前景抠图 | 单帧 < 1s |
| `utils/ffmpeg.py` | FFmpeg ≥6.0 | `xfade` 转场（dissolve/fade/wipe）、`concat`、`amix` 混音、`subtitles` 烧录、响度归一化（loudnorm） | — |
| 音频 | librosa/soundfile | 时长对齐、静音填充 | — |

> 双人同屏合成总预算：2 主体 × 约 80 帧（5 秒 @16fps），CPU（8 核基准）**≤ 3 分钟/镜头**，与 PRD F5 验收一致；超限时的优化手段：降分辨率抠图、帧间 mask 复用。

转场映射表（剧本 `TransitionType` → FFmpeg）：

| TransitionType | FFmpeg 实现 | 时长 |
|---|---|---|
| CUT | concat 直连 | 0 |
| DISSOLVE | `xfade=transition=dissolve` | 0.5s |
| FADE | `xfade=transition=fade`（首尾另加 `fade` 滤镜） | 0.8s |
| SCENERY | scenery 镜头本身插入两镜之间 + 前后各 0.5s dissolve | — |

---

## 9. 目录结构

```
D:\FastVideo3\
├── PRD.md / TDD.md / README.md
├── pyproject.toml                 # 打包与工具配置
├── requirements.txt / environment.yaml
├── docker-compose.yml             # PostgreSQL(进阶可选)
├── .env.example / .gitignore
├── Makefile                       # setup/lint/test/run 快捷目标（Linux/macOS；Windows 用等价的 scripts/*.ps1）
├── scripts/                       # setup.sh / setup.ps1 一键环境
├── configs/
│   ├── default.yaml               # provider 路由 + 并发/预算/评审门控
│   └── ablations/                 # 教学消融配置(无资产锚定/换供应商...)
├── prompts/                       # 提示词模板(版本化: scriptwriter.v1.md ...)
├── pricing.yaml                   # 各模型单价(成本台账依据)
├── fastvideo3/
│   ├── core/                      # config/schema/storage/cost/jobs
│   ├── providers/                 # base/deepseek/dashscope/siliconflow/kling/edge_tts/fake/factory
│   ├── modules/                   # m1_scripting ... m8_review
│   ├── graph/                     # state/nodes/edges/builder
│   ├── app/                       # cli.py / streamlit_app.py
│   └── utils/                     # ffmpeg/yolo/matting/audio/fetch_models
├── tests/                         # 镜像结构; providers/fakes.py 全局复用
├── assets/bgm/                    # 免版权音乐
├── models/                        # yolov8n-seg.pt(setup 下载)
└── output/                        # 运行产物 <run_id>/ + fastvideo3.db
```

---

## 10. 配置与密钥管理

`.env.example`（设计稿）：

```bash
# ── 最小 KEY 集(两把跑全功能) ──
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DASHSCOPE_API_KEY=sk-xxxx
# ── 可选供应商 ──
SILICONFLOW_API_KEY=
KLING_ACCESS_KEY=
KLING_SECRET_KEY=
# ── 运行配置 ──
FV3_PROFILE=edu-sqlite          # edu-sqlite | prod-postgres
FV3_REVIEW_MODE=gated           # gated(人工审片) | auto(全自动)
FV3_BUDGET_LIMIT_CNY=50
# ── 进阶数据库(prod-postgres 时生效) ──
FV3_PG_HOST=localhost
FV3_PG_PORT=5432
FV3_PG_DB=fastvideo3
FV3_PG_USER=fv3
FV3_PG_PASSWORD=fv3_change_me
```

`configs/default.yaml` 的 provider 路由示例：

```yaml
providers:
  text:  {default: deepseek, fallback: dashscope_qwen}
  t2i:   {default: dashscope_wan26, fallback: siliconflow_flux}
  i2v:   {default: dashscope_wan22_i2v, fallback: siliconflow_wan22}
  talk:  {default: dashscope_wan22_s2v, fallback: kling_lipsync}
  tts:   {default: edge_tts, fallback: dashscope_cosyvoice}
  vlm:   {default: dashscope_qwen_vl_max}
generation:
  aspect_ratio: "9:16"          # 竖屏短剧默认
  shot_duration_sec: 5
  talk_mode: s2v_direct         # s2v_direct | v2v_overlay | audio_overlay
limits:
  max_parallel_jobs: 4          # 全局上限; 与供应商配额取小
  per_provider_concurrency: {dashscope: 3, kling: 2}   # 按账号实际配额调整
  i2v_poll_timeout_min: 30
```

**画质档位（2026-08-30 增补）**：`--quality high` / `FV3_QUALITY=high` 触发
`configs/profile_<quality>.yaml` overlay 深度合并进 default.yaml（`_deep_merge`）。
high 档：`wanx2.1-t2i-plus` + `wan2.2-i2v-plus@1080P`、成片 1080×1920、CRF20、
全局 `style_suffix` 风格锚（中英双语真人电影锚）+ `negative_prompt`（禁卡通/3D）、
场景+角色稳定种子（m3 `_stable_seed`）固脸、cover 裁切防黑边（normalize_clip）、
eq+unsharp 调色、程序化 BGM（`scripts/make_bgm.py` → `assets/bgm/`）、
-16 LUFS 响度归一（safe_encode 内 loudnorm + `-ar 44100` 拉回采样率）、
短剧样式烧录字幕（libass 脚本单位、Fontname 无空格约束）。续跑经 run_meta.quality 恢复档位。

**cinema 档（商业级，2026-08-30）**：`--quality cinema` → `configs/profile_cinema.yaml`：
`wan2.7-i2v`（media 数组新协议，`driving_audio` 原生口型，绕开 s2v 账号门控）+
`wan2.7-r2v`（参考生视频：角色参考图像素级一致，m4 角色动作镜头自动路由，prompt 用
"图1/图2"指代）+ `candidates_per_shot=2` 的 best-of-N（VLM 抽帧打分取优，商业"抽卡"工程化）。
实测 90 秒成片 ¥113.35（预算用 --budget 300）。Provider：`DashScopeWan27Base` 系三件套，
本地素材统一走 `utils/dashscope_upload.py` 临时 OSS。

---

## 11. 可靠性与成本控制

| 场景 | 策略 |
|---|---|
| API 瞬时失败/限流 | tenacity 3 次指数退避；429 专门长退避；连续失败切 `fallback_provider` |
| 异步任务卡死 | 轮询超时 30min 判失败 → 重试（预算内） |
| 进程中断/断网 | Checkpointer 恢复图状态；`core.jobs` 台账识别"未完成 job"，续跑时**先查旧 job 状态再决定重提交**（不重复花钱） |
| 镜头级失败 | 隔离重试 ≤2；仍失败 → 人工处理队列（不阻塞其他镜头） |
| 成本失控 | `pricing.yaml` 实时记账；`FV3_BUDGET_LIMIT_CNY` 熔断；成本看板预警 80% |
| 口型高成本 | `talk_mode` 三档配置：默认 `s2v_direct` 且由编剧控制对话镜头占比；一键降级 `audio_overlay` 全片（对话镜头回退 I2V+音频直叠）；成本看板 80% 预警 |
| 内容审核拒绝 | 改写措辞重试（m1 剧本）/ 调整提示词重试（m2–m4），各 ≤2 次，仍失败转人工；错误码识别供应商审核类响应，不计入普通失败重试 |

---

## 12. 测试与 CI 自检设计（对应构建流程第 4 步）

### 12.1 测试分层

| 层 | 内容 | 运行条件 |
|---|---|---|
| 单元测试 | 每模块逻辑 + Schema 校验 + 转场参数拼装等 | 纯离线，Fake Provider |
| Provider 契约测试 | 每个 Provider 实现同一组契约用例（respx mock HTTP） | 离线 |
| API 冒烟（`@pytest.mark.api`） | 真实 KEY 最小调用（1 张图/1 段视频） | CI nightly + secrets |
| 图级冒烟 | Fake 模式跑完整图（主题→成片，占位素材） | 每次 PR |
| 环境测试 | `doctor` 各检查项 | 本地 |

### 12.2 每模块 Definition of Done（CI 绿灯标准）

1. `ruff check` 0 告警；`mypy fastvideo3` 通过
2. 单测覆盖率 ≥ 80%（该模块文件）；Fake 模式下该模块独立可运行
3. `fastvideo3 stage <module>` 用 Fake 跑通并产出约定目录结构的产物
4. 错误路径（供应商失败/超时/预算超限）有用例覆盖
5. README 段落更新：本模块的教学说明（输入/输出/可消融变量）

### 12.3 GitHub Actions 流水线（`.github/workflows/ci.yml` 设计）

```
lint      → ruff check + ruff format --check
type      → mypy fastvideo3
unit      → pytest -m "not api" --cov ≥ 80%
smoke     → Fake 全图冒烟(生成占位素材走完整流水线, 验证图拓扑与 FFmpeg)
nightly   → pytest -m api (真实 KEY, secrets 注入, 最小用例, 成本 < ¥5)
```

---

## 13. 开发计划与系统联调（对应构建流程第 3/5 步）

### 13.1 模块开发顺序（每步：实现 → CI 自检绿 → 下一步）

| 阶段 | 交付 | 退出标准 |
|---|---|---|
| P0 脚手架 | pyproject/environment.yaml/requirements/docker-compose/config/.env.example/目录骨架 + `doctor` | `fastvideo3 doctor` 全绿 |
| P1 基础与 Provider | core 全部 + providers（含 Fake）+ pricing | Provider 契约测试全绿；**实测校准 `pricing.yaml`（各模型当前单价 + 单片成本模拟）** |
| P2 业务模块 | m1 → m2 → m3 → m4 → m5 → m6 → m7 → m8 逐个 | 每个模块独立 DoD 达标 |
| P3 编排 | graph/（state/nodes/edges/builder）+ 检查点 + 审片门 | Fake 全图冒烟通过 |
| P4 接入 | CLI 全命令 + Streamlit 面板 | 两入口均可驱动全流程 |
| P5 联调 | 见 13.2 | 验收清单全过 |

### 13.2 系统联调用例（IT-1 ~ IT-5）

| 用例 | 内容 | 通过标准 |
|---|---|---|
| IT-1 | **Fake 全链路**：无 KEY 主题→成片 | 占位素材成片产出，图事件流完整，耗时 < 3min |
| IT-2 | **真实 API 冒烟**：主题"校园喜剧"→30 秒微短剧（6 镜头） | 成片可播放；成本 < ¥15；含 ≥1 对话镜头口型 |
| IT-3 | **全量短剧**：3 分钟/36 镜头，含双人同屏场景 | 成片时长偏差 ≤15%；转场 ≥3 种；双人同屏正常；成本 ≤ ¥150（默认口型模式）；另跑一遍 `audio_overlay` 模式 ≤ ¥50 |
| IT-4 | **断点续跑演练**：I2V 等待期间杀进程 → `resume` | 无重复提交已成功任务；成本无重复计账 |
| IT-5 | **教学验收**：按新生文档搭环境 + 跑 IT-1 | 30 分钟内完成；完成一次供应商切换对比 |

### 13.3 联调验收总清单

- [ ] `fastvideo3 doctor` 全绿
- [ ] IT-1 ~ IT-5 全部通过
- [ ] 成本台账与供应商账单抽样核对一致
- [ ] 全流程日志/事件流可回放；无 KEY 明文泄漏（grep 扫描）
- [ ] README/教学文档与最终行为一致

---

## 14. 附录

### 14.1 关键 API 文档索引（2026-08 调研）

| 供应商 | 能力 | 文档 |
|---|---|---|
| 百炼 | 文生图 wan2.6-t2i | [万相-文生图V2版API参考](https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference) |
| 百炼 | 图生视频 wan2.2/2.x-i2v | [万相图生视频-首帧API参考（2.1-2.6）](https://help.aliyun.com/zh/model-studio/legacy-image-to-video-api-reference/) |
| 百炼 | 图生视频 wan2.7-i2v | [万相2.7-图生视频API参考](https://help.aliyun.com/zh/model-studio/image-to-video-general-api-reference) |
| 百炼 | 数字人/口型 wan2.2-s2v | [数字人wan2.2-s2v视频生成API参考](https://help.aliyun.com/zh/model-studio/wan-s2v-api) |
| 百炼 | 口型替换 videoretalk | [VideoRetalk API参考](https://help.aliyun.com/zh/model-studio/videoretalk-api) |
| 百炼 | 临时文件上传（oss:// URL） | [上传本地文件获取临时URL](https://help.aliyun.com/zh/model-studio/get-temporary-file-url) |
| 百炼 | 视觉理解 qwen-vl | [DashScope API 参考](https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope) |
| 百炼 | 模型总览 | [选择模型](https://help.aliyun.com/zh/model-studio/models) |
| 可灵 | 口型/视频 API | [Kling AI API](https://kling.ai/dev) |
| HeyGen | Avatar IV API | [HeyGen Developers](https://developers.heygen.com/) |
| LangGraph | 编排框架 | [LangGraph](https://www.langchain.com/langgraph) |

### 14.2 教学映射（模块 ↔ 课程章节）

| 课程章节 | 对应模块 | 核心知识点 |
|---|---|---|
| 第 1 讲 结构化输出 | m1 + core.schema | LLM JSON 约束、Pydantic 校验、剧本迭代 |
| 第 2 讲 角色一致性 | m2 + m3 | 资产锚定、参考图注入、消融实验设计 |
| 第 3 讲 视频扩散应用 | m4 | I2V 异步任务、提示词工程、并发与限流 |
| 第 4 讲 语音与口型 | m5 + m6 | TTS 选型、S2V 原理、本地视觉处理管线 |
| 第 5 讲 剪辑合成 | m7 | FFmpeg 滤镜图、转场设计、混音 |
| 第 6 讲 多智能体编排 | graph/ | LangGraph 状态图、Supervisor/Critic、检查点、HITL |
| 第 7 讲（进阶）互操作 | providers 封装 | A2A Agent Card 封装、MCP 工具接入 |

### 14.3 与 FastVideo2 的继承关系

| FastVideo2 | FastVideo3 |
|---|---|
| main.py 单体 | core/modules/graph/app 分层 |
| 本地 FLUX + Wan + MultiTalk | 全 API（百炼/DeepSeek），MultiTalk → wan2.2-s2v |
| 插入空镜+硬拼接 | 转场引擎（xfade 四种）+ 剧本驱动 |
| 无一致性机制 | 资产库 + 参考图 + VLM 评审 |
| 无断点、无成本控制 | Checkpointer + 成本台账 + 熔断 |
| edge-tts | 保留为默认免费 TTS ✅ |
| YOLOv8-seg + rembg 双人同屏 | **原样保留（本地 CPU）** ✅ |
