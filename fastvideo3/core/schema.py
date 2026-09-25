"""数据契约：全链路结构化模型（TDD §4）。

Schema-first 设计：LLM 输出、产物注册表、任务台账全部经由这些模型校验。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ───────────────────────── 剧本侧 ─────────────────────────
class BaseInfo(BaseModel):
    """一次创作的基础输入。"""

    topic: str
    style: str = "都市轻喜剧"
    culture_background: str = "当代中国"
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
    text: str = Field(..., max_length=60, description="单句台词，建议 ≤30 个汉字")
    emotion: str = "neutral"


class Shot(BaseModel):
    shot_id: str
    scene_id: str
    type: ShotType
    duration_sec: float = 5.0  # 镜头时长(I2V/S2V 定长输出)
    characters: list[str] = Field(default_factory=list)  # character_id 列表
    action: str  # 动作/画面描述
    camera: str = ""  # 镜头语言(推/拉/摇/移/特写)
    dialogues: list[DialogueLine] = Field(default_factory=list)
    transition_out: TransitionType = TransitionType.CUT
    visual_prompt: str = ""  # 供 T2I/I2V 的最终提示词


class Scene(BaseModel):
    scene_id: str
    title: str
    location: str
    time_of_day: str = "日"
    summary: str = ""
    shots: list[Shot] = Field(default_factory=list)


class Character(BaseModel):
    character_id: str
    name: str
    appearance: str  # 外貌固化描述(全片逐字复用)
    voice_id: str = ""  # TTS 音色


class Script(BaseModel):
    base_info: BaseInfo
    title: str = ""  # 片名（编剧输出；缺省回退为主题）
    characters: list[Character] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    version: int = 1

    # ── 便捷遍历 ──
    def all_shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]

    def character_map(self) -> dict[str, Character]:
        return {c.character_id: c for c in self.characters}

    def shot_count(self) -> int:
        return len(self.all_shots())


class ProjectConfig(BaseModel):
    """一次生成任务的全部运行配置（写入 run_meta.json，续跑时原样恢复）。"""

    run_id: str
    base_info: BaseInfo
    profile: Literal["edu-sqlite", "prod-postgres"] = "edu-sqlite"
    review_mode: Literal["gated", "auto"] = "auto"
    talk_mode: Literal["s2v_direct", "v2v_overlay", "audio_overlay"] = "s2v_direct"
    budget_limit_cny: float = 150.0
    max_parallel_jobs: int = 4
    fake: bool = False
    quality: str = "economy"  # economy | high（续跑时按原档位恢复模型栈）


# ───────────────────────── 资产与产物侧 ─────────────────────────
class Asset(BaseModel):
    asset_id: str
    kind: Literal["character", "scene"]
    owner_id: str  # character_id 或 scene_id
    uri: str
    meta: dict = Field(default_factory=dict)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunJob(BaseModel):
    """异步 API 任务台账条目（持久化于数据库，续跑时避免重复提交）。"""

    job_id: str
    run_id: str = ""
    capability: str = ""
    provider: str = ""
    params: dict = Field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    output_uri: str | None = None
    error: str | None = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class ReviewResult(BaseModel):
    target: str = ""  # 被评审对象标识
    passed: bool = True
    score: float = 0.9
    feedback: str = ""
    rubric: str = ""


class CostEntry(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capability: str = ""
    provider: str = ""
    model: str = ""
    amount_cny: float = 0.0
    note: str = ""


class AudioLine(BaseModel):
    """m5 产物：一句对白的音频与时长对齐信息。"""

    line_id: str
    shot_id: str
    character_id: str
    text: str
    wav_uri: str
    duration_sec: float = 0.0
    padded: bool = False  # 是否需要 m7 补静帧
