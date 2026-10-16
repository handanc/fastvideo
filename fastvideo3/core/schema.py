"""数据契约：全链路结构化模型（TDD §4）。

Schema-first 设计：LLM 输出、产物注册表、任务台账全部经由这些模型校验。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


class CharacterLook(BaseModel):
    """结构化外貌卡：把"外貌"从单一自由文本升级为可校验的属性位。

    背景：appearance 是 LLM 写的自由文本，经 m3 蒸馏/改写后"黄色头盔"这类关键
    属性会蒸发（实测主角头盔出现过 4 种颜色）。属性位供 P1-3 蒸馏校验与 P1-6
    一致性度量共用；老 script.json 无该字段照常反序列化（续跑安全）。
    """

    face: str = ""  # 脸型/年龄/肤色
    hair: str = ""  # 发型+发色
    eyes: str = ""
    build: str = ""
    outfit: str = ""  # 服装主色+款式
    headwear: str = ""  # 头部配件（头盔/帽子/发饰）——实测最易漂移项
    palette: list[str] = Field(default_factory=list)  # 主色板（供 Tier-B 色相校验）
    distinctive: list[str] = Field(default_factory=list)  # 关键辨识特征


class Character(BaseModel):
    character_id: str
    name: str
    appearance: str  # 外貌固化描述(全片逐字复用)
    voice_id: str = ""  # TTS 音色（留空则由 m5 按性别/年龄确定性分配）
    # 性别/年龄段：音色的**唯一**语义依据。此前只有 voice_id 一个自由文本位，
    # 实测 LLM 全都不填 → m5 退化成 md5 哈希抽签 → 唐僧拿女声，"人物音频很奇怪"
    # 的一半原因就在这。缺失时 m5 再从 appearance/name 关键词兜底推断。
    gender: str = ""  # male | female | "" (自由文本也接受：男/女/他/她)
    age_band: str = ""  # child | teen | adult | elder | ""
    look: CharacterLook | None = None  # 结构化外貌卡（high/cinema 由 m1 抽取）
    canonical_tokens: list[str] = Field(default_factory=list)  # 蒸馏/改写后必须逐字保留的词


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


_ALIAS_MIN = 2  # 单字别名（"王"）认错人概率太高，不参与匹配


def _name_aliases(name: str) -> list[str]:
    """角色名的可检索别名：全名 + 长度 ≥2 的尾部（"孙悟空"→"悟空"、"猪八戒"→"八戒"）。"""
    stripped = name.strip()
    return [stripped] + [stripped[-n:] for n in range(len(stripped) - 1, _ALIAS_MIN - 1, -1)]


def effective_cast(script: Script, shot: Shot) -> tuple[list[str], list[str]]:
    """镜头画面里"有谁"：声明的 `characters` + 台词说话人 + 画面/镜头文字里点名的人。

    为什么需要：`characters`（在场）是 LLM 写的，实测会漏标。run_20261007_130730 的开场
    镜头 `characters=[]`，可 action 写着"女王在台阶上回眸一笑"、台词说话人就是女王；
    收尾全景同样标成空，action 里却点名四人。m3 只按 `characters` 取参考图，这些人
    的脸就完全由模型自由发挥 —— 定妆照是米色短裙的年轻女王，画出来是橙色曳地礼服的
    另一个人，一致性从源头就没建立。

    返回 `(名单, 其中属于漏标的部分)`。顺序：声明的在前，说话人次之，文字点名的最后
    —— 已锁脸的角色绝不会被漏标的挤掉参考图槽位（`ref_max_chars` 有限时这点很重要）。

    别名匹配只认"全片唯一"的别名（两个角色共享同一别名时整个放弃：宁可不补，也不能
    认错人）。自由文本的覆盖面天然有限（"女王"既不是"女儿国国王"的前缀也不是后缀），
    所以台词说话人这条结构化信号优先。
    """
    by_id = script.character_map()
    found = list(shot.characters)
    for d in shot.dialogues:
        if d.character_id in by_id and d.character_id not in found:
            found.append(d.character_id)

    text = f"{shot.action} {shot.camera}"
    alias_owner: dict[str, str] = {}
    ambiguous: set[str] = set()
    for cid, ch in by_id.items():
        for alias in _name_aliases(ch.name):
            if alias in alias_owner and alias_owner[alias] != cid:
                ambiguous.add(alias)
                continue
            alias_owner[alias] = cid
    for alias, cid in sorted(alias_owner.items(), key=lambda kv: -len(kv[0])):
        if alias in ambiguous or cid in found or alias not in text:
            continue
        found.append(cid)
    return found, [c for c in found if c not in shot.characters]


def script_fingerprint(script: Script) -> str:
    """剧本内容指纹：血缘台账用它标记"这张图是按哪版剧本生成的"。

    排除 `version`（每次保存自增）与 `visual_prompt`（m3 会回写它）—— 否则
    "保存一次无实质修改的剧本"会让全部下游作废，用户每点一次保存就烧一遍钱。
    """
    import hashlib
    import json

    payload = script.model_dump(mode="json", exclude={"version"})
    for sc in payload.get("scenes", []):
        for sh in sc.get("shots", []):
            sh.pop("visual_prompt", None)
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class ProjectConfig(BaseModel):
    """一次生成任务的全部运行配置（写入 run_meta.json，每一步都按它原样恢复）。"""

    run_id: str
    base_info: BaseInfo
    profile: Literal["edu-sqlite", "prod-postgres"] = "edu-sqlite"
    # 口型**不再是可选项**：wan3.0 生成场景片段时原生带台词口型。字段保留两个用途：
    # ① 读旧 run 的 run_meta（历史值原样反序列化，续跑不炸）；② 新 run 恒写 scripted_talk。
    talk_mode: str = "scripted_talk"
    # 音轨同理恒为 model（嘴型与声音同轨；换 TTS 母带实测领先嘴型 0～2.2s 且偏移不固定）
    voice_source: str = "model"
    # 老请求体提交过 talk_mode/talk_model/voice_source 时留痕（接受但忽略，不能静默丢）
    ignored_legacy_fields: list[str] = []
    budget_limit_cny: float = 150.0
    max_parallel_jobs: int = 4
    fake: bool = False
    # 分辨率档 = 配置档（480P | 720P | 1080P）：一致性深度、抽卡数、单价量级、
    # 是否自动重拍都由它带出来（configs/profile_<档>.yaml）。**这是唯一的规格旋钮** ——
    # 旧字段 `quality`（economy/high/cinema）已删：它与分辨率一对一，留两个就能配出
    # "cinema + 480P" 这种自相矛盾的组合（talkmode 里曾专门为它写特例）。
    # 空串只出现在 2026-10 之前的老 run 上，由 `config.resolution_of_meta` 按旧档位名换算。
    resolution_tier: str = ""
    ablation: str = ""  # 消融配置名（必须原样恢复，否则配置向量对不上）
    # run 级模型选择（能力 → 模型名；空值＝跟随画质档位）。**必须落进 meta 并原样恢复**：
    # 只存在 .env 里是全局开关，改一次就把三个档位所有新 run 一起换掉、且没人记得改回来
    # （实测出现过"档位表写 wan2.2-i2v-plus、实际跑 wan3.0-video"）。恢复入口唯一：
    # `core/config.apply_run_overrides`（CLI 与 API 共用一份，不各写一遍）。
    model_overrides: dict[str, str] = Field(default_factory=dict)
    # 溯源：这次点名的模型**是否也写回了 .env**（成为之后新建任务的默认）。
    # 这件事改变了机器上的全局配置，必须留下痕迹，否则"这台机器怎么突然换模型了"无法回答。
    model_env_written: list[str] = Field(default_factory=list)
    model_env_error: str = ""


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
    review_error: bool = False  # 评审自身异常被放行（欠费/抽帧失败等）——UI 黄标警示
    # ── 反"伪造通过"字段（全带默认值：老 final_review.json 仍可 model_validate）──
    verified: bool = True  # False = 该分数不是模型真实给出的（解析失败/欠费/无候选分）
    judge: str = ""  # 评审模型名（qwen-vl-max 等），便于跨 run 对齐基线
    raw: str = ""  # 模型原文（截断），供人工复核"分数为什么是这个"
    attributes: dict = Field(default_factory=dict)  # 结构化属性位（一致性度量用）
    gate_error: bool = False  # 账号欠费/未开通导致评审未执行（交付放行但必须可见）


class ConsistencyVerdict(BaseModel):
    """一致性度量的严格输出：全字段必填、无默认值。

    与 ReviewResult 的宽容语义相反——度量字段一旦有默认值，"模型没回答"就会被
    静默填成"通过"，正是此前 score 恒 0.7 的病灶。bool 位做宽松收敛（模型可能
    返回 "true"/"True" 字符串），但缺失即校验失败。
    """

    face: bool
    hair: bool
    outfit: bool
    headwear: bool
    distinctive_hits: list[str]
    photoreal: bool
    score: float
    reason: str

    @field_validator("face", "hair", "outfit", "headwear", "photoreal", mode="before")
    @classmethod
    def _coerce_bool(cls, v: object) -> object:
        if isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "yes", "1", "是", "一致"):
                return True
            if low in ("false", "no", "0", "否", "不一致"):
                return False
        return v

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: object) -> object:
        f = float(v)  # type: ignore[arg-type]
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"score 越界: {f}")
        return f


class RegionBoxes(BaseModel):
    """口幅度量的取样框（归一化 0~1 的 x0,y0,x1,y1）。

    为什么必须由视觉模型给：口型度量原先用一个写死的比例框（画面下缘中心），
    实测特写/过肩/全身混合分镜下**框经常不在嘴上** —— 在同一素材上"嘴区"帧间
    运动幅度 0.8 而整脸 4.4、背景 4.2，即框落在静止区域。用这种尺子判口型会把
    好片段误杀（触发付费重做）或把坏片段放过，所以先定位再测量，定位不到就
    明确返回"无结论"，绝不给一个看起来像结论的数字。
    eyes 框是给"同一张脸上的非嘴部区域"做对照：若嘴区相关不高于眼区对照，
    说明测到的只是头部/画面运动，不是语音驱动的口型。
    """

    has_person: bool
    lips: tuple[float, float, float, float]
    eyes: tuple[float, float, float, float]
    confidence: float
    reason: str

    @field_validator("has_person", mode="before")
    @classmethod
    def _coerce_bool(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "是", "有")
        return v

    @field_validator("lips", "eyes", mode="before")
    @classmethod
    def _coerce_box(cls, v: object) -> object:
        """接受 [..]/{"x0":..} 两种写法，补齐越界与反序（框仍可测量的前提下尽量挽救）。"""
        if isinstance(v, dict):
            seq = (v.get("x0", 0), v.get("y0", 0), v.get("x1", 1), v.get("y1", 1))
        elif isinstance(v, (list, tuple)) and len(v) >= 4:
            seq = tuple(v[:4])
        else:
            raise ValueError(f"框格式非法: {v!r}")
        vals: list[float] = []
        for x in seq:
            if isinstance(x, str) and x.strip().endswith("%"):
                x = float(x.strip()[:-1]) / 100.0
            vals.append(float(x))  # type: ignore[arg-type]
        # 像素制换算必须**整框判一次**：逐元素判会让同一个框的 x0 按 /100、x1 按 /1000
        # 缩放，框被拉成对角线（模型常回 0~100 或 0~1000 两种坐标制）。
        peak = max(abs(v) for v in vals)
        if peak > 1.5:
            divisor = 1000.0 if peak > 500.0 else 100.0
            vals = [v / divisor for v in vals]
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        x0, y0, x1, y1 = vals
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return (x0, y0, x1, y1)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_conf(cls, v: object) -> float:
        f = float(v)  # type: ignore[arg-type]
        if f > 1.0:
            f = f / 100.0
        return min(max(f, 0.0), 1.0)

    @staticmethod
    def area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def sane(self) -> bool:
        """框要"像个嘴/眼框"：非退化、不占满全画面、不互相完全重合。"""
        if not self.has_person or self.confidence < 0.4:
            return False
        la, ea = self.area(self.lips), self.area(self.eyes)
        if not (0.0005 <= la <= 0.25 and 0.0005 <= ea <= 0.25):
            return False
        return self.lips != self.eyes  # 眼框=嘴框 说明模型没分清，对照失效


class StyleVerdict(BaseModel):
    """立绘媒介判别：拦截"动漫立绘污染真人短剧全链路"（实测发生过）。"""

    photoreal: bool
    medium: Literal["photo", "anime", "3d", "illustration"]
    score: float

    @field_validator("photoreal", mode="before")
    @classmethod
    def _coerce_bool(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "是")
        return v

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: object) -> object:
        f = float(v)  # type: ignore[arg-type]
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"score 越界: {f}")
        return f


class CostEntry(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capability: str = ""
    provider: str = ""
    model: str = ""
    amount_cny: float = 0.0
    note: str = ""
    # 结构化请求指纹：事后只看 cost.json 就能重建"这次 API 实际发了什么"
    # （seed/ref_count/ref_ids/route/verified…）。此前 wan2.7-image 0 次、vlm 0 条
    # 之所以能瞒住，正是因为请求内容不落账。老 cost.json 缺该字段照常加载。
    meta: dict = Field(default_factory=dict)


class AudioLine(BaseModel):
    """m5 产物：一句对白的音频与时长对齐信息。"""

    line_id: str
    shot_id: str
    character_id: str
    text: str
    wav_uri: str
    duration_sec: float = 0.0
    padded: bool = False  # 是否需要 m7 补静帧
