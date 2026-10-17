"""配置加载：.env 环境变量 + configs/*.yaml 供应商路由（TDD §10）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.getenv("FV3_OUTPUT_DIR", PROJECT_ROOT / "output"))
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models"

#: 分辨率 ↔ 内部画质档的**单一映射**。面板上用户只选分辨率（480P/720P/1080P），
#: 档位继续在后端驱动"一致性深度 + 单价量级 + 是否自动重拍"这些用户不该逐条记的东西。
#: 为什么保留档位而不是删掉：`consistency_gate: reshoot`（自动重拍）是**花钱**的动作，
#: 480P 教学档量出问题只该报告不该烧钱；而三类闸门本身（结构性 / 技术 QC / 身份客观层）
#: 与分辨率无关，任何档都跑 —— 见 configs/default.yaml 的注释。
#: 目标时长的合法区间（秒）。**单一来源**：API 的 pydantic 校验、CLI 的参数校验、
#: 前端表单都引用它 —— 三处各写一份就是"面板能填、后端 422"那种对不上。
#: 下限 1 而不是 0：0 会让剧本没有目标长度、成本预估归零，且熔断判据（`exceeded()` 用
#: `>=`）在第一步就跳闸；上限 3600 对齐"一集短剧"的量级。
DURATION_MIN = 1
DURATION_MAX = 3600

RESOLUTIONS: tuple[str, ...] = ("480P", "720P", "1080P")
#: 分辨率 = 配置档名（`configs/profile_<tier>.yaml`）。**用户只有一个旋钮**：选分辨率。
#: "一致性深度 / 抽卡数 / 单价量级 / 是否自动重拍"这些由它带出来，不用逐条记。
#: 480P 没有 overlay 文件（= default.yaml 本体），720P/1080P 各叠一层。
#:
#: 为什么不再有 economy/high/cinema 这层名字：它与分辨率一对一，两个旋钮表达同一件事，
#: 就必然出现"cinema + 480P"这种自相矛盾的组合（talkmode 里曾专门为它写特例）。
#: 一个旋钮 = 不可能配错。
PROFILE_FOR_RESOLUTION: dict[str, str] = {t: t for t in RESOLUTIONS}
#: 2026-10 之前的 run_meta 里写着旧档位名。**只用于读旧数据**，新代码一律用分辨率；
#: 不留这张表，老 run 续跑时会拿不到档位 → 静默按 480P 跑（一致性闸门全关，成片脸会漂）。
LEGACY_QUALITY_TO_RESOLUTION: dict[str, str] = {
    "economy": "480P",
    "high": "720P",
    "cinema": "1080P",
}
DEFAULT_RESOLUTION = "720P"


#: 分辨率 × 画幅 → 成片尺寸（宽, 高）。**显式列出来而不按比例算**：480P 档沿用项目既有
#: 约定 540×960（不是 480×853），按短边反推会和存量 run 的尺寸对不上。
#: 为什么必须有这张表：档位 yaml 里写着各自的 resolution（profile_720P 是 1080×1920），
#: 面板显示"720P"却按 1080P 出片就是**骗人** —— 分辨率是用户唯一看得见的规格承诺。
RESOLUTION_SIZES: dict[tuple[str, str], tuple[int, int]] = {
    ("480P", "9:16"): (540, 960),
    ("480P", "16:9"): (960, 540),
    ("480P", "1:1"): (540, 540),
    ("720P", "9:16"): (720, 1280),
    ("720P", "16:9"): (1280, 720),
    ("720P", "1:1"): (720, 720),
    ("1080P", "9:16"): (1080, 1920),
    ("1080P", "16:9"): (1920, 1080),
    ("1080P", "1:1"): (1080, 1080),
}


def apply_resolution(cfg: "AppConfig", resolution_tier: str) -> "AppConfig":
    """让**分辨率标签真的决定成片尺寸与 I2V 档位**（就地改 cfg 并返回它）。

    档位 yaml 只负责"一致性深度 / 抽卡数 / 单价量级 / 是否自动重拍"这些用户不该逐条记的
    东西；规格承诺（分辨率）由面板选的那一项说了算。两者分工写死在这里，避免
    "面板显示 720P、实际按 high 档的 1080P 出片"这种对不上。
    """
    tier = (resolution_tier or "").strip().upper()
    if tier not in RESOLUTIONS:
        return cfg  # 认不出来就不动（不猜规格）
    size = RESOLUTION_SIZES.get((tier, cfg.aspect_ratio))
    if size:
        cfg.resolution = size
    cfg.i2v_resolution = tier
    return cfg


def resolution_of_meta(meta: dict | None) -> str:
    """run_meta → 分辨率档位（**唯一**的恢复入口，CLI 与 API 都走它）。

    新 run 写 `resolution_tier`；2026-10 之前的 run 只有旧档位名 `quality`，按兼容表换算。
    两者都没有就按默认档 —— 不猜、不报错，因为续跑一个老 run 不该被一个显示字段卡住。
    """
    m = meta or {}
    tier = str(m.get("resolution_tier") or "").strip().upper()
    if tier in RESOLUTIONS:
        return tier
    legacy = str(m.get("quality") or "").strip().lower()
    return LEGACY_QUALITY_TO_RESOLUTION.get(legacy, DEFAULT_RESOLUTION)


class ProviderRouting(BaseModel):
    """单类能力的供应商路由：默认 + 备选。"""

    default: str
    fallback: str | None = None


class AppConfig(BaseModel):
    """应用级配置（环境变量 + yaml 合并后的运行时视图）。"""

    # ── API 密钥 ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    dashscope_api_key: str = ""
    dashscope_text_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # wan2.2-s2v（对口型主通道）的异步任务端点。官方文档要求"华北2（北京）地域 + 该地域
    # 的 API Key"，并给出业务空间专属写法 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1
    # 默认用公共端点（同一把 KEY 已实测可调用其它百炼模型）；若该模型在公共端点报
    # url/access 类错误，把本项换成业务空间端点即可，无需改代码（提交与轮询同端点）。
    dashscope_video_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    siliconflow_api_key: str = ""
    kling_access_key: str = ""
    kling_secret_key: str = ""
    # fal.ai（MeiGen InfiniteTalk 口型通道的托管方）。KEY 在 fal.ai dashboard →
    # API Keys 生成后**手动**写进 .env 的 FAL_KEY。口型主通道没有它就无法构造，
    # 因此它不是"可选第三方"，而是当前 talk 路由的硬依赖（见 providers/fal.py）。
    fal_api_key: str = ""
    fal_acceleration: str = "regular"  # none | regular | high（fal 的加速档，影响质量/速度）

    # ── 文本模型特性 ──
    #: 文本/兼容端点的读超时秒数。默认 180 是给"一次写出整部 30 秒剧本"留的量；
    #: 名字发错端点时不该靠重试硬等（那是 9 分钟黑盒），但慢模型确实需要它。
    text_timeout_sec: float = 180.0
    enable_thinking: bool = False  # qwen3 系思考模式(流式); 结构化 JSON 场景默认关闭

    # ── 运行模式 ──
    profile: str = "edu-sqlite"  # edu-sqlite | prod-postgres
    # 口型模式：取值语义与它调用的通道**同出一表**（`core/talkmode.py`），不在这里重复定义，
    # 也不再由 `providers.talk` 路由决定（路由只在形态相容时作为覆盖/兜底来源）。
    #   s2v_direct     = wan2.2-s2v（图片+语音+提示词 → 口型视频）
    #   v2v_overlay    = videoretalk（wan3.0 出动作片段 → 再重加口型）
    #   scripted_talk  = wan3.0-video（图片+提示词 → 口型+动作一体）
    #   audio_overlay  = 不做口型，m7 只叠音轨
    # 口型**不再是可选项**：wan3.0 生成场景片段时原生带台词口型（音频与嘴型同轨），
    # 独立口型通道（s2v/videoretalk）与"不做口型"都已下线。字段保留只为读旧 run 的
    # run_meta（历史值原样反序列化），`load_config` 会把它钉回 scripted_talk。
    talk_mode: str = "scripted_talk"
    # 音轨来源同样不再可选：场景片段的嘴型是对着模型自己的声音画的，换回 TTS 母带
    # 实测领先嘴型 0～2.2s 且偏移不固定（无法用固定延迟修）。字段保留读旧 run 用。
    voice_source: str = "model"
    fake: bool = False

    # ── 预算与并发 ──
    budget_limit_cny: float = 150.0
    max_parallel_jobs: int = 4
    per_provider_concurrency: dict[str, int] = Field(default_factory=dict)
    i2v_poll_timeout_min: int = 30
    retry_per_shot: int = 2

    # ── 生成参数 ──
    aspect_ratio: str = "9:16"
    shot_duration_sec: float = 5.0
    resolution: tuple[int, int] = (540, 960)
    storyboard_sample_rate: float = 0.5
    max_script_revisions: int = 1  # 台词句长违规后的重写上限（不是 AI 评审回路，那条已删）

    # ── 配置档 = 分辨率档（480P/720P/1080P，见 configs/profile_*.yaml）──
    # 档位标签就是 `i2v_resolution` 本身：再存一个同义字段，就会出现两者不一致时
    # "看板显示 720P、实际按 1080P 出片"这类对不上（这条 bug 修过一次）。
    ablation: str = ""  # 消融配置名（写入 run_meta，续跑原样恢复）
    use_r2v_for_characters: bool = False  # 角色动作镜头走 wan2.7-r2v 参考生视频（像素级一致）
    use_reference_keyframes: bool = False  # 首帧带角色立绘参考图生成（硬锁脸，wan2.7-image）
    require_reference_keyframes: bool = False  # 硬门控：主供应商不支持参考图时直接报错（720P/1080P）
    ref_max_chars: int = 1  # 首帧最多带几张角色参考图（双人镜头=2；探单通过前保持 1）
    keyframe_seed_scope: Literal["cast", "scene_cast", "shot"] = "cast"
    # 种子作用域：scene_cast（旧行为）会把"场景"编进种子 → 角色换场景即换采样起点，
    # 实测主角 4 个场景 4 张脸；cast = 同角色跨场景同种子。
    asset_style_gate: Literal["off", "warn", "block"] = "off"  # 立绘写实风格闸门
    look_extraction: bool = False  # m1 抽取结构化外貌卡（蒸馏校验/一致性度量的属性来源）
    consistency_metric: Literal["off", "sample", "full"] = "off"  # m9 一致性度量档位
    # 场景概念图参与下游生成的方式：off=不用；scenery_only=只给空镜（默认，零身份风险）；
    # all=含角色镜头也带（必须等 m9 证明人脸达标率不降，且带自动回退）
    scene_ref_mode: Literal["off", "scenery_only", "all"] = "scenery_only"
    # 剧本稿自动整理：确定性层（draft_norm，不花钱）之后仍有格式问题时，允许调一次
    # TEXT 只改排版（逐字校验内容没被改，失败就拒绝保存而不是静默改词）。
    # 关掉它就等于"只接受标准排版"，报错按行号回给人。
    draft_llm_format: bool = True
    # ── 语音预算系数（core/voice_budget.py）：全是**软校验**的参数，不做门禁 ──────────
    # 这一段描述的是"这份稿子拍出来大概多长、哪一镜念不完"，用来在花钱之前给出容量判断；
    # 片段内部怎么分配时长是 wan3.0 的自由度，不由这些数字控制，也不回写用户写的任何时长。
    # 语速口径与 m6 的 TTS 档同源：改这里就等于改全链路的语音假设，别再引入第二份。
    voice_cps: float = 4.5                  # 中文语音语速（字/秒）
    # 最小预估时长 = round(字数 / voice_cps) + voice_breath_sec（用户裁定的公式）
    voice_breath_sec: float = 1.0           # 每镜呼吸余量（也起到"每镜固定开销"的作用）
    voice_beat_sec: float = 1.2             # 一个可拍动作单元的演出秒
    voice_hold_sec: float = 0.6             # 静默内容的额外留白上限（凝视/落泪/末镜，见 voice_budget）
    clip_soft_cap_sec: float = 25.0         # 单次调用的软预算；服务端硬顶 30s 见 scenegroup.MAX_CLIP_SEC
    voice_nonspeech_band_low: float = 0.30  # 非语音占比下限（低于它=念稿）
    voice_nonspeech_band_high: float = 0.45  # 非语音占比上限（高于它=空转）
    # 成片前一致性闸门：off=不拦；reshoot=不达标镜头自动带参考图重做（≤2 轮）后仍不达标
    # 则 quality_verdict 标不合格。720P/1080P 档强制 reshoot，480P 教学档只报告不烧钱。
    consistency_gate: Literal["off", "reshoot"] = "off"
    # 双人（多人）对白镜头的口型策略：
    #   overlay     = 运动片段 + 音轨叠加，**不模拟口型**（默认）
    #   s2v_merged  = 全部台词拼成单音轨喂口型模型
    # 默认 overlay 的原因：API 侧口型模型都是"单音轨驱动整幅画面"，没有参考项目
    # MultiTalk 那种"每人一条音轨 + bbox 绑定"的能力；单音轨喂双人画面会让**没在
    # 说话的人**也跟着动嘴，观感比无口型更糟。想赌合并音轨效果可显式设 s2v_merged。
    two_person_talk: Literal["overlay", "s2v_merged"] = "overlay"
    # 口型同步闸门：嘴部运动×音频包络相关系数低于该值 → 判"口型与语音不同步"，
    # 标 stale 待重做（gate=reshoot 时进自动重做）。实测无口型叠音片段 ≈ 0.0-0.12，
    # 真口型片段应显著更高；阈值保守取 0.25，宁可漏拦不误删。
    lipsync_min_corr: float = 0.25
    # 视频类通道（i2v/talk/r2v）单价上限（元/秒）。用户红线：口型视频成本不应超过
    # ¥5/s。装配 provider 时按 pricing.yaml 校验，超上限直接报错而非静默烧钱。
    max_video_cost_per_sec: float = 5.0
    consistency_min_face: float = 0.8  # 逐镜 face 属性位达标率阈值
    consistency_min_outfit: float = 0.8
    candidates_per_shot: int = 1  # >1 时 best-of-N + VLM 选优（商业管线"抽卡"）
    asset_candidates: int = 1  # 角色立绘 best-of-N（源头防立绘崩，全链路继承放大）
    auto_reshoot: bool = False  # auto 终审未过时自动重拍 ≤2 镜（默认关，防成本失控）
    i2v_resolution: str = "480P"  # wan 系 I2V 档位：480P | 720P | 1080P
    # 首尾帧生视频：给需要运动片段的镜头额外生成一张「动作结束态」尾帧，i2v 时一起发。
    # 官方原文"用首帧（或首帧+尾帧）锁定画面起止，模型负责补全中间的运动过程"——
    # 只给首帧时结束状态全靠模型自由发挥，镜头末尾姿态/构图乱飘、切下一镜接不上。
    # 代价：每个需要运动片段的镜头多一次 t2i（约 ¥0.25，走免费额度模型时 ¥0）。
    # 只对**支持 last_frame 的通道**生效（wan2.7/wan3.0 的 media 协议）；wan2.2 旧协议
    # 与 r2v 参考生视频拿不到（官方 first/last_frame 与 reference_image 互斥），
    # 那种情况自动退回单首帧并记事件，绝不阻断交付。
    use_last_frames: bool = True
    # 场景级生成（wan3.0 一次调用出一个场景的连贯多镜头片段）的输入形态：
    #   auto        = 按 scenegroup.choose_input_mode 的确定性规则选（默认）
    #   reference   = 强制参考生视频（立绘+场景图+音色样本；≈90% 的主干）
    #   first_last  = 强制首尾帧（像素级锁起止；≈10% 的串联/单镜头/重拍）
    #   off         = 关掉场景级生成，回到逐镜头路径（480P/A链/B链本来就走那边）
    # 官方硬约束：reference_* 与 first/last_frame **互斥**，所以这是"二选一"不是"叠加"。
    scene_clip_mode: Literal["auto", "reference", "first_last", "off"] = "auto"
    style_suffix: str = ""  # 全局风格锚：追加到所有文生图提示词尾部（统一全片画风）
    negative_prompt: str = ""  # 全局负向提示词（防卡通化/防劣化）
    color_grade: bool = False  # 成片轻度调色（对比/饱和/锐化）
    loudnorm: bool = False  # 交付响度归一 -16 LUFS（竖屏平台标准）
    encode_crf: int = 23  # 中间编码质量（交付档 20）
    encode_preset: str = "ultrafast"
    output_fps: int = 24  # 成片帧率（此前 ffmpeg 层硬编码 24）
    intermediate_crf: int = 18  # 中转产物 CRF 上限（normalize/静帧等，减少代际累积损伤）
    bgm_volume: float = 0.15

    # ── 路由与模型 ──
    routing: dict[str, ProviderRouting] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    # .env 能力级模型名覆盖（FV3_TEXT_MODEL / FV3_T2I_MODEL / FV3_I2V_MODEL /
    # FV3_TALK_MODEL / FV3_VLM_MODEL / FV3_TTS_MODEL）：优先级高于 yaml 的 models:，
    # 并且会连带把 provider 改道到该模型真正使用的协议（见 providers/protocols.py）。
    model_overrides: dict[str, str] = Field(default_factory=dict)

    # ── 数据库 ──
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "fastvideo3"
    pg_user: str = "fv3"
    pg_password: str = ""

    @property
    def db_url(self) -> str:
        if self.profile == "prod-postgres":
            return (
                f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{OUTPUT_DIR / 'fastvideo3.db'}"

    def model_name(self, key: str, default: str = "") -> str:
        return self.models.get(key, default)

    def model_for(self, capability: str, provider_key: str, default: str = "") -> str:
        """某能力实际调用的模型名：.env 覆盖 > yaml models[provider_key] > 内置默认。

        factory 必须走这个入口而不是直接 `models.get(...)`，否则 .env 里设的
        `FV3_T2I_MODEL` 之类会被静默忽略（用户改了 .env 却没生效最难查）。
        """
        return self.model_overrides.get(capability) or self.models.get(provider_key, default)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """档位 overlay 深度合并进基础 yaml（只改差异项，其余继承）。"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def normalize_resolution(raw: Any) -> str:
    """任意写法 → 合法分辨率档（认不出就按默认档，旧档位名走兼容表）。"""
    tier = str(raw or DEFAULT_RESOLUTION).strip().upper()
    if tier not in RESOLUTIONS:
        tier = LEGACY_QUALITY_TO_RESOLUTION.get(tier.lower(), DEFAULT_RESOLUTION)
    return tier


#: 一致性相关的 generation 位（`configs/*.yaml` 决定的那几项）。
FACE_LOCK_KEYS = (
    "use_reference_keyframes",
    "require_reference_keyframes",
    "consistency_gate",
    "consistency_metric",
)


@lru_cache(maxsize=32)
def _profile_flags_cached(configs_dir: str, tier: str) -> dict[str, Any]:
    yml = _load_yaml(Path(configs_dir) / "default.yaml")
    overlay = _load_yaml(Path(configs_dir) / f"profile_{PROFILE_FOR_RESOLUTION[tier]}.yaml")
    if overlay:
        _deep_merge(yml, overlay)
    gen = yml.get("generation") or {}
    return {k: gen.get(k) for k in FACE_LOCK_KEYS}


def profile_flags(resolution: str) -> dict[str, Any]:
    """这一档**在 configs 里**的一致性承诺（不读 .env、不读环境变量）。

    为什么单开一个入口，而不是让调用方 `load_config(resolution=...)` 再取字段：
    建任务时的拦截（`model_catalog.ref_image_warning`）和面板的档位说明都要回答同一句
    话——「这一档到底锁不锁脸」。走 load_config 会把 `.env` 与进程环境变量卷进来，
    于是同一份 configs 在两台机器上能给出两个答案，而一致性承诺不该跟着机器漂。
    """
    tier = normalize_resolution(resolution)
    yml = _load_yaml(CONFIGS_DIR / "default.yaml")
    overlay = _load_yaml(CONFIGS_DIR / f"profile_{PROFILE_FOR_RESOLUTION[tier]}.yaml")
    if overlay:
        _deep_merge(yml, overlay)
    gen = yml.get("generation") or {}
    return {k: gen.get(k) for k in FACE_LOCK_KEYS}


def tier_locks_faces(resolution: str) -> bool:
    """这一档是否要求「首帧带角色参考图」（= 同一个角色用同一张角色图出片）。"""
    return bool(profile_flags(resolution).get("use_reference_keyframes"))


def tier_hard_gates_faces(resolution: str) -> bool:
    """路由撑不起锁脸时是否**在花钱前报错**（false = 只会记事件、成片照常交付）。"""
    return bool(profile_flags(resolution).get("require_reference_keyframes"))


def tier_auto_reshoots(resolution: str) -> bool:
    """这一档是否会为不达标镜头**自动重拍**（要花钱，只有 720P/1080P 开）。"""
    return profile_flags(resolution).get("consistency_gate") == "reshoot"


def _resolve_ablation(name: str) -> Path:
    """消融配置路径解析：必须落在 configs/ 内（防路径穿越，同 api.py 的三重防护风格）。

    接受裸名（"ref_only"）、带目录名（"ablations/ref_only"）或绝对路径三种写法。
    """
    cand = Path(name)
    if not cand.is_absolute():
        if not cand.parts or cand.parts[0] != "ablations":
            cand = CONFIGS_DIR / "ablations" / name
        else:
            cand = CONFIGS_DIR / name
        if not cand.suffix:
            cand = cand.with_suffix(".yaml")
    resolved = cand.resolve()
    if not resolved.is_relative_to(CONFIGS_DIR.resolve()):
        raise ValueError(f"消融配置必须位于 configs/ 内: {name}")
    if not resolved.exists():
        raise ValueError(f"消融配置不存在: {resolved}")
    return resolved


def load_config(
    config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
    resolution: str | None = None,
    ablation: str | None = None,
) -> AppConfig:
    """构建运行时配置。优先级：显式参数 > 环境变量 > .env 文件 > yaml > 默认值。

    用 dotenv_values 只读不注入 —— 避免污染进程环境变量（测试隔离的关键）。
    分辨率档（`--resolution` / `FV3_RESOLUTION`）通过 `configs/profile_<tier>.yaml`
    overlay 生效（480P 没有 overlay，就是 default.yaml 本体）。
    消融配置（--ablation）在档**之后**再叠一层：若当 base 传入会被档反向
    覆盖（overlay 顺序陷阱），消融就测不到它想测的开关。
    """
    dotenv_vals = dotenv_values(dotenv_path or PROJECT_ROOT / ".env")

    yml = _load_yaml(Path(config_path) if config_path else CONFIGS_DIR / "default.yaml")
    # 认不出的写法（含旧档位名）→ 走兼容表，再不行按默认档：配置读不出来就整条流水线停摆，
    # 不值得为一个显示字段这么做；但换算过程要在事件里看得见（调用方负责）。
    # 归一化只有一份（`normalize_resolution`）：档位决定一致性承诺，两处各漂一次就会出现
    # "面板说这档锁脸、真正跑的却是另一档"。
    tier = normalize_resolution(
        resolution
        or os.getenv("FV3_RESOLUTION")
        or dotenv_vals.get("FV3_RESOLUTION")  # type: ignore[assignment]
        or str((yml.get("generation") or {}).get("i2v_resolution") or DEFAULT_RESOLUTION)
    )
    overlay = _load_yaml(CONFIGS_DIR / f"profile_{PROFILE_FOR_RESOLUTION[tier]}.yaml")
    if overlay:
        _deep_merge(yml, overlay)
    if ablation:
        _deep_merge(yml, _load_yaml(_resolve_ablation(ablation)))
    gen = yml.get("generation", {})
    limits = yml.get("limits", {})
    routing = {cap: ProviderRouting(**route) for cap, route in (yml.get("providers") or {}).items()}

    def env(key: str, default: str = "") -> str:
        v = os.getenv(key)
        if not v:
            v = dotenv_vals.get(key)  # type: ignore[assignment]
        return v or default

    size = tuple(gen.get("resolution", [540, 960]))

    # 口型模式曾是可选项（s2v_direct/v2v_overlay/scripted_talk/audio_overlay），由此踩过的
    # 坑（.env 点名 wan3.0 而 talk_mode 还是 s2v_direct → m6 喂音频 → 全镜头降级无口型）
    # 随选项一起消失：talk_mode 现在恒为 scripted_talk，`.env` 的 FV3_TALK_MODEL 只负责
    # 在 wan3.0-video 与 -prime 之间选模型（见 model_overrides["talk"]）。

    cfg = AppConfig(
        deepseek_api_key=env("DEEPSEEK_API_KEY"),
        deepseek_base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        dashscope_api_key=env("DASHSCOPE_API_KEY"),
        dashscope_text_base_url=env(
            "DASHSCOPE_TEXT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        dashscope_video_base_url=env(
            "FV3_S2V_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        ),
        enable_thinking=env("FV3_ENABLE_THINKING", "0") in ("1", "true", "yes"),
        siliconflow_api_key=env("SILICONFLOW_API_KEY"),
        kling_access_key=env("KLING_ACCESS_KEY"),
        kling_secret_key=env("KLING_SECRET_KEY"),
        fal_api_key=env("FAL_KEY"),
        fal_acceleration=env("FAL_ACCELERATION", "regular"),
        profile=env("FV3_PROFILE", "edu-sqlite"),
        # 口型/音轨不再可选：wan3.0 场景片段自带口型，音轨用模型原声（嘴型与声音同轨）。
        # .env 的 FV3_TALK_MODEL 仍有效 —— 它选的是 wan3.0-video 还是 -prime。
        talk_mode="scripted_talk",
        voice_source="model",
        fake=env("FV3_FAKE", "0") in ("1", "true", "yes"),
        budget_limit_cny=float(env("FV3_BUDGET_LIMIT_CNY", "0") or 150),
        max_parallel_jobs=int(limits.get("max_parallel_jobs", 4)),
        per_provider_concurrency=limits.get("per_provider_concurrency", {}),
        i2v_poll_timeout_min=int(limits.get("i2v_poll_timeout_min", 30)),
        retry_per_shot=int(limits.get("retry_per_shot", 2)),
        aspect_ratio=gen.get("aspect_ratio", "9:16"),
        shot_duration_sec=float(gen.get("shot_duration_sec", 5)),
        resolution=(int(size[0]), int(size[1])),  # type: ignore[index]
        storyboard_sample_rate=float(gen.get("storyboard_sample_rate", 0.5)),
        max_script_revisions=int(gen.get("max_script_revisions", 1)),
        ablation=ablation or "",
        use_r2v_for_characters=bool(gen.get("use_r2v_for_characters", False)),
        use_reference_keyframes=bool(gen.get("use_reference_keyframes", False)),
        require_reference_keyframes=bool(gen.get("require_reference_keyframes", False)),
        ref_max_chars=int(gen.get("ref_max_chars", 1)),
        keyframe_seed_scope=gen.get("keyframe_seed_scope", "cast"),
        asset_style_gate=gen.get("asset_style_gate", "off"),
        look_extraction=bool(gen.get("look_extraction", False)),
        consistency_metric=gen.get("consistency_metric", "off"),
        scene_ref_mode=gen.get("scene_ref_mode", "scenery_only"),
        draft_llm_format=bool(gen.get("draft_llm_format", True)),
        voice_cps=float(gen.get("voice_cps", 4.5)),
        voice_breath_sec=float(gen.get("voice_breath_sec", 1.0)),
        voice_beat_sec=float(gen.get("voice_beat_sec", 1.2)),
        voice_hold_sec=float(gen.get("voice_hold_sec", 0.6)),
        clip_soft_cap_sec=float(gen.get("clip_soft_cap_sec", 25.0)),
        voice_nonspeech_band_low=float(gen.get("voice_nonspeech_band_low", 0.30)),
        voice_nonspeech_band_high=float(gen.get("voice_nonspeech_band_high", 0.45)),
        consistency_gate=gen.get("consistency_gate", "off"),
        two_person_talk=gen.get("two_person_talk", "overlay"),
        lipsync_min_corr=float(gen.get("lipsync_min_corr", 0.25)),
        max_video_cost_per_sec=float(
            env("FV3_MAX_VIDEO_COST_PER_SEC", "") or gen.get("max_video_cost_per_sec", 5.0)
        ),
        consistency_min_face=float(gen.get("consistency_min_face", 0.8)),
        consistency_min_outfit=float(gen.get("consistency_min_outfit", 0.8)),
        candidates_per_shot=int(gen.get("candidates_per_shot", 1)),
        asset_candidates=int(gen.get("asset_candidates", 1)),
        auto_reshoot=bool(gen.get("auto_reshoot", False)),
        i2v_resolution=tier,
        use_last_frames=bool(gen.get("use_last_frames", True)),
        scene_clip_mode=gen.get("scene_clip_mode", "auto"),
        style_suffix=gen.get("style_suffix", ""),
        negative_prompt=gen.get("negative_prompt", ""),
        color_grade=bool(gen.get("color_grade", False)),
        loudnorm=bool(gen.get("loudnorm", False)),
        encode_crf=int(gen.get("encode_crf", 23)),
        encode_preset=gen.get("encode_preset", "ultrafast"),
        output_fps=int(gen.get("output_fps", 24)),
        intermediate_crf=int(gen.get("intermediate_crf", 18)),
        bgm_volume=float(gen.get("bgm_volume", 0.15)),
        routing=routing,
        models=yml.get("models", {}),
        model_overrides={
            cap: v
            for cap, v in (
                ("text", env("FV3_TEXT_MODEL")),
                ("t2i", env("FV3_T2I_MODEL")),
                ("i2v", env("FV3_I2V_MODEL")),
                ("r2v", env("FV3_R2V_MODEL")),
                ("talk", env("FV3_TALK_MODEL")),
                ("tts", env("FV3_TTS_MODEL")),
                ("vlm", env("FV3_VLM_MODEL")),
            )
            if v  # 未设置的项不进表，避免空串把 yaml 的模型名覆盖掉
        },
        pg_host=env("FV3_PG_HOST", "localhost"),
        pg_port=int(env("FV3_PG_PORT", "5432") or 5432),
        pg_db=env("FV3_PG_DB", "fastvideo3"),
        pg_user=env("FV3_PG_USER", "fv3"),
        pg_password=env("FV3_PG_PASSWORD", ""),
    )
    # 成片尺寸与档位标签必须一起落定：只设一个就会出现「看板显示 720P、实际按 1080P 出片」
    return apply_resolution(cfg, tier)


def apply_run_overrides(cfg: AppConfig, meta: dict | None) -> AppConfig:
    """把 run_meta.json 里的 run 级模型选择盖到 cfg 上（**唯一恢复入口**）。

    优先级：run 选择 > `.env` 的 FV3_*_MODEL > yaml 的 `models:`。
    之所以 run 级最高：用户在这次生成里点名要的模型，不该被机器上那份全局 .env 顶掉
    —— 反过来，没点名的能力继续跟随 .env/yaml，行为与改造前完全一致。

    必须只有一份实现：CLI 的 `_cfg_from_meta` 与 API 的 `_cfg_for_run` 是两条独立路径，
    各写一遍就会出现"同一个 run 在网页和命令行下用的是不同模型"（本项目反复出现的
    静默分歧根因）。
    """
    chosen = (meta or {}).get("model_overrides") or {}
    clean = {k: v for k, v in chosen.items() if isinstance(v, str) and v.strip()}
    if clean:
        cfg.model_overrides = {**cfg.model_overrides, **clean}
    return cfg


@lru_cache
def get_config() -> AppConfig:
    """全局缓存的配置实例（测试中用 load_config() 直接构建）。"""
    return load_config()
