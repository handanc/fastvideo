"""配置加载：.env 环境变量 + configs/*.yaml 供应商路由（TDD §10）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.getenv("FV3_OUTPUT_DIR", PROJECT_ROOT / "output"))
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models"


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
    siliconflow_api_key: str = ""
    kling_access_key: str = ""
    kling_secret_key: str = ""

    # ── 文本模型特性 ──
    enable_thinking: bool = False  # qwen3 系思考模式(流式); 结构化 JSON 场景默认关闭

    # ── 运行模式 ──
    profile: str = "edu-sqlite"  # edu-sqlite | prod-postgres
    review_mode: str = "auto"  # auto | gated
    talk_mode: str = "s2v_direct"  # s2v_direct | v2v_overlay | audio_overlay
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
    max_script_revisions: int = 2

    # ── 画质档位（economy=教学 | high=交付 | cinema=商业级，见 configs/profile_*.yaml）──
    quality: str = "economy"
    use_r2v_for_characters: bool = False  # 角色动作镜头走 wan2.7-r2v 参考生视频（像素级一致）
    use_reference_keyframes: bool = False  # 首帧带角色立绘参考图生成（硬锁脸，wan2.7-image）
    candidates_per_shot: int = 1  # >1 时 best-of-N + VLM 选优（商业管线"抽卡"）
    asset_candidates: int = 1  # 角色立绘 best-of-N（源头防立绘崩，全链路继承放大）
    script_review: bool = False  # auto 模式剧本 critic 评审（high/cinema 开启）
    auto_reshoot: bool = False  # auto 终审未过时自动重拍 ≤2 镜（默认关，防成本失控）
    i2v_resolution: str = "480P"  # wan 系 I2V 档位：480P | 720P | 1080P
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


def load_config(
    config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
    quality: str | None = None,
) -> AppConfig:
    """构建运行时配置。优先级：显式参数 > 环境变量 > .env 文件 > yaml > 默认值。

    用 dotenv_values 只读不注入 —— 避免污染进程环境变量（测试隔离的关键）。
    画质档位（--quality / FV3_QUALITY）通过 configs/profile_<quality>.yaml overlay 生效。
    """
    dotenv_vals = dotenv_values(dotenv_path or PROJECT_ROOT / ".env")

    yml = _load_yaml(Path(config_path) if config_path else CONFIGS_DIR / "default.yaml")
    quality = (
        quality
        or os.getenv("FV3_QUALITY")
        or dotenv_vals.get("FV3_QUALITY")  # type: ignore[assignment]
        or (yml.get("generation") or {}).get("quality", "economy")
    )
    if quality not in ("", "economy"):
        overlay = _load_yaml(CONFIGS_DIR / f"profile_{quality}.yaml")
        if overlay:
            _deep_merge(yml, overlay)
    gen = yml.get("generation", {})
    limits = yml.get("limits", {})
    routing = {cap: ProviderRouting(**route) for cap, route in (yml.get("providers") or {}).items()}

    def env(key: str, default: str = "") -> str:
        v = os.getenv(key)
        if not v:
            v = dotenv_vals.get(key)  # type: ignore[assignment]
        return v or default

    resolution = tuple(gen.get("resolution", [540, 960]))

    return AppConfig(
        deepseek_api_key=env("DEEPSEEK_API_KEY"),
        deepseek_base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        dashscope_api_key=env("DASHSCOPE_API_KEY"),
        dashscope_text_base_url=env(
            "DASHSCOPE_TEXT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        enable_thinking=env("FV3_ENABLE_THINKING", "0") in ("1", "true", "yes"),
        siliconflow_api_key=env("SILICONFLOW_API_KEY"),
        kling_access_key=env("KLING_ACCESS_KEY"),
        kling_secret_key=env("KLING_SECRET_KEY"),
        profile=env("FV3_PROFILE", "edu-sqlite"),
        review_mode=env("FV3_REVIEW_MODE", gen.get("review_mode", "auto")),
        talk_mode=env("FV3_TALK_MODE", gen.get("talk_mode", "s2v_direct")),
        fake=env("FV3_FAKE", "0") in ("1", "true", "yes"),
        budget_limit_cny=float(env("FV3_BUDGET_LIMIT_CNY", "0") or 150),
        max_parallel_jobs=int(limits.get("max_parallel_jobs", 4)),
        per_provider_concurrency=limits.get("per_provider_concurrency", {}),
        i2v_poll_timeout_min=int(limits.get("i2v_poll_timeout_min", 30)),
        retry_per_shot=int(limits.get("retry_per_shot", 2)),
        aspect_ratio=gen.get("aspect_ratio", "9:16"),
        shot_duration_sec=float(gen.get("shot_duration_sec", 5)),
        resolution=(int(resolution[0]), int(resolution[1])),  # type: ignore[index]
        storyboard_sample_rate=float(gen.get("storyboard_sample_rate", 0.5)),
        max_script_revisions=int(gen.get("max_script_revisions", 2)),
        quality=quality or "economy",
        use_r2v_for_characters=bool(gen.get("use_r2v_for_characters", False)),
        use_reference_keyframes=bool(gen.get("use_reference_keyframes", False)),
        candidates_per_shot=int(gen.get("candidates_per_shot", 1)),
        asset_candidates=int(gen.get("asset_candidates", 1)),
        script_review=bool(gen.get("script_review", False)),
        auto_reshoot=bool(gen.get("auto_reshoot", False)),
        i2v_resolution=gen.get("i2v_resolution", "480P"),
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
        pg_host=env("FV3_PG_HOST", "localhost"),
        pg_port=int(env("FV3_PG_PORT", "5432") or 5432),
        pg_db=env("FV3_PG_DB", "fastvideo3"),
        pg_user=env("FV3_PG_USER", "fv3"),
        pg_password=env("FV3_PG_PASSWORD", ""),
    )


@lru_cache
def get_config() -> AppConfig:
    """全局缓存的配置实例（测试中用 load_config() 直接构建）。"""
    return load_config()
