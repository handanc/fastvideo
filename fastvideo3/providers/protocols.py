"""模型名 → 调用协议 → provider 实现 的解析表。

为什么需要这一层：`configs/*.yaml` 的 `models:` 只把"provider 键 → 模型名"绑定，
而 provider 键同时**决定了 HTTP 端点与请求体协议**。所以在 .env 里把 t2i 模型改成
`qwen-image-3.0` 而路由仍是 `dashscope_t2i`（万相异步 text2image 端点）时，请求会
打到错误的端点直接 400 —— 模型名与协议必须一起解析。

三级解析优先级（高 → 低）：
1. `configs/model_protocols.yaml`（`scripts/probe_models.py` 的实测结论回填处）
2. 环境变量 `FV3_T2I_PROTOCOL` / `FV3_I2V_PROTOCOL` / `FV3_TALK_PROTOCOL`
3. 内置命名模式表（按百炼现有模型命名习惯推断）

推断不出（None）时**保持路由里写的 provider 不变**，不做猜测性改道。
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

import yaml

from fastvideo3.core.config import CONFIGS_DIR

# ── 协议标识 ──
T2I_ASYNC = "async_text2image"        # /services/aigc/text2image/image-synthesis（异步任务）
T2I_MULTIMODAL = "multimodal_sync"    # /services/aigc/multimodal-generation/generation（同步 messages）
I2V_LEGACY = "i2v_legacy"             # video-synthesis + input.img_url（wan2.2 系）
I2V_MEDIA = "i2v_media_array"         # video-synthesis + input.media[{type:first_frame|reference_image|driving_audio}]
TALK_S2V = "s2v"                      # image_url + audio_url 数字人口型
TALK_RETALK = "videoretalk"           # video_url + audio_url 重对口型
TALK_MEDIA_AUDIO = "i2v_driving_audio"  # media[{first_frame},{driving_audio}]（**仅 wan2.7 系**）
# wan3.0 的口型**不吃音频**：官方 media 枚举里没有 driving_audio，台词写在提示词里由模型自己
# 发声（"提示词重点描述动态过程、剧情、台词与运镜"）。把它当 driving_audio 发会得到真实 400，
# 项目里存着原文（tests/test_talk_fallback.py）—— 所以这条必须是独立协议，不能复用上一条。
TALK_SCRIPTED = "wan30_scripted_talk"  # media[{first_frame(,last_frame)}] + 台词进 prompt
TALK_FAL_I2V = "fal_infinitalk_image"  # fal.ai：image_url + audio_url + prompt → 说话视频
TALK_FAL_V2V = "fal_infinitalk_v2v"    # fal.ai：video_url + audio_url + prompt → 重对口型
# 文本这一族两家都是 OpenAI 兼容 `/chat/completions`，但**域名与 KEY 不同**。
# 2026-10-14 事故：面板点名 `deepseek-v4-flash-0731`，而路由是 `dashscope_qwen` →
# 这个名字被发到百炼端点，连上但不返回，180s × 3 次重试 = 9 分钟后才报 ReadTimeout。
# 名字里就带着它属于哪家，所以让协议表把它配对到正确的 provider，而不是等 API 超时来告诉你。
TEXT_OPENAI_DEEPSEEK = "openai_compat_deepseek"    # https://api.deepseek.com
TEXT_OPENAI_DASHSCOPE = "openai_compat_dashscope"  # 百炼 OpenAI 兼容端点

# (模型名 glob, 能力, 协议)：顺序敏感，越具体越靠前
_PATTERNS: list[tuple[str, str, str]] = [
    ("deepseek*", "text", TEXT_OPENAI_DEEPSEEK),
    ("qwen*", "text", TEXT_OPENAI_DASHSCOPE),
    ("qwen-image*", "t2i", T2I_MULTIMODAL),
    ("wan2.7-image*", "t2i", T2I_MULTIMODAL),
    ("wan2.6-t2i*", "t2i", T2I_ASYNC),
    ("wanx2.1-t2i*", "t2i", T2I_ASYNC),
    ("wan3.0-video-prime*", "i2v", I2V_MEDIA),
    ("wan3.0-video*", "i2v", I2V_MEDIA),
    ("wan2.7-i2v*", "i2v", I2V_MEDIA),
    ("wan2.7-r2v*", "i2v", I2V_MEDIA),
    ("wan2.2-i2v*", "i2v", I2V_LEGACY),
    ("wan2.2-s2v*", "talk", TALK_S2V),
    ("videoretalk*", "talk", TALK_RETALK),
    # fal.ai 托管的 InfiniteTalk（端点 ID 就是模型名）。*video-to-video 必须排在
    # *infinitalk* 之前：模式表顺序敏感，越具体越靠前，否则 v2v 端点会被判成 i2v。
    ("*infinitalk*video-to-video*", "talk", TALK_FAL_V2V),
    ("*infinitalk*", "talk", TALK_FAL_I2V),
    # wan3.0：台词写进提示词、模型自发声（**没有 driving_audio**，别按 2.7 那套发）
    ("wan3.0-video*", "talk", TALK_SCRIPTED),
    # wan2.7-i2v 才有 media[].type=driving_audio 这条原生音频驱动口型
    ("wan2.7-i2v*", "talk", TALK_MEDIA_AUDIO),
]

# (能力, 协议) → factory 里的 provider 名
_PROVIDER_FOR: dict[tuple[str, str], str] = {
    ("text", TEXT_OPENAI_DEEPSEEK): "deepseek",
    ("text", TEXT_OPENAI_DASHSCOPE): "dashscope_qwen",
    ("t2i", T2I_ASYNC): "dashscope_t2i",
    ("t2i", T2I_MULTIMODAL): "dashscope_wan27_image",
    ("i2v", I2V_LEGACY): "dashscope_wan22_i2v",
    ("i2v", I2V_MEDIA): "dashscope_wan27_i2v",
    ("talk", TALK_S2V): "dashscope_wan22_s2v",
    ("talk", TALK_RETALK): "dashscope_videoretalk",
    ("talk", TALK_MEDIA_AUDIO): "dashscope_wan27_audio",
    ("talk", TALK_SCRIPTED): "dashscope_wan30_talk",
    # 同一个 provider 类内部按输入形态选端点（有 video→v2v，只有 image→i2v），
    # 所以两种协议都映射到 fal_infinitalk：改道不会改变行为，只是把端点名认对。
    ("talk", TALK_FAL_I2V): "fal_infinitalk",
    ("talk", TALK_FAL_V2V): "fal_infinitalk",
}

_ENV_KEYS = {
    "text": "FV3_TEXT_PROTOCOL",
    "t2i": "FV3_T2I_PROTOCOL",
    "i2v": "FV3_I2V_PROTOCOL",
    "r2v": "FV3_R2V_PROTOCOL",
    "talk": "FV3_TALK_PROTOCOL",
}


def _probe_table(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIGS_DIR / "model_protocols.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001  # 表坏了退回推断，不阻塞运行
        return {}


def infer_protocol(
    model: str, capability: str, *, env: dict[str, str] | None = None, probe: dict | None = None
) -> str | None:
    """解析模型应走的协议；未知返回 None（调用方保持原路由，不猜）。"""
    if not model:
        return None
    probe = probe if probe is not None else _probe_table()
    # 1) 探测结论（按 "capability/model" 或裸模型名两种键）
    for key in (f"{capability}/{model}", model):
        hit = (probe.get("protocols") or {}).get(key) or probe.get(key)
        if isinstance(hit, str) and hit:
            return hit
    # 2) 环境变量显式声明
    from_env = (env if env is not None else os.environ).get(_ENV_KEYS.get(capability, ""))
    if from_env:
        return from_env.strip()
    # 3) 内置命名模式
    for pattern, cap, protocol in _PATTERNS:
        if cap == capability and fnmatch.fnmatch(model.lower(), pattern.lower()):
            return protocol
    return None


def provider_for(capability: str, protocol: str | None) -> str | None:
    """协议 → 应使用的 provider 名（None 表示不改道）。"""
    if not protocol:
        return None
    if capability == "r2v":  # 参考生视频只有 media 数组一种形态
        return "dashscope_wan27_r2v"
    return _PROVIDER_FOR.get((capability, protocol))


def known_protocols(capability: str) -> list[str]:
    return sorted({p for (c, p) in _PROVIDER_FOR if c == capability})


def describe(model: str, capability: str) -> str:
    """给 doctor / 探测脚本用的一行人话说明。"""
    proto = infer_protocol(model, capability)
    if proto is None:
        return f"{model}: 协议未知（保持 yaml 路由的 provider）"
    return f"{model}: {proto} → {provider_for(capability, proto)}"
