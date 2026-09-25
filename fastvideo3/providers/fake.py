"""Fake Provider 全家桶：零成本离线跑通全流程（TDD §1.2 / PRD 术语表）。

教学定位：学生第一次运行、单元测试、CI 冒烟全部走 Fake 模式；
理解流水线后再切换真实供应商 —— 这正是 Provider 模式的价值。
"""

from __future__ import annotations

import hashlib
import json
import uuid
import wave
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fastvideo3.core.schema import BaseInfo, ReviewResult, Script
from fastvideo3.providers.base import (
    I2VProvider,
    ImageResult,
    T2IProvider,
    TalkProvider,
    TextProvider,
    TTSProvider,
    VideoJobStatus,
    VLMProvider,
)
from fastvideo3.utils.ffmpeg import image_to_video
from fastvideo3.utils.image import make_placeholder_image

# 调色板：不同 prompt 生成不同颜色，便于肉眼区分"资产"
_PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#355070"]


def _color_of(text: str) -> str:
    return _PALETTE[int(hashlib.md5(text.encode()).hexdigest(), 16) % len(_PALETTE)]


# ───────────────────────── 文本（剧本/评审） ─────────────────────────
def fake_script(base: BaseInfo) -> Script:
    """确定性地构造一个结构合法的小剧本：3 场景、对话/动作/风景俱全。"""
    topic = base.topic or "测试主题"
    char_a = {
        "character_id": "c1",
        "name": "小明",
        "appearance": "25岁短发男生，白衬衫牛仔裤",
        "voice_id": "zh-CN-YunxiNeural",
    }
    char_b = {
        "character_id": "c2",
        "name": "小红",
        "appearance": "24岁长发女生，红色连衣裙",
        "voice_id": "zh-CN-XiaoxiaoNeural",
    }

    def shot(sid: str, scene: str, stype: str, action: str, dialogues=None, trans="cut"):
        return {
            "shot_id": sid,
            "scene_id": scene,
            "type": stype,
            "duration_sec": base.target_duration_sec and 5.0,
            "characters": [d["character_id"] for d in (dialogues or [])] or ["c1"],
            "action": action,
            "camera": "中景",
            "dialogues": dialogues or [],
            "transition_out": trans,
        }

    scenes = [
        {
            "scene_id": "s1",
            "title": "开场",
            "location": "城市街道",
            "time_of_day": "日",
            "summary": f"{topic}：故事开始",
            "shots": [
                shot("s1_01", "s1", "scenery", "清晨的城市街道空镜，阳光洒落", trans="dissolve"),
                shot(
                    "s1_02",
                    "s1",
                    "dialogue",
                    "小明边走边打电话，表情兴奋",
                    [
                        {
                            "character_id": "c1",
                            "text": f"关于{topic}，我有个大计划！",
                            "emotion": "happy",
                        }
                    ],
                ),
            ],
        },
        {
            "scene_id": "s2",
            "title": "冲突",
            "location": "咖啡馆",
            "time_of_day": "日",
            "summary": "两人相遇，观点碰撞",
            "shots": [
                shot(
                    "s2_01",
                    "s2",
                    "dialogue",
                    "小红坐在窗边，抬头看向小明",
                    [
                        {
                            "character_id": "c2",
                            "text": "你这计划听起来就不靠谱。",
                            "emotion": "doubt",
                        }
                    ],
                    "cut",
                ),
                shot("s2_02", "s2", "action", "小明手舞足蹈地比划解释", [], "dissolve"),
            ],
        },
        {
            "scene_id": "s3",
            "title": "解决",
            "location": "天台",
            "time_of_day": "黄昏",
            "summary": "达成共识，望向远方",
            "shots": [
                shot(
                    "s3_01",
                    "s3",
                    "dialogue",
                    "两人并肩站立，小红微笑着点头",
                    [{"character_id": "c2", "text": "好吧，那就一起试试！", "emotion": "happy"}],
                ),
                shot("s3_02", "s3", "scenery", "黄昏的城市天际线，镜头缓缓拉远", trans="fade"),
            ],
        },
    ]
    return Script.model_validate(
        {
            "base_info": base.model_dump(),
            "title": base.topic or "测试短剧",
            "characters": [char_a, char_b],
            "scenes": scenes,
        }
    )


class FakeTextProvider(TextProvider):
    """按请求的 json_schema 返回合法样本；支持注入脚本化回复（测试评审回路用）。"""

    name = "fake_text"
    model = "fake-llm"

    def __init__(self, scripted: list[str] | None = None, base_info: BaseInfo | None = None):
        self.scripted = list(scripted or [])
        self.base_info = base_info

    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        if self.scripted:
            return self.scripted.pop(0)
        if json_schema is Script:
            base = self.base_info or BaseInfo(topic="测试主题")
            return fake_script(base).model_dump_json()
        if json_schema is ReviewResult:
            return ReviewResult(
                passed=True, score=0.9, feedback="Fake 评审通过", rubric="fake"
            ).model_dump_json()
        if json_schema is not None:
            return json.dumps(_sample_model(json_schema), ensure_ascii=False)
        return "这是 Fake 文本模型的回复。"


def _sample_model(model: type[BaseModel]) -> dict:
    """为任意 Pydantic 模型递归构造合法样本（通用兜底）。"""
    from typing import Literal, get_args, get_origin

    def value_for(field_type: Any) -> Any:
        from enum import Enum as _Enum

        origin = get_origin(field_type)
        if field_type is str:
            return "示例"
        if field_type is int:
            return 1
        if field_type is float:
            return 1.0
        if field_type is bool:
            return True
        if isinstance(field_type, type) and issubclass(field_type, _Enum):
            return list(field_type)[0].value
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return _sample_model(field_type)
        args = get_args(field_type)
        if origin is Literal:
            return args[0]
        if origin is list and args:
            return [value_for(args[0])]
        if origin is dict:
            return {}
        if str(field_type).startswith("typing.Union") or field_type is type(None):  # X | None
            inner = [a for a in args if a is not type(None)]
            return value_for(inner[0]) if inner else None
        return "示例"

    out = {}
    for name, field in model.model_fields.items():
        if not field.is_required():
            if field.default_factory is not None:
                out[name] = field.default_factory()  # type: ignore[call-arg]
            else:
                out[name] = field.default
            continue
        out[name] = value_for(field.annotation)
    return out


# ───────────────────────── 图像 ─────────────────────────
class FakeT2IProvider(T2IProvider):
    name = "fake_t2i"
    model = "fake-t2i"

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first  # 测试重试用

    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: Path | None = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        if self.fail_first > 0:
            self.fail_first -= 1
            from fastvideo3.providers.base import TransientAPIError

            raise TransientAPIError("fake transient failure")
        dest.parent.mkdir(parents=True, exist_ok=True)
        make_placeholder_image(dest, label=prompt[:16], color=_color_of(prompt))
        return ImageResult(uri=dest, meta={"prompt": prompt, "fake": True})


# ───────────────────────── 视频（I2V / Talk） ─────────────────────────
class _FakeVideoJobMixin:
    """内存任务表：submit 立即出片（由首帧图做 5 秒动图）。"""

    def __init__(self) -> None:
        self._jobs: dict[str, tuple[Path, Path]] = {}  # job_id -> (dest, source_image)
        self._fail_first = 0

    def _fail_if_needed(self) -> None:
        if self._fail_first > 0:
            self._fail_first -= 1
            from fastvideo3.providers.base import TransientAPIError

            raise TransientAPIError("fake transient failure")

    async def _poll_impl(self, job_id: str) -> VideoJobStatus:
        dest, _ = self._jobs[job_id]
        return VideoJobStatus(status="succeeded", progress=1.0, output_uri=dest)


class FakeI2VProvider(I2VProvider, _FakeVideoJobMixin):
    name = "fake_i2v"
    model = "fake-i2v"

    def __init__(self, fail_first: int = 0):
        super().__init__()
        self._fail_first = fail_first

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, prompt: str = "", **params: Any
    ) -> str:
        self._fail_if_needed()
        job_id = uuid.uuid4().hex[:12]
        dest.parent.mkdir(parents=True, exist_ok=True)
        image_to_video(image, dest, duration=2.0)  # Fake 用 2 秒片段, 加快冒烟
        self._jobs[job_id] = (dest, image)
        return job_id

    async def poll(self, job_id: str) -> VideoJobStatus:
        return await self._poll_impl(job_id)


class FakeTalkProvider(TalkProvider, _FakeVideoJobMixin):
    name = "fake_talk"
    model = "fake-talk"

    def __init__(self, fail_first: int = 0):
        super().__init__()
        self._fail_first = fail_first

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, audio: Path, **params: Any
    ) -> str:
        self._fail_if_needed()
        job_id = uuid.uuid4().hex[:12]
        dest.parent.mkdir(parents=True, exist_ok=True)
        image_to_video(image, dest, duration=2.0, audio=audio)
        self._jobs[job_id] = (dest, image)
        return job_id

    async def poll(self, job_id: str) -> VideoJobStatus:
        return await self._poll_impl(job_id)


# ───────────────────────── 语音 ─────────────────────────
class FakeTTSProvider(TTSProvider):
    name = "fake_tts"
    model = "fake-tts"

    async def synthesize(
        self, text: str, *, voice: str = "", emotion: str = "", dest: Path
    ) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        duration = max(1.0, min(4.5, len(text) * 0.18))  # 每字约 0.18s
        _write_silence_wav(dest, duration)
        return dest


def _write_silence_wav(path: Path, duration: float, sr: int = 16000) -> None:
    import struct

    n = int(duration * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<h", 0) * n)


# ───────────────────────── 视觉评审 ─────────────────────────
class FakeVLMProvider(VLMProvider):
    name = "fake_vlm"
    model = "fake-vlm"

    def __init__(self, always_pass: bool = True):
        self.always_pass = always_pass

    async def review(
        self, media: list[Path], *, rubric: str, dest: Path | None = None
    ) -> ReviewResult:
        return ReviewResult(
            target=media[0].name if media else "",
            passed=self.always_pass,
            score=0.9 if self.always_pass else 0.3,
            feedback="Fake VLM 评审",
            rubric=rubric,
        )
