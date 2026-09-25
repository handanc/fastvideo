"""对口型云端 API 真实冒烟（nightly，需真实 KEY + 模型已开通）。

运行：pytest -m api
前置：百炼控制台开通 wan2.2-s2v / videoretalk，且账户余额正常；
未开通时提交会抛 AccountGateError（m6 会自动降级，不阻塞成片）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastvideo3.core.config import load_config
from fastvideo3.providers import dashscope
from fastvideo3.providers.base import poll_until_done

pytestmark = pytest.mark.api


def _media() -> tuple[Path, Path, Path]:
    """复用最近一次 run 的首帧/音频/片段作为输入（无 run 时跳过）。"""
    from fastvideo3.core.config import OUTPUT_DIR

    runs = sorted(
        (p for p in OUTPUT_DIR.glob("run_*") if (p / "audio_map.json").exists()), reverse=True
    )
    if not runs:
        pytest.skip("无历史 run 产物可复用")
    import json

    run = runs[0]
    line = json.loads((run / "audio_map.json").read_text(encoding="utf-8"))[0]
    wav = Path(line["wav_uri"])
    frame = run / "storyboard" / f"shot_{line['shot_id']}.png"
    clip = run / "clips" / f"shot_{line['shot_id']}.mp4"
    if not (wav.exists() and frame.exists()):
        pytest.skip("历史产物不全")
    return wav, frame, clip


async def test_s2v_lipsync_via_oss_upload(tmp_path):
    """wan2.2-s2v：oss 直传 + OssResourceResolve 头 → 真口型视频。"""
    cfg = load_config()
    wav, frame, _ = _media()
    talk = dashscope.DashScopeTalkProvider(cfg.dashscope_api_key, cfg.models["dashscope_s2v"])
    job = await talk.submit(dest=tmp_path / "s2v.mp4", image=frame, audio=wav)
    st = await poll_until_done(talk.poll, job, timeout_sec=900, fast_interval=5, slow_interval=15)
    assert st.status == "succeeded" and st.output_uri and st.output_uri.stat().st_size > 10_000


async def test_videoretalk_lipsync_via_oss_upload(tmp_path):
    """videoretalk：I2V 片段 + 音频 → 口型替换（原 FastVideo 对口型阶段的云端版）。"""
    cfg = load_config()
    wav, _, clip = _media()
    if not clip.exists():
        pytest.skip("无 I2V 片段可复用")
    talk = dashscope.DashScopeRetalkProvider(cfg.dashscope_api_key, cfg.models["dashscope_retalk"])
    job = await talk.submit(dest=tmp_path / "retalk.mp4", video=clip, audio=wav)
    st = await poll_until_done(talk.poll, job, timeout_sec=900, fast_interval=5, slow_interval=15)
    assert st.status == "succeeded" and st.output_uri and st.output_uri.stat().st_size > 10_000
