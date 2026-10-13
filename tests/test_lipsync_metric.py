"""口型度量必须自证"量到了嘴"，否则一律无结论（不许拿噪声触发付费重做）。

回归背景（2026-10-03 实测）：旧实现用写死的比例框（画面下缘中心），在真实分镜
（特写/过肩/全身混合）下根本不在嘴上 —— 同一素材三区运动幅度为 嘴 0.8 / 整脸 4.4 /
背景 4.2，框落在静止区域。用这把尺子判过的 "wan2.7-i2v 没有口型" 是不成立的结论，
而 verdict 一旦进 reshoot 回路就是拿噪声决定重做（每次几元到几十元）。

本文件锁住三件事：
1. 取样框由视觉模型定位，定位不到 → verdict=inconclusive，**不标 stale**；
2. 即使定位到了，嘴区运动幅度不显著高于眼区对照 → 仍判无结论（自我怀疑优先）；
3. 口型通道的官方硬规格（wan2.2-s2v 只到 720P / 音频 <20s）在提交前钳位或止损，
   不再让"配置越界"看起来像"模型不支持对口型"。
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.schema import RegionBoxes
from fastvideo3.utils import lipsync as ls

FPS = 6
N_FRAMES = 48
W, H = 64, 96


def _write_video(path: Path, on: list[bool], movers) -> None:
    """在 movers 指定的区域内做逐帧明暗跳变（=该区域在动），其余保持静止。

    movers 可以是单个框，或 [(框, 幅度), ...]：幅度用来构造"嘴几乎不动、别处狂动"
    这种真实误框形态（皮尔逊与尺度无关，那种情况下 corr 会**很高**，只能靠
    运动幅度对照识破 —— 见 lipsync.AMP_DOMINANCE）。
    """
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    if isinstance(movers, tuple):
        movers = [(movers, 140)]

    d = path.parent / f"_src_{path.stem}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        im = Image.new("L", (W, H), 120)
        px = im.load()
        for box, span in movers:
            x0, y0, x1, y1 = (int(W * box[0]), int(H * box[1]),
                              int(W * box[2]), int(H * box[3]))
            # 静止 = 恒定 128（帧间差 0）；运动 = 在 span 幅度内跳变 →
            # 运动幅度由 span 唯一控制（早先写成 40↔128 的明暗翻转，span 就被架空了）
            level = 128 if not on[i] else min(255, 100 + (i * 37) % span)
            for y in range(y0, max(y1, y0 + 1)):
                for x in range(x0, max(x1, x0 + 1)):
                    px[x, y] = level
        im.convert("RGB").save(d / f"f_{i:03d}.png")
    run_ffmpeg(
        ["-framerate", str(FPS), "-i", str(d / "f_%03d.png"),
         "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)]
    )


def _write_wav(path: Path, on: list[bool]) -> None:
    rate = 16000
    per = rate // FPS
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b""
        for i, is_on in enumerate(on):
            amp = 9000 if is_on else 0
            frames += b"".join(
                int(amp * math.sin(2 * math.pi * 220 * t / rate)).to_bytes(2, "little",
                                                                           signed=True)
                for t in range(per)
            )
        w.writeframes(frames)


ON = [i % 8 < 4 for i in range(N_FRAMES)]          # 说话/停顿交替
SHIFT = [i % 8 < 4 for i in range(4, N_FRAMES + 4)]  # 与画面运动错开 4 格


class StubVLM:
    """只回定位框的视觉模型桩：review_json 按 schema 严格构造（与真 provider 同语义）。"""

    name = "stub_vlm"
    model = "stub"

    def __init__(self, boxes: dict | None) -> None:
        self.boxes = boxes
        self.calls = 0

    async def review_json(self, media, *, schema, rubric, dest=None):
        self.calls += 1
        assert "嘴" in rubric, "定位 rubric 必须真的点名嘴部（模板没渲染出来就是空跑）"
        if self.boxes is None:
            raise ValueError("视觉模型不配合")
        return schema(**self.boxes)


def _lips_at(moving_box: tuple[float, ...]) -> dict:
    return {
        "has_person": True, "lips": tuple(moving_box),
        "eyes": (0.34, 0.20, 0.66, 0.32), "confidence": 0.9, "reason": "测试桩",
    }


# ── 1. 定位框 schema 自身的健壮性（模型常回像素制/反序/百分比） ──

def test_region_boxes_normalizes_common_model_outputs():
    b = RegionBoxes(has_person=True, lips=[320, 580, 680, 800],  # 0~1000 像素制
                    eyes=["34%", "40%", 0.66, 0.52], confidence=90, reason="ok")
    assert b.lips == (0.32, 0.58, 0.68, 0.80)
    assert b.eyes[0] == pytest.approx(0.34) and b.eyes[3] == pytest.approx(0.52)
    assert b.confidence == 0.9
    assert b.sane()


def test_region_boxes_rejects_useless_boxes():
    assert not RegionBoxes(has_person=False, lips=(0, 0, 0, 0), eyes=(0, 0, 0, 0),
                           confidence=0.1, reason="没人").sane()
    # 眼框 == 嘴框 → 对照失效
    assert not RegionBoxes(has_person=True, lips=(0.3, 0.6, 0.6, 0.75),
                           eyes=(0.3, 0.6, 0.6, 0.75), confidence=0.9, reason="抄的").sane()
    # 占满全画面 = 没定位
    assert not RegionBoxes(has_person=True, lips=(0.0, 0.0, 1.0, 1.0),
                           eyes=(0.3, 0.4, 0.6, 0.5), confidence=0.9, reason="整图").sane()


# ── 2. 定位成功时才允许出结论 ──

def test_diag_synced_when_box_actually_on_moving_mouth(tmp_path):
    v = tmp_path / "sync.mp4"
    _write_video(v, ON, ls.MOUTH_BOX)
    w = tmp_path / "sync.wav"
    _write_wav(w, ON)
    boxes = RegionBoxes(**_lips_at(ls.MOUTH_BOX))
    d = ls.lipsync_diag(v, w, tmp_path / "wd", boxes=boxes, min_corr=0.25)
    assert d["located"] is True
    assert d["verdict"] == "synced", d
    assert d["corr"] > 0.3


def test_diag_no_conclusion_without_localization(tmp_path):
    """没定位（无 vlm / 定位失败）→ 必须 inconclusive，且不得伪造一个像结论的数字。"""
    v = tmp_path / "sync.mp4"
    _write_video(v, ON, ls.MOUTH_BOX)
    w = tmp_path / "sync.wav"
    _write_wav(w, ON)
    d = ls.lipsync_diag(v, w, tmp_path / "wn", boxes=None, min_corr=0.25)
    assert d["located"] is False and d["verdict"] == "inconclusive"
    assert "未定位" in d["reason"]


def test_diag_refuses_when_mouth_box_is_not_where_motion_is(tmp_path):
    """框标错了地方（运动在别处）→ 无结论。

    这正是旧实现的行为：写死框落在静止区，运动幅度 0.8 而整脸 4.4，
    当时它照样给了一个 0.24 的"分数"。现在必须自证失败。
    """
    moving = (0.70, 0.10, 0.95, 0.30)  # 画面右上角在动（不是嘴）
    v = tmp_path / "mis.mp4"
    _write_video(v, ON, moving)
    w = tmp_path / "mis.wav"
    _write_wav(w, ON)
    boxes = RegionBoxes(**_lips_at(ls.MOUTH_BOX))  # 声称嘴在下缘中心，其实那儿不动
    d = ls.lipsync_diag(v, w, tmp_path / "wm", boxes=boxes, min_corr=0.25)
    assert d["verdict"] == "inconclusive", d
    assert "信号不足" in d["reason"] or "幅度" in d["reason"], d["reason"]


def test_diag_refuses_when_sampled_region_barely_moves(tmp_path):
    """定位框没落在真正会动的嘴上 → 无结论，**不是**"不同步"。

    这是最贵的一种误判：框取样的区域只有编码噪声在动（幅度 7 vs 眼区几十），
    corr 自然低，按老逻辑就是"口型不同步 → 付费重做"，而镜头本身可能完全没问题。
    运动幅度对照（AMP_DOMINANCE）专门拦这一类：先证明量到了嘴，再谈同不同步。
    """
    lips_area = (0.32, 0.58, 0.68, 0.80)
    eyes_area = (0.34, 0.20, 0.66, 0.32)
    v = tmp_path / "weak.mp4"
    _write_video(v, ON, [(lips_area, 8), (eyes_area, 150)])  # 嘴区几乎不动，眼区狂动
    w = tmp_path / "weak.wav"
    _write_wav(w, ON)
    boxes = RegionBoxes(has_person=True, lips=lips_area, eyes=eyes_area,
                        confidence=0.9, reason="测试桩")
    d = ls.lipsync_diag(v, w, tmp_path / "ww", boxes=boxes, min_corr=0.25)
    assert d["corr"] is not None and d["corr"] < 0.25, d  # 单看 corr 会误判成"不同步"
    assert (d["mouth_amp"] or 0) < (d["eyes_amp"] or 0), d
    assert d["verdict"] == "inconclusive" and "幅度" in d["reason"], d


def test_diag_unsynced_when_motion_and_audio_misaligned(tmp_path):
    v = tmp_path / "desync.mp4"
    _write_video(v, ON, ls.MOUTH_BOX)
    w = tmp_path / "desync.wav"
    _write_wav(w, SHIFT)
    boxes = RegionBoxes(**_lips_at(ls.MOUTH_BOX))
    d = ls.lipsync_diag(v, w, tmp_path / "wds", boxes=boxes, min_corr=0.3)
    assert d["verdict"] == "unsynced", d


# ── 3. m6 侧：只有"定位到 + 判不同步"才标 stale（省钱安全线） ──

@pytest.mark.asyncio
async def test_m6_reports_lipsync_but_never_marks_stale(cfg, storage):
    """口型度量**只报告、不标 stale**：它在真实素材上会假阴性，据此重做就是烧钱重抽合格镜头。

    实测依据（同一条 videoretalk 产物）：人眼对着逐帧响度条确认嘴随语音开合，
    而度量只给 原始 corr 0.163 / 偏相关 0.113（阈值 0.25）→ 会被判"不同步"。
    假阴性比假阳性贵得多（反复重抽），所以自动重做回路里不留口型这条判据。
    """
    from fastvideo3.modules.m6_talk import _lipsync_check

    on = [i % 8 < 4 for i in range(N_FRAMES)]
    shifted = [i % 8 < 4 for i in range(4, N_FRAMES + 4)]
    cfg.lipsync_min_corr = 0.3

    # 定位到了嘴、且与音频错开 → 仍然只记录
    clip = storage.path("talk", "shot_s1_01.mp4")
    _write_video(clip, on, ls.MOUTH_BOX)
    wav = storage.path("audio", "shot_s1_01_mix.wav")
    _write_wav(wav, shifted)
    d = await _lipsync_check(cfg, storage, "s1_01", clip, wav, method="lipsync",
                             vlm=StubVLM(_lips_at(ls.MOUTH_BOX)))
    assert d["located"] is True and d["corr"] is not None
    assert not storage.is_stale("talk/shot_s1_01.mp4"), "度量不得自动标 stale（会触发付费重做）"
    ev = [e for e in storage.read_events() if e.get("status") == "lipsync_check"]
    assert ev and "verdict" in ev[-1] and "pcorr" in ev[-1], "事件要留下可审计的判定与统计量"
    assert any(e.get("status") == "lipsync_report" for e in storage.read_events())

    # 无 vlm 定位 → 无结论，同样不标 stale
    clip2 = storage.path("talk", "shot_s1_02.mp4")
    _write_video(clip2, on, ls.MOUTH_BOX)
    wav2 = storage.path("audio", "shot_s1_02_mix.wav")
    _write_wav(wav2, shifted)
    d2 = await _lipsync_check(cfg, storage, "s1_02", clip2, wav2, method="lipsync", vlm=None)
    assert d2["verdict"] == "inconclusive"
    assert not storage.is_stale("talk/shot_s1_02.mp4")

    # 叠音降级：只记录
    clip3 = storage.path("clips", "shot_s1_03.mp4")
    _write_video(clip3, on, ls.MOUTH_BOX)
    wav3 = storage.path("audio", "shot_s1_03_mix.wav")
    _write_wav(wav3, on)
    await _lipsync_check(cfg, storage, "s1_03", clip3, wav3, method="overlay")
    assert not storage.is_stale("clips/shot_s1_03.mp4")


def test_lipsync_never_enters_auto_reshoot_gate(cfg):
    """闸门里口型一律不触发重做（含 located+unsynced），但脸/衣不一致照旧触发。"""
    from fastvideo3.modules.m9_metrics import gate_failures

    cfg.consistency_gate = "reshoot"
    cfg.lipsync_min_corr = 0.25
    row = lambda extra, **kw: dict({"shot": "s1_01", "verified": True, "face": True,
                                    "outfit": True, "hue_dist": 10.0}, **extra, **kw)
    # 口型被判"不同步"且已定位 → 仍不进重做清单
    assert gate_failures({"per_shot": [row({"lipsync": {"method": "lipsync", "corr": 0.02,
                                  "located": True, "verdict": "unsynced"}})]}, cfg) == []
    # 未定位 → 更不该进
    assert gate_failures({"per_shot": [row({"lipsync": {"method": "lipsync", "corr": 0.02}})]},
                         cfg) == []
    # 但人脸漂移必须仍然被拦（别把这条一起改坏了）
    bad = gate_failures({"per_shot": [row({"face": False, "reason": "脸漂移"},
                                          lipsync=None)]}, cfg)
    assert len(bad) == 1 and bad[0]["face"] is False


# ── 4. 免模型滑窗扫描：让数据自己指出"有没有随语音运动的块" ──

def test_scan_finds_the_box_that_moves_with_speech(tmp_path):
    """扫描能证明"存在随语音运动的区域"，且该区域与真区重合（宏块级精度）。"""
    v = tmp_path / "scan_sync.mp4"
    truth = (0.30, 0.55, 0.52, 0.68)
    _write_video(v, ON, truth)
    w = tmp_path / "scan_sync.wav"
    _write_wav(w, ON)
    d = ls.scan_lipsync(v, w, tmp_path / "ws", min_corr=0.25)
    assert d["verdict"] == "synced", d
    assert d["corr"] > d["null_corr"], d  # 必须显著高于乱序对照
    bx = d["best_box"]
    # 与真区有实际重叠即可：h.264 以 16×16 宏块编码，小画面(64×96)下定位粒度
    # 天生只有约 1/4 画面宽 —— 所以扫描回答"有没有口型证据"，精确定位仍靠 VLM。
    ox = min(bx[2], truth[2]) - max(bx[0], truth[0])
    oy = min(bx[3], truth[3]) - max(bx[1], truth[1])
    assert ox > 0 and oy > 0, f"扫描框 {bx} 与真区 {truth} 不相交"


def test_scan_refuses_to_call_camera_motion_lipsync(tmp_path):
    """全程等速运动（无语音节律）→ 不许出"synced"，且必须给出零分布对照。"""
    v = tmp_path / "pan.mp4"
    _write_video(v, [True] * N_FRAMES, (0.1, 0.1, 0.9, 0.9))
    w = tmp_path / "pan.wav"
    _write_wav(w, ON)  # 音频有节律，画面没有 → 无口型证据
    d = ls.scan_lipsync(v, w, tmp_path / "wp", min_corr=0.25)
    assert d["verdict"] != "synced", d
    assert d["null_corr"] is not None


def test_scan_reports_none_when_nothing_moves(tmp_path):
    v = tmp_path / "still.mp4"
    _write_video(v, [False] * N_FRAMES, (0.3, 0.6, 0.5, 0.7))
    w = tmp_path / "still_scan.wav"
    _write_wav(w, ON)
    d = ls.scan_lipsync(v, w, tmp_path / "wst")
    assert d["verdict"] == "inconclusive" and d["corr"] is None, d


# ── 5. 口型通道官方硬规格（防"配置越界"伪装成"模型不行"） ──

def test_s2v_resolution_is_clamped_to_its_official_ceiling():
    from fastvideo3.providers.base import clamp_resolution
    from fastvideo3.providers.dashscope import DashScopeTalkProvider

    s2v = DashScopeTalkProvider("sk-x", "wan2.2-s2v")
    # 官方写"时长 <20s"（严格小于）；19.5 是留的换气/取整余量，见 provider 注释
    assert s2v.max_resolution == "720P" and s2v.max_audio_sec == 19.5
    # cinema 档位是 1080P：必须钳到 720P，否则提交即 400，看着像"口型模型不行"
    assert clamp_resolution(s2v, "1080P") == ("720P", True)
    assert clamp_resolution(s2v, "720P") == ("720P", False)
    assert clamp_resolution(s2v, "480P") == ("480P", False)
    # 上限未知就不猜；未知档位名原样透传
    assert clamp_resolution(s2v, "") == ("", False)

    class _NoCap:
        max_resolution = None

    assert clamp_resolution(_NoCap(), "1080P") == ("1080P", False)


@pytest.mark.asyncio
async def test_m6_clamps_resolution_and_stops_on_overlong_audio(cfg, storage, jobs, ledger,
                                                                tmp_path):
    """提交参数按规格钳位；混音超 20s 直接止损（不空转 3 次重试）。"""
    from fastvideo3.modules.m6_talk import _s2v_one
    from fastvideo3.providers.base import CapabilityMismatchError, ProviderError

    class CapProbeS2V:
        name = "dashscope_wan22_s2v"
        model = "wan2.2-s2v"
        max_resolution = "720P"
        max_audio_sec = 20.0
        prefers_16k_audio = False
        submitted: dict = {}

        async def submit(self, *, dest, image, audio, video=None, prompt="",
                         resolution="720P", duration=0):
            type(self).submitted = {"resolution": resolution, "duration": duration}
            raise ProviderError("离线：不真提交")

        async def poll(self, ext_id):
            raise AssertionError

    talk = CapProbeS2V()
    cfg.i2v_resolution = "1080P"
    wav = storage.path("audio", "shot_s1_01_mix.wav")
    _write_wav(wav, ON)  # 48 格 @6fps = 8s
    clip = storage.path("talk", "shot_s1_01.mp4")

    with pytest.raises(ProviderError):
        await _s2v_one(cfg, talk, tmp_path / "first.png", wav, clip, ledger, jobs,
                       storage, "s1_01")
    assert talk.submitted.get("resolution") == "720P", "必须按 s2v 官方上限钳位"
    assert any(e.get("status") == "resolution_clamped" for e in storage.read_events())

    # 超 20s 的混音：永久错误 → 一次都不提交，直接 CapabilityMismatchError 止损
    long_wav = storage.path("audio", "shot_s1_02_mix.wav")
    _write_wav(long_wav, [i % 8 < 4 for i in range(160)])  # ≈26.7s
    talk.submitted = {}
    with pytest.raises(CapabilityMismatchError):
        await _s2v_one(cfg, talk, tmp_path / "first.png", long_wav, clip, ledger, jobs,
                       storage, "s1_02")
    assert talk.submitted == {}, "超长音频属官方规格外，重试无意义，不许提交"
