"""百炼 wan2.2-s2v 口型通道的契约测试（respx 全 mock，零计费）。

这些用例逐条对着官方 API 参考（help.aliyun.com/zh/model-studio/wan-s2v-api，
文档更新 2026-09-22）写，因为它们守的全是"看起来像模型不行、其实是接错了"的坑：

1. 端点：必须是 `/services/aigc/image2video/video-synthesis`。曾经打在
   `/services/aigc/video-generation/video-synthesis` 上，服务端回 `400 url error`，
   报文只说"url"，于是**整条 ¥0.5-0.9/秒的专用口型通道被误判成不能对口型**。
2. 地域/端点可配：官方要求华北2（北京）；公共端点不通时应能换业务空间端点而不改代码。
3. 出参形态：s2v 成功响应里 `output.results` 是**对象** `{"video_url": …}`，而 t2i 是
   **数组**。写死数组会在任务成功那一刻抛 KeyError(0) —— 钱花了、产物"失败"了。
4. 输入硬规格：图片宽高 [400,7000]px、音频 <15MB 且 <20s（超限是永久错误，先查再传，
   不把几 MB 传上临时 OSS 之后才吃 400）。
5. 分档计价：480P ¥0.5/秒、720P ¥0.9/秒，按实际发送档记（笼统价必把一边记错）。
6. 超长对白：官方唯一的正确做法是"切成 <20s 的多段分别生成再拼接" → m6 必须自动做，
   否则一个累计超上限的镜头**永远拿不到口型**。
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from fastvideo3.core.cost import price_of, price_of_provider, price_of_route
from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m6_talk import plan_talk_groups, talk_for_shot
from fastvideo3.providers.base import (
    CapabilityMismatchError,
    clamp_resolution,
    price_key_for,
)
from fastvideo3.providers.dashscope import DashScopeTalkProvider, _first_media_url
from fastvideo3.providers.fake import (
    FakeT2IProvider,
    FakeTalkProvider,
    fake_script,
)

_DS = "https://dashscope.aliyuncs.com/api/v1"
_WS = "https://ws-test.cn-beijing.maas.aliyuncs.com/api/v1"
_S2V_PATH = "/services/aigc/image2video/video-synthesis"


def _png(path: Path, w: int = 1024, h: int = 1792) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (120, 90, 70)).save(path)
    return path


def _wav(path: Path, seconds: float = 5.0, rate: int = 16000) -> Path:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(2500 * math.sin(2 * math.pi * 300 * i / rate)))
                for i in range(n)
            )
        )
    return path


def _mock_oss() -> None:
    """百炼临时 OSS 三步流（getPolicy + 表单直传）。"""
    respx.get(f"{_DS}/uploads").mock(
        return_value=Response(
            200,
            json={
                "data": {
                    "upload_host": "https://oss-test.aliyuncs.com",
                    "upload_dir": "tmp/x",
                    "oss_access_key_id": "ak",
                    "signature": "sig",
                    "policy": "pol",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            },
        )
    )
    respx.post("https://oss-test.aliyuncs.com").mock(return_value=Response(200))


def _provider(base_url: str | None = None) -> DashScopeTalkProvider:
    return DashScopeTalkProvider("sk-test", "wan2.2-s2v", base_url=base_url)


# ─────────────────────────── 端点与协议 ───────────────────────────
@pytest.mark.asyncio
async def test_submit_uses_image2video_endpoint_and_oss_header(tmp_path):
    """端点钉死在 image2video（官方创建任务地址），并带 oss:// 解析头。"""
    img, audio = _png(tmp_path / "f.png"), _wav(tmp_path / "mix.wav")
    with respx.mock:
        _mock_oss()
        route = respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(200, json={"output": {"task_id": "t-1"}})
        )
        wrong = respx.post(f"{_DS}/services/aigc/video-generation/video-synthesis").mock(
            return_value=Response(400, json={"message": "url error, please check url"})
        )
        job = await _provider().submit(
            dest=tmp_path / "o.mp4", image=img, audio=audio, resolution="720P", prompt="她说"
        )
        assert job == "t-1"
        assert route.call_count == 1 and wrong.call_count == 0
        req = route.calls.last.request
        assert req.headers["X-DashScope-Async"] == "enable"
        assert req.headers["X-DashScope-OssResourceResolve"] == "enable"
        body = json.loads(req.content)
        assert body["model"] == "wan2.2-s2v"
        # 输入必须是临时 OSS 公网 URL（本模型拒绝 base64 data URI → 400 url error）
        for key in ("image_url", "audio_url"):
            assert str(body["input"][key]).startswith("oss://"), body["input"]
        # resolution 必须显式发送：没有 parameters 块时服务端按默认 480P 出片，
        # 对白镜头会混进 1080P 成片（上采样发糊的主因之一）
        assert body["parameters"]["resolution"] == "720P"
        # 时长由音频决定，s2v 没有 duration 参数 —— 发过去就是 400
        assert "duration" not in body["parameters"] and "duration" not in body["input"]


@pytest.mark.asyncio
async def test_business_space_base_url_used_for_submit_and_poll(tmp_path):
    """换业务空间端点时，提交与**轮询**都必须是同一个端点（否则查不到任务）。"""
    img, audio = _png(tmp_path / "f.png"), _wav(tmp_path / "mix.wav")
    p = _provider(base_url=_WS)
    with respx.mock:
        _mock_oss()
        respx.post(f"{_WS}{_S2V_PATH}").mock(
            return_value=Response(200, json={"output": {"task_id": "t-ws"}})
        )
        respx.get(f"{_DS}/tasks/t-ws").mock(
            return_value=Response(404, json={"message": "task not exist"})
        )
        poll = respx.get(f"{_WS}/tasks/t-ws").mock(
            return_value=Response(200, json={"output": {"task_status": "RUNNING"}})
        )
        await p.submit(dest=tmp_path / "o.mp4", image=img, audio=audio, resolution="480P")
        st = await p.poll("t-ws")
        assert poll.called and st.status == "running"


# ─────────────────────────── 出参形态 ───────────────────────────
def test_first_media_url_accepts_object_and_list_results():
    """s2v 的 `output.results` 是对象、t2i 是数组 —— 两种都要认。

    写死成数组时，任务 SUCCEEDED 那一刻 `results[0]` 直接 KeyError(0)：
    计费已经发生、产物就在响应里，我们却报"失败"。
    """
    obj = {"results": {"video_url": "https://x/1.mp4"}}
    lst = {"results": [{"url": "https://x/2.png"}]}
    flat = {"video_url": "https://x/3.mp4"}
    assert _first_media_url(obj) == "https://x/1.mp4"
    assert _first_media_url(lst) == "https://x/2.png"
    assert _first_media_url(flat) == "https://x/3.mp4"
    assert _first_media_url({"results": {}}) is None


@pytest.mark.asyncio
async def test_poll_succeeds_with_object_results_and_downloads(tmp_path):
    dest = tmp_path / "talk" / "shot_x.mp4"
    p = _provider()
    p._job_dest["t-ok"] = dest
    with respx.mock:
        respx.get(f"{_DS}/tasks/t-ok").mock(
            return_value=Response(
                200,
                json={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": {"video_url": "https://cdn/1.mp4"},
                    },
                    "usage": {"duration": 5.2, "size": "512*640", "fps": 16, "SR": 480},
                },
            )
        )
        respx.get("https://cdn/1.mp4").mock(return_value=Response(200, content=b"videobytes"))
        st = await p.poll("t-ok")
        assert st.status == "succeeded" and st.output_uri == dest
        assert dest.read_bytes() == b"videobytes"


@pytest.mark.asyncio
async def test_poll_maps_failed_and_unknown_to_failed(tmp_path):
    p = _provider()
    p._job_dest["t-bad"] = tmp_path / "x.mp4"
    with respx.mock:
        respx.get(f"{_DS}/tasks/t-bad").mock(
            return_value=Response(
                200,
                json={"output": {"task_status": "FAILED", "code": "InvalidImage.Size",
                                 "message": "image size out of range"}},
            )
        )
        st = await p.poll("t-bad")
        assert st.status == "failed" and "image size" in (st.error or "")


# ─────────────────────────── 输入硬规格 ───────────────────────────
@pytest.mark.asyncio
async def test_rejects_video_input_instead_of_silently_ignoring_it(tmp_path):
    """s2v 不接受视频输入。收到 video 必须**响亮失败**，不能静默丢掉运动片段。

    静默丢的代价是实测过的形态：v2v_overlay 下 m4 为对白镜头花了 i2V 的钱，s2v 把片段
    当不存在、从首帧另出一段 —— 台账看着"口型已完成"，画面却是重画的。
    """
    img, audio = _png(tmp_path / "f.png"), _wav(tmp_path / "mix.wav")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"motion")
    with respx.mock:  # 不该发出任何网络请求
        with pytest.raises(CapabilityMismatchError) as e:
            await _provider().submit(
                dest=tmp_path / "o.mp4", image=img, audio=audio, video=clip
            )
        assert "不接受视频输入" in str(e.value) or "talk_mode" in str(e.value)


@pytest.mark.asyncio
async def test_rejects_image_smaller_than_official_minimum(tmp_path):
    img, audio = _png(tmp_path / "small.png", 320, 240), _wav(tmp_path / "mix.wav")
    with respx.mock:
        with pytest.raises(CapabilityMismatchError) as e:
            await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=audio)
        assert "400" in str(e.value)  # ≥400px 的官方下限


@pytest.mark.asyncio
async def test_rejects_oversized_audio(tmp_path):
    img = _png(tmp_path / "f.png")
    audio = _wav(tmp_path / "big.wav", 0.2)
    audio.write_bytes(audio.read_bytes() + b"\x00" * int(16 * 1024 * 1024))  # >15MB
    with respx.mock:
        with pytest.raises(CapabilityMismatchError) as e:
            await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=audio)
        assert "15MB" in str(e.value)


def test_official_spec_attributes():
    """官方规格写成类属性：钳位/止损/预估都读这些值，写错就等于对外说谎。"""
    p = _provider()
    assert p.max_resolution == "720P"
    assert p.supports_video_input is False
    assert p.max_audio_sec == 19.5  # 官方"时长 <20s"（严格小于）留余量
    assert p.max_audio_mb == 15.0 and (p.min_image_px, p.max_image_px) == (400, 7000)
    assert p.prefers_16k_audio is True  # 中文 wav2vec2 输入规格
    assert clamp_resolution(p, "1080P") == ("720P", True)
    assert clamp_resolution(p, "480P") == ("480P", False)


@pytest.mark.asyncio
async def test_resolution_never_goes_out_illegal(tmp_path):
    """超上限的档名靠上钳到 720P；完全认不出的按官方默认 480P —— 绝不把非法值发出去。"""
    img, audio = _png(tmp_path / "f.png"), _wav(tmp_path / "mix.wav")
    with respx.mock:
        _mock_oss()
        route = respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(200, json={"output": {"task_id": "t"}})
        )
        await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=audio,
                                 resolution="2K")
        assert json.loads(route.calls.last.request.content)["parameters"]["resolution"] == "720P"
        route2 = respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(200, json={"output": {"task_id": "t2"}})
        )
        await _provider().submit(dest=tmp_path / "o2.mp4", image=img, audio=audio,
                                 resolution="高清")
        assert json.loads(route2.calls.last.request.content)["parameters"]["resolution"] == "480P"


# ─────────────────────────── 分档计价 ───────────────────────────
def test_s2v_pricing_is_tiered_and_matches_official_prices():
    p = _provider()
    assert p.price_key("480P") == "wan2.2-s2v@480p"
    assert p.price_key("1080P") == "wan2.2-s2v@720p"  # 档位名归一到上限
    assert p.worst_case_price_key() == "wan2.2-s2v@720p"
    assert price_of("wan2.2-s2v@480p", "talk", 1.0) == 0.50  # 官方 480P 价
    assert price_of("wan2.2-s2v@720p", "talk", 1.0) == 0.90  # 官方 720P 价
    # pricing.yaml 的预估表与 provider 的红线键必须一致（两处漂移=预估说谎）
    assert price_of_provider("dashscope_wan22_s2v", "talk", 1.0) == price_of(
        p.worst_case_price_key(), "talk", 1.0
    )
    # 预估按实际会发的那一档：economy 480P 不该被按 720P 记账
    assert price_of_route("dashscope_wan22_s2v", "talk", "480P") == 0.50
    assert price_of_route("dashscope_wan22_s2v", "talk", "1080P") == 0.90


# ─────────────────── 超长对白：分段生成 + 拼接 ───────────────────
def test_plan_talk_groups_cuts_only_at_sentence_boundaries():
    """按官方 <20s 上限分组：累计含句间换气，且绝不切断单句。"""
    from fastvideo3.core.schema import AudioLine

    def line(i: int, dur: float) -> AudioLine:
        return AudioLine(
            line_id=f"s1_01_{i}", shot_id="s1_01", character_id="c1",
            text=f"第{i}句", wav_uri=f"audio/line_s1_01_{i}.wav", duration_sec=dur,
        )

    lines = [line(i, 6.0) for i in range(1, 5)]  # 4×6s + 换气 > 19.5s
    groups = plan_talk_groups(lines, 19.5)
    assert [len(g) for g in groups] == [3, 1], groups
    assert [ln.line_id for g in groups for ln in g] == [f"s1_01_{i}" for i in range(1, 5)]
    # 换气间隔也占时长：3×6 + 2×0.25 = 18.5 可容，4×6 + 3×0.25 = 24.75 超
    assert sum(float(ln.duration_sec) for ln in groups[0]) + 0.25 * 2 <= 19.5
    # 上限未知/不限 → 单组（不猜规格）
    assert len(plan_talk_groups(lines, 0)) == 1
    # 单句本身就超长：仍单独成组，交给提交前的规格检查响亮报错（不静默截断音频）
    huge = [line(1, 30.0)]
    assert plan_talk_groups(huge, 19.5) == [huge]


class _S2VLike(FakeTalkProvider):
    """按 s2v 真实契约建模的桩：图片+音频、有分段上限、产物用 ffmpeg 真生成。

    真生成是必须的 —— m6 出片后还会跑身份漂移检查与口型度量，桩文件（几十字节）
    会让那两步抛异常被 except 吞成 failed_fallback，"测试通过"反而掩盖真实路径。
    """

    name = "dashscope_wan22_s2v"
    model = "wan2.2-s2v"
    supports_video_input = False
    prefers_16k_audio = False  # 真通道是 True（上面已断言）；这里免叠一次 ffmpeg
    max_resolution = "720P"
    max_audio_sec = 3.0  # 故意设小，逼出"分段生成 + 拼接"这条分支

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def price_key(self, resolution=None, *, has_video=False):
        tier = "480p" if str(resolution or "").startswith("48") else "720p"
        return f"{self.model}@{tier}"

    async def submit(self, dest, image=None, audio=None, video=None, **params):
        if video is not None:
            raise CapabilityMismatchError("s2v 不接受视频输入")
        self.calls.append({"audio": Path(audio), "resolution": params.get("resolution")})
        return await super().submit(dest, image, audio, **params)


@pytest.mark.asyncio
async def test_long_dialogue_is_split_then_stitched_into_one_shot_clip(
    cfg, storage, ledger, jobs
):
    """累计对白超通道上限：分段出片 → 拼成一个镜头片段 → 事件与台账都按段留痕。"""
    from fastvideo3.core.schema import AudioLine
    from fastvideo3.modules.m2_assets import build_assets
    from fastvideo3.modules.m3_storyboard import build_storyboard

    cfg.fake = False
    cfg.talk_mode = "s2v_direct"
    cfg.i2v_resolution = "720P"
    script = fake_script(BaseInfo(topic="t"))
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    frames = await build_storyboard(cfg, FakeT2IProvider(), script, assets, storage, ledger)

    talk = _S2VLike()  # 桩上限 3.0s
    shot = next(s for s in script.all_shots() if s.dialogues)
    # 自己构造"累计超上限"的对白：4 句 ×0.8s + 句间 0.25s 换气 = 3.5s > 3.0s
    # （不靠 fake 剧本的台词长度凑数：那会随 fake_script 变化而这条用例要的是确定性）
    shot_lines = []
    for i in range(4):
        w = _wav(storage.path("audio", f"line_{shot.shot_id}_{i}.wav"), 0.8)
        shot_lines.append(
            AudioLine(
                line_id=f"{shot.shot_id}_{i}", shot_id=shot.shot_id, character_id="c1",
                text=f"第{i + 1}句", wav_uri=str(w), duration_sec=0.8,
            )
        )
    assert sum(float(ln.duration_sec) for ln in shot_lines) > talk.max_audio_sec
    dest = storage.path("talk", f"shot_{shot.shot_id}.mp4")
    got = await talk_for_shot(
        cfg, talk, Path(frames[shot.shot_id]), shot_lines, dest,
        ledger, jobs, storage, shot.shot_id, prompt="她在说话",
    )
    assert Path(got) == dest and dest.exists()
    # 1) 确实分成了多段（每段一次提交），且段产物在盘
    assert len(talk.calls) >= 2, f"上限 3s 的对白应当分段：{len(talk.calls)}"
    segs = sorted(storage.dir("talk").glob(f"shot_{shot.shot_id}_g*.mp4"))
    assert len(segs) == len(talk.calls)
    # 2) 拼接后的时长 ≈ 各段之和（口型片段必须覆盖整段对白，不能只剩第一段）
    from fastvideo3.utils.ffmpeg import probe_duration

    stitched = probe_duration(dest)
    assert stitched == pytest.approx(sum(probe_duration(s) for s in segs), abs=0.6), stitched
    # 3) 分段后驱动音频落到规范路径，且与拼接出的视频时间轴一致（否则换音轨会错位）
    from fastvideo3.utils.audio import wav_duration

    driver = storage.path("audio", f"shot_{shot.shot_id}_driver.wav")
    assert driver.exists()
    # 驱动音频必须覆盖**全部**台词（4×0.8 + 3×0.25 换气 ≈ 3.95s），m7 用它换掉
    # 模型回传的音轨 —— 用错文件（比如整镜头混音）会让后段口型与声音错位。
    assert wav_duration(driver) == pytest.approx(sum(ln.duration_sec for ln in shot_lines)
                                                 + 0.25 * (len(shot_lines) - 1), abs=0.25)
    # 4) 分段这件事必须可见（事件）+ 每段单独计费（台账 note 带段号，不互相覆盖）
    evs = [e for e in storage.read_events() if e.get("status") == "s2v_split"]
    assert evs and evs[-1]["shot"] == shot.shot_id and evs[-1]["groups"] == len(segs)
    notes = [e.note for e in ledger.entries if e.capability == "talk"]
    assert notes and all("_g" in n for n in notes), notes  # note 带段号（tag 同用作 job_id 后缀）
    assert len(set(notes)) == len(notes), f"分段台账被覆盖：{notes}"
    # 4) 分段按实际发送档计价
    models = {e.model for e in ledger.entries if e.capability == "talk"}
    assert models == {"wan2.2-s2v@720p"}, models


def test_price_key_for_routes_s2v_by_tier_through_m6_helper():
    p = _provider()
    assert price_key_for(p, "480P") == "wan2.2-s2v@480p"
    assert price_key_for(p, "720P") == "wan2.2-s2v@720p"
    # 桩 provider 没有 price_key 时回落到模型名（不炸记账）
    class Legacy:
        model = "videoretalk"

    assert price_key_for(Legacy(), "720P") == "videoretalk"


@pytest.mark.asyncio
async def test_no_money_spent_on_guard_rejection(tmp_path):
    """规格检查必须在**上传之前**：先传几 MB 到临时 OSS 再报错，等于白跑一趟。"""
    img, audio = _png(tmp_path / "small.png", 100, 100), _wav(tmp_path / "mix.wav")
    with respx.mock(assert_all_called=False):
        uploads = respx.get(f"{_DS}/uploads").mock(
            return_value=Response(200, json={"data": {"upload_host": "https://oss-test"}})
        )
        with pytest.raises(CapabilityMismatchError):
            await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=audio)
        assert uploads.call_count == 0, "规格检查必须在上传之前"


# ─────────────────────────── 其它 ───────────────────────────
def test_missing_key_message_points_at_env(tmp_path):
    from fastvideo3.providers.base import ProviderError

    with pytest.raises(ProviderError, match="DASHSCOPE_API_KEY"):
        DashScopeTalkProvider("", "wan2.2-s2v")


def test_i2v_not_needed_for_dialogue_shots_under_s2v_direct():
    """s2v_direct 下对白镜头**不**过 m4：省下那份 i2V 钱（也是模式定义的一部分）。"""
    from fastvideo3.modules.m4_shots import shots_need_i2v

    script = fake_script(BaseInfo(topic="t"))
    dialogue = {s.shot_id for s in script.all_shots() if s.dialogues}
    need = {s.shot_id for s in shots_need_i2v(script, "s2v_direct")}
    assert dialogue and not (dialogue & need), "s2v_direct 不该为对白镜头出运动片段"


# ─────────────── 零成本账号探测（doctor --probe） ───────────────
@pytest.mark.asyncio
async def test_probe_activation_ok_without_creating_task():
    """探针用**不可达的媒体 URL**：400 DataInspection、不建任务、不计费。"""
    from fastvideo3.providers.dashscope import probe_model_activation

    with respx.mock:
        route = respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(
                400,
                json={"code": "InvalidParameter.DataInspection",
                      "message": "Unable to download the media resource during the "
                                 "data inspection process.", "request_id": "r1"},
            )
        )
        status, detail = await probe_model_activation("sk-x", "wan2.2-s2v")
        assert status == "ok" and "未建任务" in detail
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "wan2.2-s2v" and body["parameters"]["resolution"] == "480P"
        assert "probe.invalid" in body["input"]["image_url"]


@pytest.mark.asyncio
async def test_probe_activation_distinguishes_not_activated_from_bad_params():
    """同为 HTTP 400，"Model not exist" 是**账号未开通**（重试无意义），不是参数问题。"""
    from fastvideo3.providers.dashscope import probe_model_activation

    with respx.mock:
        respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(400, json={"code": "InvalidParameter",
                                             "message": "Model not exist."})
        )
        status, detail = await probe_model_activation("sk-x", "wan2.2-s2v")
        assert status == "arrears" and "开通" in detail


@pytest.mark.asyncio
async def test_probe_activation_cancels_if_the_server_surprisingly_accepts():
    """校验若被放宽到接受探针请求，必须**立刻 cancel**，不留排队的视频任务。"""
    from fastvideo3.providers.dashscope import probe_model_activation

    with respx.mock:
        respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(200, json={"output": {"task_id": "t-live",
                                                         "task_status": "PENDING"}})
        )
        cancel = respx.post(f"{_DS}/tasks/t-live/cancel").mock(
            return_value=Response(200, json={"code": "SUCCESS"})
        )
        status, detail = await probe_model_activation("sk-x", "wan2.2-s2v")
        assert status == "ok" and "取消" in detail
        assert cancel.call_count == 1


@pytest.mark.asyncio
async def test_probe_activation_key_and_network_states():
    from fastvideo3.providers.dashscope import probe_model_activation

    assert (await probe_model_activation(""))[0] == "no_key"
    with respx.mock:
        respx.post(f"{_DS}{_S2V_PATH}").mock(
            return_value=Response(401, json={"code": "InvalidApiKey", "message": "bad key"})
        )
        assert (await probe_model_activation("sk-bad"))[0] == "invalid_key"
    with respx.mock:
        respx.post(f"{_DS}{_S2V_PATH}").mock(side_effect=httpx.ConnectError("boom"))
        assert (await probe_model_activation("sk-x"))[0] == "unreachable"
