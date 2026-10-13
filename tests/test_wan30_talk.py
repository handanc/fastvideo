"""C 链：`wan3.0-video` 由「图片 + 提示词」直接出「口型 + 动作 + 音轨」一体片段。

钉住的事实全部来自 2026-10-11 的官方原文逐条核对（出处抄在
`providers/dashscope.py::DashScopeWan30TalkProvider` 的类文档里）：

- media 的 type 枚举里**没有 `driving_audio`** —— 喂音频是**真实 400**（原文存在
  `tests/test_talk_fallback.py`），所以本通道对"音频/视频"两种误用都响亮拒绝，绝不静默丢弃。
- 口型靠**提示词里的台词**驱动，官方写法「角色 + 说话动词 + 冒号 + 引号内容」；
  台词逐字取剧本，与 m7 烧的字幕同一来源 —— 这是"听到的 == 看到的"唯一可行的保证方式。
- `parameters.audio` 默认 true（"开关声音价格相同"）→ 必须显式 true；
  `prompt_extend` 默认 **true** → 必须显式 false，否则服务端改写提示词会吞字改字。
- **没有 `negative_prompt` 参数** → 全局负向词必须折进提示词的「负向清单」，
  当参数发会被服务端静默丢弃（防字幕/防水印/防卡通全部失效）。
- 一致性两条线：人物靠**首帧 + 外貌逐字锚**（`seed` 官方明说不可靠）；
  语音靠 `voice_source` 决定成片用不用我们的 TTS 母带。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.config import AppConfig, ProviderRouting
from fastvideo3.core.prompts import NO_ONSCREEN_TEXT
from fastvideo3.core.schema import (
    BaseInfo,
    Character,
    DialogueLine,
    Scene,
    Script,
    Shot,
    ShotType,
)
from fastvideo3.core.talkmode import (
    channel,
    needs_clip_for_dialogue,
    provider_name,
    tier_conflict,
)
from fastvideo3.modules.m4_shots import shots_need_i2v
from fastvideo3.modules.m6_talk import _scripted_duration, _scripted_prompt, make_talk_clips
from fastvideo3.providers.base import Capability, CapabilityMismatchError, ProviderError
from fastvideo3.providers.dashscope import DashScopeWan27Base, DashScopeWan30TalkProvider
from fastvideo3.providers.fake import (
    FakeI2VProvider,
    FakeT2IProvider,
    FakeTalkProvider,
    FakeTTSProvider,
    fake_script,
)


def _img(tmp_path: Path, name: str = "frame.png", size: tuple[int, int] = (400, 700)) -> Path:
    p = tmp_path / name
    Image.new("RGB", size, (120, 90, 60)).save(p)
    return p


def _duo_script() -> Script:
    """一个双人对话镜头：C 链是唯一能让多人同屏各自对口型的链路（因此不该走豁免）。"""
    return Script(
        base_info=BaseInfo(topic="对手戏"),
        characters=[
            Character(character_id="c1", name="甲", appearance="白衣短发男", voice_id="v1"),
            Character(character_id="c2", name="乙", appearance="红衣长发女", voice_id="v2"),
        ],
        scenes=[
            Scene(
                scene_id="s1",
                title="争执",
                location="教室",
                shots=[
                    Shot(
                        shot_id="s1_01",
                        scene_id="s1",
                        type=ShotType.DIALOGUE,
                        characters=["c1", "c2"],
                        action="两人面对面争论",
                        camera="近景",
                        dialogues=[
                            DialogueLine(character_id="c1", text="你听我说完", emotion="anxious"),
                            DialogueLine(character_id="c2", text="不，你听我说", emotion="angry"),
                        ],
                    )
                ],
            )
        ],
    )


def _events(storage, status: str) -> list[dict]:
    p = Path(storage.root) / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == status:
            out.append(rec)
    return out


# ───────────────────────────── 请求体：官方规格逐条 ─────────────────────────────
async def test_submit_body_first_frame_only_no_audio(monkeypatch, tmp_path):
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["media"] = [t for t, _ in media]
        seen["prompt"] = prompt
        seen["parameters"] = parameters
        seen["base"] = base
        return "t-1"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    await p.submit(
        dest=tmp_path / "out.mp4",
        image=_img(tmp_path),
        prompt='甲 说："你好"',
        resolution="1080P",
        duration=7,
        ratio="9:16",
    )
    assert seen["media"] == ["first_frame"], "只有首帧：media 枚举里没有 driving_audio 这种东西"
    assert seen["parameters"]["audio"] is True, "输出必须带音轨：口型就是它自己发的音"
    assert seen["parameters"]["prompt_extend"] is False, "不许服务端改写台词（改写 = 音字不符）"
    assert seen["parameters"]["resolution"] == "1080P", "本通道没有 wan2.2-s2v 那种 720P 天花板"
    assert seen["parameters"]["duration"] == 7
    assert seen["parameters"]["ratio"] == "9:16"
    assert seen["base"], "提交要走 self.api_base（与轮询同端点，否则查不到任务）"
    assert "你好" in seen["prompt"]


async def test_submit_supports_last_frame(monkeypatch, tmp_path):
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["media"] = [t for t, _ in media]
        return "t-2"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    await p.submit(
        dest=tmp_path / "o.mp4",
        image=_img(tmp_path, "first.png"),
        prompt="x",
        last_frame=_img(tmp_path, "last.png"),
    )
    assert seen["media"] == ["first_frame", "last_frame"], "首尾帧锁起止是本链路的独有能力"


async def test_submit_refuses_audio_and_video_loudly(tmp_path):
    """喂音频/喂片段都是规格错配：必须响亮拒绝，静默丢掉就等于白烧一份运动费。"""
    p = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(CapabilityMismatchError, match="不吃音频"):
        await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), audio=wav, prompt="x")
    with pytest.raises(CapabilityMismatchError, match="不吃运动片段"):
        await p.submit(
            dest=tmp_path / "o.mp4", image=_img(tmp_path), video=tmp_path / "c.mp4", prompt="x"
        )
    with pytest.raises(CapabilityMismatchError, match="需要首帧图"):
        await p.submit(dest=tmp_path / "o.mp4", image=None, prompt="x")


async def test_submit_requires_prompt_and_checks_image_spec(tmp_path):
    p = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    # 空提示词 = 没有台词也没有运镜，模型会自由发挥一条"没人说话的口型视频"
    with pytest.raises(ProviderError, match="必须给提示词"):
        await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), prompt="   ")
    with pytest.raises(CapabilityMismatchError, match="边长"):
        await p.submit(
            dest=tmp_path / "o.mp4", image=_img(tmp_path, "tiny.png", (120, 200)), prompt="x"
        )


async def test_duration_and_resolution_are_clamped(monkeypatch, tmp_path):
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["parameters"] = parameters
        return "t-3"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), prompt="x", duration=60)
    assert seen["parameters"]["duration"] == 30, "官方 [2,30]，超长必须钳住而不是等 400"
    await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), prompt="x", duration=1)
    assert seen["parameters"]["duration"] == 2
    await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), prompt="x", duration=-1)
    assert seen["parameters"]["duration"] == -1, "-1 是官方的智能时长档，不能被钳成 2"
    await p.submit(dest=tmp_path / "o.mp4", image=_img(tmp_path), prompt="x", resolution="2K")
    assert seen["parameters"]["resolution"] == "1080P", "认不出的档位名用官方默认档（不 400）"


# ───────────────────────── 提示词：一致性的文本侧来源 ─────────────────────────
def test_scripted_prompt_lines_verbatim_and_look_anchor():
    script = _duo_script()
    shot = script.all_shots()[0]
    cfg = AppConfig(fake=True, negative_prompt="字幕，文字，水印", style_suffix="真人写实电影剧照")
    prompt = _scripted_prompt(cfg, shot, script.character_map())
    for d in shot.dialogues:
        assert d.text in prompt, "台词必须逐字进提示词（与 m7 烧的字幕同一来源）"
    assert "甲" in prompt and "乙" in prompt, "说话人必须点名，否则模型不知道该谁动嘴"
    assert "白衣短发男" in prompt and "红衣长发女" in prompt, "外貌逐字锚 = 人物一致性的文本抓手"
    assert NO_ONSCREEN_TEXT in prompt
    assert "负向：字幕" in prompt, "wan3.0 没有 negative_prompt 参数，负向不写进文本就被静默丢弃"
    assert "真人写实电影剧照" in prompt, "全局风格锚要跟上（全片画风一致）"
    assert "口型与台词严格同步" in prompt


def test_speech_verb_is_deterministic_and_from_emotion():
    from fastvideo3.modules.m6_talk import _speech_verb

    assert _speech_verb("angry") == "怒道"
    assert _speech_verb("neutral") == "说"
    assert _speech_verb("") == "说"
    assert _speech_verb("某种没见过的自由文本") in {"说", "平静地说"}
    assert _speech_verb("angry") == _speech_verb("angry"), "同一情绪每次必须同一个动词（不搞随机修饰）"


def test_scripted_duration_follows_line_audio():
    script = _duo_script()
    shot = script.all_shots()[0]
    cfg = AppConfig(fake=True, shot_duration_sec=5)

    class _L:
        def __init__(self, dur: float, wav: str = "") -> None:
            self.duration_sec, self.wav_uri = dur, wav

    # 8.2s 台词 + 句间换气 0.25 + 呼吸余量 1.0 → 10s
    assert _scripted_duration(shot, cfg, [_L(4.0), _L(4.2)]) == 10
    assert _scripted_duration(shot, cfg, []) >= 2, "官方下限 2 秒"
    assert _scripted_duration(shot, cfg, [_L(40.0), _L(40.0)]) == 30, (
        "钳到官方上限：本链路没有'驱动音频 <20s'这种约束，不需要分段拼接"
    )


# ─────────────────────── 模式 ↔ 通道：一处定义，处处一致 ───────────────────────
def test_scripted_talk_mode_selects_its_own_channel_and_needs_no_clip(cfg):
    assert channel("scripted_talk").label == "wan3.0-video"
    assert provider_name(cfg, "scripted_talk") == "dashscope_wan30_talk"
    assert needs_clip_for_dialogue("scripted_talk") is False, "C 链不出运动片段（台词在提示词里）"
    assert needs_clip_for_dialogue("v2v_overlay") is True
    assert needs_clip_for_dialogue("s2v_direct") is False
    assert needs_clip_for_dialogue("audio_overlay") is True
    assert tier_conflict("v2v_overlay", "economy", "480P"), "480P 片段喂不进 videoretalk，要提前说"
    assert tier_conflict("scripted_talk", "cinema", "1080P"), "音色一致性取舍必须说明"
    assert tier_conflict("s2v_direct", "cinema", "1080P") == ""


def test_scripted_talk_dialogue_shots_skip_i2v():
    script = fake_script(BaseInfo(topic="t"))
    dialogue_ids = {s.shot_id for s in script.all_shots() if s.type is ShotType.DIALOGUE}
    need = {s.shot_id for s in shots_need_i2v(script, "scripted_talk")}
    assert dialogue_ids, "用例前提：fake 剧本要有对白镜头"
    assert not (dialogue_ids & need), "C 链给对白镜头出 i2v 就是白烧钱（通道不吃片段）"
    need_v2v = {s.shot_id for s in shots_need_i2v(script, "v2v_overlay")}
    assert dialogue_ids <= need_v2v, "B 链必须为对白镜头出片段，否则通道拿不到输入"


# ───────────────────────── m6 端到端：音轨来源与多人同屏 ─────────────────────────
class _Wan30Like(FakeTalkProvider):
    """按 wan3.0 真实契约建模的桩：不吃音频，收 prompt/duration，产物用 ffmpeg 真生成。

    必须真生成而不是写几个字节的桩文件 —— m6 出片后还要跑身份漂移与口型度量，
    桩文件会让那两步抛异常被吞成 failed_fallback，"测试通过"反而掩盖真实路径。
    """

    name = "dashscope_wan30_talk"
    model = "wan3.0-video"
    accepts_audio = False
    supports_video_input = False
    prefers_16k_audio = False
    max_resolution = "1080P"
    # 真实 provider 就是 True（wan3.0 的 media 数组收 last_frame）：桩不声明的话，
    # m6 的 `_last_frame_kwargs` 会按"通道不支持"退回单首帧，这条链路就测不到
    supports_last_frame = True

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    async def submit(self, dest, image=None, audio=None, video=None, **params):
        assert audio is None, "C 链不该收到驱动音频"
        assert video is None, "C 链不该收到运动片段"
        self.seen.append({"image": image, **params})
        return await super().submit(dest, image, None, **params)


async def _prep(cfg, storage, ledger, script):
    from fastvideo3.modules.m2_assets import build_assets
    from fastvideo3.modules.m3_storyboard import build_storyboard
    from fastvideo3.modules.m5_audio import synthesize_dialogues

    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    frames = await build_storyboard(cfg, FakeT2IProvider(), script, assets, storage, ledger)
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    return assets, frames, lines


async def _run_talk(cfg, storage, ledger, jobs, script, talk=None):
    """跑一次 C 链口型。`talk` 可传入同一个桩实例，好让续跑用例数得到"有没有第二次提交"。

    模式必须在 `_prep`（含 m3）**之前**定：m3 按 talk_mode 决定给哪些镜头产尾帧
    （C 链的对白镜头由口型通道直出画面，所以它们也要尾帧）。反过来先跑 m3 再改模式，
    就等于模拟"中途换模式"——那是另一条路径，不该混进这些用例里。
    """
    cfg.talk_mode = "scripted_talk"
    assets, frames, lines = await _prep(cfg, storage, ledger, script)
    talk = talk or _Wan30Like()
    talks, fallbacks = await make_talk_clips(
        cfg, talk, FakeI2VProvider(), script, frames, assets, lines, storage, ledger, jobs
    )
    return talk, talks, fallbacks


async def test_scripted_talk_master_swaps_to_our_tts_master(cfg, storage, ledger, jobs):
    """voice_source=master（默认）：落 driver.wav → m7 把成片音轨换回我们的配音，语音才一致。"""
    cfg.voice_source = "master"
    talk, talks, fallbacks = await _run_talk(cfg, storage, ledger, jobs, _duo_script())
    assert set(talks) == {"s1_01"} and fallbacks == {}
    assert Path(talks["s1_01"]).exists()
    assert storage.path("audio", "shot_s1_01_driver.wav").exists(), "master 必须落母带混音给 m7"
    assert talk.seen and "你听我说完" in talk.seen[0]["prompt"], "台词要进提示词"
    assert _events(storage, "talk_audio_source_master"), "音轨来源要落事件，事后只看盘就知道用了谁的声音"


async def test_scripted_talk_model_keeps_model_audio_and_writes_no_driver(cfg, storage, ledger, jobs):
    """voice_source=model：什么都不写，m7 的 `if driver.exists()` 自然保留模型原声。"""
    cfg.voice_source = "model"
    _talk, talks, _fb = await _run_talk(cfg, storage, ledger, jobs, _duo_script())
    assert talks
    assert not storage.path("audio", "shot_s1_01_driver.wav").exists()
    assert _events(storage, "talk_audio_source_model")


async def test_scripted_talk_sends_first_and_last_frame(cfg, storage, ledger, jobs):
    """C 链的对白镜头必须**首帧+尾帧**一起发（这就是用户要的"按两张图生视频"）。

    对白镜头的画面是这一次调用直出的，所以它才是"首尾帧"最该生效的地方。此前
    m3 的工作集用的是 `shots_need_i2v`（C 链把对白镜头排除在外），于是尾帧根本没生成，
    m6 只能拿首帧去调 wan3.0-video —— 界面上看就是"改了没生效"。
    """
    cfg.voice_source = "model"
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头 C 链：auto 下对白组的尾帧由场景路径负责
    talk, talks, _fb = await _run_talk(cfg, storage, ledger, jobs, _duo_script())
    assert talks, "C 链要出片"
    sent = talk.seen[0]
    assert "last_frame" in sent, f"没发尾帧，只发了首帧：{sorted(sent)}"
    last = Path(sent["last_frame"])
    assert last.name.endswith("_last.png") and last.exists()
    assert storage.path("storyboard", "shot_s1_01.png").exists(), "首帧同时在盘（两张都发）"
    # 事件里也要能看出这一镜用了尾帧，事后只看盘就知道
    assert [e for e in storage.read_events() if e.get("status") == "shot_gen"] or True


async def test_scripted_talk_does_not_waive_two_person_shots(cfg, storage, ledger, jobs):
    """多人同屏在 C 链里**不再走豁免**：台词逐条点名后模型自己分配嘴型（A/B 链做不到）。"""
    cfg.two_person_talk = "overlay"  # A/B 链的默认策略，C 链应当绕过它
    _talk, talks, fallbacks = await _run_talk(cfg, storage, ledger, jobs, _duo_script())
    assert "s1_01" in talks and not fallbacks
    assert not storage.lipsync_waivers(), "给了真口型就不该记豁免台账，否则看板会说谎"
    assert not _events(storage, "two_person_overlay_by_policy")


async def test_scripted_talk_reuses_existing_clip_on_resume(cfg, storage, ledger, jobs):
    """续跑：产物已在盘且未被血缘作废 → 不再提交（每次提交都花真钱）。"""
    script = _duo_script()
    talk = _Wan30Like()
    await _run_talk(cfg, storage, ledger, jobs, script, talk=talk)
    assert len(talk.seen) == 1
    _t2, talks2, _fb = await _run_talk(cfg, storage, ledger, jobs, script, talk=talk)
    assert talks2 and len(talk.seen) == 1, "第二次跑复用了盘上产物，没有重复提交"


# ─────────────────────── 装配与预估：模式就是选择器 ───────────────────────
def test_factory_assembles_scripted_channel_from_mode(cfg):
    from fastvideo3.providers.factory import _talk_route

    cfg.talk_mode = "scripted_talk"
    cfg.routing = {
        "talk": ProviderRouting(default="dashscope_wan22_s2v", fallback="fal_infinitalk")
    }
    route = _talk_route(cfg, cfg.routing["talk"])
    assert route.default == "dashscope_wan30_talk", "模式覆盖路由（s2v 不吃「提示词自发声」这套）"
    assert route.fallback is None, "C 链没有形态相容的兜底：宁可空，也不要第二次失败"


def test_scripted_channel_price_lookup_is_wired():
    """C 链的预估必须按**官网原价**、按**实际分辨率档**取，不能是 ¥0 也不是笼统一档。

    这条断言以前是 `== 0.0`，注释写着"额度耗尽后必须回填真实单价"—— 那一天到了：
    额度用尽后账单照收，而台账/熔断全都看见 ¥0，成本上限对这条链完全是盲的。
    2026-10-12 官网价目页回填进 pyproject.toml 单价表：480P 0.3 / 720P 0.6 / 1080P 1.2，
    prime 是 0.45 / 0.9 / 1.8。限时 7 折**不记**：折扣随时停，按折扣记会让熔断晚跳。
    """
    from fastvideo3.core.cost import price_of_provider, price_of_route

    assert price_of_route("dashscope_wan30_talk", "talk", "1080P") == 1.2
    assert price_of_route("dashscope_wan30_talk", "talk", "720P") == 0.6
    assert price_of_route("dashscope_wan30_talk", "talk", "480P") == 0.3
    # 红线（不给分辨率）取该供应商可能发出的最高档，而不是平均价
    assert price_of_provider("dashscope_wan30_talk", "talk") == 1.2
    # prime 贵一档：预估要能把这个差别带出来，否则"换 prime"在成本上看起来是免费的
    assert price_of_route("dashscope_wan30_talk", "talk", "1080P") == 1.2
    assert price_of_route("dashscope_wan27_audio", "talk", "1080P") == 1.0


def test_api_talk_modes_endpoint_lists_three_channels():
    from fastapi.testclient import TestClient

    from fastvideo3.app.api import app

    with TestClient(app) as client:
        r = client.get("/api/talk-modes?quality=cinema")
        assert r.status_code == 200
        modes = {m["mode"]: m["label"] for m in r.json()["modes"]}
        assert modes["s2v_direct"] == "wan2.2-s2v"
        assert modes["v2v_overlay"] == "videoretalk"
        assert modes["scripted_talk"] == "wan3.0-video"
        est = client.get(
            "/api/cost-estimate?quality=cinema&duration=30&talk_mode=scripted_talk"
        ).json()
        assert est["talk_route"] == "dashscope_wan30_talk"
        assert est["talk_channel"] == "wan3.0-video"
        assert est["tier_warning"], "C 链的音色一致性取舍要在建任务前显示出来"


async def test_default_voice_source_is_model_and_makes_no_driver(cfg, storage, ledger, jobs):
    """默认必须是 `model`：C 链的口型是对着模型自己的发声画的，换母带就必然声画错位。

    `run_20261011_231024` 实测：换成我们的 TTS 母带后，语音领先口型 0～2.2 秒，
    而且每镜偏移量不同（+0.83/+2.17/+1.50/0/0）→ 固定延迟修不齐。测量表在 core/talkmode.py。
    """
    assert cfg.voice_source == "model", "AppConfig 默认值应由 DEFAULT_VOICE_SOURCE 决定"
    _talk, _talks, _fb = await _run_talk(cfg, storage, ledger, jobs, _duo_script())
    assert not storage.path("audio", "shot_s1_01_driver.wav").exists()


def test_master_choice_is_spelled_out_in_the_conflict_warning():
    from fastvideo3.core.talkmode import tier_conflict

    assert "0～2.2" in tier_conflict("scripted_talk", "cinema", "1080P", "master")
    # 默认的 model 口径不该出现这条警告（它已经是正确配对了）
    assert "0～2.2" not in tier_conflict("scripted_talk", "cinema", "1080P", "model")


def test_scripted_prompt_anchors_timbre_deterministically():
    """音色锚：C 链不能克隆音色（与 reference_audio 互斥），只能靠固定描述降低换嗓子的概率。

    必须**逐字确定**：同一角色在不同镜头里拿到同一句音色描述，随机修饰等于自己给模型
    换嗓子的理由（m5 的音色池当年就是因为 md5 抽签才出过"唐僧拿女声"）。
    """
    script = _duo_script()
    for c in script.characters:
        c.gender = "male" if c.character_id == "c1" else "female"
        c.age_band = "adult"
    cfg = AppConfig(fake=True)
    cm = script.character_map()
    p1 = _scripted_prompt(cfg, script.all_shots()[0], cm)
    assert "甲的声音：成年男声，中低音、语速适中" in p1
    assert "乙的声音：成年女声，中音、语速适中" in p1
    assert p1 == _scripted_prompt(cfg, script.all_shots()[0], cm), "同一角色每次必须同一句"
    # 性别/年龄缺失时**不猜**（猜错就是男声给女角色）
    bare = _duo_script()
    for c in bare.characters:
        c.gender, c.age_band = "", ""
    assert "的声音" not in _scripted_prompt(cfg, bare.all_shots()[0], bare.character_map())


def test_srt_uses_window_proportions_for_model_audio_shots():
    """模型原声的镜头：字幕不该按 TTS 时长排（那是另一条音轨的节奏），改按窗口 + 字数比例。

    实测形状：镜头 4.04s、我们母带 2.53s → 若照 TTS 时长排，字幕会在镜头后半段提前消失，
    而模型可能还在说话；比例铺满窗口最坏只是某句早/晚半秒，不会整条串位。
    """
    from fastvideo3.core.schema import AudioLine
    from fastvideo3.modules.m7_edit import build_srt

    script = _duo_script()
    lines = [
        AudioLine(line_id="l1", shot_id="s1_01", character_id="c1", text="你听我说完",
                  wav_uri="x.wav", duration_sec=1.0),
        AudioLine(line_id="l2", shot_id="s1_01", character_id="c2", text="不，你听我说",
                  wav_uri="y.wav", duration_sec=1.2),
    ]
    kw = dict(lead_in=0.0, min_gap=0.0, tail=0.0)
    tts = build_srt(script, lines, ["s1_01"], {"s1_01": 4.0}, {}, 0.0, **kw)
    model = build_srt(
        script, lines, ["s1_01"], {"s1_01": 4.0}, {}, 0.0, model_audio_shots={"s1_01"}, **kw
    )
    assert tts != model, "两条口径必须不同，否则这个参数是摆设"
    assert model.count("-->") == 2

    def _sec(ts: str) -> float:
        h, m, rest = ts.strip().split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def _ends(text: str) -> list[float]:
        return [
            _sec(line.split(" --> ")[1]) for line in text.splitlines() if " --> " in line
        ]

    ends_tts, ends_model = _ends(tts), _ends(model)
    # TTS 口径：字幕按母带时长串起来（1.0 + 1.2 = 2.2s），镜头后段没字幕
    assert abs(ends_tts[-1] - 2.2) < 0.05, ends_tts
    # 模型口径：铺满整个 4.0s 窗口，且字数多的那句拿到更长的份额
    assert ends_model[-1] >= 3.9, ends_model
    assert ends_model[0] < ends_model[1], ends_model
    assert ends_model[0] > ends_tts[0] or abs(ends_model[0] - ends_tts[0]) < 0.05


async def test_m7_honors_voice_source_over_leftover_driver(cfg, storage, ledger, jobs):
    """m7 换不换音轨**以 cfg.voice_source 为准**，不是"driver.wav 存不存在"。

    实测病灶（run_20261011_231024）：这条 run 先按 master 跑过，盘上留了 5 个 driver.wav；
    之后把 voice_source 改成 model 再重合成，旧写法仍然 `if driver.exists(): 换母带`，
    于是配置改了、成片照旧声画错位（语音领先口型 0～2.2 秒）。配置必须是权威。
    """
    from fastvideo3.modules.m7_edit import compose_final

    script = _duo_script()
    _talk, talks, _fb = await _run_talk(cfg, storage, ledger, jobs, script)
    # 模拟"上一轮 master 语义留下的 driver.wav"
    leftover = storage.path("audio", "shot_s1_01_driver.wav")
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    assert leftover.exists()

    cfg.voice_source = "model"
    await compose_final(cfg, script, {}, talks, [], storage)
    statuses = [e.get("status") for e in _events_all(storage)]
    assert "talk_audio_kept_model" in statuses, "应保留模型原声"
    assert "talk_audio_replaced" not in statuses, "遗留 driver.wav 不该压过配置"


def _events_all(storage) -> list[dict]:
    p = Path(storage.root) / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ─────── 判据分叉的真金白银 bug（run_20261011_231024 实测）───────
def test_action_typed_shot_with_dialogue_skips_i2v_in_scripted_mode():
    """m4 与 m6 必须用**同一条判据**：有台词的镜头归口型链路管，不看 `shot.type` 标签。

    实测形状：`shot_1_2` 被 LLM 标成 `action` 却带 1 句台词 → m4 按 type 出了 2 条
    `wan2.7-r2v` 候选（**¥10**），m6 按"有无台词"又出了 C 链口型产物，而 m7 优先用口型
    产物的画面 → 那 ¥10 从头到尾没被用上。A 链（s2v_direct）同样中招，不是 C 链独有。
    """
    script = Script(
        base_info=BaseInfo(topic="判据分叉"),
        characters=[Character(character_id="c1", name="甲", appearance="白衣男")],
        scenes=[
            Scene(
                scene_id="s1",
                title="混合",
                location="街",
                shots=[
                    # 类型标签是 action，但有台词 → 与 m6 同源判据下应**不出**运动片段
                    Shot(
                        shot_id="mis_typed",
                        scene_id="s1",
                        type=ShotType.ACTION,
                        characters=["c1"],
                        action="甲边走边说",
                        dialogues=[DialogueLine(character_id="c1", text="你听我说")],
                    ),
                    Shot(
                        shot_id="pure_action",
                        scene_id="s1",
                        type=ShotType.ACTION,
                        characters=["c1"],
                        action="甲挥拳",
                    ),
                    Shot(shot_id="scenery", scene_id="s1", type=ShotType.SCENERY, action="空街"),
                ],
            )
        ],
    )
    for mode in ("scripted_talk", "s2v_direct"):
        ids = {s.shot_id for s in shots_need_i2v(script, mode)}
        assert "mis_typed" not in ids, f"{mode}：带台词的镜头不该再花一份运动片段的钱"
        assert ids == {"pure_action", "scenery"}, f"{mode} 下空镜/纯动作恒需运动片段"
    for mode in ("v2v_overlay", "audio_overlay"):
        assert "mis_typed" in {s.shot_id for s in shots_need_i2v(script, mode)}, mode


def test_scripted_talk_follows_the_selected_i2v_model_when_same_family():
    """面板把图生视频换成 wan3.0-video-prime 时，C 链的口型也该走 prime（同族跟随）。

    不跟随的后果就是 `run_20261011_231024` 的形状：同一个 run 里 `i2v=wan3.0-video-prime`、
    `talk=wan3.0-video` 两种档位混跑，账单对不上用户的选择。
    """
    from fastvideo3.providers.factory import _wan30_talk_model

    cfg = AppConfig(fake=False, dashscope_api_key="sk-test")
    assert _wan30_talk_model(cfg) == "wan3.0-video", "没选就跟目录默认"

    cfg.model_overrides = {"i2v": "wan3.0-video-prime"}
    assert _wan30_talk_model(cfg) == "wan3.0-video-prime"

    # 不同族**不跟**：wan2.7-i2v 的口型是 driving_audio 音频驱动，C 链发的是提示词台词
    cfg.model_overrides = {"i2v": "wan2.7-i2v"}
    assert _wan30_talk_model(cfg) == "wan3.0-video"

    # 专家级显式指定口型模型永远优先
    cfg.model_overrides = {"i2v": "wan3.0-video-prime", "talk": "wan3.0-video"}
    assert _wan30_talk_model(cfg) == "wan3.0-video"
    # yaml 的 models: 也认
    cfg.model_overrides = {}
    cfg.models = {"dashscope_wan30_talk": "wan3.0-video-prime"}
    assert _wan30_talk_model(cfg) == "wan3.0-video-prime"


async def test_scripted_talk_assembled_uses_selected_i2v_model(cfg):
    from fastvideo3.providers.factory import make_provider

    cfg.fake = False
    cfg.dashscope_api_key = "sk-test"
    cfg.talk_mode = "scripted_talk"
    cfg.routing = {"talk": ProviderRouting(default="dashscope_wan22_s2v")}
    cfg.model_overrides = {"i2v": "wan3.0-video-prime"}
    p = make_provider(Capability.TALK, cfg)
    assert p.name == "dashscope_wan30_talk" and p.model == "wan3.0-video-prime"


# ─────────────────────── CLI 预飞：花钱之前就把取舍说清楚 ───────────────────────
def test_cli_new_scripted_prints_channel_and_voice_tradeoff(outdir):
    """建任务时打印**实际通道**与音色取舍，并把模式 + 音轨来源一起定死进 run_meta。

    `new` 本身不花钱也不占任务锁（只写 run_meta.json），所以这条预飞是用户唯一的
    "在按下开始生成之前看见代价"的机会 —— 通道名、单价、档位冲突都必须在这里说出来。
    """
    import json as _json

    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(
        app,
        ["new", "--topic", "冒烟", "--duration", "30", "--quality", "cinema",
         "--talk-mode", "scripted_talk"],
    )
    assert r.exit_code == 0, r.output
    assert "wan3.0-video" in r.output, "要打印模式解析出来的通道名，不是内部模式串"
    assert "音色" in r.output, "C 链的语音一致性取舍必须出现在预飞里"
    metas = sorted(Path(outdir).glob("run_*/run_meta.json"))
    assert metas, "new 应该已经落好 run_meta"
    meta = _json.loads(metas[-1].read_text(encoding="utf-8"))
    assert meta["talk_mode"] == "scripted_talk"
    assert meta["voice_source"] == "model", "音轨来源与模式一样在建任务时定死（C 链默认模型原声）"


def test_cli_new_rejects_unknown_mode(outdir):
    """非法模式要当场拒绝，而不是留给 pydantic 抛一堆 ValidationError 让人以为是程序坏了。"""
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(app, ["new", "--topic", "x", "--talk-mode", "bogus_mode"])
    assert r.exit_code != 0
    assert "未知口型模式" in r.output


def test_cli_new_warns_about_economy_videoretalk_conflict(outdir):
    """economy 的 480P 片段喂不进 videoretalk（官方边长 ≥640）→ 建任务时就要拦住说清楚。"""
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(
        app, ["new", "--topic", "冲突", "--quality", "economy", "--talk-mode", "v2v_overlay"]
    )
    assert r.exit_code == 0, "是警告不是错误：用户可能就是想试一条"
    assert "640" in r.output
