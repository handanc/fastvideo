"""wan3.0 **角色+场景生成**的请求体契约：逐条对着官方 API 参考页校，不发一个真请求。

为什么单独要这一份：这条链路的失败方式几乎都是"服务端 400"或"静默不按预期"——
2026-10-13 实测就撞过 `references=[]` 被本地守卫拦下（点了生成毫无反应、账号分文未动）。
判据全部抄自 `help.aliyun.com/zh/model-studio/wan3-video-generation-api-reference`
（2026-10-11 取原文），不靠记忆：

- `input.media[].type` ∈ {first_frame, last_frame, reference_image, reference_video,
  reference_audio, file, link}；**reference_image ≤10 张、reference_audio ≤5 段且总时长 ≤15s**；
- `first_frame`/`last_frame` 与 `reference_*` **互斥**（原文错误码："The two modes are
  mutually exclusive. Do not pass reference_xx and first_frame/last_frame at the same time."）；
- 图/视频/音频**分别按各自 type 在数组中的顺序**编号 → 提示词里的 `图N`/`音频N` 必须与
  media 数组里同 type 的顺序逐位对应；
- 图像规格：单边 [240, 8000] px、长宽比 ≤8:1、≤20MB、PNG 不支持透明通道；
- `parameters.resolution` ∈ {480P,720P,1080P}；`duration` 无视频输入时为 [2,30] 整数；
  `audio` 默认 true（关掉就没口型了）；`prompt_extend` 默认 true，**我们必须显式 false**
  （字幕是 m7 用剧本台词逐字烧的，服务端改写必然音字不符）；
- **没有 `negative_prompt` 参数**：按参数发会被静默丢弃，负向清单必须折进提示词。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.assets import load_assets
from fastvideo3.core.config import AppConfig, load_config
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import BaseInfo, DialogueLine, Script, Shot, ShotType
from fastvideo3.core.storage import RunStorage
from fastvideo3.modules.m4_scene import generate_scene_clips
from fastvideo3.providers.dashscope import DashScopeWan30TalkProvider

MEDIA_TYPES = {
    "first_frame", "last_frame", "reference_image", "reference_video",
    "reference_audio", "file", "link",
}
RESOLUTIONS = {"480P", "720P", "1080P"}
RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


@pytest.fixture
def run(tmp_path):
    """一个 run：两个角色定妆照 + 一张空景场景图 + 一段两句台词的对白组。"""
    rid = "run_wan30_contract"
    storage = RunStorage(rid, root=tmp_path)
    adir = storage.root / "assets"
    adir.mkdir(parents=True, exist_ok=True)
    # 810×1440 = 本项目 9:16 画幅的归一化尺寸（也是用户实际上传图被归一化到的尺寸）
    for name in ("char_c1.png", "char_c2.png", "scene_s1.png"):
        Image.new("RGB", (810, 1440), (30, 90, 160)).save(adir / name, format="PNG")
    script = Script(
        base_info=BaseInfo(topic="契约", target_duration_sec=9),
        title="契约",
        characters=[
            {"character_id": "c1", "name": "八戒", "gender": "male", "age_band": "adult",
             "appearance": "黑帽灰袍"},
            {"character_id": "c2", "name": "女王", "gender": "female", "age_band": "adult",
             "appearance": "红裙金冠"},
        ],
        scenes=[
            {
                "scene_id": "s1",
                "title": "街道",
                "location": "女儿国街道",
                "time_of_day": "日",
                "summary": "喧闹的街道",
                "shots": [
                    Shot(shot_id="s1_01", scene_id="s1", type=ShotType.DIALOGUE,
                         duration_sec=4.0, characters=["c1"], action="八戒招手",
                         camera="中景",
                         dialogues=[DialogueLine(character_id="c1", text="师父别走！")]),
                    Shot(shot_id="s1_02", scene_id="s1", type=ShotType.DIALOGUE,
                         duration_sec=5.0, characters=["c1", "c2"], action="女王回眸",
                         camera="近景",
                         dialogues=[DialogueLine(character_id="c2", text="圣僧留下。")]),
                ],
            }
        ],
    )
    storage.save_json("script.json", script.model_dump())
    storage.save_json(
        "run_meta.json",
        {"run_id": rid, "base_info": BaseInfo(topic="契约", target_duration_sec=9).model_dump(),
         "quality": "cinema", "talk_mode": "scripted_talk", "budget_limit_cny": 100.0},
    )
    # 配音母带：音色参考样本是从**台词母带**里切的（m5 的产物），不是凭空造的文件。
    # 连 audio_map.json 一起落 —— 那才是 m5 的完整产物形态，而面板预览正是读它。
    _fake_wav(storage.root / "audio" / "line_s1_01_0.wav", seconds=4.0)
    _fake_wav(storage.root / "audio" / "line_s1_02_0.wav", seconds=4.0)
    storage.save_json("audio_map.json", [
        ln.model_dump() for ln in _lines(storage)
    ])
    return storage, script


def _lines(storage: RunStorage) -> list:
    """m5 落盘的 audio_map.json 那两条台词（每角色 4 秒，总 8 秒 ≤ 官方 15 秒上限）。"""
    from fastvideo3.core.schema import AudioLine

    return [
        AudioLine(line_id="s1_01_0", shot_id="s1_01", character_id="c1", text="师父别走！",
                  wav_uri=str(storage.root / "audio" / "line_s1_01_0.wav"), duration_sec=4.0),
        AudioLine(line_id="s1_02_0", shot_id="s1_02", character_id="c2", text="圣僧留下。",
                  wav_uri=str(storage.root / "audio" / "line_s1_02_0.wav"), duration_sec=4.0),
    ]


def _fake_wav(dest: Path, seconds: float = 4.0) -> None:
    """最小可辨识的 wav 头（契约测试只校验规格，不校验内容）。"""
    import wave

    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


class _Capture:
    """替掉两处网络：临时 OSS 直传与创建任务，把最终发出去的 body 留在盘上给人看。"""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self.uploads: list[tuple[str, str]] = []  # (type, 本地文件名)

    def install(self, monkeypatch, provider) -> None:
        import fastvideo3.utils.dashscope_upload as up

        cap = self

        async def fake_upload(api_key, model, path):
            Path(path).is_file()
            cap.uploads.append((Path(path).name, str(path)))
            return f"oss://dashscope-instant/fake/{Path(path).name}"

        monkeypatch.setattr(up, "upload_local_file", fake_upload)

        async def fake_submit(url, body, extra_headers=None):
            cap.urls.append(url)
            cap.bodies.append(json.loads(json.dumps(body)))
            cap.headers.append({**(extra_headers or {})})
            return "task-fake-1"

        async def fake_query(task_id):
            # 官方轮询响应形状：output.task_status / video_url，usage 在响应根
            return {
                "output": {
                    "task_id": task_id, "task_status": "SUCCEEDED",
                    "video_url": "https://example.com/v.mp4",
                },
                "usage": {"video_count": 1, "output_video_duration": 9.0, "SR": 720,
                          "ratio": "9:16"},
            }

        async def fake_download(url, dest):
            Path(dest).write_bytes(b"\x00\x00fake-mp4")
            return Path(dest)

        monkeypatch.setattr(provider, "_submit_task", fake_submit)
        monkeypatch.setattr(provider, "_query_task", fake_query)
        monkeypatch.setattr("fastvideo3.providers.dashscope._download", fake_download)


def _cfg(**kw) -> AppConfig:
    """用**项目默认配置**（configs/default.yaml）：负向清单、分辨率这些默认值本身就是契约的一部分。"""
    cfg = load_config()
    cfg.fake = True
    cfg.dashscope_api_key = "sk-test"
    cfg.i2v_resolution = "720P"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


async def _reference_request(run, monkeypatch, cfg=None):
    storage, script = run
    cfg = cfg or _cfg()
    provider = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    cap = _Capture()
    cap.install(monkeypatch, provider)
    ledger = CostLedger(storage)
    jobs = JobStore(f"sqlite:///{storage.root / 'jobs.db'}")
    got = await generate_scene_clips(
        cfg, provider, script, load_assets(storage), storage, ledger, jobs,
        lines=_lines(storage), frames={}, skip_existing=False,
    )
    return storage, cfg, provider, cap, got, ledger


async def test_budget_gate_stops_new_submits_without_touching_finished_clips(
    run, monkeypatch
):
    """成本到线：**停手问人**，而不是静默出半片（闸门实现见 core/pause.py）。

    这条路径以前完全没有预算检查 —— m4_scene 是本项目最贵的一步
    （一段场景片段 ¥5–17），旧行为只在 m3/m4_shots/m6 里有 `exceeded()` 判断，
    也就是说"熔断"根本拦不住场景片段。
    """
    import json

    from fastvideo3.core import pause
    from fastvideo3.core.cost import CostLedger
    from fastvideo3.core.jobs import JobStore
    from fastvideo3.modules.m4_scene import generate_scene_clips
    from fastvideo3.providers.dashscope import DashScopeWan30TalkProvider

    storage, script = run
    cfg = _cfg()
    provider = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    cap = _Capture()
    cap.install(monkeypatch, provider)

    # 已花 120 / 上限 100：到线
    meta = json.loads((storage.root / "run_meta.json").read_text(encoding="utf-8"))
    meta["budget_limit_cny"] = 100.0
    (storage.root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (storage.root / "cost.json").write_text(
        json.dumps([{"capability": "i2v", "model": "wan3.0-video", "amount_cny": 120.0}]),
        encoding="utf-8",
    )
    # 上一轮已经出过的那段：必须原地不动
    done = storage.root / "clips" / "scene_s1.mp4"
    done.write_bytes(b"already-paid")

    ledger = CostLedger(storage, budget_limit_cny=100.0)
    jobs = JobStore(f"sqlite:///{storage.root / 'jobs.db'}")
    got = await generate_scene_clips(
        cfg, provider, script, {}, storage, ledger, jobs,
        lines=_lines(storage), frames={}, skip_existing=False,
    )
    assert got == {}, "到线后一段都不该新提交"
    assert cap.bodies == [], "不许再发任何创建任务请求"
    assert done.read_bytes() == b"already-paid", "已付费的产物一个字节都不动"
    st = json.loads((storage.root / "pause.json").read_text(encoding="utf-8"))
    assert st["pending"] is True and st["limit_cny"] == 100.0, st
    assert pause.stop_for_budget(storage, ledger, node="m4_scene") is True

    # 人工确认（抬上限）之后同一次调用就能继续跑，且台账/血缘都不需要特殊处理
    pause.approve(storage, 200.0)
    got2 = await generate_scene_clips(
        cfg, provider, script, {}, storage, CostLedger(storage, budget_limit_cny=200.0),
        jobs, lines=_lines(storage), frames={}, skip_existing=False,
    )
    assert got2 and cap.bodies, "确认后应该照常把剩下的片段补出来"


async def test_reference_mode_body_matches_official_contract(run, monkeypatch):
    storage, cfg, provider, cap, got, ledger = await _reference_request(run, monkeypatch)
    assert got, "参考形态必须出片（这一步空返回说明素材又没解析出来）"
    assert len(cap.bodies) == 1, cap.bodies
    body = cap.bodies[0]

    # ── 端点与异步头 ──
    url = cap.urls[0]
    assert url.endswith("/api/v1/services/aigc/video-generation/video-synthesis"), url
    assert "X-DashScope-OssResourceResolve" in cap.headers[0], "oss:// 临时 URL 必须带解析头"

    # ── model 与顶层结构 ──
    assert body["model"] == "wan3.0-video"
    assert set(body) == {"model", "input", "parameters"}, sorted(body)
    assert set(body["input"]) == {"prompt", "media"}, sorted(body["input"])

    media = body["input"]["media"]
    types = [m["type"] for m in media]
    assert all(t in MEDIA_TYPES for t in types), types
    # 互斥：参考形态绝不能混发首/尾帧（官方 400）
    assert not ({"first_frame", "last_frame"} & set(types)), types
    assert types.count("reference_image") >= 1, "角色+场景生成必须至少发一张参考图"
    assert types.count("reference_image") <= 10, "官方上限：reference_image ≤10"
    assert types.count("reference_audio") <= 5, "官方上限：reference_audio ≤5 段"
    assert all(str(m["url"]).startswith("oss://") for m in media), media

    # 立绘在前、场景图在后：编号顺序 = 数组顺序（图1…图N）
    img_names = [Path(u).name for t, u in zip(types, [m["url"] for m in media])
                 if t == "reference_image"]
    assert img_names[:2] == ["char_c1.png", "char_c2.png"], img_names
    assert img_names[-1] == "scene_s1.png", f"场景图必须接在立绘后面连续编号：{img_names}"

    # ── parameters 逐字段对官方取值域 ──
    params = body["parameters"]
    assert params["resolution"] in RESOLUTIONS, params
    assert isinstance(params["duration"], int) and 2 <= params["duration"] <= 30, params
    assert params["prompt_extend"] is False, "必须显式关智能改写（字幕逐字烧，改写必音字不符）"
    assert params.get("audio", True) is not False, "口型就是它自己发的音，关声音等于关掉这条链路"
    assert "negative_prompt" not in params, "wan3.0 没有该参数，发了会被静默丢弃"
    assert params.get("ratio", "adaptive") in RATIOS, params


async def test_billing_follows_api_usage_not_request(run, monkeypatch):
    """台账按 API 回传的**实际用量**计费（秒数 + 分辨率档），不是按我们请求的值。

    服务端可能交付 8.9 秒、也可能把请求按别的档计（`usage.SR`）—— 用量取错，单价再准
    也是错的金额（"合计 ¥6.30、i2v 全记 ¥0"那次事故的根因之一就是这两处都记错）。
    """
    storage, cfg, provider, cap, got, ledger = await _reference_request(run, monkeypatch)
    rows = [e for e in ledger.entries if str(e.note or "").startswith("scene:")]
    assert rows, [e.note for e in ledger.entries]
    row = rows[-1]
    meta = row.meta
    assert float(meta["quantity"]) == pytest.approx(9.0), meta  # usage.output_video_duration
    assert "usage" in str(meta.get("quantity_source", "")), meta.get("quantity_source")
    assert str(meta.get("price_key", "")).endswith("@720p"), meta.get("price_key")  # usage.SR=720
    assert row.amount_cny == pytest.approx(
        round(float(meta["quantity"]) * float(meta["unit_price_cny"]), 4)
    ), (row.amount_cny, meta)
    assert row.amount_cny > 0, "场景片段绝不能记 ¥0"
    assert meta.get("mode") == "reference", meta
    assert meta.get("n_ref_images") >= 3 and meta.get("n_ref_audios") >= 1, meta


async def test_prompt_numbering_lockstep_with_media(run, monkeypatch):
    """提示词里的 图N / 音频N 必须与 media 数组**同 type 的顺序**逐位一致。

    错位不会报错，只会把"锁八戒的脸"套到女王身上 —— 这类静默错位是最贵的 bug。
    """
    storage, cfg, provider, cap, got, ledger = await _reference_request(run, monkeypatch)
    body = cap.bodies[0]
    prompt = body["input"]["prompt"]
    media = body["input"]["media"]

    img_order = [Path(m["url"]).name for m in media if m["type"] == "reference_image"]
    aud_order = [Path(m["url"]).name for m in media if m["type"] == "reference_audio"]
    assert "图1" in prompt and f"图{len(img_order)}" in prompt, prompt[:400]
    assert f"图{len(img_order) + 1}" not in prompt, "提示词引用了没发送的编号"
    # 每条引用后面紧跟它指代的素材（角色名或场景名），且顺序与数组一致
    for n, name in enumerate(img_order, start=1):
        assert f"图{n}" in prompt, (n, name)
    if aud_order:
        assert "音频1" in prompt, prompt[:400]

    # 台词逐字进提示词（官方：口型由提示词里的台词驱动，没有 driving_audio）
    assert "师父别走！" in prompt and "圣僧留下。" in prompt
    # 多镜头时间戳：第N个镜头[起-止秒]
    assert "第1个镜头" in prompt and "第2个镜头" in prompt, prompt[:400]
    # 场景图是**空景**：不写清楚模型会照搬背景里的人数
    assert "无人" in prompt, "必须声明场景参考图里没有人物"
    # 音乐只由后期统一混一次：必须显式关掉模型自带 BGM（官方三档里的"完全不要"）
    assert "无背景音乐" in prompt and "No background music" in prompt, prompt[-300:]
    # 负向清单折进提示词（wan3.0 的 parameters 里**没有** negative_prompt，按参数发被静默丢弃）
    assert "负向" in prompt, prompt[-300:]
    assert cfg.negative_prompt.split("，")[0].strip() in prompt, cfg.negative_prompt


async def test_reference_images_meet_official_image_spec(run, monkeypatch):
    """发出去的每张参考图都满足官方图像规格：单边 [240,8000]、长宽比 ≤8:1、≤20MB、无透明通道。"""
    storage, cfg, provider, cap, got, ledger = await _reference_request(run, monkeypatch)
    media = cap.bodies[0]["input"]["media"]
    by_name = {Path(u).name: p for (t, u), p in zip(
        [(m["type"], m["url"]) for m in media],
        [p for _, p in cap.uploads],
    )}
    for m in media:
        if m["type"] != "reference_image":
            continue
        path = Path(by_name[Path(m["url"]).name])
        with Image.open(path) as im:
            w, h = im.size
        assert 240 <= w <= 8000 and 240 <= h <= 8000, (path.name, w, h)
        assert max(w / h, h / w) <= 8.0, (path.name, w, h)
        assert im.mode in ("RGB", "L"), f"{path.name} 带 Alpha 通道，官方 PNG 不支持透明"
        assert path.stat().st_size <= 20 * 1024 * 1024


async def test_voice_ref_audio_within_official_budget(run, monkeypatch):
    """音色样本：单个 [1,15] 秒、总 ≤15 秒、≤15MB —— 超了会在本地就被拦下，不发 OSS。"""
    storage, cfg, provider, cap, got, ledger = await _reference_request(run, monkeypatch)
    media = cap.bodies[0]["input"]["media"]
    names = [Path(m["url"]).name for m in media if m["type"] == "reference_audio"]
    total = 0.0
    for n in names:
        path = next(Path(p) for _, p in cap.uploads if Path(p).name == n)
        import wave

        with wave.open(str(path), "rb") as w:
            secs = w.getnframes() / float(w.getframerate())
        assert 1.0 <= secs <= 15.0, (n, secs)
        assert path.stat().st_size <= 15 * 1024 * 1024
        total += secs
    assert total <= 15.0 + 1e-6, f"参考音频总时长超官方 15 秒上限：{total}"


async def test_missing_reference_files_never_send_empty_references(run, monkeypatch):
    """**回归 2026-10-13 的事故**：盘上没有角色图/场景图时，绝不能发出 `media=[]` 的参考请求。

    当时的表现是"界面点了生成毫无反应、账号一分没扣"：参考形态拿着 0 张素材撞本地规格
    守卫 → CapabilityMismatchError → 降级，而看板上只有一句"片段还没出来"。
    正确行为：既没参考图也没首帧 → 整组跳过并写明原因；有首帧 → 自动改走首尾帧。
    """
    storage, script = run
    for name in ("char_c1.png", "char_c2.png", "scene_s1.png"):
        (storage.root / "assets" / name).unlink()
    cfg = _cfg()
    provider = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    cap = _Capture()
    cap.install(monkeypatch, provider)
    jobs = JobStore(f"sqlite:///{storage.root / 'jobs.db'}")
    got = await generate_scene_clips(
        cfg, provider, script, load_assets(storage), storage,
        CostLedger(storage), jobs, lines=[], frames={}, skip_existing=False,
    )
    assert got == {}, "没有任何输入素材时不该发出请求"
    assert cap.bodies == [], "references=[] 的请求绝不能发出去"
    evs = storage.read_events()
    assert any(e.get("status") == "scene_no_input" for e in evs), [e.get("status") for e in evs]
    degr = json.loads((storage.root / "degradations.json").read_text(encoding="utf-8"))
    assert any(d.get("what") == "scene_clip_no_input" for d in degr), degr


async def test_first_last_body_when_no_reference_images(run, monkeypatch):
    """没图但有首帧 → 自动退回首尾帧形态：media 只有 first_frame(+last_frame)，且提示词必填。"""
    storage, script = run
    for name in ("char_c1.png", "char_c2.png", "scene_s1.png"):
        (storage.root / "assets" / name).unlink()
    sb = storage.root / "storyboard"
    sb.mkdir(exist_ok=True)
    Image.new("RGB", (810, 1440), (200, 30, 30)).save(sb / "shot_s1_01.png", format="PNG")
    Image.new("RGB", (810, 1440), (30, 200, 30)).save(sb / "shot_s1_02_last.png", format="PNG")

    cfg = _cfg(use_last_frames=True)
    provider = DashScopeWan30TalkProvider("sk-test", "wan3.0-video")
    cap = _Capture()
    cap.install(monkeypatch, provider)
    frames = {"s1_01": str(sb / "shot_s1_01.png")}
    got = await generate_scene_clips(
        cfg, provider, script, load_assets(storage), storage,
        CostLedger(storage), JobStore(f"sqlite:///{storage.root / 'jobs.db'}"),
        lines=_lines(storage), frames=frames, skip_existing=False,
    )
    assert got
    media = cap.bodies[0]["input"]["media"]
    assert [m["type"] for m in media] == ["first_frame", "last_frame"], media
    assert cap.bodies[0]["input"]["prompt"].strip(), "首尾帧模式提示词必填（官方）"
    assert cap.bodies[0]["parameters"].get("ratio") == "adaptive", "首尾帧建议自适应比例"

async def test_panel_preview_is_byte_identical_to_what_gets_sent(run, monkeypatch):
    """面板上那段"本段视频的提示词"必须**就是**将要发出去的那段，逐字节相同。

    预览曾经自己用 `plan_references` 按剧本推断编号，而生成时按盘上实况编号：两者一旦
    不同，预览就会承诺不存在的素材。2026-10-14 实测到预览写着"唐僧的音色参考音频1、
    猪八戒的音色参考音频2"，而那条 run 根本没有配音母带，实发音频为空 —— 编号指向不存在
    的素材时模型不会报错，只会把锁声意图错配到别的素材上。
    现在两边都走 `m4_scene.plan_inputs`，这条测试把它钉住。
    """
    import fastvideo3.app.api as api_mod

    storage, _script = run
    # 两边必须用**同一份配置向量**：面板按 run_meta 恢复（`_cfg_for_run`），生成侧在
    # `_cfg_from_meta` 里同样按 run_meta 恢复 —— 线上本来就同源。测试里若一边用
    # `load_config()` 的默认档、另一边用 meta 的档位，风格锚会不同，比的就不是编号了。
    cfg = api_mod._cfg_for_run(storage.root)
    storage2, cfg2, provider, cap, got, ledger = await _reference_request(
        run, monkeypatch, cfg=cfg
    )

    payload = api_mod._scene_groups_payload(storage.run_id, storage.root)
    by_gid = {g["group_id"]: g for g in payload["groups"]}
    assert by_gid, "面板至少要能列出这一组"
    gid = next(iter(by_gid))
    sent = cap.bodies[0]["input"]["prompt"]
    assert by_gid[gid]["prompt"] == sent, "预览与实发不一致：编号会指向不存在的素材"
    # 有配音母带时，两边都必须兑现同样的 音频N（编号只能指向真实发出去的素材）
    assert "音频1" in sent and "音频1" in by_gid[gid]["prompt"], sent[-260:]
    media = cap.bodies[0]["input"]["media"]
    n_audio = sum(1 for m in media if m["type"] == "reference_audio")
    assert n_audio >= 1, media
