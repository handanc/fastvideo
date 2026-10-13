"""三类闸门（2026-10-12 裁定：与分辨率无关，任何档都必须跑）+ 分辨率/中途改模型/删成片。

1. 结构性闸门：产物存在性、规格合规、成本熔断（已有测试覆盖大半，这里补规格与分辨率映射）
2. 技术 QC：黑帧/静帧/静音/时长偏差/黑边（utils/qc.py，纯 ffmpeg+PIL，零 API）
3. 身份一致性客观层：SFace 人脸 embedding（utils/faceid.py）—— 权重缺失必须**优雅降级**，
   绝不抛错、绝不误判（"没量到"≠"不合格"，否则就会像口型度量那样制造假阴性去触发付费重拍）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.config import (
    RESOLUTION_TO_QUALITY,
    quality_for_resolution,
    resolution_for_quality,
)
from fastvideo3.utils import ffmpeg as ff
from fastvideo3.utils import qc
from fastvideo3.utils.faceid import COSINE_THRESHOLD, available, identity_similarity, summarize


# ─────────────────────────── 技术 QC：判据必须真抓得住 ───────────────────────────
def _solid(tmp_path: Path, name: str, color: tuple[int, int, int], size=(64, 112)) -> Path:
    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return p


def test_qc_flags_black_and_silent_clip(tmp_path):
    """全黑 + 无音轨 = 两条硬缺陷，必须进 failed（不是 warning）。"""
    black = _solid(tmp_path, "black.png", (0, 0, 0))
    clip = tmp_path / "black.mp4"
    ff.image_to_video(black, clip, duration=2.0)
    r = qc.check_media(clip, tmp_path / "w", label="黑屏测试")
    assert any("黑帧" in f for f in r["failed"]), r
    assert any("静音" in f or "没有音轨" in str(r["metrics"].get("audio")) for f in r["failed"]), r
    assert r["metrics"]["audio"]["has_audio"] is False


def test_qc_passes_a_normal_clip_with_audio(tmp_path):
    """有画面有声音的正常片段不该被误判（假阳性会让用户不信任这一层）。"""
    from fastvideo3.utils.audio import concat_wavs  # noqa: F401  仅为确认音频工具可用

    img = _solid(tmp_path, "mid.png", (120, 90, 60))
    wav = tmp_path / "tone.wav"
    ff.run_ffmpeg(
        ["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "16000",
         "-ac", "1", str(wav)]
    )
    clip = tmp_path / "ok.mp4"
    ff.image_to_video(img, clip, duration=2.0, audio=wav)
    r = qc.check_media(clip, tmp_path / "w", expected_duration=2.0, label="正常片段")
    assert r["failed"] == [], r
    assert r["metrics"]["audio"]["has_audio"] is True


def test_qc_flags_duration_deviation(tmp_path):
    """时长偏差 >15% 必须拦：字幕窗口是按镜头时长排的，偏了就是音画字幕三者错位。"""
    img = _solid(tmp_path, "x.png", (80, 80, 80))
    clip = tmp_path / "short.mp4"
    ff.image_to_video(img, clip, duration=2.0)
    r = qc.check_media(clip, tmp_path / "w", expected_duration=5.0, label="时长偏差")
    assert any("时长偏差" in f for f in r["failed"]), r
    assert r["metrics"]["duration_deviation"] > qc.DURATION_TOL


def test_qc_duration_severity_depends_on_what_is_compared(tmp_path):
    """同样是时长偏差 17%，判级必须不同 —— 否则 QC 会狼来了。

    - 素材 vs 它自己的驱动音频 → **failed**（字幕窗口会与画面错位，是真缺陷）
    - 成片 vs 用户填的目标时长 → **warning**（目标只是意图，实际由镜头数×每镜时长决定；
      30s 目标跑出 24s 是常态，判死会让几乎每条 run 都红，真该拦的黑帧/静音就没人看了）
    """
    img = _solid(tmp_path, "d.png", (80, 80, 80))
    clip = tmp_path / "d.mp4"
    ff.image_to_video(img, clip, duration=2.0)
    # 用 2.0s 的素材比 3.0s 的预期 → 偏差 33%，稳稳超过 15% 容忍（贴近阈值会因
    # 容器时长取整而抖动，测试要离阈值远一点）
    strict = qc.check_media(clip, tmp_path / "a", expected_duration=3.0, label="素材")
    loose = qc.check_media(
        clip, tmp_path / "b", expected_duration=3.0, strict_duration=False, label="成片"
    )
    assert any("时长偏差" in f for f in strict["failed"]), strict
    # 同一份素材、同一个偏差：换成"意图"口径后只剩警告（这条素材没音轨，
    # 所以 failed 里仍会有"静音" —— 那是另一条判据，不该被这次的口径改动影响）
    assert not any("时长偏差" in f for f in loose["failed"]), loose
    assert any("时长偏差" in w for w in loose["warnings"]), loose
    assert any("静音" in f for f in loose["failed"]), "静音判据不受时长口径影响"


def test_qc_spec_checks_size_and_aspect(tmp_path):
    img = _solid(tmp_path, "v.png", (60, 60, 60), size=(54, 96))
    clip = tmp_path / "v.mp4"
    ff.image_to_video(img, clip, duration=1.0)
    bad = qc.check_spec(clip, size=(1080, 1920), fps=24, aspect="9:16")
    assert any("分辨率" in f for f in bad["failed"]), bad
    good = qc.check_spec(clip, size=(54, 96), fps=24, aspect="9:16")
    assert good["failed"] == [], good
    wrong_aspect = qc.check_spec(clip, size=(54, 96), fps=24, aspect="16:9")
    assert any("画幅" in f for f in wrong_aspect["failed"]), wrong_aspect


def test_qc_report_separates_not_checked_from_passed(tmp_path):
    """`not_checked` 必须单列：它是"没量"，不是"量了没问题"。

    这正是 VLM 终审的老毛病 —— 解析失败落 0.6 兜底分并 passed=True，异常被写成"通过"。
    """
    img = _solid(tmp_path, "y.png", (90, 90, 90))
    clip = tmp_path / "y.mp4"
    ff.image_to_video(img, clip, duration=1.0)
    rep = qc.qc_report([qc.check_media(clip, tmp_path / "w", label="x")])
    assert rep["passed"] in (True, False)
    assert any("文字/水印" in n for n in rep["not_checked"]), rep
    assert rep["not_checked"], "没有 OCR 后端时必须明说没检，而不是当作通过"


def test_qc_still_frame_is_warning_not_failure_unless_allowed(tmp_path):
    """静帧是**警告**不是失败：m4 的 `_still_fallback` 会合法产出静帧兜底镜头。"""
    img = _solid(tmp_path, "s.png", (70, 70, 70))
    clip = tmp_path / "s.mp4"
    ff.image_to_video(img, clip, duration=2.0)
    r = qc.check_media(clip, tmp_path / "w", label="静帧")
    assert r["metrics"]["frames"].get("frozen") is True
    assert not any("不动" in f for f in r["failed"]), "静帧不该直接判死"
    assert any("不动" in w for w in r["warnings"]), r
    r2 = qc.check_media(clip, tmp_path / "w2", allow_still=True, label="合法静帧")
    assert r2["metrics"].get("still_allowed") is True


# ─────────────────── 身份客观层：缺权重必须优雅降级 ───────────────────
def test_faceid_degrades_without_weights(tmp_path):
    """权重不在（本机下载不通）时：available=False、verdict=inconclusive、**不抛错**。"""
    a = _solid(tmp_path, "a.png", (200, 180, 160))
    b = _solid(tmp_path, "b.png", (30, 60, 90))
    r = identity_similarity(a, b)
    assert isinstance(r, dict)
    if not available():
        assert r["available"] is False and r["verdict"] == "inconclusive"
        assert r["cosine"] is None
        assert "权重" in r["reason"]
    else:  # 权重齐了就真的量：纯色块检不到人脸 → 仍然是 inconclusive，不是 different
        assert r["verdict"] in ("inconclusive", "different", "same")


def test_faceid_summarize_excludes_inconclusive_from_rate():
    """`inconclusive` 不进达标率分母：把"没量到"算成不合格就会触发付费重拍。"""
    rows = [
        {"cosine": 0.71, "verdict": "same", "available": True},
        {"cosine": 0.12, "verdict": "different", "available": True},
        {"cosine": None, "verdict": "inconclusive", "available": True},
    ]
    s = summarize(rows)
    assert s["faceid_measured"] == 2
    assert s["faceid_same_rate"] == 0.5
    assert s["faceid_min_cosine"] == 0.12
    empty = summarize([{"cosine": None, "verdict": "inconclusive", "available": False}])
    assert empty["faceid_same_rate"] is None and empty["faceid_measured"] == 0


def test_gate_uses_objective_layer_even_when_vlm_untrusted():
    """VLM 自校准锚不可信时，客观层（SFace）仍应能拦 —— 以前这种情况一律放弃重做。"""
    from fastvideo3.core.config import AppConfig
    from fastvideo3.modules.m9_metrics import gate_failures

    cfg = AppConfig(fake=True, consistency_gate="reshoot")
    report = {
        "judge_trusted": False,  # VLM 不可信
        "per_shot": [
            {"shot": "s1", "verified": False, "face": None, "outfit": None,
             "face_verdict": "different", "face_cosine": 0.11},
            {"shot": "s2", "verified": False, "face": None, "outfit": None,
             "face_verdict": "inconclusive", "face_cosine": None},
        ],
    }
    bad = gate_failures(report, cfg)
    assert [b["shot"] for b in bad] == ["s1"], "客观判不同人要拦；inconclusive 不拦"
    assert bad[0]["evidence"] == "sface"
    assert COSINE_THRESHOLD == 0.363, "阈值是 OpenCV 对 SFace 的官方推荐值，不许私调"
    # 度量关闭时不烧钱
    cfg2 = AppConfig(fake=True, consistency_gate="off")
    assert gate_failures(report, cfg2) == []


# ─────────── 多人同框的身份判定：只比 characters[0] 会烧钱 ───────────
def _stub_faces(monkeypatch, tmp_path, vectors: dict[str, list[float]]):
    """把检测/嵌桩成"文件内容 = 向量名"，测试于是与权重、cv2、下载全都无关。

    真实向量只在 `_cosine` 那条纯算术路上被用到，所以这样桩出来的判定就是线上判定。
    """
    import numpy as np

    from fastvideo3.utils import faceid as _f

    def _img(path):
        return type("Img", (), {"tag": Path(path).read_text(encoding="utf-8")})()

    def _faces(img):
        """有向量 → 一张脸；向量写成空列表 → 检不到脸（背影/远景/空镜）。"""
        if img.tag not in vectors:
            return []
        return [img.tag] if vectors[img.tag] else []

    monkeypatch.setattr(_f, "available", lambda: True)
    monkeypatch.setattr(_f, "_read_bgr", _img)
    monkeypatch.setattr(_f, "_faces", _faces)
    monkeypatch.setattr(_f, "_feature", lambda img, face: np.array(vectors[face], dtype=float))
    return _f


def _write_img(tmp_path, name: str, tag: str) -> Path:
    p = tmp_path / f"{name}.png"
    p.write_text(tag, encoding="utf-8")
    return p


def test_multi_ref_same_when_any_declared_character_matches(monkeypatch, tmp_path):
    """双人镜头里第一个人这一帧正好侧脸（余弦掉到阈值下）→ **不能**判身份漂移。

    线上真实假阳性（2026-10-12 权重就位后首次真跑）：同帧 唐僧 0.306 / 国王 0.4286，
    旧规则只比 characters[0]=唐僧 → different → 白花一次重拍钱。新规则比全部角色，
    只要有一个对得上就算 same。
    """
    f = _stub_faces(monkeypatch, tmp_path, {"A": [1.0, 0.0], "B": [0.0, 1.0], "frame": [1.0, 0.0]})
    refs = {"tang_seng": _write_img(tmp_path, "r1", "A"), "guo_wang": _write_img(tmp_path, "r2", "B")}
    r = f.identity_similarity_multi(refs, _write_img(tmp_path, "fr", "frame"))
    assert r["verdict"] == "same", r
    assert r["best_char"] == "tang_seng" and r["cosine"] == pytest.approx(1.0, abs=1e-6)
    # 逐角色证据都留着：报告要能解释"谁对上了、谁这一帧没量到"
    assert r["per_char"]["guo_wang"]["verdict"] == "different"
    assert r["per_char"]["tang_seng"]["ref_faces"] == 1


def test_multi_ref_different_only_when_no_character_matches(monkeypatch, tmp_path):
    """有脸但一张都对不上任何声明角色 → 这才是真的该拦（整批脸换了 / 换成别人了）。"""
    f = _stub_faces(monkeypatch, tmp_path, {"A": [1.0, 0.0], "B": [0.0, 1.0], "frame": [0.0, -1.0]})
    # frame = [0,-1]：与 B 反平行、与 A 垂直 → 两个角色都 ≤ 阈值
    refs = {"a": _write_img(tmp_path, "r1", "A"), "b": _write_img(tmp_path, "r2", "B")}
    r = f.identity_similarity_multi(refs, _write_img(tmp_path, "fr", "frame"))
    assert r["verdict"] == "different" and r["cosine"] is not None
    assert "没有一个" in r["reason"] or "没有一张" in r["reason"], r["reason"]


def test_multi_ref_no_face_is_inconclusive(monkeypatch, tmp_path):
    """背影/远景/空镜（检不到脸）→ inconclusive，不进闸门、不进达标率分母。"""
    f = _stub_faces(monkeypatch, tmp_path, {"A": [1.0, 0.0], "frame_noface": []})
    refs = {"a": _write_img(tmp_path, "r1", "A")}
    r = f.identity_similarity_multi(refs, _write_img(tmp_path, "fr", "frame_noface"))
    assert r["available"] is True and r["verdict"] == "inconclusive"
    assert r["frame_faces"] == 0 and r["cosine"] is None


def test_multi_ref_missing_cards_and_unavailable_layer_never_raise(monkeypatch, tmp_path):
    """没有任何角色图 / 客观层不可用：返回 inconclusive + 说明，绝不抛错（度量不该拖垮交付）。"""
    from fastvideo3.utils import faceid as _f

    r = _f.identity_similarity_multi({}, tmp_path / "nope.png")
    assert r["verdict"] == "inconclusive" and "参考图" in r["reason"]

    monkeypatch.setattr(_f, "available", lambda: False)
    refs = {"a": _write_img(tmp_path, "r1", "A")}
    r2 = _f.identity_similarity_multi(refs, _write_img(tmp_path, "fr", "frame"))
    assert r2["available"] is False and r2["verdict"] == "inconclusive"


# ─────────────────────────── 分辨率 ↔ 档位映射 ───────────────────────────
def test_resolution_mapping_is_single_source_and_reversible():
    assert list(RESOLUTION_TO_QUALITY) == ["480P", "720P", "1080P"]
    assert quality_for_resolution("720P") == "high"
    assert quality_for_resolution("1080p") == "cinema", "大小写要宽容"
    assert quality_for_resolution("", "economy") == "economy"
    assert quality_for_resolution("4K", "high") == "high", "认不出来不猜，回落"
    for res, q in RESOLUTION_TO_QUALITY.items():
        assert resolution_for_quality(q) == res


def test_api_resolutions_endpoint_lists_three():
    from fastapi.testclient import TestClient

    from fastvideo3.app.api import app

    with TestClient(app) as c:
        r = c.get("/api/resolutions")
        assert r.status_code == 200
        rows = r.json()["resolutions"]
        assert [x["resolution"] for x in rows] == ["480P", "720P", "1080P"]
        by = {x["resolution"]: x for x in rows}
        assert by["1080P"]["output_size"] == [1080, 1920]
        assert by["1080P"]["auto_reshoot"] is True
        assert by["480P"]["auto_reshoot"] is False, "480P 量但不自动烧钱重拍"
        assert by["480P"]["cost_30s"][0] < by["1080P"]["cost_30s"][0]


def test_resolution_label_actually_drives_output_size():
    """面板写 720P 就必须真出 720P —— 档位 yaml 的数字不能盖过用户看得见的规格承诺。

    这是本轮真抓到的 bug：720P 映射到 `high` 档，而 `profile_high.yaml` 的 resolution 是
    1080×1920，于是"选 720P、出 1080P"，成本与规格双双对不上。
    """
    from fastvideo3.core.config import RESOLUTION_SIZES, apply_resolution, load_config

    for tier, size in RESOLUTION_SIZES.items():
        res, aspect = tier
        cfg = apply_resolution(load_config(quality=quality_for_resolution(res)), res)
        cfg.aspect_ratio = aspect
        cfg = apply_resolution(cfg, res)  # 画幅变了要重算（尺寸表是 分辨率×画幅 两维）
        assert tuple(cfg.resolution) == size, (tier, cfg.resolution)
        assert cfg.i2v_resolution == res, (tier, cfg.i2v_resolution)
    # 认不出来的分辨率不动 cfg（不猜规格）
    cfg = load_config(quality="cinema")
    before = tuple(cfg.resolution)
    apply_resolution(cfg, "4K")
    assert tuple(cfg.resolution) == before


def test_create_run_720p_restores_720p_size(outdir, monkeypatch):
    """建任务时选 720P → 落 meta → 续跑恢复出来的 cfg 也是 720×1280（两条路径同一口径）。"""
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        r = c.post("/api/runs", json={"topic": "720", "resolution": "720P", "fake": True})
        assert r.status_code == 201, r.text
        rid = r.json()["run_id"]
        cfg = api_mod._cfg_for_run(outdir / rid)
        assert tuple(cfg.resolution) == (720, 1280), cfg.resolution
        assert cfg.i2v_resolution == "720P"


def test_api_resolutions_sizes_are_truthful():
    from fastapi.testclient import TestClient

    from fastvideo3.app.api import app

    with TestClient(app) as c:
        rows = {x["resolution"]: x for x in c.get("/api/resolutions").json()["resolutions"]}
    assert rows["480P"]["output_size"] == [540, 960]
    assert rows["720P"]["output_size"] == [720, 1280], "面板显示的尺寸必须是真的"
    assert rows["1080P"]["output_size"] == [1080, 1920]
    assert rows["720P"]["i2v_resolution"] == "720P"


def test_create_run_with_resolution_persists_tier_and_maps_quality(outdir, monkeypatch):
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        r = c.post("/api/runs", json={"topic": "分辨率", "resolution": "720P", "fake": True})
        assert r.status_code == 201, r.text
        meta = json.loads((outdir / r.json()["run_id"] / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["resolution_tier"] == "720P"
        assert meta["quality"] == "high", "分辨率决定内部档位"


def test_cli_new_accepts_resolution(outdir):
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(app, ["new", "--topic", "CLI分辨率", "--resolution", "1080P", "--fake"])
    assert r.exit_code == 0, r.output
    meta = json.loads(sorted(Path(outdir).glob("run_*/run_meta.json"))[-1].read_text(encoding="utf-8"))
    assert meta["resolution_tier"] == "1080P" and meta["quality"] == "cinema"
    bad = CliRunner().invoke(app, ["new", "--topic", "x", "--resolution", "4K"])
    assert bad.exit_code != 0 and "未知分辨率" in bad.output


# ─────────────────── 中途改模型 / 删成片 ───────────────────
@pytest.fixture
def client(outdir, monkeypatch):
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


def _mk(client, outdir, **over):
    body = {"topic": "中途改模型", "quality": "cinema", "fake": True, **over}
    r = client.post("/api/runs", json=body)
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]
    return rid, outdir / rid


def test_patch_models_mid_run_updates_meta(client, outdir):
    rid, root = _mk(client, outdir)
    r = client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"i2v": "wan2.7-i2v"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_overrides"] == {"i2v": "wan2.7-i2v"}
    assert body["effective"]["i2v"] == "wan2.7-i2v"
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["model_overrides"] == {"i2v": "wan2.7-i2v"}
    # 再改一次：合并而不是覆盖丢失（空串＝撤回该项）
    client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"t2i": "wan2.7-image"}})
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["model_overrides"] == {"i2v": "wan2.7-i2v", "t2i": "wan2.7-image"}
    client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"i2v": ""}})
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["model_overrides"] == {"t2i": "wan2.7-image"}, "空串＝撤回，回到跟随档位"


def test_patch_models_rejects_bad_name(client, outdir):
    rid, _ = _mk(client, outdir)
    r = client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"i2v": "sora-2"}})
    assert r.status_code == 400 and "不支持模型" in r.json()["detail"]


def test_delete_final_removes_video_and_reopens_the_step(client, outdir):
    rid, root = _mk(client, outdir)
    (root / "final").mkdir(parents=True, exist_ok=True)
    (root / "final" / "final.mp4").write_bytes(b"\x00" * 32)
    (root / "final" / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    (root / "final" / "qc_report.json").write_text("{}", encoding="utf-8")
    r = client.delete(f"/api/runs/{rid}/final")
    assert r.status_code == 200, r.text
    assert "final/final.mp4" in r.json()["removed"]
    assert not (root / "final" / "final.mp4").exists()
    assert not (root / "final" / "qc_report.json").exists(), "旧 QC 报告不能留着冒充合格"
    # 再删一次 → 404（没有成片可删）
    assert client.delete(f"/api/runs/{rid}/final").status_code == 404


def test_duration_bounds_are_one_to_3600_on_both_paths(client, outdir):
    """时长 1~3600 的边界在 API 与 CLI 上是同一份（config.DURATION_MIN/MAX）。

    钉住边界而不只是"能跑"：下限 0 会让剧本没有目标长度、成本预估归零，且熔断判据
    （`exceeded()` 用 `>=`）在第一步就跳闸 —— 表现是"点了没反应"，最难查。
    """
    from fastvideo3.core.config import DURATION_MAX, DURATION_MIN

    assert (DURATION_MIN, DURATION_MAX) == (1, 3600)
    for bad in (0, -5, 3601):
        r = client.post("/api/runs", json={"topic": "x", "duration": bad, "fake": True})
        assert r.status_code == 422, (bad, r.status_code)
    for good in (1, 3600):
        r = client.post("/api/runs", json={"topic": "x", "duration": good, "fake": True})
        assert r.status_code == 201, (good, r.text)

    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    runner = CliRunner()
    bad = runner.invoke(app, ["new", "--topic", "x", "--duration", "0", "--fake"])
    assert bad.exit_code != 0 and "1~3600" in bad.output
    ok = runner.invoke(app, ["new", "--topic", "x", "--duration", "3600", "--fake"])
    assert ok.exit_code == 0, ok.output


def test_model_choices_includes_talk_models_with_their_modes(client, outdir):
    """口型模型与其它模型同栏显示，但**落盘字段是 talk_mode**（不是 model_overrides.talk）。

    为什么不让它进 model_overrides：通道与输入形态绑死（s2v 只吃图片+音频、videoretalk 只吃
    视频、wan3.0 只吃图片+提示词），而模式还决定"要不要为对白镜头出运动片段、成片音轨用谁的、
    字幕按谁的时间轴排"。两处各存一份就是本项目反复出问题的形状（两次实测：对白镜头先烧
    一份运动片段的钱、再 100% 降级成无口型）。
    """
    r = client.get("/api/model-choices?quality=cinema")
    assert r.status_code == 200
    talk = r.json()["talk"]
    # 下拉的值是**模型名**（"不做口型"为空串），mode 由后端解析 —— 前端不自己维护映射
    assert talk["field"] == "talk_model"
    by_label = {c["label"]: c for c in talk["choices"]}
    assert by_label["wan2.2-s2v"]["mode"] == "s2v_direct"
    assert by_label["videoretalk"]["mode"] == "v2v_overlay"
    assert by_label["wan3.0-video"]["mode"] == "scripted_talk"
    # 同一条链的优速版：mode 相同、模型名不同（只有落进 model_overrides['talk'] 才区分得出）
    assert by_label["wan3.0-video-prime"]["mode"] == "scripted_talk"
    assert by_label["wan3.0-video-prime"]["value"] == "wan3.0-video-prime"
    assert by_label["不做口型"]["mode"] == "audio_overlay"
    assert by_label["不做口型"]["value"] == ""
    # 顺序：优速版排在"不做口型"之前
    labels = [c["label"] for c in talk["choices"]]
    assert labels.index("wan3.0-video-prime") < labels.index("不做口型"), labels
    assert by_label["videoretalk"]["needs_clip_for_dialogue"] is True
    assert by_label["wan3.0-video"]["voices_script"] is True
    assert talk["switch_note"], "中途换口型模型的代价必须在选择处就说清楚"


def test_patch_talk_model_mid_run_updates_mode_and_channel(client, outdir):
    """生成过程中换口型模型：写进 run_meta.talk_mode，续跑时按它装配对应通道。"""
    rid, root = _mk(client, outdir)
    # 用模型名换（前端显示的就是模型名）
    r = client.patch(f"/api/runs/{rid}/models", json={"talk_mode": "wan3.0-video"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["talk_mode"] == "scripted_talk"
    assert body["talk_channel"] == "wan3.0-video"
    assert body["talk_provider"] == "dashscope_wan30_talk"
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["talk_mode"] == "scripted_talk"
    # 续跑恢复出来的 cfg 必须装配成同一条通道（cli._cfg_from_meta / api._cfg_for_run 同源）
    import fastvideo3.app.api as api_mod
    from fastvideo3.core.talkmode import provider_name

    cfg = api_mod._cfg_for_run(root)
    assert cfg.talk_mode == "scripted_talk"
    assert provider_name(cfg, cfg.talk_mode) == "dashscope_wan30_talk"
    # 换回 A 链也一样成立
    r2 = client.patch(f"/api/runs/{rid}/models", json={"talk_mode": "wan2.2-s2v"})
    assert r2.json()["talk_provider"] == "dashscope_wan22_s2v"


def test_patch_talk_model_empty_string_means_no_lipsync(client, outdir, monkeypatch, tmp_path):
    """面板选「不做口型」提交**空串**：它是显式意图，不是"没填"。

    早先这里用 `.strip()` 判空，空串被当缺席 → talk_mode 保持原样、保存后什么都没变，
    而这条 run 之后还在按旧通道花钱出对口型。

    `.env` 里**必须**留着一个 FV3_TALK_MODEL：`current_value` 走的是"run 点名 > .env >
    模式默认"，只有 .env 有残留值时，"audio_overlay 不该显示任何模型"这条才真的在被测。
    """
    import fastvideo3.app.api as api_mod

    env = tmp_path / ".env"
    env.write_text("DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=wan3.0-video-prime\n", encoding="utf-8")
    real_load = api_mod.load_config
    monkeypatch.setattr(
        api_mod, "load_config", lambda **kw: real_load(**{**kw, "dotenv_path": env})
    )

    rid, root = _mk(client, outdir)
    client.patch(f"/api/runs/{rid}/models", json={"talk_model": "wan3.0-video-prime"})
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["model_overrides"]["talk"] == "wan3.0-video-prime"

    r = client.patch(f"/api/runs/{rid}/models", json={"talk_model": ""})
    assert r.status_code == 200, r.text
    assert r.json()["talk_mode"] == "audio_overlay"
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["talk_mode"] == "audio_overlay"
    # 旧点名必须**撤掉**：留着它，面板会把 current_value 读成那个模型（overrides 优先于
    # 模式默认），于是"显示 wan3.0-video-prime、实际不做口型"
    assert "talk" not in meta["model_overrides"], meta["model_overrides"]
    talk = client.get("/api/model-choices", params={"run_id": rid}).json()["talk"]
    assert talk["current"] == "audio_overlay"
    assert talk["current_value"] == ""
    assert talk["current_label"] == "不做口型"
    # 字段缺席 = 不改（不能被上面的空串语义带偏）
    keep = client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"t2i": "wan2.7-image"}})
    assert keep.json()["talk_mode"] == "audio_overlay"


def test_patch_talk_model_rejects_unknown_and_warns_on_conflict(client, outdir):
    rid, _ = _mk(client, outdir, resolution="480P")
    bad = client.patch(f"/api/runs/{rid}/models", json={"talk_mode": "sora-2"})
    assert bad.status_code == 400 and "未知口型模型" in bad.json()["detail"]
    # 480P + videoretalk：片段边长 <640 会被服务端拒 → 允许改，但必须把代价说出来
    warn = client.patch(f"/api/runs/{rid}/models", json={"talk_mode": "videoretalk"})
    assert warn.status_code == 200
    assert "640" in warn.json()["talk_warning"]
    bad_voice = client.patch(f"/api/runs/{rid}/models", json={"voice_source": "both"})
    assert bad_voice.status_code == 400


def test_talk_model_choice_drives_mode_and_assembled_provider(client, outdir):
    """选「wan3.0-video-prime」→ 模式 scripted_talk + 模型覆盖，装配出的就是 prime。

    同一条链有多个模型时，只落 mode 区分不出选了哪个 —— 所以模型名必须进
    `model_overrides['talk']`，并由 `_wan30_talk_model` 优先采用。
    """
    r = client.post(
        "/api/runs",
        json={"topic": "优速口型", "quality": "cinema", "fake": True,
              "talk_model": "wan3.0-video-prime"},
    )
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]
    meta = json.loads((outdir / rid / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["talk_mode"] == "scripted_talk"
    assert meta["model_overrides"]["talk"] == "wan3.0-video-prime"

    import fastvideo3.app.api as api_mod
    from fastvideo3.providers.base import Capability
    from fastvideo3.providers.factory import make_provider

    cfg = api_mod._cfg_for_run(outdir / rid)
    assert cfg.talk_mode == "scripted_talk"
    p = make_provider(Capability.TALK, cfg)
    assert p.name == "dashscope_wan30_talk" and p.model == "wan3.0-video-prime"


def test_talk_model_empty_string_means_no_lipsync(client, outdir):
    """下拉里「不做口型」的值是空串 —— 必须与"没提供该字段"区分开。"""
    r = client.post(
        "/api/runs",
        json={"topic": "不要口型", "quality": "cinema", "fake": True, "talk_model": ""},
    )
    assert r.status_code == 201, r.text
    meta = json.loads((outdir / r.json()["run_id"] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["talk_mode"] == "audio_overlay"
    assert "talk" not in (meta["model_overrides"] or {})


def test_unknown_talk_model_is_rejected_with_choices(client, outdir):
    r = client.post(
        "/api/runs",
        json={"topic": "x", "quality": "cinema", "fake": True, "talk_model": "sora-2"},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "未知口型模型" in detail and "wan3.0-video-prime" in detail


def test_patch_models_still_works_without_talk_field(client, outdir):
    """只改文本/图像模型时不该被口型那段逻辑影响（talk_mode 保持原值）。"""
    rid, root = _mk(client, outdir)
    before = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))["talk_mode"]
    r = client.patch(f"/api/runs/{rid}/models", json={"model_overrides": {"text": "qwen3.7-plus"}})
    assert r.status_code == 200, r.text
    assert r.json()["talk_mode"] == before
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["talk_mode"] == before
    assert meta["model_overrides"] == {"text": "qwen3.7-plus"}


def test_board_exposes_tech_qc(client, outdir):
    rid, root = _mk(client, outdir)
    (root / "final").mkdir(parents=True, exist_ok=True)
    (root / "final" / "qc_report.json").write_text(
        json.dumps({"passed": False, "failed": [{"item": "成片", "problem": "静音"}],
                    "warnings": [], "not_checked": ["文字/水印"], "items": 1}),
        encoding="utf-8",
    )
    b = client.get(f"/api/runs/{rid}/board").json()
    assert b["tech_qc"]["passed"] is False
    assert b["tech_qc"]["failed"][0]["problem"] == "静音"
