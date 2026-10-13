"""开始生成（一次一步）：进度状态机 / 全局单任务锁 / 剧本编辑 / 删除产物 / 全链路。

用户语义的硬判据：
- 每次点「开始生成」只完成最靠前的未完成步骤；
- 删了角色图不上传 → 点逐步只补那一个 → **再点一次**才进下一步；
- 编辑剧本只作废受影响的下游，其他镜头不重做；
- 换/删场景图只作废该场景的下游，其他场景一个字节不动；
- 已有任务在跑时，任何新建/续跑/单步请求一律 409。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import progress as prog
from fastvideo3.core import tasklock
from fastvideo3.core.config import load_config
from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.storage import RunStorage


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[attr-defined]
        yield c


def _cfg():
    return load_config(dotenv_path="none.env", quality="economy")


def _progress(outdir, run_id):
    root = outdir / run_id
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    return prog.compute_progress(RunStorage(run_id, root=outdir), meta, _cfg())


def test_progress_walks_five_steps_in_order(outdir):
    root, script = _mk_run(outdir, "run_20260101_000020", with_final=False)
    # 清掉产物，从空 run 开始
    for p in (root / "assets").glob("*.png"):
        p.unlink()
    for p in (root / "storyboard").glob("*.png"):
        p.unlink()
    for p in (root / "clips").glob("*.mp4"):
        p.unlink()
    rep = _progress(outdir, "run_20260101_000020")
    assert rep.next.id == "script" or rep.next.id == "chars"  # 剧本在盘 → 直接到角色图
    (root / "script.json").unlink()
    rep = _progress(outdir, "run_20260101_000020")
    assert rep.next.id == "script"


def test_deleted_portrait_blocks_advance_until_refilled(outdir):
    """删了角色图不上传 → 下一步仍是 chars 且 missing 只有它；补回才推进。"""
    root, script = _mk_run(outdir, "run_20260101_000021")
    (root / "assets" / "char_c1.png").unlink()
    rep = _progress(outdir, "run_20260101_000021")
    assert rep.next.id == "chars" and rep.next.missing == ["c1"]
    Image.new("RGB", (8, 8), (7, 7, 7)).save(root / "assets" / "char_c1.png")
    rep = _progress(outdir, "run_20260101_000021")
    # _mk_run 只造了首帧、没造尾帧 → 角色补齐后先回 frames 步补尾帧，再进镜头视频。
    # 尾帧缺失算 frames 的缺口是刻意的：否则没有任何一步会去补它，m4/m6 永远退回单首帧。
    assert rep.next.id == "frames", rep.next
    assert set(rep.next.missing), "缺口要点名要补尾帧的镜头"


def test_step_endpoint_spawns_and_lock_rejects_second(client, outdir):
    _mk_run(outdir, "run_20260101_000022")
    r = client.post("/api/runs/run_20260101_000022/step")
    assert r.status_code == 202, r.text
    assert client.spawned[-1][-2:] == ["step", "run_20260101_000022"]
    # API 在 spawn 前就持锁（子进程认领）；测试里子进程被替掉不运行，锁仍在 → 后续全 409
    r2 = client.post("/api/runs/run_20260101_000022/step")
    assert r2.status_code == 409
    r3 = client.post(
        "/api/runs/run_20260101_000022/script/regenerate",
        json={"prompt": "换个思路", "mode": "rewrite"},
    )
    assert r3.status_code == 409, "剧本重生成也要被同一把锁拦住（一次只烧一份钱）"
    # 建任务不花钱 → 不该被锁拦住（此前 e2e 分支会在建任务时就 409）
    assert client.post("/api/runs", json={"topic": "另一个任务"}).status_code == 201
    assert client.get("/api/task").json()["busy"] is True
    tasklock.release(outdir, "run_20260101_000022")
    assert client.get("/api/task").json()["busy"] is False


def test_create_run_does_not_spawn(client, outdir):
    r = client.post("/api/runs", json={"topic": "新任务", "fake": True})
    assert r.status_code == 201
    assert client.spawned == []  # 建任务不启动任何生成
    rid = r.json()["run_id"]
    meta = json.loads((outdir / rid / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["base_info"]["topic"] == "新任务"
    rep = client.get(f"/api/runs/{rid}/steps").json()
    assert rep["next"]["id"] == "script"


def test_put_script_validates_and_invalidates_downstream(client, outdir):
    root, script = _mk_run(outdir, "run_20260101_000023")
    payload = json.loads((root / "script.json").read_text(encoding="utf-8"))
    # 悬空引用必须被拒且不落盘
    bad = json.loads(json.dumps(payload))
    bad_talk = next(sh for sc in bad["scenes"] for sh in sc["shots"] if sh.get("dialogues"))
    bad_talk["dialogues"][0]["character_id"] = "ghost"
    r = client.put("/api/runs/run_20260101_000023/script", json=bad)
    assert r.status_code == 422
    assert "ghost" not in (root / "script.json").read_text(encoding="utf-8")
    # 改一句台词：只作废该镜头的配音/口型/成片，不动其他镜头
    good = json.loads(json.dumps(payload))
    shot = next(sh for sc in good["scenes"] for sh in sc["shots"] if sh.get("dialogues"))
    shot["dialogues"][0]["text"] = "换一句台词"
    r = client.put("/api/runs/run_20260101_000023/script", json=good)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 2 and (root / "script.bak.json").exists()
    hit = [x for x in body["items"] if x.startswith(("audio/line_", "talk/", "final/"))]
    assert hit and all(shot["shot_id"] in x or x.startswith("final/") for x in hit)
    # 再保存一次无实质修改（m3 回写 visual_prompt 不算修改）→ 不应再作废
    storage = RunStorage("run_20260101_000023", root=outdir)
    storage.clear_stale("final/final.mp4")
    r2 = client.put("/api/runs/run_20260101_000023/script", json=good)
    assert r2.json()["invalidated"] == 0


def test_delete_portrait_invalidates_only_its_shots(client, outdir):
    root, script = _mk_run(outdir, "run_20260101_000024")
    storage = RunStorage("run_20260101_000024", root=outdir)
    # 先建立血缘：假装 c1 出镜的帧消费过这张定妆照
    shots_c1 = [sh.shot_id for sc in script.scenes for sh in sc.shots if "c1" in sh.characters]
    for sid in shots_c1:
        storage.record_lineage(f"storyboard/shot_{sid}.png", ["assets/char_c1.png"])
    r = client.delete("/api/runs/run_20260101_000024/artifacts/assets/char_c1.png")
    assert r.status_code == 200, r.text
    assert not (root / "assets" / "char_c1.png").exists()
    # 候选图保留（删主图后仍可 ✔ 选用），但 take+1 让重生成换一张脸
    assert list((root / "assets").glob("_cand_char_c1_*.png"))
    takes = json.loads((root / "takes.json").read_text(encoding="utf-8"))
    assert takes["c1"] == 1
    stale = storage.stale_set()
    assert all(f"storyboard/shot_{sid}.png" in stale for sid in shots_c1)
    other = [sh.shot_id for sc in script.scenes for sh in sc.shots if "c1" not in sh.characters]
    assert not any(f"storyboard/shot_{sid}.png" in stale for sid in other)  # 其他镜头不动


def test_delete_script_sends_next_step_back_to_script(client, outdir):
    """删除剧本 → 主按钮回到「开始生成 · 生成剧本」，且旧产物不会被新剧本悄悄复用。"""
    root, script = _mk_run(outdir, "run_20260101_000026")
    storage = RunStorage("run_20260101_000026", root=outdir)
    # 上一轮的结论文件：它们描述的是即将被删掉的这部剧本，必须一起消失
    for name in ("script_review.json", "consistency_report.json",
                 "final_review.json", "run_summary.json"):
        (root / name).write_text('{"passed": true}', encoding="utf-8")
    # 用户手工上传过定妆照 → 删剧本不能把它标作废（重做会直接覆盖上传的图）
    storage.save_json("uploads.json", {"assets/char_c1.png": {"ts": "x"}})

    r = client.delete("/api/runs/run_20260101_000026/script")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["removed"] == "script.json" and not (root / "script.json").exists()
    # 前端 notice 直接引用这两个字段（归档名 / 被清的结论文件），字段名不能漂
    assert body["archived"].startswith("script.deleted.") and (root / body["archived"]).is_file()
    assert set(body["cleared"]) == {
        "script_review.json", "consistency_report.json", "final_review.json", "run_summary.json"
    }
    assert list(root.glob("script.deleted.*.json"))  # 归档，不是抹掉
    for gone in ("script_review.json", "consistency_report.json",
                 "final_review.json", "run_summary.json"):
        assert not (root / gone).exists(), gone

    # 前端主按钮文案的唯一来源：steps 的 next
    rep = client.get("/api/runs/run_20260101_000026/steps").json()
    assert rep["next"]["id"] == "script" and rep["next"]["label"] == "生成剧本"
    assert rep["next"]["missing"] == ["script.json"]
    assert [s["state"] for s in rep["steps"][1:]] == ["pending"] * 5

    # 下游全量标 stale（只标不删），否则新剧本沿用同一批 id 时会复用旧脸/旧帧
    sid = script.all_shots()[0].shot_id
    stale = storage.stale_set()
    assert {f"storyboard/shot_{sid}.png", f"clips/shot_{sid}.mp4",
            f"assets/scene_{script.scenes[0].scene_id}.png", "final/final.mp4"} <= stale
    assert "assets/char_c1.png" not in stale  # 上传过的图不受牵连
    assert (root / "storyboard" / f"shot_{sid}.png").exists()
    edges = json.loads((root / "lineage.json").read_text(encoding="utf-8"))["edges"]
    assert edges[f"storyboard/shot_{sid}.png"]["reason"] == "script_deleted"


def test_delete_script_blocked_while_busy_or_absent(client, outdir):
    root, _ = _mk_run(outdir, "run_20260101_000027")
    assert tasklock.acquire(outdir, "run_20260101_000027", "step")[0]
    r = client.delete("/api/runs/run_20260101_000027/script")
    assert r.status_code == 409  # 生成中删剧本 = 把正在写的剧本抽走
    assert (root / "script.json").exists()
    tasklock.release(outdir, "run_20260101_000027")

    _mk_run(outdir, "run_20260101_000028", with_script=False)
    r2 = client.delete("/api/runs/run_20260101_000028/script")
    assert r2.status_code == 404  # 没剧本可删，别把"删除失败"当成"已删除"


def test_scene_image_invalidation_is_scene_local(client, outdir):
    """换场景图只作废该场景下游：其他场景的帧 mtime 与内容都不变。"""
    root, script = _mk_run(outdir, "run_20260101_000025")
    storage = RunStorage("run_20260101_000025", root=outdir)
    s1 = script.scenes[0].scene_id
    s2 = script.scenes[1].scene_id
    for sc in script.scenes:
        for sh in sc.shots:
            storage.record_lineage(
                f"storyboard/shot_{sh.shot_id}.png", [f"assets/scene_{sc.scene_id}.png"]
            )
    before = {
        p.name: (p.stat().st_mtime_ns, p.read_bytes())
        for p in (root / "storyboard").glob("*.png")
    }
    r = client.put(
        f"/api/runs/run_20260101_000025/artifacts/assets/scene_{s1}.png",
        content=_png(),
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 200
    stale = storage.stale_set()
    s1_shots = {sh.shot_id for sh in script.scenes[0].shots}
    s2_shots = {sh.shot_id for sh in script.scenes[1].shots}
    assert all(f"storyboard/shot_{s}.png" in stale for s in s1_shots)
    assert not any(f"storyboard/shot_{s}.png" in stale for s in s2_shots)
    for name, (mt, blob) in before.items():
        p = root / "storyboard" / name
        assert p.stat().st_mtime_ns == mt and p.read_bytes() == blob, f"{name} 被误动"
    assert s2 not in json.dumps(sorted(stale))


def _png(w=200, h=400, color=(10, 200, 10)):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


async def test_five_step_chain_fills_only_gaps(cfg, outdir, tmp_path):
    """离线全链路：逐步点五次跑通；中途删一张角色图 → 回到 chars 且只补那一个。"""
    from fastvideo3.app.cli import _run_step
    from fastvideo3.core.stages import build_context

    cfg.fake = True
    cfg.use_reference_keyframes = True  # 让首帧真的消费定妆照，血缘才有 char 边
    run_id = "run_step_chain"
    storage = RunStorage(run_id, root=outdir)
    storage.save_json(
        "run_meta.json",
        {
            "run_id": run_id,
            "base_info": BaseInfo(topic="逐步链路", target_duration_sec=15).model_dump(),
            "fake": True, "quality": "economy", "talk_mode": "s2v_direct",
            "budget_limit_cny": 100.0,
        },
    )
    meta = storage.load_json("run_meta.json")
    ctx = build_context(cfg, storage)
    seen_steps: list[str] = []
    for _ in range(8):
        rep = prog.compute_progress(storage, meta, cfg)
        if rep.next is None:
            break
        await _run_step(ctx, storage, cfg, meta, rep.next)
        seen_steps.append(rep.next.id)
    # 六步：分镜首帧独立成步（付视频钱前的最后人工检查点）
    assert seen_steps == ["script", "chars", "scenes", "frames", "videos", "compose"]
    assert (storage.root / "final" / "final.mp4").exists()

    # 中途删一张角色图 → 状态机回到 chars，且只补 c1
    (storage.root / "assets" / "char_c1.png").unlink()
    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "chars" and rep.next.missing == ["c1"]
    await _run_step(ctx, storage, cfg, meta, rep.next)
    assert (storage.root / "assets" / "char_c1.png").exists()
    # 补完空缺后**再点一次**才推进；且因为首帧消费过旧定妆照，它们已被血缘标 stale
    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "frames"  # 先重画首帧（不烧视频钱），人工可在此拦截
    stale_shots = set(rep.next.stale)  # frames 步的 stale 存的是 shot_id
    c1_shots = {
        sh.shot_id
        for sc in Script.model_validate(storage.load_json("script.json")).scenes
        for sh in sc.shots
        if "c1" in sh.characters
    }
    other_shots = {
        sh.shot_id
        for sc in Script.model_validate(storage.load_json("script.json")).scenes
        for sh in sc.shots
        if "c1" not in sh.characters
    }
    assert stale_shots and stale_shots <= c1_shots, (
        f"只应作废消费过 c1 的镜头：stale={sorted(stale_shots)} c1={sorted(c1_shots)}"
    )
    assert not (stale_shots & other_shots), "其他角色的镜头不该被作废"
    await _run_step(ctx, storage, cfg, meta, rep.next)  # 只重做 stale 的首帧
    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "videos"  # 首帧换了 → 消费过旧帧的片段/口型沿血缘作废
    await _run_step(ctx, storage, cfg, meta, rep.next)
    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "compose"  # 成片也沿血缘被标 stale → 最后重拼
    await _run_step(ctx, storage, cfg, meta, rep.next)
    assert prog.compute_progress(storage, meta, cfg).next is None


def test_lock_released_when_endpoint_errors_after_acquire(client, outdir, monkeypatch):
    """持锁后、spawn 前抛异常必须还锁：否则死锁卡住全系统（实测发生过）。"""
    from fastvideo3.core import progress as prog_mod

    _mk_run(outdir, "run_20260101_000026")

    def boom(*a, **k):
        raise RuntimeError("compute_progress 炸了")

    # api 端点内部 `from fastvideo3.core import progress as prog` → 打模块属性即可
    orig = prog_mod.compute_progress
    monkeypatch.setattr(prog_mod, "compute_progress", boom)
    with pytest.raises(RuntimeError):  # TestClient 默认把服务端异常抛回测试
        client.post("/api/runs/run_20260101_000026/step")
    assert client.get("/api/task").json()["busy"] is False, "异常后锁没释放 → 全系统死锁"
    # 锁已还：恢复 compute_progress 后再次请求应能正常进入（不再 409）
    monkeypatch.setattr(prog_mod, "compute_progress", orig)
    r2 = client.post("/api/runs/run_20260101_000026/step")
    assert r2.status_code == 202
    tasklock.release(outdir, "run_20260101_000026")


def test_delete_clip_or_talk_allowed_for_redo(client, outdir):
    """删降级片段/口型段是"强制重出单镜头"的入口：白名单要含 clips/talk。"""
    root, script = _mk_run(outdir, "run_20260101_000033")
    sid = script.scenes[0].shots[0].shot_id
    r = client.delete(f"/api/runs/run_20260101_000033/artifacts/clips/shot_{sid}.mp4")
    assert r.status_code == 200, r.text
    assert not (root / "clips" / f"shot_{sid}.mp4").exists()
    # 扩展名与阶段必须匹配：clips 不收 png，assets 不收 mp4
    assert client.delete(
        f"/api/runs/run_20260101_000033/artifacts/clips/shot_{sid}.png"
    ).status_code == 400
    assert client.delete(
        "/api/runs/run_20260101_000033/artifacts/assets/char_c1.mp4"
    ).status_code == 400
    # 删除片段后该镜头回到 videos 步缺口
    rep = client.get("/api/runs/run_20260101_000033/steps").json()
    videos = next(s for s in rep["steps"] if s["id"] == "videos")
    assert sid in videos["missing"] or sid in videos["stale"]


def test_dialogue_shot_without_talk_is_not_done(outdir):
    """对白镜头缺 talk 产物 = 未完成；m6 降级出的叠音 clips **不算**完成。

    这是"全片没有对口型"那个系统性缺陷的根因：旧判据把降级 clips 当成有视频，
    于是口型通道一失败，整片带着零口型镜头被标 done、被合成、被当成合格交付。
    """
    root, script = _mk_run(outdir, "run_20260101_000040", with_final=False)
    dlg = [sh.shot_id for sc in script.scenes for sh in sc.shots if sh.dialogues]
    assert dlg, "用例前提：fake 剧本要有对白镜头"
    sid = dlg[0]
    (root / "talk" / f"shot_{sid}.mp4").unlink()  # 模拟 m6 降级：clips 在、talk 不在
    rep = _progress(outdir, "run_20260101_000040")
    videos = next(s for s in rep.steps if s.id == "videos")
    # 判据是"这一步没完成、且点名了这个镜头"。state 可能是 pending 而不是 gap：
    # _mk_run 不产尾帧 → frames 步先卡住，后面的步骤一律 pending（顺序推进的语义）。
    assert videos.state != "done" and sid in videos.missing, "降级叠音不能算完成"
    (root / "talk" / f"shot_{sid}.mp4").write_bytes(b"x")
    rep = _progress(outdir, "run_20260101_000040")
    videos = next(s for s in rep.steps if s.id == "videos")
    assert sid not in videos.missing


def test_audio_overlay_is_the_only_explicit_waiver(outdir):
    """显式选 audio_overlay 才算"我不要口型"；其余模式缺 talk 一律是缺口。"""
    root, script = _mk_run(outdir, "run_20260101_000041", with_final=False)
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    meta["talk_mode"] = "audio_overlay"
    (root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                        encoding="utf-8")
    for p in (root / "talk").glob("*.mp4"):
        p.unlink()
    rep = _progress(outdir, "run_20260101_000041")
    videos = next(s for s in rep.steps if s.id == "videos")
    # fixture 不造台词 wav，所以 lines 仍可能是缺口；要点是**对白镜头不再因缺 talk 报缺口**
    dlg = [sh.shot_id for sc in script.scenes for sh in sc.shots if sh.dialogues]
    assert not (set(dlg) & set(videos.missing)), videos.missing


def test_lipsync_coverage_and_verdict_merge(outdir):
    """覆盖率口径 + 判定并入：high/cinema 缺口即不合格，economy 只报告不判死。"""
    from fastvideo3.core.progress import apply_lipsync_verdict, lipsync_coverage

    root, script = _mk_run(outdir, "run_20260101_000042", with_final=False)
    dlg = [sh.shot_id for sc in script.scenes for sh in sc.shots if sh.dialogues]
    (root / "talk" / f"shot_{dlg[0]}.mp4").unlink()
    cov = lipsync_coverage(RunStorage("run_20260101_000042", root=outdir), script,
                           "v2v_overlay")
    assert cov["required"] and cov["missing"] == [dlg[0]]
    assert cov["with_lipsync"] == len(dlg) - 1

    v = apply_lipsync_verdict({"passed": True}, cov, "cinema")
    assert v["passed"] is False and v["lipsync_missing"] == [dlg[0]]
    v = apply_lipsync_verdict({"passed": True}, cov, "economy")
    assert v["passed"] is True, "教学档不判不合格，但覆盖率仍写进 run_summary/看板"
    v = apply_lipsync_verdict(None, {"required": False, "missing": [], "dialogue_shots": 3,
                                     "with_lipsync": 0}, "cinema")
    assert v.get("passed") is None, "audio_overlay 豁免时不该伪造判定"


def test_board_exposes_lipsync_coverage(client, outdir):
    root, script = _mk_run(outdir, "run_20260101_000043")
    dlg = [sh.shot_id for sc in script.scenes for sh in sc.shots if sh.dialogues]
    (root / "talk" / f"shot_{dlg[0]}.mp4").unlink()
    d = client.get("/api/runs/run_20260101_000043/board").json()
    cov = d["lipsync_coverage"]
    assert cov["required"] is True and cov["missing"] == [dlg[0]]


def test_two_person_overlay_is_recorded_waiver_not_endless_gap(outdir):
    """双人镜头按策略叠音 = 显式豁免：不算缺口（否则「开始生成」无限补不出）。"""
    from fastvideo3.core.progress import lipsync_coverage
    from fastvideo3.core.storage import RunStorage

    root, script = _mk_run(outdir, "run_20260101_000044", with_final=False)
    dlg = [sh.shot_id for sc in script.scenes for sh in sc.shots if sh.dialogues]
    sid = dlg[0]
    (root / "talk" / f"shot_{sid}.mp4").unlink()
    storage = RunStorage("run_20260101_000044", root=outdir)
    # 未豁免时是缺口
    cov = lipsync_coverage(storage, script, "v2v_overlay")
    assert sid in cov["missing"]
    # m6 双人策略落豁免后：缺口清空、豁免单列，判定不因它不合格
    storage.record_lipsync_waiver(sid, "two_person_overlay_policy")
    cov = lipsync_coverage(storage, script, "v2v_overlay")
    assert cov["missing"] == [] and cov["waived"] == [sid]
    rep = _progress(outdir, "run_20260101_000044")
    videos = next(s for s in rep.steps if s.id == "videos")
    assert sid not in videos.missing, "豁免镜头不该卡住「开始生成」"
    from fastvideo3.core.progress import apply_lipsync_verdict

    v = apply_lipsync_verdict({"passed": True}, cov, "cinema")
    assert v["passed"] is True, "豁免是决定，不是不合格"


async def test_missing_last_frames_are_a_frames_step_gap(cfg, outdir):
    """尾帧**从没生成过**（m3 那趟被欠费/审核打断）→ frames 步必须算缺口，点一次能补上。

    用户实测的失败形状：8 个首帧在盘、尾帧只有 1 个，而步骤条显示"完成" ——
    于是没有任何一步会再去补尾帧，m4/m6 永远退回单首帧，界面上只看到"首尾帧改造没生效"。
    尾帧**已作废**（stale）的那条路本来就能补，缺的是"从未存在"这一条。
    """
    from fastvideo3.app.cli import _run_step
    from fastvideo3.core.stages import build_context

    cfg.fake = True
    cfg.talk_mode = "scripted_talk"  # C 链：对白镜头的画面也由口型通道出 → 同样要尾帧
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头路径
    cfg.use_last_frames = True
    run_id = "run_last_gap"
    storage = RunStorage(run_id, root=outdir)
    storage.save_json(
        "run_meta.json",
        {
            "run_id": run_id,
            "base_info": BaseInfo(topic="尾帧缺口", target_duration_sec=15).model_dump(),
            "fake": True, "quality": "economy", "talk_mode": "scripted_talk",
            "budget_limit_cny": 100.0,
        },
    )
    meta = storage.load_json("run_meta.json")
    ctx = build_context(cfg, storage)
    for _ in range(4):  # 剧本 → 角色图 → 场景图 → 分镜（首帧+尾帧）
        rep = prog.compute_progress(storage, meta, cfg)
        if rep.next is None or rep.next.id == "videos":
            break
        await _run_step(ctx, storage, cfg, meta, rep.next)

    sb = storage.root / "storyboard"
    lasts = sorted(p.name for p in sb.glob("*_last.png"))
    assert lasts, "前提：C 链下分镜那趟应同时产出尾帧"
    first_before = {p.name: p.read_bytes() for p in sb.glob("shot_*.png") if "_last" not in p.name}
    for p in sb.glob("*_last.png"):
        p.unlink()  # 模拟"尾帧那趟被打断"：首帧在盘、尾帧从未存在

    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next is not None and rep.next.id == "frames", (
        f"尾帧缺失必须算 frames 步的缺口，实际下一步是 {rep.next and rep.next.id}"
    )
    assert set(rep.next.missing), "缺口清单要点名要补的镜头"
    await _run_step(ctx, storage, cfg, meta, rep.next)
    assert sorted(p.name for p in sb.glob("*_last.png")) == lasts, "点一次就该把尾帧补齐"
    # 首帧一个字节都不该动：补尾帧不该重画首帧（那是白花一次 t2i 钱）
    after = {p.name: p.read_bytes() for p in sb.glob("shot_*.png") if "_last" not in p.name}
    assert after == first_before, "补尾帧时重画了首帧"
    assert prog.compute_progress(storage, meta, cfg).next.id != "frames", "补完就不该再报缺口"
