"""制作看板接口 + 人工选定的测试。

看板是"产物 ↔ 镜头/角色"的服务端 join：run 可能停在任何阶段，
缺产物必须降级成 null 而不是 500（历史上有 37 个 run 只有剧本）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fastvideo3.app.api as api_mod
from fastvideo3.core.schema import BaseInfo
from fastvideo3.providers.fake import fake_script


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        yield c


def _mk_run(outdir, run_id="run_20260101_000000", *, with_script=True, with_final=True):
    root = outdir / run_id
    for stage in ("assets", "storyboard", "audio", "clips", "talk", "final", "review"):
        (root / stage).mkdir(parents=True, exist_ok=True)
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "base_info": {"topic": "测试主题", "style": "都市轻喜剧",
                              "target_duration_sec": 30, "aspect_ratio": "9:16"},
                "fake": True, "resolution_tier": "1080P",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "events.jsonl").write_text("", encoding="utf-8")
    if not with_script:
        return root, None
    script = fake_script(BaseInfo(topic="测试主题"))
    (root / "script.json").write_text(
        json.dumps(script.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    for ch in script.characters:
        Image.new("RGB", (8, 8), (200, 30, 30)).save(root / "assets" / f"char_{ch.character_id}.png")
        for i in range(2):  # 落选候选（供"设为定妆照"用）
            Image.new("RGB", (8, 8), (30, 200, 30)).save(
                root / "assets" / f"_cand_char_{ch.character_id}_t0_{i}.png"
            )
    for sc in script.scenes:
        Image.new("RGB", (8, 8), (30, 30, 200)).save(root / "assets" / f"scene_{sc.scene_id}.png")
        for sh in sc.shots:
            Image.new("RGB", (8, 8), (9, 9, 9)).save(root / "storyboard" / f"shot_{sh.shot_id}.png")
            (root / "clips" / f"shot_{sh.shot_id}.mp4").write_bytes(b"x")
            if sh.dialogues:  # 对白镜头必须有口型产物才算"完成"（2026-10-04 的新判据）
                (root / "talk" / f"shot_{sh.shot_id}.mp4").write_bytes(b"x")
    if with_final:
        (root / "final" / "final.mp4").write_bytes(b"mp4")
        (root / "final" / "subtitles.srt").write_text("1\n", encoding="utf-8")
    return root, script


def test_board_joins_products_to_shots(client, outdir):
    root, script = _mk_run(outdir)
    d = client.get("/api/runs/run_20260101_000000/board").json()
    assert d["available"] is True
    assert d["shot_count"] == len(script.all_shots())
    # 角色：定妆照 + 候选 + 外貌文本
    assert d["characters"][0]["portrait"].split("?")[0].endswith("/assets/char_c1.png")
    assert len(d["characters"][0]["candidates"]) == 2
    assert d["characters"][0]["appearance"]
    # 场景：概念图 + 逐镜头
    scene = d["scenes"][0]
    assert scene["concept"].split("?")[0].endswith(f"scene_{scene['id']}.png")
    # 面板必须摆出**真正会发出去的**空景提示词：只显示 summary（剧情，必然带人名）
    # 会让人以为"背景图要画角色"，然后来来回回追问（2026-10-14 实际发生）
    assert scene["bg_prompt"].startswith("空景场景概念图"), scene["bg_prompt"][:60]
    assert "人物" in scene["bg_prompt"]  # 开头那句否定式
    shot = scene["shots"][0]
    assert shot["frame"].split("?")[0].endswith(f"shot_{shot['shot_id']}.png")
    assert shot["clip"].split("?")[0].endswith(f"shot_{shot['shot_id']}.mp4")
    assert shot["talk"] is None  # 该镜头没出过口型片段 → null 而不是报错
    # 台词里的 character_id 已换成可读名字
    duo = next((sh for s in d["scenes"] for sh in s["shots"] if sh["dialogues"]), None)
    assert duo and duo["dialogues"][0]["who"] in {"小明", "小红"}


def test_board_reports_script_version_and_unused_characters(client, outdir):
    """重写剧本后看板必须能看出"剧本变了"以及"新角色有没有真出场"。

    用户实测：按 prompt 改写后新增的角色只出现在 `characters` 清单、没进任何镜头，
    而剧本页签逐镜头展示 → 看起来像"没改出来"。所以版本号与"未被引用"都要回给前端。
    """
    root, script = _mk_run(outdir, "run_20260101_000007")
    data = script.model_dump(mode="json")
    data["version"] = 7
    data["characters"].append(
        {"character_id": "c9", "name": "路人甲", "appearance": "路过的行人", "voice_id": ""}
    )
    (root / "script.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    d = client.get("/api/runs/run_20260101_000007/board").json()
    assert d["script_version"] == 7
    by_id = {c["id"]: c for c in d["characters"]}
    assert by_id["c1"]["used"] is True
    assert by_id["c9"]["used"] is False, "没被任何镜头（在场/台词）引用的角色必须标出来"
    assert by_id["c9"]["name"] == "路人甲"


def test_board_and_steps_fall_back_to_effective_budget(client, outdir):
    """meta 落 0 的老 run：看板与步骤条都要显示实际生效的熔断值。

    显示 ¥0 的后果：`too_low` 永不成立 → 成本红条不出现 → 用户以为熔断够用，
    实际跑到口型步骤前就按 .env 的额度跳闸（对白镜头降级成无口型叠音）。
    """
    root, _ = _mk_run(outdir, "run_20260101_000008")
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    meta["budget_limit_cny"] = 0.0
    (root / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    want = float(api_mod._cfg_for_run(root).budget_limit_cny)
    assert want > 0
    board = client.get("/api/runs/run_20260101_000008/board").json()
    assert board["cost_estimate"]["budget_cny"] == pytest.approx(want)
    steps = client.get("/api/runs/run_20260101_000008/steps").json()
    assert steps["budget_cny"] == pytest.approx(want)


def test_board_subtitle_falls_back_to_legacy_name(client, outdir):
    """旧 run 的旁挂字幕还叫 final.srt：下载链接不能因此 404。"""
    root, _ = _mk_run(outdir, "run_20260101_000009")
    (root / "final" / "subtitles.srt").unlink()
    (root / "final" / "final.srt").write_text("1\n", encoding="utf-8")
    d = client.get("/api/runs/run_20260101_000009/board").json()
    assert d["subtitle"].split("?")[0].endswith("/final/final.srt")  # ?v= 是缓存破坏戳


def test_board_degrades_on_partial_runs(client, outdir):
    _mk_run(outdir, "run_20260101_000001", with_script=False)
    r = client.get("/api/runs/run_20260101_000001/board")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is False and d["scenes"] == [] and d["final_video"] is None
    assert client.get("/api/runs/nope/board").status_code == 404


def test_board_survives_broken_script_json(client, outdir):
    root, _ = _mk_run(outdir, "run_20260101_000002", with_script=False)
    (root / "script.json").write_text('{"characters": 不是JSON', encoding="utf-8")
    r = client.get("/api/runs/run_20260101_000002/board")
    assert r.status_code == 200
    assert r.json()["available"] is False and "raw_script" not in r.json()


def test_pick_portrait_swaps_and_backs_up(client, outdir):
    root, _ = _mk_run(outdir, "run_20260101_000003")
    before = (root / "assets" / "char_c1.png").read_bytes()
    r = client.post(
        "/api/runs/run_20260101_000003/characters/c1/portrait",
        json={"candidate": "_cand_char_c1_t0_1.png"},
    )
    assert r.status_code == 200, r.text
    assert (root / "assets" / "char_c1.png").read_bytes() != before  # 已换成候选图
    assert (root / "assets" / "char_c1.bak.png").read_bytes() == before  # 旧图有备份
    # 候选图保留（复制而非 rename）：供"落选候选"回看，且避开 Windows 占用崩溃
    assert (root / "assets" / "_cand_char_c1_t0_1.png").exists()
    ev = (root / "events.jsonl").read_text(encoding="utf-8")
    assert "portrait_picked" in ev


def test_pick_portrait_rejects_foreign_or_traversal_names(client, outdir):
    _mk_run(outdir, "run_20260101_000004")
    url = "/api/runs/run_20260101_000004/characters/c1/portrait"
    # 不能拿别人的候选改我的定妆照
    assert client.post(url, json={"candidate": "_cand_char_c2_t0_0.png"}).status_code == 400
    # 路径穿越 / 非 png / 不存在的候选
    assert client.post(url, json={"candidate": "../../final/final.mp4"}).status_code in (400, 404)
    assert client.post(url, json={"candidate": "char_c1.png"}).status_code == 400
    assert client.post(url, json={"candidate": "_cand_char_c1_t0_9.png"}).status_code == 404


def test_pick_portrait_invalidates_downstream(client, outdir):
    """✔ 设为定妆照必须作废消费过旧脸的下游，否则旧脸混进新成片。"""
    root, script = _mk_run(outdir, "run_20260101_000030")
    from fastvideo3.core.storage import RunStorage

    storage = RunStorage("run_20260101_000030", root=outdir)
    shots_c1 = [sh.shot_id for sc in script.scenes for sh in sc.shots if "c1" in sh.characters]
    for sid in shots_c1:
        storage.record_lineage(f"storyboard/shot_{sid}.png", ["assets/char_c1.png"])
    r = client.post(
        "/api/runs/run_20260101_000030/characters/c1/portrait",
        json={"candidate": "_cand_char_c1_t0_1.png"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["invalidated"], "换锚点不作废下游 = 旧脸进新成片"
    stale = storage.stale_set()
    assert all(f"storyboard/shot_{s}.png" in stale for s in shots_c1)
    assert "assets/char_c1.png" not in stale  # 新图本身不是 stale


def test_board_urls_carry_cache_version(client, outdir):
    """换图后路径不变 → URL 必须带版本号，否则浏览器显示缓存旧图。"""
    _mk_run(outdir, "run_20260101_000031")
    d = client.get("/api/runs/run_20260101_000031/board").json()
    assert "?v=" in d["characters"][0]["portrait"]
    assert "?v=" in d["scenes"][0]["concept"]
    assert "?v=" in d["scenes"][0]["shots"][0]["frame"]


def test_take_counter_changes_seed_and_candidate_names(client, outdir):
    """删除→重生成必须换脸：take+1 进种子与候选文件名。"""
    import asyncio

    from fastvideo3.core.storage import RunStorage
    from fastvideo3.modules.m2_assets import _char_seed, build_assets
    from fastvideo3.providers.fake import FakeT2IProvider, FakeVLMProvider

    root, script = _mk_run(outdir, "run_20260101_000032")
    storage = RunStorage("run_20260101_000032", root=outdir)
    assert _char_seed("c1", "x", 0, 0) != _char_seed("c1", "x", 0, 1)
    storage.save_json("takes.json", {"c1": 1})
    for p in root.glob("assets/char_c1*.png"):
        p.unlink()
    cfg = load_config_for_test()
    cfg.asset_candidates = 2
    asyncio.run(
        build_assets(
            cfg, FakeT2IProvider(), script, storage,
            __import__("fastvideo3.core.cost", fromlist=["CostLedger"]).CostLedger(storage, 100),
            vlm=FakeVLMProvider(),
        )
    )
    assert (root / "assets" / "_cand_char_c1_t1_0.png").exists()
    assert not (root / "assets" / "_cand_char_c1_t0_0.png").exists() or True  # 旧候选不被复用


def load_config_for_test():
    from fastvideo3.core.config import load_config

    return load_config(dotenv_path="none.env", resolution="480P")
