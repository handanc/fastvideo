"""剧本的人工三动作：改稿（可读文本）/ 删除 / 按 prompt 重写。

锁住的语义：
- 剧本稿是给人读的文本，保存走**本地确定性解析**，不靠 LLM，也不让人面对 JSON；
- 校验分三层：格式问题带**行号**、引用问题走 `_script_cross_check`、台词 >60 字由
  pydantic 拒；软规则（建议 ≤30 字、时长偏差 >15%）只警告不阻断；
- 人工保存与生成路径共用同一套工程兜底（转场多样性自动补齐 —— 以前只在生成时跑）；
- `revise` 留着旧剧本以便字段级作废；`rewrite` 归档 + 全量作废；
- 生成中一律 409。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import progress as prog
from fastvideo3.core import tasklock
from fastvideo3.core.schema import Script
from fastvideo3.core.storage import RunStorage

RUN = "run_20260101_000060"


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[attr-defined]
        yield c


@pytest.fixture(autouse=True)
def _clean_lock(outdir):
    """测试里 `_spawn` 被换成记录器，没有子进程会释放锁 —— 每个用例开局先清一次，
    否则上一个用例留下的锁会把后面的用例全部打成 409（假故障）。"""
    tasklock.release(outdir)
    yield


@pytest.fixture
def run(outdir):
    """每个用例自己造 run：`outdir` 是按用例隔离的 tmp 目录，跨用例不复用。"""
    root, script = _mk_run(outdir, RUN)
    return root, script


def _storage(outdir) -> RunStorage:
    return RunStorage(RUN, root=outdir)


def _saved(outdir) -> Script:
    return Script.model_validate(_storage(outdir).load_json("script.json"))


def _draft(client) -> str:
    r = client.get(f"/api/runs/{RUN}/script/text")
    assert r.status_code == 200, r.text
    return r.json()["text"]


def _first_dialogue_line(text: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith("  ") and "）: " in ln)


def _tail(cmd: list[str], n: int) -> list[str]:
    return cmd[-n:]


# ───────────────────────── 读剧本稿 ─────────────────────────
def test_get_script_text_returns_readable_draft(client, run):
    root, script = run
    r = client.get(f"/api/runs/{RUN}/script/text")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == script.version
    text = body["text"]
    assert text.splitlines()[0].startswith("《")
    assert "# 角色" in text and "## " in text and "### " in text
    assert "外貌:" in text
    assert "visual_prompt" not in text and "必留词" not in text  # 不暴露给人改的东西


def test_get_script_text_404_without_script(client, outdir):
    _mk_run(outdir, RUN, with_script=False)
    assert client.get(f"/api/runs/{RUN}/script/text").status_code == 404


# ───────────────────────── 保存剧本稿 ─────────────────────────
def test_put_script_text_roundtrip_invalidates_nothing(client, run):
    """原样存回去不该作废任何东西（否则"打开编辑就烧钱"）。"""
    root, _ = run
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _draft(client)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["invalidated"] == 0, body["items"]
    assert body["version"] == 2
    assert (root / "script.bak.json").is_file()


def test_put_script_text_edit_one_line_invalidates_only_that_shot(client, run):
    root, script = run
    text = _draft(client)
    target = next(sh for sh in script.all_shots() if sh.dialogues)
    idx = text.index(f"### {target.shot_id} ")
    line = next(
        ln for ln in text[idx:].splitlines() if ln.startswith("  ") and "）: " in ln
    )
    edited = text.replace(line, line.rsplit(": ", 1)[0] + ": 换一句短的", 1)
    assert edited != text
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": edited})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["invalidated"] > 0
    hit = [x for x in body["items"] if not x.startswith("final/")]
    assert hit and all(f"_{target.shot_id}" in x or target.shot_id in x for x in hit), body["items"]
    others = [sh.shot_id for sh in script.all_shots() if sh.shot_id != target.shot_id]
    assert not any(f"shot_{oid}." in x for oid in others for x in hit), "其他镜头不该被牵连"
    saved = _saved(root.parent)
    shot = next(sh for sh in saved.all_shots() if sh.shot_id == target.shot_id)
    assert any(d.text == "换一句短的" for d in shot.dialogues)


def test_put_script_text_rejects_bad_format_with_line_number(client, run):
    root, _ = run
    text = _draft(client)
    broken = text.replace("# 角色", "# 角色\n  这句没有说话人也没有情绪括号\n", 1)
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": broken})
    assert r.status_code == 422, r.text
    detail = json.dumps(r.json()["detail"], ensure_ascii=False)
    assert "第" in detail and "行" in detail, detail
    assert _saved(root.parent).version == 1, "被拒的稿子不能落盘"


def test_put_script_text_rejects_unknown_character_reference(client, run):
    text = _draft(client)
    broken = text.replace("在场: ", "在场: ghost, ", 1)
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": broken})
    assert r.status_code == 422, r.text
    assert "ghost" in json.dumps(r.json()["detail"], ensure_ascii=False)


def test_put_script_text_rejects_overlong_line_by_schema(client, run):
    """>60 字是 schema 硬上限：必须拒绝（TTS 与口型时长会直接错）。"""
    text = _draft(client)
    line = _first_dialogue_line(text)
    broken = text.replace(line, f"  c1（平静）: {'这' * 65}", 1)
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": broken})
    assert r.status_code == 422, r.text


def test_put_script_text_warns_on_soft_rules_without_blocking(client, run):
    """建议 ≤30 字：放行，但必须把后果说给人听。"""
    text = _draft(client)
    line = _first_dialogue_line(text)
    edited = text.replace(line, f"  c1（平静）: {'话' * 35}", 1)  # 35 字：超建议值、未破 60 硬上限
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": edited})
    assert r.status_code == 200, r.text
    assert any("超长" in w for w in r.json()["warnings"]), r.json()["warnings"]


def test_put_script_text_autofills_transition_variety(client, run):
    """人工把转场改成全硬切 → 保存时确定性兜底补齐，并告知补了几处（F6 ≥3 种）。"""
    text = _draft(client)
    flat = "\n".join(
        ln.replace(" · 叠化", " · 切").replace(" · 空镜转场", " · 切").replace(" · 淡入淡出", " · 切")
        for ln in text.splitlines()
    )
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": flat})
    assert r.status_code == 200, r.text
    types = {sh.transition_out.value for sh in _saved(run[0].parent).all_shots()}
    assert len(types) >= 3, f"F6 要求 ≥3 种转场，实际 {types}"
    assert any("转场多样性" in n for n in r.json()["notes"]), r.json()["notes"]


def test_put_script_text_blocked_while_generating(client, run):
    assert tasklock.acquire(run[0].parent, "other_run", "step")[0]
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _draft(client)})
    assert r.status_code == 409
    tasklock.release(run[0].parent, "other_run")


# ───────────────────────── 按 prompt 重写 ─────────────────────────
def test_regenerate_revise_keeps_script_and_spawns_child(client, run):
    root, _ = run
    r = client.post(
        f"/api/runs/{RUN}/script/regenerate",
        json={"prompt": "第三幕节奏太慢，多加一个误会", "mode": "revise"},
    )
    assert r.status_code == 202, r.text
    assert (root / "script.json").is_file(), "revise 必须留着旧剧本，否则算不出字段级作废"
    assert _tail(client.spawned[-1], 6) == [
        "script-regen", RUN, "--prompt", "第三幕节奏太慢，多加一个误会", "--mode", "revise",
    ]
    tasklock.release(root.parent, RUN)


def test_regenerate_rewrite_archives_and_invalidates_all(client, run):
    root, script = run
    st = _storage(root.parent)
    st.record_lineage(f"storyboard/shot_{script.all_shots()[0].shot_id}.png", ["assets/char_c1.png"])
    r = client.post(
        f"/api/runs/{RUN}/script/regenerate",
        json={"prompt": "换成科幻题材重来", "mode": "rewrite"},
    )
    assert r.status_code == 202, r.text
    assert not (root / "script.json").is_file()
    assert list(root.glob("script.deleted.*.json"))
    prompt_file = json.loads((root / "script_prompt.json").read_text(encoding="utf-8"))
    assert prompt_file == {"prompt": "换成科幻题材重来", "mode": "rewrite"}
    assert st.stale_set(), "推倒重写必须把下游全量标 stale"
    assert _tail(client.spawned[-1], 6) == [
        "script-regen", RUN, "--prompt", "换成科幻题材重来", "--mode", "rewrite",
    ]
    tasklock.release(root.parent, RUN)


def test_regenerate_revise_without_script_is_404(client, outdir):
    _mk_run(outdir, RUN, with_script=False)
    r = client.post(
        f"/api/runs/{RUN}/script/regenerate", json={"prompt": "改一下", "mode": "revise"}
    )
    assert r.status_code == 404
    assert client.spawned == []


def test_regenerate_rejects_empty_prompt_and_bad_mode(client, run):
    assert client.post(
        f"/api/runs/{RUN}/script/regenerate", json={"prompt": "", "mode": "revise"}
    ).status_code == 422
    assert client.post(
        f"/api/runs/{RUN}/script/regenerate", json={"prompt": "x", "mode": "nope"}
    ).status_code == 422
    assert client.spawned == []


def test_regenerate_blocked_while_busy(client, run):
    assert tasklock.acquire(run[0].parent, "other_run", "step")[0]
    r = client.post(
        f"/api/runs/{RUN}/script/regenerate", json={"prompt": "改一下", "mode": "revise"}
    )
    assert r.status_code == 409
    tasklock.release(run[0].parent, "other_run")


# ───────────────────────── 逐步路径消费 prompt ─────────────────────────
async def test_step_consumes_script_prompt(cfg, outdir):
    """「推倒重写」的子进程万一崩了，意见不能丢：再点「开始生成」仍带着同一条 prompt。"""
    from fastvideo3.app.cli import _run_step
    from fastvideo3.core.stages import build_context

    run_id = "run_step_prompt"
    storage = RunStorage(run_id, root=outdir)
    meta = {
        "run_id": run_id,
        "base_info": {
            "topic": "带意见重试", "style": "都市轻喜剧",
            "target_duration_sec": 15, "aspect_ratio": "9:16",
        },
        "fake": True, "quality": "economy", "talk_mode": "audio_overlay",
        "budget_limit_cny": 100.0,
    }
    storage.save_json("run_meta.json", meta)
    storage.save_json("script_prompt.json", {"prompt": "多加一个误会", "mode": "rewrite"})

    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "script" and rep.next.label == "生成剧本"
    ctx = build_context(cfg, storage)
    produced = await _run_step(ctx, storage, cfg, meta, rep.next)
    assert produced == ["script.json"]
    assert storage.exists("script.json")
    assert not storage.exists("script_prompt.json"), "意见用完必须清除，否则会粘在下次重生成上"
