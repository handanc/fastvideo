"""成本闸门（到线停手问人）+ 语音预算的时长下限 + 单段预算线的切法。

三条产品语义（用户裁定）：
1. **预估**超限只出红幅警告（本来就有）；**实时累计**到线才停下等人工确认 ——
   旧行为是"熔断后静默跳过后面的付费提交"，步骤照样报完成，用户看完成片才知道
   口型/片段一段没出（2026-10-04 实测）。
2. 确认继续 = **抬高上限**（只清标记不抬上限，下一段又立刻停，弹窗会变成死循环）；
   上限写进 `run_meta.budget_limit_cny`，CLI 与 API 共读同一份。
3. 时长是软规则：不拦保存、不拦生成；但镜头时长**同时决定** `duration` 与提示词里的
   分镜时间戳，写短了会被吞句尾 → 只把"装不下自己台词"的镜头**抬**到最小可行时长
   （封顶 25s），人写长了的一律不动。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import pause, tasklock, voice_budget
from fastvideo3.core.config import load_config
from fastvideo3.core.scenegroup import CLIP_CAP_SEC, MAX_CLIP_SEC, group_scenes
from fastvideo3.core.schema import BaseInfo, DialogueLine, Script, Shot, ShotType, TransitionType
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.fake import fake_script

RUN = "run_20260101_000088"


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[arg-type]
        yield c


@pytest.fixture(autouse=True)
def _free_lock(outdir):
    tasklock.release(outdir)
    yield


@pytest.fixture
def run(outdir):
    root, _ = _mk_run(outdir, RUN, with_final=False)
    return root


def _storage(root) -> RunStorage:
    return RunStorage(RUN, root=root.parent)


class _Ledger:
    """只喂 `total()`/`budget_limit_cny` 两个字段，避免测试真去写 cost.json。"""

    def __init__(self, total: float, limit: float = 50.0) -> None:
        self._total, self.budget_limit_cny = total, limit

    def total(self) -> float:
        return self._total


def _set_limit(root, value: float) -> None:
    p = root / "run_meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["budget_limit_cny"] = value
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _write_cost(root, total: float) -> None:
    (root / "cost.json").write_text(
        json.dumps([{"capability": "i2v", "amount_cny": total, "model": "wan3.0-video"}]),
        encoding="utf-8",
    )


def _events(root) -> list[dict]:
    out = []
    for ln in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


# ─────────────────────── ① 闸门状态机 ───────────────────────
def test_gate_stays_open_below_limit(run):
    st = _storage(run)
    assert pause.stop_for_budget(st, _Ledger(10.0, 50.0), node="t") is False
    assert not (run / "pause.json").exists()
    assert pause.state(st, spent=10.0, fallback_limit=50.0)["pending"] is False


def test_gate_closes_at_limit_and_asks_once(run):
    st = _storage(run)
    assert pause.stop_for_budget(st, _Ledger(50.0, 50.0), node="m4_scene", unit="scene_1") is True
    assert pause.stop_for_budget(st, _Ledger(52.0), node="m4_scene", unit="scene_2") is True
    asks = [e for e in _events(run) if e.get("status") == "budget_pause_ask"]
    assert len(asks) == 1, "每次到线只问一次，别把事件流刷成一堵墙"
    assert asks[0]["unit"] == "scene_1", asks
    st2 = pause.state(st, spent=52.0, fallback_limit=50.0)
    assert st2["pending"] is True and st2["asked_by"] == "m4_scene:scene_1"
    assert (run / "pause.json").is_file()


def test_approve_raises_limit_and_reopens_the_valve(run):
    st = _storage(run)
    pause.stop_for_budget(st, _Ledger(51.0, 50.0), node="m4_scene")
    pause.approve(st, 80.0)
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["budget_limit_cny"] == 80.0, "上限必须落在 run_meta（CLI 与 API 共同的恢复源）"
    # 现读 run_meta：51 < 80 → 放行
    assert pause.stop_for_budget(st, _Ledger(51.0, 50.0), node="m4_scene") is False
    assert pause.state(st, spent=51.0)["pending"] is False
    # 再花到新线 → 再问一次（每一笔加钱都要人点头）
    assert pause.state(st, spent=85.0)["pending"] is True


def test_approve_never_lowers_the_limit(run):
    """批准只会抬线：传 0 或更小的值不能把上限悄悄清零（清零=第一步就跳闸的老坑）。"""
    st = _storage(run)
    _set_limit(run, 50.0)
    pause.approve(st, 0)
    assert json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["budget_limit_cny"] == 50.0


# ─────────────────────── ② Web 路径 ───────────────────────
def test_step_refused_until_human_confirms(client, run):
    _set_limit(run, 50.0)
    _write_cost(run, 52.0)
    r = client.post(f"/api/runs/{RUN}/step")
    assert r.status_code == 409, r.text
    assert "已达上限" in r.text and "确认" in r.text, r.text
    assert client.spawned == [], "拦住之前绝不能 spawn 子进程（那等于照花）"

    ok = client.post(f"/api/runs/{RUN}/budget/continue", json={"new_limit_cny": 90})
    assert ok.status_code == 200, ok.text
    assert ok.json()["new_limit_cny"] == 90
    r2 = client.post(f"/api/runs/{RUN}/step")
    assert r2.status_code == 202, r2.text
    assert client.spawned, "确认后要真的接着跑"


def test_continue_suggests_a_limit_when_left_blank(client, run):
    _set_limit(run, 50.0)
    _write_cost(run, 60.0)
    r = client.post(f"/api/runs/{RUN}/budget/continue", json={})
    assert r.status_code == 200, r.text
    new = r.json()["new_limit_cny"]
    assert new > 60.0, f"新上限必须高于已花，否则下一段立刻再停：{new}"


def test_continue_rejects_not_raising(client, run):
    _set_limit(run, 50.0)
    _write_cost(run, 52.0)
    r = client.post(f"/api/runs/{RUN}/budget/continue", json={"new_limit_cny": 40})
    assert r.status_code == 400, r.text


def test_steps_payload_carries_the_gate(client, run):
    """前端每 3 秒读 /steps：闸门必须在这个载荷里，否则弹窗永远不会出现。"""
    _set_limit(run, 50.0)
    _write_cost(run, 55.0)
    body = client.get(f"/api/runs/{RUN}/steps").json()
    assert body["budget_gate"]["pending"] is True
    assert body["budget_gate"]["spent_cny"] == 55.0
    assert body["spent_cny"] == 55.0


# ─────────────────────── ③ 时长下限（软规则里唯一的写动作）───────────────────────
def _script(lines_per_shot: list[int], durations: list[float]) -> Script:
    shots = []
    for i, (n, dur) in enumerate(zip(lines_per_shot, durations), start=1):
        shots.append(
            Shot(
                shot_id=f"shot_1_{i}", scene_id="scene_1", type=ShotType.DIALOGUE,
                duration_sec=dur, action="两个人在茶馆里对坐，气氛僵住。",
                characters=["c1", "c2"],
                transition_out=TransitionType.CUT,
                dialogues=[
                    DialogueLine(character_id="c1", text="这句话正好十个字呢", emotion="calm")
                    for _ in range(n)
                ],
            )
        )
    sc = fake_script(BaseInfo(topic="测试", target_duration_sec=90))
    sc.scenes = [type(sc.scenes[0])(scene_id="scene_1", title="对坐", location="茶馆",
                                       time_of_day="日", shots=shots)]
    return sc


def test_floor_raises_only_never_lowers():
    cfg = load_config(dotenv_path="none.env")
    # 第 1 镜：1 句 9 字≈念 2s，写 1s → 该抬；第 2 镜：写 20s（远超需要）→ 不许动
    sc = _script([1, 1], [1.0, 20.0])
    out, notes = voice_budget.apply_duration_floor(sc, cfg)
    a, b = out.all_shots()
    assert a.duration_sec > 1.0, notes
    assert b.duration_sec == 20.0, "人写长了是节奏，一处都不许动"
    assert any("shot_1_1" in n for n in notes) and not any("shot_1_2" in n for n in notes)
    assert notes[0].startswith("镜头时长按语音预算"), notes[0]


def test_floor_leaves_the_comfort_zone_alone():
    """需要 ≤ 写的 ≤ 25s：一个字节都不动 —— 这一段区间里时长是人的节奏。"""
    cfg = load_config(dotenv_path="none.env")
    sc = _script([1], [8.0])          # 一句 9 字≈念 2s，人写 8s
    out, notes = voice_budget.apply_duration_floor(sc, cfg)
    assert out.all_shots()[0].duration_sec == 8.0, notes
    assert notes == [], notes


def test_floor_clamps_oversized_shots_to_the_cap():
    cfg = load_config(dotenv_path="none.env")
    sc = _script([1], [40.0])
    out, notes = voice_budget.apply_duration_floor(sc, cfg)
    assert out.all_shots()[0].duration_sec == CLIP_CAP_SEC, notes
    assert any("封顶" in n for n in notes), notes


def test_floor_caps_at_clip_soft_cap_and_still_warns():
    cfg = load_config(dotenv_path="none.env")
    sc = _script([9], [1.0])  # 9 句 81 字 ≈ 18s + 留白 → 需要 >25? 用更多字保证超线
    sc.scenes[0].shots[0].dialogues = [
        DialogueLine(character_id="c1", text="这是一句很长很长的台词用来把预算撑爆啊" * 3,
                     emotion="calm")
        for _ in range(4)
    ]
    out, notes = voice_budget.apply_duration_floor(sc, cfg)
    dur = out.all_shots()[0].duration_sec
    assert dur <= CLIP_CAP_SEC, f"封顶 {CLIP_CAP_SEC}s，不许一镜 40s：{dur}"
    assert any("封顶" in n for n in notes), notes
    assert any("建议拆成两镜" in n for n in notes), notes


def _draft_with_overload(client) -> tuple[str, str]:
    """取标准稿，把某个对白镜头改成"1s + 三句长台词"（明显装不下），返回 (稿子, 镜号)。"""
    import re

    lines = client.get(f"/api/runs/{RUN}/script/text").json()["text"].splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("### ") and "对白" in ln)
    target = lines[idx].split()[1]
    lines[idx] = re.sub(r"[\d.]+s", "1s", lines[idx])
    # 在该镜头末尾（下一个标题行之前）插三句长台词，把这一镜的语音预算顶到 5s 以上
    nxt = next((i for i in range(idx + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    sid = next(
        re.match(r"^\s+([^\s:：()（）]+)", ln).group(1)
        for ln in lines[idx:nxt]
        if ln.startswith("  ") and "）:" in ln
    )
    lines[nxt:nxt] = [
        f"  {sid}（calm）: 这句加上前面两句，三句加起来肯定超过一个五秒镜头能说完的量了不是吗",
        f"  {sid}（calm）: 再说一句，让这一镜的语音预算明显超过原来的时长，逼出时长下限",
        f"  {sid}（calm）: 第三句同样长，为的是让模型和校验器都无法假装这一镜说得完啊",
    ]
    return "\n".join(lines), target


def test_floor_raise_invalidates_the_covering_scene_clip(client, run):
    """时长被抬 = 内容变了 → 该场景片段必须沿血缘作废，否则新时长永远进不了视频。

    （`diff_invalidation` 的比较元组里含 `duration_sec`，这条钉住它真的生效。）
    """
    draft, target = _draft_with_overload(client)
    old = Script.model_validate(json.loads((run / "script.json").read_text(encoding="utf-8")))
    gid = next(g.group_id for g in group_scenes(old) if target in g.shot_ids)
    clip = run / "clips" / f"scene_{gid}.mp4"
    clip.write_bytes(b"mp4old")  # 只对盘上真实存在的产物标 stale（防幽灵状态）

    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": draft})
    assert r.status_code == 200, r.text
    saved = Script.model_validate(json.loads((run / "script.json").read_text(encoding="utf-8")))
    shot = next(sh for sh in saved.all_shots() if sh.shot_id == target)
    assert shot.duration_sec > 5.0, f"三句长台词的镜头还停在 {shot.duration_sec}s"
    assert any("语音预算" in n for n in r.json().get("notes") or []), r.json()
    assert clip.relative_to(run).as_posix() in r.json()["items"], r.json()["items"]


def test_floor_that_nets_out_invalidates_nothing(client, run):
    """把 5s 改成 1s，下限又抬回 5s → 净变化为零，一个产物都不该作废。"""
    import re

    lines = client.get(f"/api/runs/{RUN}/script/text").json()["text"].splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("### ") and "对白" in ln)
    lines[idx] = re.sub(r"[\d.]+s", "1s", lines[idx])
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": chr(10).join(lines)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["invalidated"] == 0, body["items"]
    assert body.get("fixes") or any("语音预算" in n for n in body.get("notes") or []), body


# ─────────────────────── ⑤ 界面逐格手填：允许写短，只告警 ───────────────────────
def test_manual_duration_edit_may_go_below_the_floor(client, run):
    """界面上把时长改小是**人的明确意图**：保存照旧、不偷偷抬回去，只给告警数据。

    与"整份剧本稿保存会抬时长下限"不矛盾：批量入口（粘稿/模型改写）里 1s 往往是格式
    转换掉出来的错值，而逐格手填是知情决定。判据都是同一个 `need_sec`。
    """
    body = client.get(f"/api/runs/{RUN}/script").json()
    shots = [sh for sc in body["scenes"] for sh in sc["shots"]]
    target = next(sh for sh in shots if sh["dialogues"])
    assert target["duration_sec"] >= 2.0
    target["duration_sec"] = 1.0
    r = client.put(f"/api/runs/{RUN}/script", json=body)
    assert r.status_code == 200, r.text
    saved = Script.model_validate(json.loads((run / "script.json").read_text(encoding="utf-8")))
    got = next(sh for sh in saved.all_shots() if sh.shot_id == target["shot_id"])
    assert got.duration_sec == 1.0, "手填的 1s 被后端偷偷抬回去了"


def test_board_exposes_the_per_shot_minimum_for_the_warning(client, run):
    """前端告警只引用后端算好的 `need_sec`：公式必须只有一份，不然提示会与实际判据漂。"""
    body = client.get(f"/api/runs/{RUN}/script").json()
    shots = [sh for sc in body["scenes"] for sh in sc["shots"]]
    target = next(sh for sh in shots if sh["dialogues"])
    cap = client.get(f"/api/runs/{RUN}/board").json()["capacity"]
    ps = cap["per_shot"][target["shot_id"]]
    chars = sum(len(d["text"]) for d in target["dialogues"])
    assert ps["chars"] == chars
    assert ps["need_sec"] >= round(chars / cap["cps"]) + 1 - 0.11, ps
    assert ps["speech_sec"] == round(chars / cap["cps"]) + 1, ps
    assert ps["lines"] == len(target["dialogues"])
    assert set(cap["per_shot"]) == {sh["shot_id"] for sh in shots}

# ─────────────────────── ④ 25s 预算线：切法必须只有一份 ───────────────────────
def test_oversized_scene_splits_at_the_soft_cap_sharing_one_plate():
    sc = fake_script(BaseInfo(topic="测试", target_duration_sec=300))
    sc.scenes = sc.scenes[:1]
    sc.scenes[0].location, sc.scenes[0].time_of_day = "天台", "日"
    sc.scenes[0].shots = [
        Shot(shot_id=f"shot_1_{i}", scene_id="scene_1", type=ShotType.ACTION,
             duration_sec=8.0, action="走位", characters=["c1"],
             transition_out=TransitionType.CUT)
        for i in range(1, 6)
    ]  # 40s，单场
    groups = group_scenes(sc)  # 默认预算线
    sid = sc.scenes[0].scene_id
    assert [g.group_id for g in groups] == [f"{sid}p1", f"{sid}p2"], [g.group_id for g in groups]
    assert all(g.duration <= CLIP_CAP_SEC for g in groups), [g.duration for g in groups]
    # 拆段共用同一张场景图（scene_ids 原样继承）：给每段另画场景图等于给同一场戏换布景
    assert all(g.scene_ids == (sid,) for g in groups)
    assert groups[1].note and "25" in groups[1].note, groups[1].note


def test_soft_cap_is_below_the_physical_one():
    assert CLIP_CAP_SEC < MAX_CLIP_SEC == 30.0


def test_no_call_site_re_hardcodes_the_cap():
    """16 处 `group_scenes()` 必须共用默认线：有一处写死 30，看板就会列 3 段而实际出 2 段。"""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    hits = subprocess.run(  # noqa: S607
        ["grep", "-rn", "max_clip_sec=", "fastvideo3/"],
        cwd=root, capture_output=True, text=True,
    ).stdout
    real = [ln for ln in hits.splitlines() if "def group_scenes" not in ln and "max_clip_sec=max_clip_sec" not in ln]
    assert not real, "有调用点自己写死了切分上限：\n" + "\n".join(real)
