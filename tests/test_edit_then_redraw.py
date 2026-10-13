"""改描述 → 删图 → 点「开始生成」：新描述必须真的进提示词。

看板上"点文字即改"（外貌 / 场景地点与氛围 / 镜头画面与镜头语言）提交后走的是
`PUT /script`：服务端校验 + 血缘作废。本文件锁住后半段承诺 —— 作废之后重画时，
m2/m3 读的是**改过的剧本**，而不是缓存里的旧描述：

- 角色外貌 → 立绘提示词（`char_portrait.v1` 的 appearance 槽）；
- 场景地点/氛围 → 场景概念图提示词；
- 镜头画面/镜头语言 → 首帧提示词（蒸馏失败退化原文，原文里就有 action）。

不删图也应重画（描述变了产物即 stale）；删图是用户的第二种操作顺序，两条路都走一遍。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from test_step_mode import _mk_run  # 复用"剧本+资产+首帧已生成"的 run 搭建

import fastvideo3.app.api as api_mod
import fastvideo3.core.stages as stages_mod
from fastvideo3.core import progress as prog
from fastvideo3.core.stages import build_context
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import Capability
from fastvideo3.providers.fake import (
    FakeT2IProvider,
    FakeTextProvider,
    FakeVLMProvider,
)


class _SpyT2I(FakeT2IProvider):
    """记录每一次文生图的提示词，用来证明"新描述进了提示词"。"""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    async def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return await super().generate(prompt, **kw)


class _NoDistillText(FakeTextProvider):
    """蒸馏返回空 → m3 退化用渲染后的元提示词（里面逐字含镜头画面/镜头语言）。"""

    async def chat(self, messages, json_schema=None, **kw):
        if json_schema is None:
            return ""
        return await super().chat(messages, json_schema=json_schema, **kw)


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        yield c


def _edit(client, rid, root, mutate):
    payload = json.loads((root / "script.json").read_text(encoding="utf-8"))
    mutate(payload)
    r = client.put(f"/api/runs/{rid}/script", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


async def test_edited_descriptions_reach_the_redraw_prompts(client, outdir, cfg, monkeypatch):
    rid = "run_edit_redraw"
    root, _ = _mk_run(outdir, rid)

    new_appearance = "全身银色铠甲，蓝色披风，左眼一道疤"
    new_location = "冰封的湖面"
    new_action = "角色在冰面上滑倒，镜头俯拍"
    new_camera = "俯拍特写"

    # 1) 看板内联编辑提交的就是这个请求
    r = _edit(client, rid, root, lambda p: (
        p["characters"][0].__setitem__("appearance", new_appearance),
        p["scenes"][0].__setitem__("location", new_location),
        p["scenes"][0]["shots"][0].__setitem__("action", new_action),
        p["scenes"][0]["shots"][0].__setitem__("camera", new_camera),
    ))
    inv = set(r["items"])
    assert "assets/char_c1.png" in inv, "改外貌必须作废定妆照"
    assert "assets/scene_s1.png" in inv, "改场景地点必须作废概念图"
    assert any("shot_s1_01" in i for i in inv), "改画面描述必须作废该镜头首帧"

    # 2) 用户的操作顺序：删掉当前图片
    for stage, name in (
        ("assets", "char_c1.png"),
        ("assets", "scene_s1.png"),
        ("storyboard", "shot_s1_01.png"),
    ):
        d = client.delete(f"/api/runs/{rid}/artifacts/{stage}/{name}")
        assert d.status_code == 200, d.text

    # 3) 点「开始生成」：chars → scenes → frames
    spy = _SpyT2I()

    def maker(cap, cfg_, work_dir=None):
        if cap is Capability.T2I:
            return spy
        if cap is Capability.TEXT:
            return _NoDistillText()
        if cap is Capability.VLM:
            return FakeVLMProvider()
        from fastvideo3.providers.factory import make_provider as real

        return real(cap, cfg_, work_dir)

    monkeypatch.setattr(stages_mod, "make_provider", maker)

    storage = RunStorage(rid, root=outdir)
    meta = storage.load_json("run_meta.json")
    ctx = build_context(cfg, storage)
    from fastvideo3.app.cli import _run_step

    steps = []
    for _ in range(4):
        rep = prog.compute_progress(storage, meta, cfg)
        if rep.next is None or rep.next.id not in ("chars", "scenes", "frames"):
            break
        await _run_step(ctx, storage, cfg, meta, rep.next)
        steps.append(rep.next.id)
    assert steps == ["chars", "scenes", "frames"], steps

    joined = "\n".join(spy.prompts)
    assert new_appearance in joined, "立绘没用新外貌描述重画"
    assert new_location in joined, "场景概念图没用新地点重画"
    assert new_action in joined and new_camera in joined, "首帧没用新画面/镜头语言重画"

    # 台账也记下了新描述（看板"外貌"列与血缘的依据）
    registry = json.loads((root / "asset_registry.json").read_text(encoding="utf-8"))
    assert registry["char_c1"]["meta"]["appearance"] == new_appearance
    assert registry["scene_s1"]["meta"]["location"] == new_location


async def test_edited_description_redraws_without_deleting(client, outdir, cfg, monkeypatch):
    """不删图也要重画：描述变了产物即 stale，「开始生成」直接补。"""
    rid = "run_edit_nodel"
    root, _ = _mk_run(outdir, rid)
    new_appearance = "红色斗篷，金色面具"
    _edit(client, rid, root, lambda p: p["characters"][0].__setitem__("appearance", new_appearance))

    spy = _SpyT2I()

    def maker(cap, cfg_, work_dir=None):
        if cap is Capability.T2I:
            return spy
        if cap is Capability.VLM:
            return FakeVLMProvider()
        from fastvideo3.providers.factory import make_provider as real

        return real(cap, cfg_, work_dir)

    monkeypatch.setattr(stages_mod, "make_provider", maker)
    storage = RunStorage(rid, root=outdir)
    meta = storage.load_json("run_meta.json")
    ctx = build_context(cfg, storage)
    from fastvideo3.app.cli import _run_step

    rep = prog.compute_progress(storage, meta, cfg)
    assert rep.next.id == "chars" and "c1" in (rep.next.stale or []) or "c1" in (
        rep.next.missing or []
    ), "改外貌后角色图步骤必须把 c1 列为待重做"
    await _run_step(ctx, storage, cfg, meta, rep.next)
    assert any(new_appearance in p for p in spy.prompts), "没删图也必须按新描述重画"
