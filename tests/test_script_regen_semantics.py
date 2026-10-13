"""「改写已有剧本」（revise）的三份真相 —— 走真实 CLI 入口。

这条路径的性质：模型只看得到**剧本稿文本**，而文本层刻意不含三个一致性锚点字段
（`shot.visual_prompt` = m3 蒸馏回写、`character.look`、`character.canonical_tokens`）。
所以"按意见改写"绝不能等于"把这三个字段清零 + 全片作废"，必须：

1. 按 id 把不可见锚点从旧剧本继承下来（与人工编辑剧本稿同一规则）；
2. 没被意见点名的镜头 → 不作废；只给新角色抽外貌卡（重抽全表 = 必留词漂移 = 全片作废）；
3. 提交失败时盘上仍是旧剧本 + 旧血缘（不许"新剧本配旧作废范围"）；
4. 模型写出的悬空角色引用必须在花钱前被拦下。

全部通过 `cli script-regen` 驱动，而不是自己复刻调用顺序 —— 顺序与传参本身就是被测对象。
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.script_text import parse_script_text, render_script_text
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import Capability
from fastvideo3.providers.fake import FakeTextProvider, fake_script


class _ModelSeesOnlyDraft(FakeTextProvider):
    """仿真"改写模式下的真实模型"：只看得见剧本稿，**只输出剧本稿**（不是 JSON）。

    模型确实拿不到 `visual_prompt`/`look`/`canonical_tokens` —— 它们不在喂给它的文本里，
    所以它交回来的稿子也不携带这些字段，由 `parse_script_text(new_text, old)` 从旧剧本继承。
    这正是本文件要锁住的场景（旧实现让模型直接产 JSON，锚点全靠 `inherit_hidden_anchors` 补救）。

    `mutate(raw_dict)` 钩子沿用旧的写法：内部把稿子解析成 dict 交给它改，再渲染回文本 ——
    这样"模型只会动文本层"这个前提仍然成立（渲染时锚点字段本来就不会出现）。
    """

    def __init__(self, old: Script, mutate=None):
        super().__init__()
        self.old = old
        self.mutate = mutate
        self.script_calls = 0
        self.look_prompts: list[str] = []   # 外貌卡抽取调用（每角色一次）

    async def chat(self, messages, json_schema=None, **kw):
        if json_schema is None:
            # 改写模式：不带 json_schema，要求返回剧本稿文本
            self.script_calls += 1
            text = messages[-1]["content"]
            assert "直接输出改写后的完整剧本稿" in text, "revise 必须要求模型改**文本**"
            assert "【修改意见】" in text, "人的意见必须随稿子一起交给模型"
            assert "不要输出 JSON" in text
            visible, problems = parse_script_text(render_script_text(self.old), self.old)
            assert not problems, problems
            raw = visible.model_dump(mode="json")
            for c in raw["characters"]:  # 文本层没交给我们的字段，模型只能给空
                c["look"] = None
                c["canonical_tokens"] = []
            for sc in raw["scenes"]:
                for sh in sc["shots"]:
                    sh["visual_prompt"] = ""
            if self.mutate:
                self.mutate(raw)
            return render_script_text(Script.model_validate(raw))
        if json_schema is Script:
            raise AssertionError("改写模式不该再向模型要 JSON（旧路径已废弃）")
        if json_schema is not None and json_schema is not Script:
            self.look_prompts.append(messages[-1]["content"])  # 抽外貌卡的调用
        return await super().chat(messages, json_schema=json_schema, **kw)


def _look(outfit: str):
    from fastvideo3.core.schema import CharacterLook

    return CharacterLook(
        face="圆脸", outfit=outfit, headwear="帽子", palette=["灰色"], distinctive=[outfit]
    )


def _seed(outdir: Path, run_id: str) -> RunStorage:
    """造一个"剧本 + 全部产物都已生成"的 run（钱已花），带三个锚点字段。"""
    storage = RunStorage(run_id, root=outdir)
    script = fake_script(BaseInfo(topic="锚点继承", target_duration_sec=30))
    script.version = 3
    for i, c in enumerate(script.characters):
        c.look = _look(("灰色僧袍", "红色连衣裙")[i % 2])
        c.canonical_tokens = ["灰色僧袍", "黑色僧帽"] if i == 0 else ["红色连衣裙"]
    for sc in script.scenes:
        for sh in sc.shots:
            sh.visual_prompt = f"蒸馏后的首帧提示词-{sh.shot_id}"
    storage.save_json("run_meta.json", {
        "run_id": run_id,
        "base_info": script.base_info.model_dump(mode="json"),
        "fake": False, "quality": "cinema", "talk_mode": "audio_overlay",
        "budget_limit_cny": 500.0,
    })
    storage.save_json("script.json", script.model_dump(mode="json"))
    for c in script.characters:
        _png(storage.path("assets", f"char_{c.character_id}.png"))
    for sc in script.scenes:
        _png(storage.path("assets", f"scene_{sc.scene_id}.png"))
        for sh in sc.shots:
            _png(storage.path("storyboard", f"shot_{sh.shot_id}.png"))
    (storage.root / "final").mkdir(exist_ok=True)
    (storage.root / "final" / "final.mp4").write_bytes(b"x")
    return storage


def _png(p: Path) -> None:
    from PIL import Image

    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (9, 9, 9)).save(p)


def _revise(outdir: Path, run_id: str, monkeypatch, mutate=None, prompt="把第二句台词改得更直接"):
    """跑真实的 `cli script-regen --mode revise`，只把文本模型换成"看得见剧本稿的模型"。"""
    from fastvideo3.app import cli

    old = Script.model_validate(RunStorage(run_id, root=outdir).load_json("script.json"))
    model = _ModelSeesOnlyDraft(old, mutate)
    import fastvideo3.core.stages as stages_mod

    monkeypatch.setattr(
        stages_mod, "make_provider",
        lambda cap, cfg_, work_dir=None: model if cap is Capability.TEXT else FakeTextProvider(),
    )
    res = CliRunner().invoke(
        cli.app, ["script-regen", run_id, "--prompt", prompt, "--mode", "revise"]
    )
    return res, model, old


def test_revise_keeps_anchors_and_narrows_invalidation(outdir, monkeypatch):
    storage = _seed(outdir, "run_regen_1")
    old = Script.model_validate(storage.load_json("script.json"))

    def mutate(raw):
        raw["scenes"][0]["shots"][1]["dialogues"][0]["text"] = "改过的台词，只动这一句。"

    res, model, _ = _revise(outdir, "run_regen_1", monkeypatch, mutate)
    assert res.exit_code == 0, res.output
    saved = Script.model_validate(storage.load_json("script.json"))

    # ① 真的改写了原始剧本（不是只改看板缓存）：版本 +1、新台词落盘、没点名的逐字保留
    assert saved.version == 4
    assert saved.scenes[0].shots[1].dialogues[0].text == "改过的台词，只动这一句。"
    assert saved.scenes[0].shots[1].action == old.scenes[0].shots[1].action
    assert saved.scenes[0].shots[0].dialogues == old.scenes[0].shots[0].dialogues
    # ② 隐藏锚点按 id 搬回来了
    assert [c.canonical_tokens for c in saved.characters] == [
        c.canonical_tokens for c in old.characters
    ], "必留词丢了 → m3 蒸馏校验与 m9 度量基准换了一套词"
    assert all(c.look is not None and c.look.outfit for c in saved.characters), "外貌卡被抹平"
    prompts = [sh.visual_prompt for sc in saved.scenes for sh in sc.shots]
    assert all(prompts), f"visual_prompt 被抹平（m3 要重新蒸馏一次）：{prompts[:3]}"
    # ③ 作废范围只覆盖被点名的那个镜头（+成片），定妆照一张都不该重画
    ev = [e for e in storage.read_events() if e.get("status") == "script_regenerated"]
    assert ev and ev[-1]["invalidated"] <= 5, f"一处台词改动拖垮全片：{ev}"
    stale = set(storage.stale_set())
    assert any("shot_s1_02" in s for s in stale)
    assert not any("shot_s1_01" in s for s in stale), "没被点名的镜头不该重做"
    assert not any("/char_" in s for s in stale), "外貌没变的角色不该重画定妆照"
    assert model.script_calls == 1, "一次改写只该调用一次剧本模型"


def test_revise_extracts_look_only_for_new_characters(outdir, monkeypatch):
    """意见里新增的角色要抽卡；已有角色的卡必须原样留着（重抽=必留词漂移=全片作废）。"""
    storage = _seed(outdir, "run_regen_2")
    old = Script.model_validate(storage.load_json("script.json"))
    old_tokens = {c.character_id: list(c.canonical_tokens) for c in old.characters}
    old_outfits = {c.character_id: c.look.outfit for c in old.characters}

    def mutate(raw):
        raw["characters"].append({
            "character_id": "c9", "name": "路人", "appearance": "路过的行人，蓝衬衫",
            "voice_id": "", "gender": "male", "age_band": "adult",
            # 模型编造的外貌卡：它看不见这个字段，只能瞎编 —— 必须被丢掉并真抽一次
            "look": {"face": "编的", "outfit": "编的外套", "distinctive": ["编的"]},
            "canonical_tokens": ["编的外套"],
        })
        raw["scenes"][0]["shots"][0]["characters"] = ["c9"]

    res, model, _ = _revise(outdir, "run_regen_2", monkeypatch, mutate, prompt="加一个路人")
    assert res.exit_code == 0, res.output
    saved = Script.model_validate(storage.load_json("script.json"))
    by_id = {c.character_id: c for c in saved.characters}
    # 新角色：模型编造的卡必须被丢掉，改走真抽（Fake 抽出来是空卡，看记账证明确实抽了）
    assert "编的外套" not in by_id["c9"].canonical_tokens, "模型编造的必留词被当真 → 度量基准是假的"
    assert by_id["c9"].look is not None and by_id["c9"].look.outfit != "编的外套"
    assert len(model.look_prompts) == 1, f"只该给新角色抽一次卡：{len(model.look_prompts)}"
    assert "路过的行人" in model.look_prompts[0]
    for cid, toks in old_tokens.items():
        assert by_id[cid].canonical_tokens == toks, "已有角色的必留词被重抽改写"
        assert by_id[cid].look.outfit == old_outfits[cid], "已有角色的外貌卡被重抽改写"


def test_revise_keeps_old_script_when_commit_fails(outdir, monkeypatch):
    """作废那一步崩掉 → 盘上必须还是旧剧本 + 旧血缘（绝不能"新剧本配旧血缘"）。"""
    from fastvideo3.core import progress as prog

    storage = _seed(outdir, "run_regen_3")

    def boom(*a, **k):
        raise RuntimeError("模拟作废中途崩掉")

    monkeypatch.setattr(prog, "invalidate_for_script_change", boom)
    res, _, old = _revise(outdir, "run_regen_3", monkeypatch, prompt="改点东西")
    assert res.exit_code != 0, "崩了必须非零退出，让看板显示这一步失败"
    on_disk = storage.load_json("script.json")
    assert on_disk["version"] == 3, "新剧本配旧血缘 = 旧脸会被当成已生成直接复用"
    assert on_disk == old.model_dump(mode="json")
    assert not storage.stale_set(), "没提交成功就不该作废任何产物"


def test_revise_after_save_failure_stays_consistent(outdir, monkeypatch):
    """崩在剧本落盘之后（抽卡失败）：新剧本必须已经带着正确的作废范围落地。"""
    from fastvideo3.core import scriptio

    storage = _seed(outdir, "run_regen_3b")

    async def boom(*a, **k):
        raise RuntimeError("模拟补抽外貌卡时崩掉")

    monkeypatch.setattr(scriptio, "refresh_looks_for_changed_chars", boom)

    def mutate(raw):
        raw["scenes"][0]["shots"][1]["dialogues"][0]["text"] = "崩在抽卡前的新台词。"

    res, _, _ = _revise(outdir, "run_regen_3b", monkeypatch, mutate, prompt="改第二句")
    assert res.exit_code != 0
    saved = Script.model_validate(storage.load_json("script.json"))
    assert saved.version == 4 and saved.scenes[0].shots[1].dialogues[0].text.endswith("新台词。")
    stale = set(storage.stale_set())
    assert any("shot_s1_02" in s for s in stale), "剧本已生效，受影响镜头必须已标待重做"


def test_revise_rejects_dangling_character_reference(outdir, monkeypatch):
    """模型最爱犯的错：镜头里写 `在场: c9` 但角色表里没有 → 落盘前必须拦住。"""
    storage = _seed(outdir, "run_regen_4")

    def mutate(raw):
        raw["scenes"][0]["shots"][0]["characters"] = ["ghost"]

    res, _, _ = _revise(outdir, "run_regen_4", monkeypatch, mutate, prompt="随便改改")
    assert res.exit_code != 0
    assert "ghost" in res.output or any(
        e.get("status") == "script_regen_failed" for e in storage.read_events()
    )
    assert Script.model_validate(storage.load_json("script.json")).version == 3, "被拒的剧本不许落盘"
