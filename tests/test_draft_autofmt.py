"""「编辑剧本稿 → 保存」必须对**任意排版**都成立。

锁住的语义（都是实测踩过的坑，见 core/draft_norm.py 的模块说明）：

1. 确定性整理层：别称值名（特效转场/定格）、字段别名（人物：/镜头语言：）、markdown 噪声
   （`**镜头 1｜3 秒｜中景跟拍**`、`- **唐僧**：描述`、`---`）、无编号镜头、无字段名的散文、
   说话人写中文名而不是 id —— 全部就地整理，**不花钱、不改一个字的用词**；
2. **幂等**：标准稿进 → 一字不变出。否则每次保存都会"整理"一遍已经是标准格式的稿子，
   给用户一屏根本没发生过的说明；
3. **行号守恒**：整理只改写行、绝不删行，报错行号仍能对上用户编辑器里的那一行；
4. 逐字校验：模型整理后必须一句台词都不丢、不加、不改（台词是成片字幕的唯一来源）；
   丢了就**拒绝保存**而不是静默接受；
5. 整理动作作为 `notes` 给人看，不作为 `problems` 挡保存。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core.draft_norm import flat, labels_of, normalize_draft, spoken_lines
from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.script_text import parse_script_text, render_script_text
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.fake import fake_script

RUN = "run_20260101_000077"
FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _no_lock(outdir):
    from fastvideo3.core import tasklock

    tasklock.release(outdir)
    yield


@pytest.fixture(autouse=True)
def _never_touch_network(monkeypatch):
    """本文件的用例都不许打真实付费接口：工厂换成假 provider。

    `run_meta.fake=true` 的 run 走的是假模式（`_cfg_for_run` 现在会恢复这个标记），
    这里再兜一层：真有代码绕过标记，也只是拿到一个 fake provider，不会烧钱。
    """
    from fastvideo3.core.config import AppConfig
    from fastvideo3.providers import factory

    real = factory.make_provider

    def _guarded(cap, cfg: AppConfig):
        if not getattr(cfg, "fake", False):  # 兜底：绝不给测试发真 key
            cfg = cfg.model_copy(update={"fake": True})
        return real(cap, cfg)

    monkeypatch.setattr(factory, "make_provider", _guarded)


@pytest.fixture
def run(outdir):
    root, script = _mk_run(outdir, RUN, with_final=False)
    return root, script


def _saved(outdir) -> Script:
    return Script.model_validate(RunStorage(RUN, root=outdir).load_json("script.json"))


def _norm(text: str, old: Script):
    return normalize_draft(text, labels=labels_of(old), title=old.title)


# ─────────────────────── ① 确定性整理层 ───────────────────────
def test_canonical_draft_is_untouched():
    """幂等：标准稿进去，一字不变出来，说明清单为空。

    这条是"每次保存都不会重排用户的稿子"的唯一保证，也是下面所有整理规则的边界。
    """
    src = fake_script(BaseInfo(topic="测试主题"))
    text = render_script_text(src)
    fixed, notes = _norm(text, src)
    assert notes == [], notes
    assert fixed == text
    script, problems = parse_script_text(fixed, src)
    assert problems == []
    assert script.model_dump(mode="json", exclude={"version"}) == src.model_dump(
        mode="json", exclude={"version"}
    )


def test_synonym_draft_saves_without_llm(run):
    """实测稿①：转场写「特效转场」「定格」、结尾多一行「字幕:」→ 现在能存。

    以前是 13 条报错（3 个不认识的值名连累出 9 条孤儿行报错），内容其实完全合法。
    """
    old = _read_old(run[0])
    draft = _fixture("draft_synonyms.md")
    fixed, notes = _norm(draft, old)
    script, problems = parse_script_text(fixed, old)
    assert problems == [], problems
    # 值名只允许出现在**镜头行**上；「画面突然定格」是用户写的画面描述，必须原样留着
    heads = [ln for ln in fixed.split("\n") if ln.startswith("### ")]
    assert not any("定格" in h or "特效转场" in h for h in heads), heads
    assert "画面突然定格" in fixed, "整理器不许改用户的用词"
    trans = {sh.transition_out.value for sh in script.all_shots()}
    assert trans <= {"cut", "dissolve", "fade", "scenery"}
    # 「字幕」整行不进生成 —— 但它那句原文必须在说明里回显，不能悄悄消失
    assert not any("取经归来，真的能娶女王吗" in sh.action for sh in script.all_shots())
    assert any("字幕" in n and "取经归来" in n for n in notes), notes
    assert any("定格" in n or "特效转场" in n for n in notes), notes
    assert any("字幕" in n for n in notes), notes


def _read_old(root: Path) -> Script:
    return Script.model_validate(json.loads((root / "script.json").read_text(encoding="utf-8")))


def test_markdown_screenplay_saves_without_llm(run):
    """实测稿②：整份 markdown 分镜本 → 结构正确、**台词一句不少**。

    这是"从别处粘一份剧本进来"的典型形状：粗体说话人、`**镜头 N｜3 秒｜运镜**` 当镜头行、
    `### Scene 1：标题` 当场景行、`- **名字**：外貌` 当角色表、`---` 分章、大段无标记散文。
    """
    old = _read_old(run[0])
    draft = _fixture("draft_markdown.md")
    fixed, notes = _norm(draft, old)
    script, problems = parse_script_text(fixed, old)
    assert problems == [], problems
    assert script.scenes, "一场都没解析出来"
    assert sum(len(s.shots) for s in script.scenes) >= 5
    # 「- **唐僧**：青年俊美和尚…」是**外貌**，绝不能被当成一句台词
    spoken = [d.text for sh in script.all_shots() for d in sh.dialogues]
    assert not any("青年俊美和尚" in t for t in spoken), spoken
    assert any("师父别走了" in t for t in spoken), spoken
    # 散文进画面（不丢字）
    assert any("衣袂飘飘" in sh.action for sh in script.all_shots())
    # 说话人中文名 → 角色 id，且 id 在角色表里（否则 m5/m6/m9 全断）
    ids = {c.character_id for c in script.characters}
    for sh in script.all_shots():
        for d in sh.dialogues:
            assert d.character_id in ids, (sh.shot_id, d.character_id)
        assert set(sh.characters) <= ids
    assert len(notes) > 10, "整理过程应该留痕"


def test_scene_location_not_polluted(run):
    """`地点：宫殿前广场，天空逐渐变暗` 只有第一段是地点。

    location 是场景图提示词的唯一地点来源，把画面信息塞进去会烧进一张跑偏的场景图。
    """
    old = _read_old(run[0])
    fixed, _ = _norm(_fixture("draft_markdown.md"), old)
    script, problems = parse_script_text(fixed, old)
    assert problems == []
    scene3 = next(s for s in script.scenes if s.scene_id == "scene_3")
    assert scene3.location.startswith("宫殿前广场")
    assert "天空逐渐变暗" not in scene3.location
    assert "天空逐渐变暗" in (scene3.summary + "".join(sh.action for sh in scene3.shots))


def test_shot_ids_stay_unique(run):
    """`**镜头 2｜…**` 之后再一个没编号的 `**结尾镜头｜2 秒｜定格**` 不能撞成同一个镜号。

    撞号的表现是"镜头 X 在场 scene_Y 内重复"，而人看到的是两个不同写法，根本联不起来。
    """
    old = _read_old(run[0])
    fixed, _ = _norm(_fixture("draft_markdown.md"), old)
    ids = [sh.shot_id for sh in Script.model_validate(
        parse_script_text(fixed, old)[0].model_dump(mode="json")).all_shots()]
    assert len(ids) == len(set(ids)), ids


def test_line_numbers_survive_drops(run):
    """整理丢行/并行之后，报错行号仍必须指用户编辑器里的那一行（所以只清空、不删行）。"""
    old = _read_old(run[0])
    draft = _fixture("draft_synonyms.md")
    fixed, _ = _norm(draft, old)
    assert len(fixed.split("\n")) == len(draft.split("\n"))
    # 「字幕:」那一行被**清空**而不是删掉：它后面的行号才没有整体前移
    src = draft.split("\n")
    idx = next(i for i, ln in enumerate(src) if ln.startswith("字幕:"))
    assert fixed.split("\n")[idx].strip() == ""
    assert fixed.split("\n")[idx + 1 :] == src[idx + 1 :]

    broken = "整体说明xyz\n" + draft          # 第 1 行：块外、无字段名、认不出说话人
    fixed2, _ = _norm(broken, old)
    _, problems = parse_script_text(fixed2, old)
    assert problems, "认不出的行必须交回解析器点名"
    assert problems[0].startswith("第 1 行"), problems
    assert fixed2.split("\n")[0].strip() == "整体说明xyz"


# ─────────────────────── ② 逐字校验 ───────────────────────
def test_spoken_lines_covers_both_shapes():
    """任意排版都能把"说出口的话"抠出来（校验台词没被模型改写的依据）。"""
    text = '  c1（excited）: 标准台词一句\n**猪八戒：** 粗体台词二句\n唐僧：普通台词三句\n'
    got = spoken_lines(text)
    assert any("标准台词一句" in g for g in got)
    assert any("粗体台词二句" in g for g in got)
    assert any("普通台词三句" in g for g in got)
    # 字段行不能算说话
    assert not any("Pan Right" in g for g in spoken_lines("运镜: Pan Right\n"))


def test_flat_ignores_punctuation():
    assert flat("师父，别走了！") == flat("师父别走了")


# ─────────────────────── ③ Web 保存路径 ───────────────────────
def test_put_script_text_accepts_markdown_draft(client, run):
    """端到端：网页上粘一份 markdown 剧本 → 点保存 → 200，并告诉用户整理做了什么。"""
    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _fixture("draft_markdown.md")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["notes"] and "整理" in body["notes"][0], body["notes"]
    assert len(body["notes"]) <= 10, "说明不能刷屏"
    assert len(body["fixes"]) > 10, "全量清单在 fixes 里，不能只给个截断版"
    script = _saved(run[0].parent)
    assert any(
        any("师父别走了" in d.text for d in sh.dialogues) for sh in script.all_shots()
    )
    # 再读回来的剧本稿必须是标准排版（可以原样再存一次）
    again = client.get(f"/api/runs/{RUN}/script/text").json()["text"]
    r2 = client.put(f"/api/runs/{RUN}/script/text", json={"text": again})
    assert r2.status_code == 200, r2.text
    assert "整理" not in "".join(r2.json().get("notes") or []), "标准稿不该再被整理一遍"


#: 只有"只改排版"那条提示词含它 —— 落盘时的外貌卡抽取等别的文本调用不能混进断言里
FORMATTER_MARK = "排版规范化器"


def _stub_formatter(monkeypatch, outdir, reply: str, sink: list[str]):
    """把这个 run 换成"真模式 + 替身文本模型"：证明走的是模型整理，而不是网络。

    两条都要：`run_meta.fake=false`（否则 `_cfg_for_run` 恢复假模式，端点根本不建 provider）
    与 `factory.make_provider` 换成替身（否则测试会拿 .env 里的真 key 打付费接口）。
    """
    from fastvideo3.providers import factory

    meta_path = outdir / RUN / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fake"] = False
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    class _Stub:
        name = "stub_fmt"
        model = "stub-model"
        last_usage: dict = {}

        async def chat(self, messages, json_schema=None, **kw):
            sink.append(messages[-1]["content"])
            return reply

    monkeypatch.setattr(factory, "make_provider", lambda cap, cfg: _Stub())


_GOOD_REFORMAT = (
    "《测试》 · 都市轻喜剧 · 30s · 9:16\n\n"
    "# 角色\nsc_1 小明 · male · adult\n  外貌: 短发男生\nsc_2 小红 · female · adult\n"
    "  外貌: 长发女生\n\n"
    "## scene_1 · 天台 · 日\n标题: 摊牌\n概要: 小明摊牌。\n\n"
    "### shot_1_1 对白 4s · 切\n在场: sc_1, sc_2\n画面: 天台上两人对坐。\n"
    "  sc_1（nervous）: 我有话对你说。\n  sc_2（calm）: 你说。\n"
)
#: 模型吞了最后一句台词（这类失败必须**响亮**：静默接受等于花钱出一条缺台词的成片）
_LOSSY_REFORMAT = _GOOD_REFORMAT.replace("  sc_2（calm）: 你说。\n", "")
#: 确定性层**救不了**的形状：没有场景头、没有镜头行，第一行还是括号说明。
_PROSE = (
    "（这段没有镜号，人只写了一句话一幕）\n"
    "天台的黄昏，两人对坐。\n"
    "小明：我有话对你说。\n"
    "小红：你说。\n"
)


def test_llm_reformat_is_used_only_when_needed(client, run, monkeypatch):
    """确定性层救不了的稿子（整篇散文、没有镜头行）才调模型；调完逐字校验台词。"""
    called: list[str] = []
    _stub_formatter(monkeypatch, run[0].parent, _GOOD_REFORMAT, called)

    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _PROSE})
    assert r.status_code == 200, r.text
    assert any(FORMATTER_MARK in c for c in called), called
    fmt = [c for c in called if FORMATTER_MARK in c]
    assert "时长" in fmt[0], "提示词必须给出标准排版"
    assert "只输出整理后的完整剧本稿" in fmt[0]
    script = _saved(run[0].parent)
    spoken = [d.text for sh in script.all_shots() for d in sh.dialogues]
    assert any("我有话对你说" in t for t in spoken), spoken
    assert any(t.startswith("你说") for t in spoken), spoken
    assert r.json().get("fixes"), "模型整理过就必须告诉用户整理成了什么样"


def test_llm_reformat_refused_when_it_drops_a_line(client, run, monkeypatch):
    """模型吞了一句台词 → 拒绝保存并点名是哪句，绝不静默接受。"""
    called: list[str] = []
    _stub_formatter(monkeypatch, run[0].parent, _LOSSY_REFORMAT, called)
    before = _saved(run[0].parent).model_dump_json()

    r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _PROSE})
    assert r.status_code == 422, r.text
    detail = json.dumps(r.json(), ensure_ascii=False)
    assert "你说" in detail, detail
    fmt = [c for c in called if FORMATTER_MARK in c]
    assert len(fmt) == 2, "带原因重试一次是设计内的（第二次要让它看到丢了哪句）"
    assert "丢" in fmt[1] or "你说" in fmt[1], fmt[1][-400:]
    # 整理失败不落盘：盘上仍是原来那份剧本，不会留下半相结构
    assert _saved(run[0].parent).model_dump_json() == before


def test_deterministic_layer_is_enough_for_real_drafts(client, run, monkeypatch):
    """两份实测稿子**不该**触发模型调用：能免费整理的绝不花钱。"""
    called: list[str] = []
    _stub_formatter(monkeypatch, run[0].parent, _GOOD_REFORMAT, called)
    for name in ("draft_markdown.md", "draft_synonyms.md"):
        r = client.put(f"/api/runs/{RUN}/script/text", json={"text": _fixture(name)})
        assert r.status_code == 200, (name, r.text)
    assert not any(FORMATTER_MARK in c for c in called), "确定性整理层没理由让模型插手"
