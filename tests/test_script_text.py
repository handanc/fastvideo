"""剧本稿 ⇄ Script 文本层：人改的是剧本，不是 JSON。

判据：
- 往返等价（渲染→解析回来的结构与原 Script 一致，不可见字段按 id 继承不丢）；
- 容错（中英文枚举混写、缺可选字段）；
- 错误必须带行号，且不落盘（problems 非空即拒绝）；
- 时长偏差这类软规则走 warnings，不阻断保存。
"""

from __future__ import annotations

from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.script_text import (
    duration_warnings,
    parse_script_text,
    render_script_text,
)
from fastvideo3.providers.fake import fake_script


def _script(target: int = 30) -> Script:
    return fake_script(BaseInfo(topic="测试主题", target_duration_sec=target))


def test_roundtrip_is_structure_preserving():
    src = _script()
    text = render_script_text(src)
    got, problems = parse_script_text(text, old=src)
    assert problems == [], problems
    # 结构逐字段相等（version 由保存路径管，文本层不碰）
    assert got.model_dump(mode="json", exclude={"version"}) == src.model_dump(
        mode="json", exclude={"version"}
    )
    assert len(text.splitlines()) > len(src.all_shots())


def test_invisible_fields_survive_edit():
    """visual_prompt / look / canonical_tokens 不进文本，但必须按 id 继承回来。"""
    src = _script()
    src.characters[0].canonical_tokens = ["白衬衫", "短发"]
    src.scenes[0].shots[0].visual_prompt = "distilled prompt for frame"
    got, problems = parse_script_text(render_script_text(src), old=src)
    assert problems == []
    assert "白衬衫" in got.characters[0].canonical_tokens
    assert got.scenes[0].shots[0].visual_prompt == "distilled prompt for frame"
    assert "visual_prompt" not in render_script_text(src)
    assert "必留词" not in render_script_text(src)


def test_appearance_edit_updates_text_but_keeps_look_for_re_extration():
    src = _script()
    src.characters[0].canonical_tokens = ["白衬衫"]
    text = render_script_text(src).replace("25岁短发男生，白衬衫牛仔裤", "25岁短发男生，蓝工装")
    got, problems = parse_script_text(text, old=src)
    assert problems == []
    assert got.characters[0].appearance == "25岁短发男生，蓝工装"
    # 旧必留词仍带着 → 由保存路径的 _refresh_looks_for_changed_chars 重抽覆盖
    assert got.characters[0].canonical_tokens == ["白衬衫"]


def test_base_info_settings_not_carried_by_text():
    """topic/culture_background/episode 属于任务设定：文本稿改不动它们。"""
    src = _script()
    text = render_script_text(src).replace("《", "《改了片名", 1)
    got, problems = parse_script_text(text, old=src)
    assert problems == []
    assert got.base_info.topic == src.base_info.topic
    assert got.base_info.culture_background == src.base_info.culture_background
    assert got.base_info.episode == src.base_info.episode


def test_style_and_aspect_edits_are_carried():
    src = _script()
    text = render_script_text(src).replace(" · 9:16", " · 16:9").replace(
        src.base_info.style, "悬疑犯罪"
    )
    got, problems = parse_script_text(text, old=src)
    assert problems == []
    assert got.base_info.aspect_ratio == "16:9"
    assert got.base_info.style == "悬疑犯罪"


def test_style_containing_middle_dot_survives_roundtrip():
    """分段符只认精确的「 · 」：用户在风格名里写间隔号不该被切成两段。"""
    src = _script()
    src.base_info.style = "都市·轻喜剧"
    got, problems = parse_script_text(render_script_text(src), old=src)
    assert problems == []
    assert got.base_info.style == "都市·轻喜剧"


def test_parser_accepts_enum_tokens_and_missing_optionals():
    src = _script()
    text = render_script_text(src)
    # 英文枚举值 + 删掉运镜行 + 删掉概要行
    text = text.replace(" 对白 ", " dialogue ").replace(" · 切", " · cut")
    text = "\n".join(
        ln for ln in text.splitlines() if not ln.startswith("运镜:") and not ln.startswith("概要:")
    )
    got, problems = parse_script_text(text, old=src)
    assert problems == []
    assert got.scenes[0].shots[0].type.value == "scenery"
    assert all(sh.camera == "" or sh.camera == "中景" for sh in got.all_shots())


def test_parse_errors_carry_line_numbers():
    src = _script()
    lines = render_script_text(src).splitlines()
    bad_cases = {
        "台词缺说话人": (3, "  没有说话人和情绪的一句"),
        "时长非法": (3, "### sX_9 对白 0s · 切"),
        "类型不认识": (3, "### sX_9 特写 5s · 切"),
        "画幅非法": (0, "《片名》  都市轻喜剧 · 30s · 4:3"),
    }
    for name, (idx, replacement) in bad_cases.items():
        broken = list(lines)
        broken[idx] = replacement
        _, problems = parse_script_text("\n".join(broken), old=src)
        assert problems, f"{name} 应当被拒绝但没有报错"
        assert any("第" in p and "行" in p for p in problems), f"{name} 报错未带行号：{problems}"


def test_duplicate_ids_are_rejected():
    src = _script()
    text = render_script_text(src)
    first_char = next(ln for ln in text.splitlines() if ln.startswith("c1 "))
    _, problems = parse_script_text(text + "\n" + first_char, old=src)
    assert any("重复" in p for p in problems)


def test_shot_without_scene_is_rejected():
    src = _script()
    _, problems = parse_script_text(
        "### s9_1 对白 5s · 切\n画面: 孤立镜头\n", old=src
    )
    assert any("不在任何场景" in p for p in problems)


def test_duration_warnings_flag_off_target_scripts():
    src = _script(target=30)
    assert duration_warnings(src) == []  # 6 镜 × 5s = 30s
    for sh in src.all_shots():
        sh.duration_sec = 20.0
    assert duration_warnings(src), "偏差 300% 必须警告"


def test_bare_header_parses_empty_but_is_not_valid_script():
    """文本层只判"格式认得"：只有片头行的稿子能解析，但没有任何场景。

    "至少一场一镜"是 `api._script_cross_check` 的职责（保存路径的第二道关），这里刻意
    不重复报错——否则两道关对同一件事各说一遍，措辞还会漂。
    """
    src = _script()
    script, problems = parse_script_text("《空》  都市轻喜剧 · 30s · 9:16\n", old=src)
    assert problems == []
    assert script.scenes == [] and script.characters == []
    assert script.title == "空"
    assert script.base_info.target_duration_sec == 30  # 设定从旧剧本继承
