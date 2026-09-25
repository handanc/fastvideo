from fastvideo3.core.prompts import render


def test_render_replaces_variables():
    out = render(
        "visual_prompt.v1",
        style="喜剧",
        location="街道",
        time_of_day="日",
        characters="- 小明",
        action="走路",
        camera="中景",
    )
    assert "喜剧" in out and "街道" in out


def test_render_missing_variable_lenient():
    out = render("visual_prompt.v1", style="喜剧")  # 其余变量缺失 → 宽容模式不抛错
    assert "喜剧" in out
