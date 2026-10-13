"""提示词模板加载器：prompts/<name>.md + {占位符} 渲染。"""

from __future__ import annotations

from fastvideo3.core.config import PROMPTS_DIR

# 画面内禁止出现任何文字：字幕的唯一来源是 m7 用**剧本台词**烧录（字体、位置、时间轴
# 全部确定）。模型自己"画"字幕的代价是实测过的：错字、漏句、字体不受控，还会与我们
# 烧的那条叠成两层。所以每一条生成请求（首帧/运动/口型）都显式禁止画面文字。
NO_ONSCREEN_TEXT = "画面内不得出现任何字幕、文字、水印、台词或标题"


def render(name: str, **variables: object) -> str:
    """渲染提示词模板；缺失变量以空串填充（宽容模式，便于教学迭代）。"""
    path = PROMPTS_DIR / f"{name}.md"
    template = path.read_text(encoding="utf-8")
    try:
        return template.format(**variables)
    except KeyError:
        # 宽容模式: 逐变量替换, 未提供的占位符留空
        out = template
        import re

        for key in set(re.findall(r"\{(\w+)\}", template)):
            out = out.replace("{" + key + "}", str(variables.get(key, "")))
        return out


def render_strict(name: str, **variables: object) -> str:
    """严格渲染：有未替换占位符即抛错。

    度量/抽取类模板必须用它 —— render() 的宽容模式会把缺失字段静默渲染成空串，
    rubric 里出现"角色外貌文字描述："后面是空的这种致命残缺而无人察觉。
    """
    import re

    path = PROMPTS_DIR / f"{name}.md"
    template = path.read_text(encoding="utf-8")
    # 不用 str.format：度量模板里带 JSON 示例（{"face": bool, ...}），format 会把
    # 字面花括号当占位符。只替换 {标识符} 形态的真占位符。
    missing = sorted({k for k in re.findall(r"\{(\w+)\}", template) if k not in variables})
    if missing:
        raise KeyError(f"模板 {name} 缺变量: {missing}（传入: {sorted(variables)}）")
    out = template
    for k, v in variables.items():
        out = out.replace("{" + k + "}", str(v))
    leftover = sorted(set(re.findall(r"\{(\w+)\}", out)))
    if leftover:
        raise KeyError(f"模板 {name} 存在未替换占位符: {leftover}")
    return out
