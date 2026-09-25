"""提示词模板加载器：prompts/<name>.md + {占位符} 渲染。"""

from __future__ import annotations

from fastvideo3.core.config import PROMPTS_DIR


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
