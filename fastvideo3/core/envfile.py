"""`.env` 的定点写回工具（只改模型相关的行，别的一律不碰）。

**为什么需要它**：多数用户只用 Web 面板，不会去动 `.env`。而 `load_config` 的优先级是
环境变量 > yaml —— 如果面板上选了模型而 `.env` 还留着上一次的手改值，用户看到的就是
"我明明选了 A，跑出来是 B"。所以面板的选择要同时做两件事：

1. 落进 `run_meta.json` —— 这条 run 的每一步（含续跑）都精确复现（`config.apply_run_overrides`）；
2. **写回 `.env`** —— 成为之后新建任务的默认。

这两条一起才是用户要的"前端选择的模型优先级最高"：本 run 由 meta 保证，以后的 run 由写回的
`.env` 保证，命令行 `fastvideo3 new` 不指定时也读到同一个值。

**为什么单独一个文件而不是就地拼字符串**：`.env` 里装着 API 密钥。
- 只替换 `FV3_*_MODEL` 这几行的**值**，其余每一行（密钥、注释、空行、排版）原样保留；
- 不打印、不返回、不上传文件内容 —— 任何把 `.env` 写进日志或 HTTP 响应的做法都是密钥泄露；
- 临时文件 + `os.replace` 原子落盘，中途出错不会留下半个被截断的 `.env`；
- 写失败（只读、被编辑器占用、权限）**绝不阻断建任务**：run_meta 已经有精确选择，
  调用方拿到错误字符串自己去提示。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastvideo3.core.config import PROJECT_ROOT

#: 能力 → 环境变量名。必须与 `config.load_config()` 读取的那几个键**一一对应**，
#: 写错了就是"面板选了但没生效"。
#: 口型（talk）也在其中：面板的「口型」下拉选的就是模型名（wan2.2-s2v / videoretalk /
#: wan3.0-video / wan3.0-video-prime），写回 `FV3_TALK_MODEL` 才能成为下次新建任务的默认。
#: ⚠ 但**模式不由这里决定**：`load_config` 会按这个模型名反推 `talk_mode`
#: （见 `talkmode.mode_for_model`），所以写回一个模型名不会造成"模式与通道各存一份"——
#: 恰恰相反，少了这层反推才会出现"通道已改道、模式还是旧的 → 全部降级无口型"。
ENV_KEY_FOR_CAP: dict[str, str] = {
    "text": "FV3_TEXT_MODEL",
    "t2i": "FV3_T2I_MODEL",
    "i2v": "FV3_I2V_MODEL",
    "talk": "FV3_TALK_MODEL",
    "r2v": "FV3_R2V_MODEL",
    "tts": "FV3_TTS_MODEL",
    "vlm": "FV3_VLM_MODEL",
}


def env_path() -> Path:
    return PROJECT_ROOT / ".env"


def _line_ending(raw: str) -> str:
    """保持原文件的换行风格：统一换行会让整个 `.env` 在 git diff 里变红，掩盖真实改动。"""
    return "\r\n" if "\r\n" in raw else "\n"


_COMMENT_TAIL = re.compile(r"(\s+#.*)$")


def _split_inline_comment(indented: str) -> str:
    """取出一行的行尾注释（`KEY=wan3.0-video   # 说明` 这种写法很常见）。

    连**注释前的空白一起**原样保留：改空格数会让这一行在 git diff 里显示成无意义的改动，
    而这个文件里通常就躺着密钥，diff 越干净越好审。
    """
    m = _COMMENT_TAIL.search(indented.rstrip())
    return m.group(1) if m else ""


def update_env_vars(values: dict[str, str], path: Path | None = None) -> tuple[list[str], str | None]:
    """把 `{环境变量名: 值}` 逐行写进 `.env`，返回 `(改动的变量名, 错误说明或 None)`。

    这是底层的**按名写回**入口（`update_env_models` 只是把"能力→模型名"翻译成变量名再进来）。
    单独开这一层是为了让 API KEY 这类**不是模型**的值走同一份行编辑代码 —— 两条"改 .env"的
    实现必然一条干净一条粗糙，而这个文件里躺着密钥，粗糙的那条最容易把别的行改坏。

    规则（都是为了不碰用户的其它配置）：
    - 只改值、只追加缺失行，**从不删除**任何行；
    - 注释掉的行（`# DASHSCOPE_API_KEY=…`）**不替他打开** —— 那是用户主动关掉的开关；
      需要生效时在文件末尾追加一条未注释的行（dotenv 里后定义覆盖前定义，语义一致）；
    - **空值不写回**：留空＝"不改/跟随现状"，把它当"清空"会把用户的 `.env` 变成未知状态；
    - 值里含空白或 `#` 的直接拒绝（不是剪掉）：dotenv 不加引号时这些字符会让值被截断，
      写进去就是个静默失效的 KEY —— 与其存坏值，不如当场说清楚。
    """
    writes = {k: (v or "").strip() for k, v in (values or {}).items()}
    writes = {k: v for k, v in writes.items() if k and v}
    if not writes:
        return [], None
    bad = [k for k, v in writes.items() if any(c.isspace() or c == "#" for c in v)]
    if bad:
        return [], f"这些变量含空格或 #，写进 .env 会被截断（不写回）：{'、'.join(bad)}"
    target = path or env_path()
    tmp = target.with_name(target.name + ".fv3tmp")
    changed: list[str] = []
    try:
        # **按字节读**：`read_text()` 会把 CRLF 归一化成 \n，那样写回去就等于重排了整个
        # .env（Windows 上这个文件多半是 CRLF，一个"只改一行"的提交会变成全文件 diff）。
        raw = target.read_bytes().decode("utf-8") if target.exists() else ""
        nl = _line_ending(raw)
        lines = raw.split(nl) if raw else []
        ends_with_newline = raw.endswith("\n")  # 原本没有尾换行就不强行加

        for key, name in writes.items():
            prefix = f"{key}="
            hit = False
            for i, line in enumerate(lines):
                indented = line.lstrip()
                if not indented.startswith(prefix):
                    continue
                lines[i] = f"{prefix}{name}{_split_inline_comment(indented)}"
                hit = True
                break
            if not hit:
                lines.append(f"{prefix}{name}")
            changed.append(key)

        if not changed:
            return [], None
        out = nl.join(lines)
        if ends_with_newline and not out.endswith(nl):
            out += nl
        tmp.write_bytes(out.encode("utf-8"))  # 与读取对称：不重编码、不翻译行尾
        os.replace(tmp, target)
        return changed, None
    except Exception as e:  # noqa: BLE001  # 写不回不能影响主流程（调用方自己去提示）
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        # 异常文本可能带路径，但**绝不带值**：值只出现在 `writes` 里，不参与消息拼装。
        return changed, f"{type(e).__name__}: {e}"


def update_env_models(models: dict[str, str], path: Path | None = None) -> tuple[list[str], str | None]:
    """把 `models`（能力→模型名）写进 `.env` 对应行，返回 `(改动的变量名, 错误说明或 None)`。

    行编辑规则见 `update_env_models` → `update_env_vars`（唯一一份实现）；这里只负责
    "能力名 → 环境变量名"的翻译，以及把空值（＝跟随画质档位）挑掉不写回。
    """
    mapped = {
        ENV_KEY_FOR_CAP[cap]: name
        for cap, name in (models or {}).items()
        if cap in ENV_KEY_FOR_CAP
    }
    return update_env_vars(mapped, path)
