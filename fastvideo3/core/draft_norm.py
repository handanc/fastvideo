"""任意排版的剧本文本 → 项目标准剧本稿（确定性整理：不调模型、不花钱、不改用词）。

为什么要有这一层：`parse_script_text` 是严格解析器，报错按行号点名。这对"看板渲染出来的
稿子"完全正确，对人从别处粘来的稿子就是灾难。两份实测输入（run_20261015_220939 / _2230xx）：

1. 内容完全合法的稿子被 13 条报错挡住，真正的原因只有 4 个：转场写了「特效转场」「定格」、
   多了一行「字幕:」—— 而一个不认识的值名会让紧跟其后的 在场/画面/运镜 全变成孤儿行，
   于是**一个错字报出四条错**；
2. 整份 markdown 分镜本：`**镜头 1｜3 秒｜中景跟拍**`、`**猪八戒：** 台词`、
   `### Scene 1：八戒赖着不走`、`- **唐僧**：青年俊美和尚…`、`---`、大段无标记散文。

人写剧本的这些写法**不是错字，是同一个意思的另一种写法**。本模块把能确定判断的部分改成
标准排版，规则只有三条：

- **只动排版与值名，不动内容**：用户写的每一个字原样保留（用词、语序、标点都不改）；
- **不确定就留着**：认不出的行原样输出，交回严格解析器点名 —— 整理器判错的后果是
  一条看得懂的报错，而不是一句被悄悄吃掉的对白；
- **行号守恒**：并走或丢掉的行改成空行，绝不删行。否则"第 116 行"指的已经不是用户
  编辑器里的那一行，报错等于没说。

`normalize_draft()` 返回 `(标准稿, 说明清单)`：说明是给人看的"我改了哪里、为什么"，
进响应的 `notes` 而不是 `problems` —— 被整理掉的东西不该挡住保存。

确定性层整理完仍解析不出来的（表格体、Fountain 体、纯小说段落），由
`m1_scripting.script_from_draft` 决定是否请模型**只改格式**，并逐字校验内容没被改。
"""

from __future__ import annotations

import re

from fastvideo3.core.schema import Script
from fastvideo3.core.script_text import (
    ASPECTS,
    SEP,
    SHOT_TYPE_ALIASES,
    SHOT_TYPE_CN,
    TRANSITION_ALIASES,
    TRANSITION_CN,
    _segments,
)

# ─────────────────────────── 词表与行形状 ───────────────────────────
#: 人手写的分隔符：`|` `｜` `/` `,` `;` `；` 都等于标准稿的「 · 」。注意不含中文顿号/逗号，
#: 否则「女王出场，慢动作推近」这种**内容**会被切碎。
_SPLIT_RE = re.compile(r"[｜|/,,;；]+")
#: 时长写法：`3 秒` / `3秒` / `约 90 秒` / `3s`
_DUR_RE = re.compile(r"(?:约|大约)?\s*(\d+(?:\.\d+)?)\s*(?:秒|[sS])(?![\w])")
#: 场景标记（只认行首：否则 shot_**3**_3 里的数字会被当成场号）
_SCENE_RE = re.compile(
    r"(?i)^(?:scene|場景|场景)[\s_\-]*(?P<n>\d+)?"
    r"|^第\s*(?P<cn>[一二三四五六七八九十]|\d+)\s*[场幕]"
)
#: 镜头标记：`shot_1_1` / `Shot 1` / `镜头 1` / `第 2 个镜头` / `结尾镜头`。
#: 「镜头」起头但后面没有编号的散文（`镜头扫过繁华的街道…`）必须排除，所以要么有 id、
#: 要么有编号、要么是「结尾镜头」这类固定说法且带时长。
_SHOT_RE = re.compile(
    r"(?i)^(?P<id>shot[\w.\-]*|s\d[\w.\-]*)(?=$|[\s:：｜|/,，])"
    r"|^(?:结尾镜头|片尾镜头|彩蛋镜头|尾镜头)(?=$|[^\w])"
    r"|^(?:分镜|镜头|镜)\s*[:：]?\s*(?P<n>\d+)"
    r"|^第\s*(?P<n2>\d+)\s*(?:个|号)?镜头"
)
#: 已经是标准写法的行，**一字不动**（幂等性：否则每次保存都会"整理"一遍标准稿，
#: 给用户一屏根本没发生过的说明）。
_CANON_SHOT = re.compile(r"^###\s+\S+\s+\S+\s+[\d.]+s\s*·\s*\S+$")
_CANON_SCENE = re.compile(r"^##\s+[A-Za-z0-9][\w.\-]*(\s*·.*)?$")
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
#: `键: 值`（键 ≤10 字、不含冒号）—— 比解析器的 `_FIELD_RE` 宽，人要写「出场人物：」这种。
_KV_RE = re.compile(r"^\s*(?P<key>[^:：\n]{1,10})[:：]\s*(?P<val>.*)$")
#: 「说话人：台词」/「说话人（情绪）：台词」。说话人必须能对上角色，否则 `运镜: Pan Right`
#: 会被当成一句对白 —— 认不出说话人就**不算台词**，宁可让解析器点名。
_SPK_RE = re.compile(
    r"^(?P<who>[^\s:：()（）【】《》]{1,16})\s*"
    r"(?:[（(](?P<emo>[^）)]{0,16})[）)])?\s*[:：]\s*(?P<text>.+)$"
)
#: 人写的字段名 → 标准字段名。标准稿只有七个字段（在场/画面/运镜/标题/概要/外貌/音色），
#: 但同一样东西有十种叫法；这里**改名而不是删除** —— 删了就等于"换个写法丢一段画面描述"。
KEY_RENAMES = {
    "镜头语言": "运镜", "运镜方式": "运镜", "景别": "运镜", "机位": "运镜", "拍摄": "运镜",
    "画面内容": "画面", "动作描述": "画面", "场景描述": "画面", "内容": "画面", "描述": "画面",
    "在场人物": "在场", "在场角色": "在场", "出场人物": "在场", "出场": "在场", "人物": "在场",
    "谁在画面": "在场",
    "简介": "概要", "梗概": "概要", "场景说明": "概要",
    "形象": "外貌", "外形": "外貌", "外观": "外貌", "长相": "外貌",
    "配音": "音色", "声线": "音色",
}
#: 「地点：女儿国街道，白天」这种行属于**场景头**（地点/时间），不是某个镜头的内容。
_PLACE_KEYS = ("地点", "取景", "场景地点", "所在地", "时间", "时段")
_TIME_WORDS = {
    "白天": "日", "日": "日", "上午": "日", "中午": "日", "下午": "日", "白天外景": "日",
    "黄昏": "黄昏", "傍晚": "黄昏", "夜": "夜", "夜晚": "夜", "深夜": "夜", "凌晨": "夜",
    "清晨": "清晨", "早上": "清晨", "晚上": "夜",
}
#: 这些键**不进生成**：成片画面里的字全部由 m7 用剧本台词逐字烧录，模型再画一遍就是两层
#: 字幕（这条 bug 修过两次，见 core.prompts.NO_ONSCREEN_TEXT）。丢掉不是丢内容，是丢一条
#: 本来就不可能生效的指令 —— 说明清单会把原文回显给人，并告诉 ta 正确做法。
DROP_KEYS = {"字幕", "花字", "屏幕文字", "画面文字", "字卡", "标题字", "水印", "字幕卡",
             "备注", "注意", "提示", "说明"}
#: 整片设定（`**时长：** 90 秒`）→ 并进片头行，不作为某个镜头的内容
META_KEYS = {
    "时长": "dur", "片长": "dur", "总时长": "dur", "目标时长": "dur",
    "画幅": "aspect", "比例": "aspect", "尺寸": "aspect",
    "风格": "style", "类型": "style", "题材": "style", "基调": "style",
}
CANON_KEYS = ("在场", "画面", "运镜", "标题", "概要", "外貌", "音色")
#: 解析器对同名字段是**覆盖**语义（后写的赢），所以每个块里这类字段只能有一条：
#: 多写的一律并进上一条，否则用户写的第二段画面会被第一段静默吃掉。
#: 台词不在其列 —— 台词一行一句，绝不能合并。
SINGLE_KEYS = ("画面", "运镜", "标题", "概要", "外貌", "音色")


def labels_of(script: object) -> dict[str, str]:
    """角色名/别名 → character_id（含 id→id 自己），供"说话人写的是中文名"时换回 id。

    台词必须挂在 id 上：m5 按 id 选音色、m6 按 id 对口型、m9 按 id 度量身份。
    而人在稿子里写「猪八戒」是天经地义的 —— 看板显示的就是中文名。所以换 id 是这一层的活，
    不能反过来要求用户背 `zhu_bajie`。
    """
    out: dict[str, str] = {}
    for c in getattr(script, "characters", None) or []:
        cid = str(getattr(c, "character_id", "") or "").strip()
        if not cid:
            continue
        out[cid] = cid
        for label in (getattr(c, "name", "") or "", cid.replace("_", "")):
            lab = str(label).strip()
            if lab and lab not in out:
                out[lab] = cid
    return out


# ─────────────────────────── 小工具 ───────────────────────────
def unmark(line: str) -> str:
    """去掉 markdown 强调与列表符号，**只动符号不动字**（`**唐僧**：` → `唐僧：`）。"""
    s = line.replace("﻿", "").replace("​", "").replace("\t", "  ")
    s = re.sub(r"^\s*[-*+•]\s+", "", s)   # 无序列表
    s = re.sub(r"^\s*>\s?", "", s)        # 引用块
    return s.replace("**", "").replace("__", "").replace("~~", "").rstrip()


def _dur(text: str) -> tuple[float, str] | None:
    """(秒数, 原文片段) —— 原文片段要带回去，因为它得从镜头里**删掉**才知道剩什么是运镜。"""
    m = _DUR_RE.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return (v, m.group(0)) if 0 < v <= 600 else None


def _core(head: str) -> str:
    """标题行去掉 `#`、空白、尾部冒号后的正文。"""
    return head.lstrip("#").strip().strip("：:").strip()


def _scene_no(core: str) -> int | None:
    m = _SCENE_RE.match(core)
    if not m:
        return None
    if m.group("n"):
        return int(m.group("n"))
    tok = (m.group("cn") or "").strip()
    if tok.isdigit():
        return int(tok)
    return _CN_NUM.get(tok)


def _shot_no(core: str) -> int | None:
    m = _SHOT_RE.match(core)
    if not m:
        return None
    for g in ("n", "n2"):
        if m.group(g):
            return int(m.group(g))
    return None


def _is_shot_core(core: str) -> bool:
    """认"这行是个镜头标记"：带 shot 型 id，或 `镜头N`/`第N个镜头`/`结尾镜头`+时长。"""
    m = _SHOT_RE.match(core)
    if not m:
        return False
    if m.group("id") or m.group("n") or m.group("n2"):
        return True
    return _dur(core) is not None


def _tokens(text: str) -> list[str]:
    """切 token：分隔符与空白都算分隔（`·` 也算，它只在片头/场景头里是字段分隔）。"""
    return [p for p in re.split(r"[｜|/,,;；·\s]+", text or "") if p.strip()]


def _shot_values_canonical(body: str) -> bool:
    """`### id 类型 Ns · 转场` 里两个取值都必须是**标准中文名**，否则要走整理分支重写。

    不检查这一步，`### shot_2_1 动作 2s · 定格` 会被当成"已经标准"原样放行 —— 形状是对的，
    值却是引擎不认识的，最后变成一条用户看不懂的报错（别称必须在整理时就被翻译掉，
    顺便写进 notes，让人知道定格被记成了切）。
    """
    m = re.match(r"^###\s+\S+\s+(?P<typ>\S+)\s+[\d.]+s\s*·\s*(?P<trn>\S+)$", body)
    if not m:
        return False
    return m.group("typ") in SHOT_TYPE_CN.values() and m.group("trn") in TRANSITION_CN.values()


def _enum_at(text: str, mapping: dict[str, str], aliases: dict[str, str]) -> tuple[str, str] | None:
    """从一段文字里认枚举值 → `(中文值名, 命中的原文 token)`。

    三级：整体匹配 → 逐 token → 别称包含（`特效转场`、`匹配剪辑转场` 这类表外长串）。
    返回命中的**原文**是因为调用方要把这个 token 从镜头里删掉。
    """
    def hit(tok: str) -> tuple[str, str] | None:
        t = tok.strip().strip("·—-#* ")
        if not t:
            return None
        if t in mapping.values():
            return t, t
        if t in mapping:
            return mapping[t], t
        val = aliases.get(t) or aliases.get(t.lower())
        if val and val in mapping:
            return mapping[val], t
        return None

    got = hit(text)
    if got:
        return got
    parts = _tokens(text)
    for p in parts:
        got = hit(p)
        if got:
            return got
    for p in parts:
        for alias, val in aliases.items():
            if len(alias) >= 2 and alias in p and val in mapping:
                return mapping[val], p
    return None


def _shot_id(core: str, scene_no: int, idx: int) -> tuple[str, str]:
    """(镜号, 命中的标记原文)。原文用于从镜头里删掉自己，避免被误当成运镜。"""
    m = _SHOT_RE.match(core)
    marker = m.group(0) if m else ""
    if m and m.group("id"):
        return m.group("id"), marker
    n = _shot_no(core) or idx
    sid = f"shot_{scene_no}_{n}" if scene_no else f"shot_{n}"
    return sid, marker


def _leftover_camera(core: str, *, drop: list[str], typ: str, trans: str) -> str:
    """镜头标记里既不是编号/时长/类型/转场的剩余段 → 镜头语言（`中景跟拍`）。

    剩余段**全部**保留（拼成一条「运镜:」）：整理器的底线是不丢字 —— 判不准它算画面还是
    算运镜，都比删掉强，用户看得见、改得动。
    """
    rest = core
    for piece in drop:
        if piece:
            rest = rest.replace(piece, " ", 1)
    for tok in (typ, trans):
        if tok:
            rest = re.sub(rf"(?<![\w]){re.escape(tok)}(?![\w])", " ", rest, count=1)
    keep = []
    for p in _tokens(rest):
        if p in ("·", "：", ":", "-", "—", "#", "*"):
            continue
        if _dur(p) or _shot_no(p) or _scene_no(p):
            continue
        if len(p) <= 30:
            keep.append(p)
    return "，".join(keep)


# ─────────────────────────── 主入口 ───────────────────────────
def normalize_draft(
    text: str, *, labels: dict[str, str] | None = None, title: str = ""
) -> tuple[str, list[str]]:
    """任意剧本文本 → `(标准剧本稿, 给人的说明)`。规则见模块 docstring。"""
    labels = dict(labels or {})
    notes: list[str] = []
    out: list[str] = []
    scene_no = 0
    shot_idx = 0
    in_roster = False
    new_chars = 0
    last_at: dict[str, int] = {}     # 当前块里的单值字段 → out 下标（多段并进同一条）
    scene_head_at: int | None = None  # 当前场景头在 out 里的下标（「地点：」行要改写它）
    seen_sids: set[str] = set()      # 已用过的镜号（重复镜号会被解析器判成"同场重复镜头"）
    meta: dict[str, str] = {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def note(no: int, msg: str) -> None:
        notes.append(f"第 {no} 行{msg}")

    def open_block() -> None:
        nonlocal scene_head_at
        last_at.clear()
        scene_head_at = None

    def emit_keyed(key: str, val: str, no: int, *, indented: bool) -> None:
        if not val:
            out.append("")
            return
        j = last_at.get(key)
        if key in SINGLE_KEYS and j is not None and out[j].strip():
            out[j] = f"{out[j].rstrip()} {val}"
            out.append("")                        # 行号守恒：只清空，不删行
            note(no, f"的「{key}」与上一行同属一块，已并成一条"
                     f"（剧本稿每块只留一条{key}，留两条解析时会被覆盖掉）")
            return
        pad = "  " if (indented or key in ("外貌", "音色")) else ""
        out.append(f"{pad}{key}: {val}")
        last_at[key] = len(out) - 1

    # 第一遍：先抠整片设定。`**时长：** 90 秒` 常写在片头**之后**，只靠后处理补不回来。
    for raw in lines:
        kv = _KV_RE.match(unmark(raw).strip())
        if not kv:
            continue
        lane = META_KEYS.get(re.sub(r"\s", "", kv.group("key")))
        if lane and lane not in meta:
            meta[lane] = kv.group("val").strip()
    head_dur = _dur(meta.get("dur", ""))
    head_aspect = next((a for a in ASPECTS if a in meta.get("aspect", "").replace("：", ":")), "")
    head_style = (meta.get("style") or "").strip()

    for no, raw in enumerate(lines, start=1):
        body = unmark(raw).strip()
        if not body:
            out.append("")
            continue
        if re.fullmatch(r"[-=—_*~·\s]+", body):
            note(no, "的分隔线（----）是排版，不参与生成，已去掉")
            out.append("")
            continue
        if body.startswith("```"):
            note(no, "的代码块围栏已去掉（剧本稿不写代码块）")
            out.append("")
            continue
        # 已经是标准写法、且取值也在取值表里的行：一字不动（幂等性的来源）
        if _CANON_SCENE.match(body):
            num = re.search(r"^##\s+(?:scene_|s)(\d+)", body)
            scene_no, shot_idx = (int(num.group(1)) if num else scene_no), 0
            out.append(body)
            scene_head_at = len(out) - 1
            in_roster = False
            continue
        if _CANON_SHOT.match(body) and _shot_values_canonical(body):
            shot_idx += 1
            m_sid = re.match(r"^###\s+(\S+)", body)
            if m_sid:
                seen_sids.add(m_sid.group(1))
            open_block()
            out.append(body)
            in_roster = False
            continue

        # ── 片头行：《片名》 · 风格 · 90s · 9:16 ──
        if body.startswith("《") or re.match(r"^#+\s*《", body):
            core = _core(body)
            m = re.match(r"^《([^》]*)》\s*(?P<rest>.*)$", core)
            name = (m.group(1).strip() if m else "") or title
            seg = _segments((m.group("rest") if m else core).strip())
            style = next((s for s in seg if not _dur(s) and s not in ASPECTS), head_style)
            dur = next((d for d in (_dur(s) for s in seg) if d), head_dur)
            aspect = next((s for s in seg if s in ASPECTS), head_aspect)
            pieces = [p for p in (style, f"{int(dur[0])}s" if dur else "", aspect) if p]
            rebuilt = f"《{name}》{SEP}{SEP.join(pieces)}" if pieces else f"《{name}》"
            out.append(rebuilt)
            if rebuilt != body:
                note(no, "的片头行已整理成「《片名》 · 风格 · 时长s · 画幅」"
                         + ("（设定值从下面的时长/画幅/风格行取）" if meta else ""))
            in_roster = False
            continue

        # ── 整片设定行：值已并进片头 ──
        kv0 = _KV_RE.match(body)
        if kv0 and re.sub(r"\s", "", kv0.group("key")) in META_KEYS:
            note(no, f"的「{kv0.group('key')}」是整片设定，已并进片头行（不参与逐镜生成）")
            out.append("")
            continue

        core = _core(body)
        looks_head = raw.lstrip().startswith("#") or bool(
            re.match(r"^[一二三四五六七八九十]+\s*[、.．]", body)
        )
        # 镜头/场景标记也常写成不带 # 的粗体行（`**镜头 1｜3 秒｜中景跟拍**`）
        is_shot_line = _is_shot_core(core) and (looks_head or _dur(core) or len(core) <= 34)
        n_sc = _scene_no(core) if (looks_head or re.match(r"(?i)^(scene|场景)", core)) else None

        # ── 镜头行 ──
        if is_shot_line and not n_sc:
            given = _shot_no(core)
            # 标了编号就跟着编号走（`镜头 2` 之后再来一个没编号的，得是 3 而不是 2），
            # 否则按顺序 +1。镜号重复是致命的：解析器会报"镜头 X 在场 scene_Y 内重复"，
            # 而人看到的是两个不同写法造成的同一个号，根本联不起来。
            shot_idx = max(given, shot_idx + 1) if given else shot_idx + 1
            open_block()
            sid, marker = _shot_id(core, scene_no, shot_idx)
            if sid in seen_sids:
                sid = f"shot_{scene_no}_{shot_idx}" if scene_no else f"shot_{shot_idx}"
                while sid in seen_sids:
                    shot_idx += 1
                    sid = f"shot_{scene_no}_{shot_idx}" if scene_no else f"shot_{shot_idx}"
                note(no, f"的镜号重复了，已改用 {sid}（同一场里镜号必须唯一）")
            seen_sids.add(sid)
            d = _dur(core)
            t_typ = _enum_at(core, SHOT_TYPE_CN, SHOT_TYPE_ALIASES)
            t_trn = _enum_at(core, TRANSITION_CN, TRANSITION_ALIASES)
            typ = t_typ[0] if t_typ else "动作"
            trans = t_trn[0] if t_trn else "切"
            line = f"### {sid} {typ} {d[0] if d else 3:g}s{SEP}{trans}"
            out.append(line)
            in_roster = False
            cam = _leftover_camera(
                core, drop=[marker, d[1] if d else ""], typ=t_typ[1] if t_typ else "",
                trans=t_trn[1] if t_trn else "",
            )
            if cam:
                out.append(f"运镜: {cam}")
                last_at["运镜"] = len(out) - 1
            if line != body or cam:
                note(no, f"的镜头行已整理成「{line.strip()}」"
                        + ("（这一行没写时长，按 3s 记）" if not d else "")
                        + (f"；「{cam[:18]}」是镜头语言，另起「运镜:」" if cam else ""))
            if t_typ and t_typ[1] != typ:
                note(no, f"的镜头类型「{t_typ[1]}」按取值写成「{typ}」")
            if t_trn and t_trn[1] != trans:
                note(no, f"的转场「{t_trn[1]}」引擎里没有这种转场，按最接近的「{trans}」记"
                         f"（引擎只有 切/叠化/淡入淡出/空镜转场）")
            continue

        # ── 场景行 ──
        if n_sc:
            scene_no, shot_idx = n_sc, 0
            open_block()
            loc = re.sub(
                r"(?i)^scene[\s_\-]*\d*|^場?场景\s*[:：]?\s*\d*|^第\s*[一二三四五六七八九十\d]+\s*[场幕]",
                "", core,
            ).strip(" ·:：-—")
            # 剩余段只有**带分隔符**才当作地点（`## scene_1 · 街道 · 日`）；
            # 「Scene 1：八戒赖着不走」那种是标题 —— 硬塞进地点会把场景图提示词带歪。
            if loc and "·" in loc:
                out.append(f"## scene_{n_sc}{SEP}{loc.strip(' ·')}")
                note(no, f"的场景标记已整理成「## scene_{n_sc} …」（后端 id 不变，只换写法）")
            else:
                out.append(f"## scene_{n_sc}")
                scene_head_at = len(out) - 1
                if loc and n_sc != scene_no or loc:
                    out.append(f"标题: {loc}")
                    last_at["标题"] = len(out) - 1
                note(no, f"的场景标记已整理成「## scene_{n_sc}」（后端 id 不变）"
                        + (f"；「{loc[:16]}」是这一场的标题，另起「标题:」" if loc else ""))
            in_roster = False
            continue

        # ── 其他标题行 ──
        if looks_head:
            if re.search(r"角色|人物", core) and "镜头" not in core:
                out.append("# 角色")
                if body != "# 角色":
                    note(no, "的「角色设定」标题已统一成「# 角色」")
                in_roster = True
                open_block()
                continue
            note(no, f"的标题「{core[:16]}」是章节排版，不参与生成，已去掉")
            out.append("")
            continue

        kv = _KV_RE.match(body)
        key = re.sub(r"\s", "", kv.group("key")) if kv else ""
        val = kv.group("val").strip() if kv else ""

        # ── 「# 角色」段：`唐僧：青年俊美和尚…` 是**外貌**，不是台词 ──
        if in_roster and not raw[:1].isspace():
            # 先认标准角色行 `c1 小明 · male · adult`：原样放行，并把名字登记进查找表
            # （后面写「小明：台词」才换得出 id）。没有这一步，标准稿会被当成"不认识的角色"
            # 而重新编号 —— 幂等性就是这么丢的。
            if not kv and re.match(r"^[^\s]+\s+\S+", body):
                cid0 = body.split()[0]
                rest_segs = _segments(body[len(cid0):].strip())
                labels.setdefault(cid0, cid0)
                if rest_segs:
                    labels.setdefault(rest_segs[0], cid0)
                out.append(body)
                last_at.clear()
                continue
            who = key or re.split(r"[：:]", body)[0].strip()
            desc = val or (re.split(r"[：:]", body, 1)[1].strip() if re.search(r"[：:]", body) else "")
            if not desc:
                out.append(raw.rstrip())     # 没有描述可放的角色行：交回解析器点名
                continue
            cid = labels.get(who) or labels.get(who.replace(" ", ""))
            if not cid and who and len(who) <= 12 and not re.search(r"[。，.!?]", who):
                # 原剧本没有这个角色：登记成新角色（id 按出场顺序补 char_N）。
                # 不凭空造 id 的话，这份稿子会因为"名字对不上 id"而彻底保存不了；
                # 造 id 的代价是**多画一张定妆照**，这是新角色本就该花的钱。
                # 但 id 绝不能撞上剧本里已有的角色 —— 撞了就是把别人的脸认领过来。
                taken = set(labels.values())
                while True:
                    new_chars += 1
                    cid = f"char_{new_chars}"
                    if cid not in taken:
                        break
                labels[who] = cid
                note(no, f"的「{who}」不在原剧本角色表里，已登记为新角色 {cid}（会画一张新的定妆照）")
            if cid:
                if not any(str(o).strip().startswith(f"{cid} ") for o in out):
                    out.append(f"{cid} {who}")
                    last_at.clear()
                if desc:
                    emit_keyed("外貌", desc, no, indented=True)
                if body != out[-1]:
                    note(no, f"的角色「{who}」已写成标准角色行（id={cid}，外貌另起一行）")
                continue
            if not kv and re.match(r"^\S+\s+\S+", body):
                out.append(body)      # 已经是 `id 名字 · male · adult` 标准角色行
                last_at.clear()
                continue

        # ── 「地点：街道，白天」改写场景头 ──
        if key in _PLACE_KEYS and scene_head_at is not None and not shot_idx:
            bits = [b.strip(" 。.") for b in re.split(r"[，,、;；]", val) if b.strip(" 。.")]
            tod = next((_TIME_WORDS[b] for b in bits if b in _TIME_WORDS), "")
            # 地点只取**第一个**非时间片段：`地点：宫殿前广场，天空逐渐变暗` 的后半是画面信息，
            # 塞进 location 会直接进场景图提示词（`location` 是场景图的唯一地点来源）。
            loc_bits = [b for b in bits if b not in _TIME_WORDS]
            loc = loc_bits[0] if loc_bits else ""
            head = f"## scene_{scene_no}"
            if loc:
                head += f"{SEP}{loc}"
            if tod:
                head += f"{SEP}{_TIME_WORDS[tod]}"
            out[scene_head_at] = head
            out.append("")
            scene_head_at = None      # 一场只有一个头：第二个「地点：」不再改写它
            note(no, f"的「{key}」是场景信息，已并进场景头「{head}」")
            for extra in loc_bits[1:]:
                emit_keyed("概要", extra, no, indented=False)
            continue

        # ── 字段行 ──
        if kv:
            if key in DROP_KEYS:
                note(no, f"的「{key}」不进生成：成片画面里的字由 m7 用剧本台词**逐字**烧录，"
                        f"模型自己画字会叠成两层。原文「{val[:20]}」已丢弃 —— "
                        f"要这句话上屏，请把它写成某个角色的台词")
                out.append("")
                continue
            if key in KEY_RENAMES:
                emit_keyed(KEY_RENAMES[key], val, no, indented=_looks_indented(raw))
                note(no, f"的「{key}」不是标准字段名，已改写成「{KEY_RENAMES[key]}:」（内容原样）")
                continue
            if key in CANON_KEYS:
                if key == "在场" and val:
                    # 「在场」必须是 character_id 列表（模式选择、角色图挑选、m9 身份度量都按它
                    # 索引），而人写的是中文名：`在场: 猪八戒、唐僧` 一句都认不出来。
                    cast, moved = _map_cast(val, labels)
                    if moved:
                        note(no, f"的在场角色名已换成 id（{moved}）")
                    val = cast
                emit_keyed(key, val, no, indented=_looks_indented(raw))
                continue
            if key in _PLACE_KEYS:
                emit_keyed("概要", val, no, indented=_looks_indented(raw))
                note(no, f"的「{key}」已写成「概要:」（场景头里已经有地点了）")
                continue

        # ── 台词行 ──
        sp = _speaker_line(body, labels)
        if sp:
            cid, emo, spoken = sp
            out.append(f"  {cid}（{emo or 'neutral'}）: {spoken}")
            if raw[:1] != " " and raw[:1] != "\t":
                note(no, f"的台词已整理成标准写法（补两空格缩进，说话人写成 id「{cid}」）")
            elif emo != (emo or "neutral"):
                note(no, "的台词没写情绪，按「neutral」记")
            continue
        if raw[:1] in (" ", "\t"):
            out.append(unmark(raw))   # 已缩进的标准台词/字段行：只去 markdown 符号
            continue

        # ── 无标记散文：按位置归到画面或概要 ──
        if shot_idx:
            emit_keyed("画面", body, no, indented=False)
            note(no, "的这段散文没有字段名，按所在位置写成「画面:」")
            continue
        if scene_no:
            emit_keyed("概要", body, no, indented=False)
            note(no, "的这段散文在场景里、镜头之前，写成「概要:」")
            continue
        out.append(raw.rstrip())      # 交回解析器点名（不猜）

    return "\n".join(out), notes


def parse_draft(
    text: str, old: Script, *, title_hint: str = ""
) -> tuple[Script, list[str], list[str]]:
    """整理 + 解析的一份实现：`(Script, 说明, 问题)`。

    网页保存、按 prompt 改写落地、CLI 三条路都必须走这里 —— 分成三份实现就会漂，
    漂的结果是"网页能存、命令行报错"这类最难查的分歧。
    问题非空时返回的 Script 只是给人回显用，**不能落盘**。
    """
    from fastvideo3.core.script_text import parse_script_text

    norm, notes = normalize_draft(
        text, labels=labels_of(old), title=title_hint or str(getattr(old, "title", "") or "")
    )
    script, problems = parse_script_text(norm, old)
    return script, notes, problems


def _map_cast(val: str, labels: dict[str, str]) -> tuple[str, str]:
    """`在场` 里的中文角色名 → id。认不出的名字**原样留着**（交回 cross_check 点名，
    它会说"引用了不存在的角色 X"，比整理器瞎猜一个 id 更可查）。"""
    parts = [p.strip() for p in re.split(r"[、，,;；/]+", val) if p.strip()]
    ids: list[str] = []
    moved: list[str] = []
    for p in parts:
        cid = labels.get(p) or labels.get(p.replace(" ", ""))
        if cid and cid not in ids:
            if cid != p:
                moved.append(f"{p}→{cid}")
            ids.append(cid)
        elif p not in ids:
            ids.append(p)
    return ", ".join(ids), "、".join(moved)


def _looks_indented(raw: str) -> bool:
    return raw[:1] in (" ", "\t")


def flat(s: str) -> str:
    """只留汉字与字母数字：比较"这句话还在不在"时，标点/空格/全半角差异不该算丢内容。"""
    return "".join(ch for ch in s if ch.isalnum() or "一" <= ch <= "鿿")


def spoken_lines(text: str) -> list[str]:
    """稿子里**说出口的话**（逐字清单，任意排版都能抠出来）。

    用途是模型整理格式后的逐字校验：它只许动排版，一句台词都不许改词、吞句、加句。
    认形状而不是认语义：`  id（情绪）: 台词`、`**猪八戒：** 台词`、`唐僧：台词` 都算，
    而字段行（在场/画面/运镜/…）与整片设定行（时长/画幅/风格/…）排除掉。
    """
    stop = set(CANON_KEYS) | set(KEY_RENAMES) | set(META_KEYS) | set(DROP_KEYS) | set(_PLACE_KEYS)
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        s = unmark(raw).strip()
        if not s or s.startswith(("#", "《")):
            continue
        m = re.match(
            r"^(?P<who>[^\s:：()（）【】《》]{1,16})\s*(?:[（(](?P<emo>[^）)]{0,16})[）)])?\s*[:：]\s*(?P<text>.+)$",
            s,
        )
        if not m:
            continue
        who = m.group("who").strip()
        if re.sub(r"\s", "", who) in stop:
            continue
        said = m.group("text").strip()
        said = re.sub(r"^[「“\"'『]+|[」”\"'』]+$", "", said).strip()
        if len(flat(said)) < 2 or said in out:
            continue
        out.append(said)
    return out


def _speaker_line(s: str, labels: dict[str, str]) -> tuple[str, str, str] | None:
    """认出「说话人：台词」/「说话人（情绪）：台词」→ `(id, 情绪, 台词)`；说话人必须认得出。"""
    m = _SPK_RE.match(s.strip())
    if not m:
        return None
    who = m.group("who").strip()
    cid = labels.get(who) or labels.get(who.replace(" ", ""))
    if not cid:
        return None
    text = m.group("text").strip()
    text = re.sub(r"^[「“\"'『]+", "", text)
    text = re.sub(r"[」”\"'』]+$", "", text).strip()
    if not text:
        return None
    return cid, (m.group("emo") or "").strip(), text
