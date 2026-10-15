"""剧本 ⇄ 可读剧本文本：让人改"剧本本身"，而不是改 JSON。

为什么要有这一层：分步生成的核心闸门是人，而人面对 `script.json` 时只能改字段名、
括号和逗号——那不是"改剧本"，那是"改数据结构"。本模块把 Script 渲染成一份固定排版的
剧本稿（场景 / 镜头 / 在场 / 画面 / 台词 / 运镜），再把人编辑后的文本确定性地解析回
Script：零 LLM 调用、逐字保留人写的台词、解析失败按行号点名。

**故意不进文本的字段**（不是给人看的，丢了又会让一致性机制退化）：
- `Shot.visual_prompt`：m3 蒸馏回写的产物，人改了下一轮也会被覆盖；
- `Character.look` / `canonical_tokens`：m1 抽取的结构化外貌卡与必留词，是 m3 蒸馏校验
  与 m9 一致性度量的基准。
两者按 `shot_id` / `character_id` 从旧剧本继承（`parse_script_text(..., old=...)`）。
外貌描述被改动时，保存路径会重抽这些角色的外貌卡（`api._refresh_looks_for_changed_chars`），
所以继承不会留下"新描述配旧必留词"的错配。

`base_info.topic / culture_background / episode` 也从旧剧本继承：它们属于新建任务时的
创作设定，不在这份稿子里改。`style / aspect_ratio / target_duration_sec` 允许改——
前两者改动后 `progress.diff_invalidation` 已判为全片作废（`__all__`），语义正确。
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from fastvideo3.core.schema import (
    BaseInfo,
    Character,
    DialogueLine,
    Scene,
    Script,
    Shot,
    ShotType,
    TransitionType,
)

# 中文标签与看板表格同一套叫法（对应 ProductionBoard.vue 的 typeLabel/transLabel）。
# 解析时中英都接受：人会照着看板写，也可能照着 JSON 的枚举值写。
SHOT_TYPE_CN = {"dialogue": "对白", "action": "动作", "scenery": "空镜"}
TRANSITION_CN = {"cut": "切", "dissolve": "叠化", "fade": "淡入淡出", "scenery": "空镜转场"}

# 常见别称 → 枚举值。转场引擎只有上面那四种能力（实现见 m7_edit），但人和模型写稿子时
# 会写「硬切」「黑场」「特效转场」「定格」——这些词**不是错字，是同一个意思的另一种说法**，
# 拒绝它们的结果是一份剧情完全合法的稿子被 12 条报错挡在门外（run_20261015_220939 实测：
# 3 个不认识的值名连累出 9 条"字段不在可识别位置"）。就近映射到最接近的一种，并告知用户。
TRANSITION_ALIASES: dict[str, str] = {
    "硬切": "cut", "直切": "cut", "跳切": "cut", "切镜": "cut", "无转场": "cut",
    "匹配剪辑": "cut", "甩镜": "cut", "定格": "cut", "freeze": "cut",
    "溶解": "dissolve", "交叉溶解": "dissolve", "缓入缓出": "dissolve", "特效转场": "dissolve",
    "特效": "dissolve", "闪白": "dissolve", "划像": "dissolve", "wipe": "dissolve",
    "淡出": "fade", "淡入": "fade", "渐隐": "fade", "渐显": "fade", "黑场": "fade",
    "黑屏": "fade", "白场": "fade",
    "空镜": "scenery", "风景转场": "scenery", "过场": "scenery", "空镜头": "scenery",
}
SHOT_TYPE_ALIASES: dict[str, str] = {
    "对话": "dialogue", "台词": "dialogue", "口播": "dialogue", "说话": "dialogue",
    "对白镜头": "dialogue", "talk": "dialogue",
    "行动": "action", "表演": "action", "情节": "action", "动作镜头": "action",
    "空镜头": "scenery", "风景": "scenery", "大远景": "scenery", "环境": "scenery",
    "转场": "scenery",
}

SEP = " · "
ASPECTS = ("9:16", "16:9", "1:1")
DURATION_TOL = 0.15  # PRD F6 验收①：成片时长与目标偏差 ≤15%

# title 用**贪婪**匹配到最后一个 》：模型经常把书名号写进 title 字段本身
# （run_20261012_184649 的 title 就是「《唐僧动凡心吗》」），渲染时再包一层就成了
# 《《…》》 —— 非贪婪正则会把它切成 title=「《唐僧动凡心吗」+ rest=「》 · 风格 · 180s · 9:16」，
# 于是"风格"被当成时长、时长被当成画幅，整份稿子判为格式错误（编辑与改写都保存不了）。
_HEADER_RE = re.compile(r"^《(?P<title>.*)》\s*(?P<rest>.*)$")
_CHARACTER_RE = re.compile(r"^(?P<id>[^\s·:：]+)\s+(?P<rest>.*)$")
_SCENE_RE = re.compile(r"^##\s+(?P<id>\S+)(?P<rest>.*)$")
_SHOT_RE = re.compile(
    r"^###\s+(?P<id>\S+)\s+(?P<type>\S+)\s+(?P<dur>[\d.]+)s\s*·\s*(?P<trans>\S+)\s*$"
)
_FIELD_RE = re.compile(r"^(?P<key>[^:：]{1,8})[:：]\s*(?P<val>.*)$")
_LINE_RE = re.compile(
    r"^\s{2,}(?P<who>[^\s:：()（）]+)\s*[（(](?P<emo>[^）)]*)[）)]\s*[:：]\s*(?P<text>.*)$"
)
_SEC_RE = re.compile(r"^([\d.]+)s$")

_SCENE_KEYS = ("标题", "概要")
_SHOT_KEYS = ("在场", "画面", "运镜")
_CHAR_KEYS = ("外貌", "音色")


def _sec(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{float(v):g}"


def _cn(mapping: dict[str, str], key: str) -> str:
    return mapping.get(key, key)


def _segments(rest: str) -> list[str]:
    """按分隔符切段。只认渲染时用的精确「 · 」，不用单「·」——否则用户在被风格名里
    写个间隔号（"都市·轻喜剧"）就会被切成两段。行首那个「·」是正则吃掉《片名》后
    留下的前导符，必须先剥掉。"""
    rest = re.sub(r"^\s*·\s*", "", rest)
    return [p.strip() for p in rest.split(SEP) if p.strip()]


def _enum_of(token: str, mapping: dict[str, str], aliases: dict[str, str]) -> str | None:
    """值名 → 枚举值（英文枚举 / 中文名 / 别称 三层）；认不出返回 None。"""
    t = (token or "").strip()
    if not t:
        return None
    if t in mapping:
        return t
    inverted = {v: k for k, v in mapping.items()}
    if t in inverted:
        return inverted[t]
    return aliases.get(t)


def _as_enum(
    raw: str,
    mapping: dict[str, str],
    aliases: dict[str, str] | None = None,
    *,
    fallback: str | None = None,
) -> str:
    """值名 → 枚举值：英文枚举、中文名、别称三层都认。

    `fallback` 给"这行整体还有救"的场景（镜头行的类型/转场写错，但镜号与时长是好的）：
    认不出时取一个安全默认值而不是抛错 —— 抛错会让整条 `###` 行作废，紧跟其后的
    在场/画面/运镜 三行于是全部变成孤儿行，一个错字报出四条错。
    """
    token = raw.strip()
    hit = _enum_of(token, mapping, aliases or {})
    if hit:
        return hit
    if fallback is not None:
        return fallback
    raise ValueError(f"无法识别「{token}」，可写 {'/'.join(mapping.values())} 或 {'/'.join(mapping)}")


def render_script_text(script: Script) -> str:
    """Script → 剧本稿。数组顺序即文本顺序，解析回来按同一顺序重建。"""
    bi = script.base_info
    # 渲染前剥掉 title 自带的书名号：模型经常把《》写进 title 字段，不剥就会渲染成
    # 《《…》》，解析时字段全错位（时长被当成风格、画幅被当成时长）→ 整份稿子保存不了。
    # 这一层与 _HEADER_RE 的贪婪匹配互为保险：一个保证渲染幂等，一个保证手写稿也能读。
    title = (script.title or bi.topic).strip().strip("《》").strip()
    out: list[str] = [
        f"《{title}》{SEP}{bi.style}{SEP}"
        f"{_sec(bi.target_duration_sec)}s{SEP}{bi.aspect_ratio}",
        "",
        "# 角色",
    ]
    if not script.characters:
        out.append("（无角色）")
    for c in script.characters:
        head = [c.character_id, c.name]
        line = " ".join(p for p in head if p)
        meta = [p for p in (c.gender, c.age_band) if p]
        if meta:
            line += SEP + SEP.join(meta)
        out.append(line)
        if c.appearance:
            out.append(f"  外貌: {c.appearance}")
        if c.voice_id:
            out.append(f"  音色: {c.voice_id}")

    for sc in script.scenes:
        out.append("")
        head = [p for p in (sc.location, sc.time_of_day) if p]
        out.append(f"## {sc.scene_id}" + (SEP + SEP.join(head) if head else ""))
        if sc.title:
            out.append(f"标题: {sc.title}")
        if sc.summary:
            out.append(f"概要: {sc.summary}")
        for sh in sc.shots:
            out.append("")
            out.append(
                f"### {sh.shot_id} {_cn(SHOT_TYPE_CN, sh.type.value)} "
                f"{_sec(sh.duration_sec)}s{SEP}{_cn(TRANSITION_CN, sh.transition_out.value)}"
            )
            if sh.characters:
                out.append("在场: " + ", ".join(sh.characters))
            out.append(f"画面: {sh.action}")
            if sh.camera:
                out.append(f"运镜: {sh.camera}")
            for d in sh.dialogues:
                out.append(f"  {d.character_id}（{d.emotion}）: {d.text}")
    return "\n".join(out).rstrip() + "\n"


def parse_script_text(text: str, old: Script) -> tuple[Script, list[str]]:
    """剧本稿 → `(Script, problems)`。

    `problems` 非空时 Script 也会尽量构造出来（方便调用方回显给人看），但**不能落盘**：
    这里只保证"格式认得"，语义正确性另有两道关——`api._script_cross_check`（角色引用
    存在、至少一场一镜）与 pydantic（台词 ≤60 字硬上限）。三层都过才算合法保存。

    `old` 提供 `base_info`（topic 等设定）与各 id 的不可见字段继承；没有旧剧本时不该
    调这里——文本稿不携带 base_info 的全部信息，构造不出可信的设定向量。
    """
    problems: list[str] = []
    old_chars = {c.character_id: c for c in old.characters}
    old_shots = {sh.shot_id: sh for sh in old.all_shots()}
    old_scenes = {s.scene_id: s for s in old.scenes}

    base_info = old.base_info.model_copy(deep=True)
    title = old.title
    characters: list[Character] = []
    scenes: list[Scene] = []
    scene: Scene | None = None
    shot: Shot | None = None
    char: Character | None = None

    def close_shot() -> None:
        nonlocal shot
        if shot is None:
            return
        if not shot.characters:
            # 「在场」空着但台词里有人说话 → 说话人就是在场的人。
            # 粘来的稿子普遍不写在场（人觉得画面里谁说话自然就谁在），而参考生视频的
            # 模式选择与角色图挑选看的正是 `characters`：不补就等于这一镜一张角色图都不带，
            # 整片的脸会一镜一个样（m9 度量到的正是这个）。
            cast: list[str] = []
            for d in shot.dialogues:
                if d.character_id and d.character_id not in cast:
                    cast.append(d.character_id)
            shot.characters = cast
        if scene is not None:
            if any(s.shot_id == shot.shot_id for s in scene.shots):
                problems.append(f"镜头 {shot.shot_id} 在场 {scene.scene_id} 内重复")
            else:
                scene.shots.append(shot)
        shot = None

    def close_scene() -> None:
        nonlocal scene
        close_shot()
        if scene is None:
            return
        if any(s.scene_id == scene.scene_id for s in scenes):
            problems.append(f"场景 {scene.scene_id} 重复")
        else:
            scenes.append(scene)
        scene = None

    for no, raw in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
        line = raw.rstrip()
        if not line.strip() or line.strip() == "（无角色）":
            continue

        if line.startswith("# 角色"):
            close_scene()
            char = None
            continue

        if line.startswith("###"):
            m = _SHOT_RE.match(line)
            if not m:
                problems.append(
                    f"第 {no} 行镜头行应为「### 镜号 类型 时长s · 转场」：{line.strip()[:40]}"
                )
                continue
            close_shot()
            char = None
            if scene is None:
                problems.append(f"第 {no} 行镜头不在任何场景内（缺「## 场景」行）")
                continue
            try:
                # 值名认不出来时**取默认值继续**，而不是放弃整条镜头行：放弃的话，紧跟在
                # 后面的 在场/画面/运镜 会全部变成孤儿行 —— 实测一个「特效转场」报出四条错，
                # 人在满屏报错里再也找不到真正那一个词（run_20261015_220939）。
                shot_type = _as_enum(
                    m["type"], SHOT_TYPE_CN, SHOT_TYPE_ALIASES, fallback="action"
                )
                trans = _as_enum(
                    m["trans"], TRANSITION_CN, TRANSITION_ALIASES, fallback="cut"
                )
                dur = float(m["dur"])
            except ValueError as e:
                problems.append(f"第 {no} 行：{e}")
                continue
            # 别称（`特效转场`→叠化、`定格`→切）**算认识**，不报错：把它们翻译成人话是
            # `draft_norm.normalize_draft` 的活（写进 notes 给人看）。这里只在连别称都对
            # 不上时才点名 —— 而且镜头行照样开起来，后面的字段不会再变成孤儿行。
            for label, token, table, enums, aliases in (
                ("镜头类型", m["type"], SHOT_TYPE_CN, shot_type, SHOT_TYPE_ALIASES),
                ("转场", m["trans"], TRANSITION_CN, trans, TRANSITION_ALIASES),
            ):
                t = str(token).strip()
                if _enum_of(t, table, aliases) is None:
                    problems.append(
                        f"第 {no} 行{label}「{t}」不认识，已按「{table[enums]}」记，"
                        f"可写 {'/'.join(table.values())}"
                    )
            if dur <= 0:
                problems.append(f"第 {no} 行镜头时长必须 > 0：{m['dur']}")
                dur = 5.0
            prev_shot = old_shots.get(m["id"])
            shot = Shot(
                shot_id=m["id"],
                scene_id=scene.scene_id,
                type=ShotType(shot_type),
                duration_sec=dur,
                action="",
                camera=prev_shot.camera if prev_shot else "",
                transition_out=TransitionType(trans),
                visual_prompt=prev_shot.visual_prompt if prev_shot else "",
            )
            continue

        if line.startswith("##"):
            m = _SCENE_RE.match(line)
            if not m:
                problems.append(f"第 {no} 行场景行应为「## 场号 地点 · 时间」：{line.strip()[:40]}")
                continue
            close_scene()
            char = None
            head = _segments(m["rest"])
            prev_scene = old_scenes.get(m["id"])
            scene = Scene(
                scene_id=m["id"],
                title=prev_scene.title if prev_scene else "",
                location=head[0] if head else (prev_scene.location if prev_scene else ""),
                time_of_day=(
                    head[1] if len(head) > 1 else (prev_scene.time_of_day if prev_scene else "日")
                ),
                summary=prev_scene.summary if prev_scene else "",
            )
            continue

        if line.startswith("《"):
            close_scene()
            char = None
            m = _HEADER_RE.match(line)
            if not m:
                problems.append(f"第 {no} 行片头应为「《片名》  风格 · 时长s · 画幅」：{line.strip()[:40]}")
                continue
            # 剥掉书名号：《片名》 的括号是**渲染格式**，不属于 title 本身。
            # 不剥就会一轮一轮累积成《《《…》》》（每次改写都再包一层）。
            title = m["title"].strip().strip("《》").strip()
            seg = _segments(m["rest"])
            if len(seg) >= 1:
                base_info = base_info.model_copy(update={"style": seg[0]})
            if len(seg) >= 2:
                mm = _SEC_RE.match(seg[1])
                if not mm:
                    problems.append(f"第 {no} 行时长要写成「30s」：{seg[1]}")
                else:
                    base_info = base_info.model_copy(update={"target_duration_sec": int(float(mm[1]))})
            if len(seg) >= 3:
                if seg[2] not in ASPECTS:
                    problems.append(f"第 {no} 行画幅只能是 {'/'.join(ASPECTS)}：{seg[2]}")
                else:
                    base_info = base_info.model_copy(update={"aspect_ratio": seg[2]})
            continue

        # ── 缩进行：角色属性 或 台词 ──
        if line[:1].isspace():
            body = line.strip()
            fm = _FIELD_RE.match(body)
            key = fm["key"].strip() if fm else ""
            val = fm["val"].strip() if fm else ""
            if char is not None and key in _CHAR_KEYS:
                if key == "外貌":
                    char.appearance = val
                else:
                    char.voice_id = val
                continue
            if shot is not None:
                dm = _LINE_RE.match(line)
                if dm:
                    text_line = dm["text"].strip()
                    if not text_line:
                        problems.append(f"第 {no} 行台词是空句")
                    else:
                        try:
                            shot.dialogues.append(
                                DialogueLine(
                                    character_id=dm["who"].strip(),
                                    text=text_line,
                                    emotion=(dm["emo"] or "").strip() or "neutral",
                                )
                            )
                        except ValidationError as e:
                            # schema 硬上限（台词 ≤60 字）也必须带行号，否则人在 200 行的
                            # 稿子里找不到是哪一句坏了。
                            problems.append(
                                f"第 {no} 行台词不合法（{len(text_line)} 字，上限 60）：{text_line[:24]}…"
                            )
                            for err in e.errors():
                                if err.get("loc", ())[-1] != "text":
                                    problems.append(f"第 {no} 行台词不合法：{err.get('msg')}")
                    continue
                if key in _SHOT_KEYS or key in _SCENE_KEYS:
                    problems.append(
                        f"第 {no} 行台词应写成「  角色id（情绪）: 内容」，不是「{key}:」：{body[:40]}"
                    )
                    continue
            problems.append(f"第 {no} 行无法解析（缩进行只能挂在角色外貌/镜头台词下）：{body[:40]}")
            continue

        # ── 顶格行：字段 或 角色行 ──
        fm = _FIELD_RE.match(line)
        key = fm["key"].strip() if fm else ""
        val = fm["val"].strip() if fm else ""
        if fm and key in _SHOT_KEYS and shot is not None:
            if key == "在场":
                shot.characters = [p.strip() for p in re.split(r"[,，、]", val) if p.strip()]
            elif key == "画面":
                shot.action = val
            else:
                shot.camera = val
            continue
        if fm and key in _SCENE_KEYS and scene is not None:
            if shot is not None:
                problems.append(f"第 {no} 行「{key}」出现在镜头块之后，它属于场景；镜头要另起「###」行")
                continue
            if key == "标题":
                scene.title = val
            else:
                scene.summary = val
            continue
        if fm:
            problems.append(f"第 {no} 行字段「{key}」不在可识别位置（在场/画面/运镜/标题/概要/外貌/音色）")
            continue
        m = _CHARACTER_RE.match(line)
        if m:
            close_scene()
            seg = _segments(m["rest"])
            name = seg[0] if seg else ""
            meta = seg[1:]
            prev_char = old_chars.get(m["id"])
            char = Character(
                character_id=m["id"],
                name=name,
                appearance=prev_char.appearance if prev_char else "",
                voice_id=prev_char.voice_id if prev_char else "",
                gender=meta[0] if len(meta) > 0 else (prev_char.gender if prev_char else ""),
                age_band=meta[1] if len(meta) > 1 else (prev_char.age_band if prev_char else ""),
                look=(
                    prev_char.look.model_copy(deep=True)
                    if prev_char and prev_char.look
                    else None
                ),
                canonical_tokens=list(prev_char.canonical_tokens) if prev_char else [],
            )
            characters.append(char)
            continue
        problems.append(f"第 {no} 行无法解析：{line.strip()[:40]}")

    close_scene()

    seen: set[str] = set()
    for c in characters:
        if c.character_id in seen:
            problems.append(f"角色 {c.character_id} 重复定义")
        seen.add(c.character_id)
        if not c.name:
            problems.append(f"角色 {c.character_id} 缺名字")

    script = Script(
        base_info=base_info if base_info else BaseInfo(topic=title),
        title=title,
        characters=characters,
        scenes=scenes,
    )
    return script, problems


_QUOTE_PAIRS = (("「", "」"), ("“", "”"), ('"', '"'), ("『", "』"), ("‘", "’"), ("'", "'"))


def quoted_spans(feedback: str) -> list[str]:
    """从人写的修改意见里抠出「必须逐字出现」的引号内容。

    为什么需要它：改写是否落地不能问模型（它会说"已按要求修改"），也不能只看版本号 ——
    run_20261012_184649 实测：用户两次要求加一句台词，模型只把 `version` 从 3 推到 4、
    顺手清空了 `title`，那句台词一个字没加，而界面回报"✔ 已按意见重写"。
    引号里的原文是最硬的判据：它要么出现在台词里，要么没有。
    """
    out: list[str] = []
    for open_q, close_q in _QUOTE_PAIRS:
        rest = feedback
        while True:
            i = rest.find(open_q)
            if i < 0:
                break
            j = rest.find(close_q, i + len(open_q))
            if j < 0:
                break
            span = rest[i + len(open_q) : j].strip()
            if len(span) >= 4 and span not in out:
                out.append(span)
            rest = rest[j + len(close_q) :]
    return out


def dialogue_texts(script: Script) -> list[str]:
    """全片台词文本（含说话人），供"引号原文有没有落地"的比对。"""
    return [
        f"{d.character_id}|{d.text}" for sh in script.all_shots() for d in sh.dialogues
    ]


def script_signature(script: Script) -> dict:
    """剧本的**语义指纹**：只含"改动会花钱"的字段，不含 version 这类记账字段。

    用它判断一次改写是不是空转 —— 只动了 version / 语序 / 无关字段，指纹不变。
    """
    return {
        "title": script.title,
        "chars": [(c.character_id, c.name, c.appearance) for c in script.characters],
        "shots": [
            (
                sh.shot_id,
                sh.type.value,
                tuple(sh.characters),
                tuple((d.character_id, d.text, d.emotion) for d in sh.dialogues),
                sh.action,
                sh.camera,
                float(sh.duration_sec or 0),
            )
            for sh in script.all_shots()
        ],
    }


def revise_landed(old: Script, new: Script, feedback: str) -> tuple[bool, list[str]]:
    """改写是否**真的落地**。返回 (是否落地, 未落地的具体原因)。

    两条确定性判据（都不再问模型）：
    1. 语义指纹没变 = 空转（模型只是把剧本重新序列化了一遍）；
    2. 意见里引号内的原文没有出现在任何台词里 = 点名的那句没加进去。

    为什么必须挡在这里：改写"成功"却什么都没改，是最贵的一种静默失效 —— 用户会以为
    已经改好了，继续点「开始生成」，然后花钱出一条不含那句台词的成片。
    """
    reasons: list[str] = []
    if script_signature(old) == script_signature(new):
        reasons.append("剧本内容一处都没变（只有版本号在动）—— 模型把改写当成了重新解析")
    missing = []
    for span in quoted_spans(feedback):
        joined = " ".join(dialogue_texts(new))
        # 逐字优先；退一步允许标点/空格差异（模型常把"，"写成","）
        flat = "".join(ch for ch in joined if ch.isalnum() or "一" <= ch <= "鿿")
        want = "".join(ch for ch in span if ch.isalnum() or "一" <= ch <= "鿿")
        if span not in joined and (not want or want not in flat):
            missing.append(span)
    if missing:
        reasons.append(
            "意见里引号内的台词没有出现在剧本中："
            + "、".join(f"「{m}」" for m in missing[:3])
        )
    return (not reasons), reasons


def inherit_hidden_anchors(old: Script | None, new: Script, *, strict: bool = False) -> Script:
    """把文本层不携带的**一致性锚点**按 id 从旧剧本搬回新剧本（就地改 `new` 并返回）。

    为什么必须有这一份：`render_script_text` 刻意不输出 `shot.visual_prompt`（m3 蒸馏后
    回写的）、`character.look` 与 `character.canonical_tokens`（度量基准与必留词），
    人是靠剧本稿改剧情的，不该看到这些工程字段。人工编辑走 `parse_script_text(text, old)`
    时由解析器负责继承；而**「按 prompt 改写」是模型直接返回 JSON**，它只看得见剧本稿，
    这些字段必然回来是空的。不搬回来会有两个后果：

    ① m3 每次重跑都要重新蒸馏一遍（白花钱），m9 的度量基准换了一套词；
    ② 更贵的是 `diff_invalidation` 会看见"5 个角色的 look 都变了" → 定妆照 + 全部
       首帧/片段/口型一起作废 —— 改一句台词拖垮全片（2026-10-07 实测：v4/v5/v6 三次
       改写各重抽 5 张卡、作废 12-16 项）。

    `strict=True`（改写模式专用）：模型**根本看不到**这些字段，它给的任何值都是编的，
    所以旧剧本里有的角色一律以旧值为准；新增角色的 look/tokens 直接清空，交给
    `extract_looks` 去真抽（否则编出来的卡会被"已有卡就别重抽"当成真的）。
    `strict=False`（兜底继承）：只在新值缺失时补旧值，不覆盖模型/人写的东西。
    """
    if old is None:
        return new
    old_chars = {c.character_id: c for c in old.characters}
    for c in new.characters:
        oc = old_chars.get(c.character_id)
        if oc is None:  # 新角色：没有旧锚点可继承
            if strict:
                c.look = None
                c.canonical_tokens = []
            continue
        if strict or c.look is None:
            c.look = oc.look.model_copy(deep=True) if oc.look is not None else c.look
        if strict or not c.canonical_tokens:
            c.canonical_tokens = list(oc.canonical_tokens or [])
    old_shots = {sh.shot_id: sh for sh in old.all_shots()}
    for sh in new.all_shots():
        osh = old_shots.get(sh.shot_id)
        if osh is None:
            if strict:
                sh.visual_prompt = ""
            continue
        if strict or not sh.visual_prompt:
            sh.visual_prompt = osh.visual_prompt or sh.visual_prompt
    return new


def duration_warnings(script: Script) -> list[str]:
    """镜头时长合计 vs 目标偏差（PRD F6 验收①在剧本侧的前置检查）。

    值得在保存时算：偏差 >15% 时 m7 拼出来的成片必然不合格，而这个数在剧本落地那一刻
    就已经定了——不该等花钱合成完才发现。
    """
    target = float(script.base_info.target_duration_sec or 0)
    if target <= 0:
        return []
    total = sum(float(sh.duration_sec or 0) for sh in script.all_shots())
    if total <= 0:
        return ["全部镜头时长为 0，成片会是空片"]
    dev = (total - target) / target
    if abs(dev) <= DURATION_TOL:
        return []
    return [
        f"镜头合计 {total:.1f}s 与目标 {target:.0f}s 偏差 {dev * 100:+.0f}%（验收要求 ≤15%）："
        f"成片时长会明显不符，请调镜头时长或增减镜头"
    ]
