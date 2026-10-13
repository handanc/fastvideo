"""场景级生成单位：一次 wan3.0 调用覆盖哪些镜头（全项目唯一一份判据）。

为什么要这一层：一个场景被拆成 4-6 秒的独立镜头各自生成时，镜头之间没有任何共享的
运动状态 —— 上一镜人站在左边、下一镜凭空出现在右边，硬切看起来就是瞬移。电影里硬切
不难看，是因为覆盖镜头之间的走位是编排过的；AI 独立抽卡没有这个前提。加转场也救不
回来（转场只能遮住切点，遮不住空间不连续）。

wan3.0 的原生解法是**多镜头叙事**：一次调用里用「序号 + 时间戳 + 分镜内容」把多个
镜头串成连贯叙事，官方明确"在镜头间保持主体、场景、氛围一致"。于是生成单位从
"镜头"上移到"场景组"，本模块就是这个单位的定义与切分规则。

硬约束（都来自官方 API/提示词文档，不是实现偏好）：
- 单次时长 `duration ∈ [2, 30]` 秒（无视频输入时）；
- 每个分镜 **2–5 秒**，各镜头首尾相接、不跳段不重叠；
- `reference_image` ≤10 张、`reference_audio` ≤5 段且总时长 ≤15 秒；
- `first_frame`/`last_frame` 与 `reference_*` **互斥** —— 所以"锁脸参考"与
  "跨场景首尾帧串联"二选一，不能同时要（选哪条由 `choose_input_mode` 决定）。

切分规则（确定性、可测试，不依赖模型自觉）：
1. **相邻场景合并**的条件：location 与 time_of_day 都相同 —— 同一时空才谈得上连贯。
   用户那条 run 的三个场景 location 全是"女儿国街道"，却被拆成三次生成，
   连贯性在剧本阶段就已经被切碎了。
2. 合并到超过 30 秒（或分镜数超过 `MAX_SHOTS_PER_CLIP`）就**开新组**：宁可两组之间
   用串联/转场衔接，也不能提交一个服务端会拒的时长。
3. 组的 `windows` 给出每个镜头在片段内的起止秒 —— 字幕排布、逐镜头 QC、身份度量、
   以及"只重拍某一个镜头"的降级路径都要靠它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 官方 duration 上限（无视频输入时取值范围 [2,30] 的整数）
MAX_CLIP_SEC = 30
MIN_CLIP_SEC = 2
#: 官方分镜指南：每段 2–5 秒
SHOT_MIN_SEC = 2.0
SHOT_MAX_SEC = 5.0
#: 一次调用里塞太多分镜，提示词就控不住了（官方综合示例是 15 秒 4 个分镜）
MAX_SHOTS_PER_CLIP = 6
#: 参考素材预算（官方上限）：`reference_image` ≤10 张；`reference_audio` ≤5 段且总时长 ≤15 秒
MAX_REF_IMAGES = 10
MAX_REF_AUDIOS = 5
MAX_REF_AUDIO_TOTAL_SEC = 15.0
#: 每个角色的音色样本长度上限。取 4 秒是"够模型抓到音色、又能在 5 段/15 秒预算里
#: 塞下 3 个角色"的折中：短剧一个场景通常 2-3 人说话，超出的角色按出场顺序截断并留痕。
VOICE_REF_SEC = 4.0


@dataclass(frozen=True)
class SceneGroup:
    """一次生成调用对应的镜头范围。"""

    group_id: str
    scene_ids: tuple[str, ...]
    shot_ids: tuple[str, ...]
    duration: float
    location: str
    time_of_day: str
    characters: tuple[str, ...]
    #: shot_id → (片段内起点秒, 终点秒)。字幕/QC/身份度量/单镜头重拍都按它定位。
    windows: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: True = 本组用上一组的尾帧当首帧（跨场景串联）。与 reference_* 互斥，
    #: 所以串联组**不能**同时发角色参考图 —— 这是官方约束，不是取舍。
    #: 被截断/拆分的说明（进事件与看板，别让用户猜为什么这里断开了）
    note: str = ""

    @property
    def shot_count(self) -> int:
        return len(self.shot_ids)


def _scene_key(scene: Any) -> tuple[str, str]:
    """判定"同一时空"的键：地点 + 时段。

    只用 location 会把"街道·日"和"街道·夜"并成一组 —— 那是两次布光，模型接不上。
    """
    return (str(scene.location or "").strip(), str(getattr(scene, "time_of_day", "") or "").strip())


def group_scenes(script: Any, *, max_clip_sec: float = MAX_CLIP_SEC) -> list[SceneGroup]:
    """**一个 scene = 一个场景视频**（用户裁定：不合并多个 scene 成一个片段）。

    为什么不再做"同地点相邻场景合并"：合并之后界面上"场景2"没有自己的卡片与模式选择，
    用户看到的是"场景2 的镜头挂在场景1 的片段下面"，顺序与归属都乱了；而连贯性的真正
    来源是**一次调用内的多镜头**，不是"把两个 scene 塞进一次调用"。把同一时空的剧情
    写进同一个 scene，是 m1 剧本提示词的职责（见 screenwriter.v1.md 的场景语义规则），
    不在这里事后合并。

    唯一的例外是**拆**不是**合**：单个 scene 超过 30 秒（官方单次上限）时按镜头边界
    拆成多段（`scene_2p1`/`scene_2p2`），否则提交必被服务端拒。拆是物理约束，不是选择。
    """
    groups: list[SceneGroup] = []
    for scene in script.scenes:
        shots = list(scene.shots or [])
        if not shots:
            continue
        groups.append(_new_group(scene, shots, _scene_key(scene), groups))
    out: list[SceneGroup] = []
    for g in groups:
        out.extend(_split_oversized(g, max_clip_sec))
    return out


def _windows(shots: list[Any], start: float = 0.0) -> tuple[dict[str, tuple[float, float]], float]:
    win: dict[str, tuple[float, float]] = {}
    t = start
    for sh in shots:
        d = float(sh.duration_sec or 0)
        win[sh.shot_id] = (round(t, 3), round(t + d, 3))
        t += d
    return win, t


def _cast(scene: Any, shots: list[Any]) -> tuple[str, ...]:
    """本组出场角色（**有序去重**）：决定发几张立绘参考、取谁的音色样本。"""
    seen: list[str] = []
    for sh in shots:
        for cid in list(getattr(sh, "characters", []) or []):
            if cid not in seen:
                seen.append(cid)
        for d in list(getattr(sh, "dialogues", []) or []):
            cid = getattr(d, "character_id", "")
            if cid and cid not in seen:
                seen.append(cid)
    return tuple(seen)


def _new_group(scene: Any, shots: list[Any], key: tuple[str, str], prev: list[SceneGroup]) -> SceneGroup:
    win, total = _windows(shots)
    # 组 id 默认就是场景 id；万一重复（剧本里 scene_id 撞了）加序号，
    # 因为它是产物文件名（`clips/scene_{gid}.mp4`）与台账 note 的键，撞了就互相覆盖
    gid = str(scene.scene_id)
    if any(g.group_id == gid for g in prev):
        gid = f"{gid}#{len(prev) + 1}"
    return SceneGroup(
        group_id=gid,
        scene_ids=(scene.scene_id,),
        shot_ids=tuple(sh.shot_id for sh in shots),
        duration=round(total, 3),
        location=key[0],
        time_of_day=key[1],
        characters=_cast(scene, shots),
        windows=win,
        note="" if total <= MAX_CLIP_SEC else f"本场景 {total:.0f}s 超过单次上限，将按镜头边界拆分",
    )


def _split_oversized(g: SceneGroup, max_clip_sec: float) -> list[SceneGroup]:
    """超预算的组按**镜头边界**拆开（绝不切断一个镜头：那会切掉半句台词）。

    拆分只可能来自"单个场景自己就超 30 秒"（跨场景合并在 `group_scenes` 里已经
    卡过预算），所以 scene_ids 原样继承，时间轴按镜头时长重排。
    """
    if g.duration <= max_clip_sec and g.shot_count <= MAX_SHOTS_PER_CLIP:
        return [g]
    durs = {sid: round(e - s, 3) for sid, (s, e) in g.windows.items()}
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_dur = 0.0
    for sid in g.shot_ids:
        d = durs.get(sid, 0.0)
        if cur and (cur_dur + d > max_clip_sec or len(cur) >= MAX_SHOTS_PER_CLIP):
            chunks.append(cur)
            cur, cur_dur = [], 0.0
        cur.append(sid)
        cur_dur += d
    if cur:
        chunks.append(cur)
    if len(chunks) == 1:
        return [g]
    out: list[SceneGroup] = []
    for i, chunk in enumerate(chunks):
        win, total = _windows_from_durs(chunk, durs)
        out.append(
            SceneGroup(
                group_id=f"{g.group_id}p{i + 1}",
                scene_ids=g.scene_ids,
                shot_ids=tuple(chunk),
                duration=round(total, 3),
                location=g.location,
                time_of_day=g.time_of_day,
                # 角色按整组继承：`reference_image` 上限 10 张，而一段短剧的在场角色
                # 远不到这个数，收窄省不下钱，反而可能把"这一段没台词但入画"的人漏掉
                characters=g.characters,
                windows=win,
                note=f"原组 {g.duration:.0f}s 超 {max_clip_sec:.0f}s 上限，"
                     f"按镜头边界拆成 {len(chunks)} 段（第 {i + 1} 段）",
            )
        )
    return out


def _windows_from_durs(shot_ids: list[str], durs: dict[str, float]) -> tuple[dict[str, tuple[float, float]], float]:
    win: dict[str, tuple[float, float]] = {}
    t = 0.0
    for sid in shot_ids:
        d = durs.get(sid, 0.0)
        win[sid] = (round(t, 3), round(t + d, 3))
        t += d
    return win, t


@dataclass(frozen=True)
class RefPlan:
    """一次场景调用要发的参考素材，以及**编号**（提示词里按编号引用：图1/图2、音频1）。

    编号必须与 media 数组顺序一致：官方规定"素材按上传顺序编号，图片、视频、音频
    分别计数"，提示词写"唐僧的形象参考图2"就必须真的把唐僧立绘放在第 2 张。
    错位不会报错，只会让模型锁错脸 —— 那是最难查的一类失真。
    """

    images: tuple[tuple[str, Any], ...]  # (编号名 图1/图2…, 角色 id)
    #: 场景图参考，编号**接在角色立绘之后**（图N 连续）：官方示例就是
    # "人物参考图 + 场景参考"混在同一个 media 数组里按上传顺序编号
    scene_images: tuple[tuple[str, Any], ...] = ()
    audios: tuple[tuple[str, str], ...] = ()  # (编号名 音频1…, 角色 id)
    dropped_images: tuple[str, ...] = ()
    dropped_audios: tuple[str, ...] = ()
    #: 有角色拿不到参考素材时必须说出来：少一张立绘 = 那个人的脸靠模型自由发挥
    note: str = ""

    def all_images(self) -> tuple[tuple[str, Any], ...]:
        """media 数组的真实顺序 = 提示词里 图N 的顺序。两者错位不会报错，只会锁错脸。"""
        return self.images + self.scene_images


def plan_references(
    group: SceneGroup,
    *,
    speakers: tuple[str, ...] | None = None,
    per_voice_sec: float = VOICE_REF_SEC,
    scene_ids: tuple[str, ...] | list[str] = (),
) -> RefPlan:
    """按官方预算分配参考素材：立绘 ≤10 张、音色 ≤5 段且总时长 ≤15 秒。

    取舍顺序是刻意的：
    - **立绘按出场顺序**取前 10 个（脸是最贵的失真项，m9 客观层就是量它）；
    - **音色按"本组有台词的角色"优先**，再按出场顺序补，最多 5 段；
      段数用完后再看总时长预算（5 段 × 4 秒 = 20 秒 > 15 秒 → 实际最多 3 段满长）。
    超预算的角色进 `dropped_*` 并写进 `note`，由调用方落事件/降级台账：
    静默丢参考素材 = 那个角色开始漂脸/漂音色，而台账和看板都看不出来。
    """
    cast = list(group.characters)
    images = tuple((f"图{i + 1}", cid) for i, cid in enumerate(cast[:MAX_REF_IMAGES]))
    dropped_images = tuple(cast[MAX_REF_IMAGES:])
    # 场景图接在立绘后面继续编号（官方：media 按上传顺序编号，图/视频/音频分别计数）。
    # 预算是**合起来** ≤10 张：立绘已经占满时场景图只能让位（脸比布景更贵）。
    scene_images: list[tuple[str, Any]] = []
    for sid in scene_ids:
        if len(images) + len(scene_images) >= MAX_REF_IMAGES:
            break
        scene_images.append((f"图{len(images) + len(scene_images) + 1}", sid))

    want_voices = [c for c in (speakers or ()) if c in cast]
    want_voices += [c for c in cast if c not in want_voices]
    audios: list[tuple[str, str]] = []
    dropped_audios: list[str] = []
    budget = MAX_REF_AUDIO_TOTAL_SEC
    for cid in want_voices:
        if len(audios) >= MAX_REF_AUDIOS or budget - per_voice_sec < 0:
            dropped_audios.append(cid)
            continue
        audios.append((f"音频{len(audios) + 1}", cid))
        budget -= per_voice_sec
    note = ""
    if dropped_images:
        note += f"立绘参考超 {MAX_REF_IMAGES} 张，未发：{'、'.join(dropped_images)}；"
    if dropped_audios:
        note += f"音色参考超预算（≤{MAX_REF_AUDIOS} 段且总 ≤{MAX_REF_AUDIO_TOTAL_SEC:g}s），未发：{'、'.join(dropped_audios)}；"
    if note:
        note += "这些角色的脸/音色由模型自由发挥，跨场景可能漂"
    # 全用关键字：RefPlan 字段顺序变过（加了 scene_images），位置参数会静默错位
    return RefPlan(
        images=images,
        scene_images=tuple(scene_images),
        audios=tuple(audios),
        dropped_images=dropped_images,
        dropped_audios=tuple(dropped_audios),
        note=note,
    )


#: 两种互斥的输入形态（官方：reference_* 与 first/last_frame 不能同发）
MODE_REFERENCE = "reference"   # 立绘+场景图+音色样本 + 多镜头提示词（主干，≈90%）
MODE_FIRST_LAST = "first_last"  # 首帧(+尾帧) + 提示词（串联/单镜头/无立绘，≈10%）


MODE_NONE = "none"


def choose_input_mode(
    group: SceneGroup,
    *,
    has_dialogue: bool = False,
    has_first_frames: bool = False,
    has_references: bool = True,
    forced: str = "auto",
) -> tuple[str, str]:
    """自动选这次调用走哪种输入形态，返回 (模式, 理由)。三条规则，按顺序判：

    1. **组内有对白 → 参考生视频**（角色图+场景图+音色样本）。有台词就需要音色参考，
       而 `reference_audio` 只有参考形态能发（与首尾帧官方互斥）；脸和声一起锁。
    2. **角色数 ≥2 → 参考生视频**。多角色同框时首尾帧只能锁两个瞬间的构图，
       锁不住"每个人是谁"；参考形态按编号逐人锁脸，避免特征串位。
    3. **其余（≤1 个角色且无对白）→ 首尾帧**。空镜/单人动作镜头没有身份与音色要锁，
       像素级锁起止构图最直接，也更便宜（少传一堆素材）。

    规则是确定性的（同一剧本同一配置永远选同一条路，结果才可复现、才可对账），
    不靠 LLM 判断：判据全是剧本里数得出来的事实（有无台词、几个角色）。

    `forced` 是人工覆盖（分镜视频界面里每组可改），覆盖时仍受**物理前提**约束：
    - 要首尾帧却没有首帧 → 退回参考；
    - 要参考却一张参考素材都没有（角色图/场景图在盘上都找不到）→ 退回首尾帧；
      两边都没有 → 返回 `MODE_NONE`，调用方必须降级并说清原因，**不能**带着空
      `references` 去撞 wan3.0 的规格守卫（实测那样只会得到"点了生成毫无反应"）。
    """
    char_count = len(group.characters)
    if forced == MODE_FIRST_LAST:
        if has_first_frames:
            return MODE_FIRST_LAST, "人工指定：首尾帧生成"
        if has_references:
            return MODE_REFERENCE, "人工指定首尾帧，但本组还没有首帧 → 退回参考生视频（场景图当参考）"
        return MODE_NONE, "人工指定首尾帧，但既没有首帧也没有参考素材 → 本组无法生成"
    if forced == MODE_REFERENCE:
        if has_references:
            return MODE_REFERENCE, "人工指定：角色图+场景图参考生成"
        if has_first_frames:
            return MODE_FIRST_LAST, "人工指定参考生成，但盘上没有可用的角色图/场景图 → 改走首尾帧"
        return MODE_NONE, "人工指定参考生成，但盘上没有参考素材也没有首帧 → 本组无法生成"
    if has_dialogue:
        if has_references:
            return MODE_REFERENCE, "规则1：组内有对白 → 参考生成（音色参考只有这条路能发，脸与声一起锁）"
        if has_first_frames:
            return MODE_FIRST_LAST, "规则1 命中但盘上没有参考素材 → 退回首尾帧（口型仍由 wan3.0 自带）"
        return MODE_NONE, "组内有对白，但既没有参考素材也没有首帧 → 本组无法生成"
    if char_count >= 2 and has_references:
        return MODE_REFERENCE, f"规则2：{char_count} 个角色同框 → 参考生成逐人锁脸"
    if has_first_frames:
        return MODE_FIRST_LAST, "规则3：单角色/无角色且无对白 → 首尾帧锁起止构图"
    if has_references:
        return MODE_REFERENCE, "规则3 命中但本组还没有首帧 → 退回参考生视频（场景图当参考）"
    return MODE_NONE, "规则3 命中，但既没有首帧也没有参考素材 → 本组无法生成"


def group_has_dialogue(script: Any, group: SceneGroup) -> bool:
    ids = set(group.shot_ids)
    return any(bool(sh.dialogues) for sh in script.all_shots() if sh.shot_id in ids)


def servable_groups(
    script: Any,
    groups: list[SceneGroup],
    *,
    talk_mode: str,
    i2v_model: str,
    scene_mode: str = "auto",
    force_scene: bool = True,
) -> list[SceneGroup]:
    """哪些组交给"wan3.0 一次调用直出一段场景视频"。

    **默认全部服务**（`force_scene=True`）：本项目的图生视频只有 wan3.0 系列，
    成片的每一段都应当是场景级产物 —— 组内多镜头由模型自己切，空间/服装/光线天然连续，
    口型也是 wan3.0 自带的。"首帧图 → 逐镜头片段"那条老路不再作为正常路径：
    它把一段戏切成若干互不连贯的短片，正是"镜头之间像瞬移"的根源。

    仍保留两个例外（判据来自 run_meta，老 run 要能续跑）：
    1. `scene_mode="off"`：一组都不服务（逐镜头兜底，仅历史 run 用）；
    2. 非 C 链（talk_mode 是历史上的 A/B 链）且环里没有 wan3.0：只服务**整组无对白**的组，
       否则 m6 与新路径会抢同一秒画面（重复付费）。
    """
    if scene_mode == "off":
        return []
    wan30_i2v = str(i2v_model or "").startswith("wan3.0")
    c_chain = talk_mode in ("", "scripted_talk")
    if not c_chain and not wan30_i2v:
        return []  # 环里没有 wan3.0：老 run 的 A/B 链，画面不归场景路径
    if force_scene and c_chain:
        return list(groups)
    # A/B 链（s2v / videoretalk 驱动对白镜头）且环里没有 wan3.0：含对白的组**不能**算覆盖，
    # 否则 m6 会被跳过 → 成片没有口型。这是"静默交付坏片"的形状，宁可重复付费也不放行。
    return [g for g in groups if not group_has_dialogue(script, g)]


def describe(groups: list[SceneGroup]) -> str:
    """一行人类可读的分组结论（CLI/看板用，避免用户对着 scene_id 猜）。"""
    return "、".join(
        f"{g.group_id}[{g.shot_count}镜/{g.duration:.0f}s/{'+'.join(g.scene_ids)}]"
        for g in groups
    )
