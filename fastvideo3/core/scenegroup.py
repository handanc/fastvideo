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
  "跨场景首尾帧串联"二选一，不能同时要（`SceneGroup.bridging` 标明本组走哪条）。

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
    bridging: bool = False
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
    """把剧本的场景切成"一次调用一组"的生成单位。

    合并只在**相邻**场景之间发生：隔着别的场景再回到同一地点，那是叙事上的回访，
    不该跟前面那组合并（合并了等于把中间那段剧情挪走）。
    """
    groups: list[SceneGroup] = []
    for scene in script.scenes:
        shots = list(scene.shots or [])
        if not shots:
            continue
        key = _scene_key(scene)
        dur = float(sum(float(sh.duration_sec or 0) for sh in shots))
        cur = groups[-1] if groups else None
        # 能与上一组合并：同地点同时段、时长与分镜数都还在预算内
        if (
            cur is not None
            and (cur.location, cur.time_of_day) == key
            and cur.duration + dur <= max_clip_sec
            and cur.shot_count + len(shots) <= MAX_SHOTS_PER_CLIP
        ):
            groups[-1] = _extend(cur, scene, shots)
            continue
        groups.append(_new_group(scene, shots, key, groups))
    # 单场景自身就超过 30 秒 → 按镜头边界拆成多组（服务端会拒超长请求，不能提交）
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


def _extend(cur: SceneGroup, scene: Any, shots: list[Any]) -> SceneGroup:
    # `_windows(start=…)` 返回的 total 是**绝对结束时刻**（已含 cur.duration），
    # 不能再加一次 —— 加过就变成 9+18=27 这种数，30 秒预算被虚耗掉一半，
    # 明明能一次生成的场景被拆成两次（连贯性和钱都白花）。
    win, total = _windows(shots, start=cur.duration)
    merged = dict(cur.windows)
    merged.update(win)
    cast = list(cur.characters)
    for cid in _cast(scene, shots):
        if cid not in cast:
            cast.append(cid)
    return SceneGroup(
        group_id=cur.group_id,
        scene_ids=cur.scene_ids + (scene.scene_id,),
        shot_ids=cur.shot_ids + tuple(sh.shot_id for sh in shots),
        duration=round(total, 3),
        location=cur.location,
        time_of_day=cur.time_of_day,
        characters=tuple(cast),
        windows=merged,
        bridging=cur.bridging,
        note=(cur.note + "；" if cur.note else "") + f"并入 {scene.scene_id}（同地点同时段）",
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
                bridging=g.bridging,
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


def choose_input_mode(
    group: SceneGroup,
    *,
    has_look_cards: bool,
    has_first_frames: bool,
    has_dialogue: bool = False,
    first_frame_stale: bool = False,
    bridging: bool = False,
    forced: str = "auto",
) -> tuple[str, str]:
    """自动选这次调用走哪种输入形态，返回 (模式, 理由)。

    判据是"**这一组最需要锁住什么**"—— 两种形态锁的东西不同：
    参考生视频锁**身份属性**（脸/服装/音色，整段全程）并让模型拥有组内镜头语言；
    首尾帧锁**起止两个瞬间的像素构图**，但中段不受控、且发不了音色参考（官方互斥）。
    按预测力排序的五个信号：

    1. **组内有几个切点**：多镜头组只有参考形态能让中段连贯 —— 首尾帧只能锁两端，
       中间 3-5 个切点完全不受控，等于放弃组内连续性。
    2. **要不要与邻组像素级衔接**：要（串联）就只能首尾帧 —— 参考形态表达不了
       "从上一组尾帧那个像素状态开始"。
    3. **有没有台词**：有台词就需要音色参考，而 `reference_audio` 只有参考形态能发；
       单镜头说话镜头走首尾帧会锁住构图却丢嗓子（跨场景换声就是这么来的）。
    4. **参考素材与首帧哪个新鲜可信**：首帧是旧立绘画的（血缘已标 stale）时，
       拿它锁起止 = 把旧脸焊进新片段，必须改走参考形态或先重画首帧。
    5. **失败代价**：首尾帧形态单镜头 2-5s（¥1-6），参考形态整组 10-30s（¥6-36）。
       代价高不改变选型（前四条更影响成片质量），但超预算时应该**拆组**而不是换形态。

    规则是确定性的（同一剧本同一配置永远选同一条路，结果才可复现、才可对账）。
    `forced` 允许配置/面板覆盖，但覆盖仍受物理前提约束（强制首尾帧却没有新鲜首帧
    → 说明原因后退回参考；强制参考而没有立绘 → 场景图也可以当参考，照走）。
    """
    fresh_first = has_first_frames and not first_frame_stale
    if forced == MODE_FIRST_LAST:
        if fresh_first:
            return MODE_FIRST_LAST, "配置指定首尾帧形态"
        return MODE_REFERENCE, (
            "配置要首尾帧，但本组首帧"
            + ("已作废（上游变更）" if first_frame_stale else "缺失")
            + "，退回参考生视频"
        )
    if forced == MODE_REFERENCE:
        return MODE_REFERENCE, "配置指定参考生视频形态"
    if bridging:
        return MODE_FIRST_LAST, "跨场景串联：首帧取自上一组尾帧（与参考素材官方互斥）"
    if first_frame_stale and has_look_cards:
        return MODE_REFERENCE, "首帧已作废（上游立绘/剧本变更）：拿它锁起止会把旧脸焊进新片段"
    if has_dialogue and has_look_cards:
        return MODE_REFERENCE, (
            "有台词：音色参考只有参考形态能发（与首尾帧互斥），脸与声一起锁"
            + ("；组内多镜头由模型自切保持连贯" if group.shot_count > 1 else "")
        )
    if group.shot_count == 1 and fresh_first:
        return MODE_FIRST_LAST, "单镜头无台词：像素级锁起止构图最直接，且少传素材更便宜"
    if has_look_cards:
        return MODE_REFERENCE, "多镜头无台词：组内连贯靠模型自切，立绘锁脸"
    if fresh_first:
        return MODE_FIRST_LAST, "本组没有立绘可参考：用首帧锁构图"
    return MODE_REFERENCE, "无立绘也无可用首帧：场景图作参考素材（聊胜于无）"


def describe(groups: list[SceneGroup]) -> str:
    """一行人类可读的分组结论（CLI/看板用，避免用户对着 scene_id 猜）。"""
    return "、".join(
        f"{g.group_id}[{g.shot_count}镜/{g.duration:.0f}s/{'+'.join(g.scene_ids)}]"
        for g in groups
    )
