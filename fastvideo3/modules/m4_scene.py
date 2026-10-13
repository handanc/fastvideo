"""场景级提示词构造：一次 wan3.0 调用生成"一个场景的连贯多镜头片段"。

严格照官方多镜头模板写（`wan3-video-generation-prompt-guide`）：

    提示词 = 总体描述 + 镜头序号 + 时间戳 + 分镜内容
    各镜头首尾相接、不跳段不重叠，每段 2–5 秒
    素材按上传顺序编号：第 1 张图是图1、第 1 段音频是音频1

官方综合示例正是"参考生视频 + 多镜头 + 台词 + 音色参考 + 风格控制"写在同一次调用里，
所以这里把四件事拼成一段：

1. **总体描述**：题材/风格 + 场景（地点、时段）+ 总时长 + 画幅 —— 模型据此定基调；
2. **参考素材编号引用**：`图1` = 角色立绘（锁脸）、`音频1` = 角色音色样本（锁声）。
   编号必须与 media 数组顺序一致（`scenegroup.RefPlan` 负责排），错位不会报错，
   只会让模型锁错脸/错音色；
3. **逐镜头分镜**：`第N个镜头[起-止秒]` + 转场词 + 景别机位 + 外貌锚 + 动作 + 台词。
   台词逐字取剧本 `DialogueLine.text`（与 m7 烧字幕同源），说话动词按情绪档确定性映射；
4. **一致性要求**：同一角色全片长相/服装/音色不变，且**场景空间连续、走位衔接** ——
   最后这句是场景级生成相对逐镜头拼接的全部意义所在，漏了它就退化成"一次画多个镜头"。

台词块与音色锚**复用 m6 的实现**（`_scripted_lines_block` / `_timbre_anchor` /
`_speech_verb`）：这两条路径（单镜头 C 链、场景级）必须逐字同源，否则同一个角色在
两种模式下会拿到不同的说话动词或音色描述，跨模式的一致性就没了。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastvideo3.core import voiceref
from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.prompts import NO_ONSCREEN_TEXT
from fastvideo3.core.scenegroup import (
    MAX_CLIP_SEC,
    MIN_CLIP_SEC,
    MODE_FIRST_LAST,
    MODE_REFERENCE,
    RefPlan,
    SceneGroup,
    choose_input_mode,
    group_scenes,
)
from fastvideo3.core.schema import AudioLine, JobStatus, Script, TransitionType
from fastvideo3.core.storage import RunStorage
from fastvideo3.modules.m6_talk import _scripted_lines_block, _timbre_anchor
from fastvideo3.providers.base import (
    AccountGateError,
    CapabilityMismatchError,
    ProviderError,
    is_protocol_error,
    poll_until_done,
    price_key_for,
)

#: 剧本的 transition_out → 提示词里的转场词。官方示例就是在分镜开头写"硬切转场，固定机位"，
#: 模型据此决定切法；不写的话它会自己挑，同一段里可能忽而叠化忽而硬切。
_TRANSITION_WORD = {
    TransitionType.CUT: "硬切转场",
    TransitionType.DISSOLVE: "叠化转场",
    TransitionType.FADE: "淡出转场",
    TransitionType.SCENERY: "以空镜过渡",
}

#: 镜头类型 → 默认景别（剧本 camera 为空时兜底，避免"没有景别"让模型自己乱选）
_TYPE_FRAMING = {
    "dialogue": "中近景",
    "action": "中景",
    "scenery": "大远景空镜",
}


def _ts(sec: float) -> str:
    """时间戳：整秒写整数，否则留一位小数（官方示例两种都出现过，混着写更易读错）。"""
    return str(int(sec)) if abs(sec - round(sec)) < 1e-6 else f"{sec:.1f}"


def _name_look(ch: Any) -> tuple[str, str]:
    """角色名与外貌锚。立绘卡缺失/字段为空时返回空串（调用方据此跳过），
    绝不写出 "None（None）" —— 那种垃圾进提示词，模型会照着画一个"叫 None 的人"。"""
    if ch is None:
        return "", ""
    return str(getattr(ch, "name", "") or ""), str(getattr(ch, "appearance", "") or "")


def _end_punct(text: str) -> str:
    """补句号，但**不重复补**：剧本里的 action 常常已经带"。"，直接 + "。" 会写出
    "不肯移动。。"。提示词里出现双句号不会报错，只会白白吃掉 token 权重。"""
    t = (text or "").strip()
    if not t:
        return ""
    return t if t[-1] in "。！？；.!?" else t + "。"


def transition_word(shot: Any) -> str:
    t = shot.transition_out
    key = t if isinstance(t, TransitionType) else TransitionType(str(t or "cut"))
    return _TRANSITION_WORD.get(key, "硬切转场")


def scene_duration_sec(group: SceneGroup) -> int:
    """本次调用的 `duration`：官方要求整数且 ∈ [2,30]。

    向上取整而不是四舍五入：少给一秒会把最后一个镜头的尾巴切掉（听感是"话没说完"），
    多给半秒只是画面末尾多一点点静止，两害相权取其轻。
    """
    import math

    return int(min(MAX_CLIP_SEC, max(MIN_CLIP_SEC, math.ceil(group.duration - 1e-6))))


def reference_line(plan: RefPlan | None, char_map: dict, scene_names: dict | None = None) -> str:
    """参考素材的编号引用段。**串联组（bridging）不发参考素材** → 返回空串。

    `first_frame`/`last_frame` 与 `reference_image`/`reference_audio` 官方明写互斥，
    所以跨场景串联（上一组尾帧当这一组首帧）与锁脸/锁声参考只能二选一。
    """
    scene_names = scene_names or {}
    if plan is None:
        return ""
    imgs = []
    for label, cid in plan.images:
        ch = char_map.get(cid)
        name = (getattr(ch, "name", "") if ch else "") or cid
        imgs.append(f"{name}的形象参考{label}")
    # 场景图接在立绘后继续编号：官方多主体示例就是"人物图N + 场景图M"混排指名，
    # 用途在提示词里切清楚，避免特征串位（把场景的配色串到人物衣服上）
    for label, sid in plan.scene_images:
        imgs.append(f"{scene_names.get(sid, sid)}的场景参考{label}")
    auds = []
    for label, cid in plan.audios:
        ch = char_map.get(cid)
        name = (getattr(ch, "name", "") if ch else "") or cid
        auds.append(f"{name}的音色参考{label}")
    parts = [p for p in ("，".join(imgs), "，".join(auds)) if p]
    if not parts:
        return ""
    return (
        "参考素材：" + "；".join(parts)
        + "。严格保持参考人物的长相、服装与音色，严格保持参考场景的空间结构与光线。"
    )


def scene_prompt(
    cfg: Any,
    group: SceneGroup,
    script: Any,
    char_map: dict,
    ref_plan: RefPlan | None = None,
    *,
    bridging: bool = False,
) -> str:
    """把一个场景组写成官方多镜头模板的提示词。

    `bridging=True` 表示本组用上一组的尾帧当首帧（跨场景串联）：此时**不能**再发
    reference_image/reference_audio（官方类型互斥），提示词里也不该出现"参考图N"，
    否则模型会去找不存在的素材。
    """
    shots = {sh.shot_id: sh for sh in script.all_shots()}
    ordered = [shots[sid] for sid in group.shot_ids if sid in shots]
    if not ordered:
        raise ValueError(f"场景组 {group.group_id} 的镜头在剧本里找不到")

    total = scene_duration_sec(group)
    # 风格取自**剧本的 base_info**（建任务时用户选的那个），不是 cfg —— cfg 上只有
    # 全局画风锚 style_suffix，没有 style 字段（写成 cfg.style 会直接 AttributeError）
    style = str(getattr(getattr(script, "base_info", None), "style", "") or "").strip()
    ratio = str(getattr(cfg, "aspect_ratio", "") or "9:16")
    screen = {"9:16": "竖屏 9:16", "16:9": "横屏 16:9", "1:1": "方形 1:1"}.get(ratio, ratio)
    head = [
        f"{style}短剧片段，共 {total} 秒，{screen}。",
        f"场景：{group.location or '室内'}，{group.time_of_day or '日'}。",
        "由下列分镜首尾相接构成一段连贯叙事，镜头之间空间连续、人物走位与上一镜头衔接，"
        "不要跳戏、不要重新布景。",
    ]
    if bridging:
        head.append("画面严格从首帧开始、到尾帧结束（首尾帧已锁定起止构图）。")
    else:
        scene_names_map = {
            sc.scene_id: (sc.title or sc.location or sc.scene_id)
            for sc in getattr(script, "scenes", [])
        }
        ref = reference_line(ref_plan, char_map, scene_names_map)
        if ref:
            head.append(ref)

    # 人物外貌锚在**总体描述里写一次**，分镜里只点名 —— 官方综合示例就是这个形状。
    # 每个分镜都重抄一遍外貌会把提示词撑长一倍（5 镜 3 人 = 15 段重复），
    # 而重复描述对一致性没有额外帮助，反而挤占了动作与台词的权重。
    cast_ids = list(dict.fromkeys(
        cid for sh in ordered for cid in list(sh.characters) + [d.character_id for d in sh.dialogues]
    ))
    people_desc = []
    for cid in cast_ids:
        nm, look = _name_look(char_map.get(cid))
        if not nm:
            continue
        people_desc.append(f"{nm}（{look}）" if look else nm)
    if people_desc:
        head.append(f"人物：{'；'.join(people_desc)}。")

    body: list[str] = []
    for i, sh in enumerate(ordered):
        start, end = group.windows.get(sh.shot_id, (0.0, 0.0))
        names = []
        for cid in dict.fromkeys(list(sh.characters) + [d.character_id for d in sh.dialogues]):
            nm, _look = _name_look(char_map.get(cid))
            if nm:
                names.append(nm)
        stype = str(getattr(sh.type, "value", "") or sh.type or "")
        framing = (sh.camera or "").strip() or _TYPE_FRAMING.get(stype, "中景")
        lines = _scripted_lines_block(sh, char_map)
        seg = [f"第{i + 1}个镜头[{_ts(start)}-{_ts(end)}秒]"]
        if i > 0:
            # 转场词取自**上一个镜头**的 transition_out（它描述的是"怎么切到下一镜"）
            seg.append(transition_word(ordered[i - 1]) + "，")
        seg.append(f"{framing}。")
        if names:
            seg.append(f"画面中人物：{'、'.join(names[:4])}。")
        action = _end_punct((sh.action or "").strip())
        seg.append(action)
        if lines:
            seg.append(f"台词：{_end_punct(lines)}")
        body.append("".join(p for p in seg if p))

    # 音色锚按**整组**给一次（不是每镜头重复）：同一角色在整段里必须是同一副嗓子
    timbre = _timbre_anchor(
        char_map,
        [cid for sh in ordered for cid in list(sh.characters) + [d.character_id for d in sh.dialogues]],
    )
    any_lines = any(_scripted_lines_block(sh, char_map) for sh in ordered)
    tail = [
        (
            "开口说话的人物口型与台词严格同步，表情自然、头部有轻微动作；"
            "没有台词的人物保持闭口，不要给不出台词的人对口型。"
            if any_lines
            else "全片无人说话，画面无口型动作。"
        ),
        f"{timbre}，同一角色在整段视频中音色不变。" if timbre else "",
        "全片保持同一角色的长相、发型、服装配色一致，不换脸、不换衣服、不改画幅。",
        (cfg.style_suffix or "").strip(),
        NO_ONSCREEN_TEXT,
    ]
    neg = (cfg.negative_prompt or "").strip()
    if neg:
        # wan3.0 的 parameters 里**没有** negative_prompt，按参数发会被静默丢弃 → 折进提示词
        tail.append(f"负向：{neg}。")
    return " ".join(p for p in (head + body + tail) if p).strip()


def _tail_frame(clip: Path, dest: Path) -> Path | None:
    """取一段视频的最后一帧（跨场景串联：上一组尾帧 = 这一组首帧）。

    失败返回 None：串联是增强项，拿不到尾帧就退成普通场景生成，不值得为它中断整步。
    """
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(["-sseof", "-0.25", "-i", str(clip), "-frames:v", "1", str(dest)])
        return dest if dest.is_file() and dest.stat().st_size > 1024 else None
    except Exception:  # noqa: BLE001
        return None


def _ordered_refs(group: SceneGroup, assets: dict) -> tuple[list[str], list[str]]:
    """本组实际有文件的立绘角色 id 与场景图 id（顺序 = 上传顺序 = 提示词编号顺序）。"""
    looks = []
    for cid in group.characters:
        a = assets.get(f"char_{cid}")
        if a is not None and Path(str(a.uri)).exists():
            looks.append(cid)
    scenes = []
    for sid in group.scene_ids:
        a = assets.get(f"scene_{sid}")
        if a is not None and Path(str(a.uri)).exists():
            scenes.append(sid)
    return looks, scenes


async def generate_scene_clips(
    cfg: AppConfig,
    provider: Any,
    script: Script,
    assets: dict,
    storage: RunStorage,
    ledger: CostLedger,
    jobs: JobStore,
    *,
    lines: list[AudioLine] | None = None,
    frames: dict[str, str] | None = None,
    groups: list[SceneGroup] | None = None,
    only: set[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, str]:
    """每个场景组**一次调用**生成连贯多镜头片段 → `clips/scene_{gid}.mp4`。

    输入形态二选一（官方互斥，`choose_input_mode` 决定）：
    - `reference`：立绘 + 场景图（`reference_image`，编号即提示词里的 图N）+ 音色样本
      （`reference_audio`，音频N）+ 多镜头提示词。主干路径 —— 组内镜头由模型自己切，
      空间/服装/光线在组内天然连续；
    - `first_last`：首帧（+尾帧）+ 提示词。用于跨场景串联、单镜头组、以及没有立绘时。

    产物除视频外还落一份 `clips/scene_{gid}.json`：组内每个镜头的时间窗、实际发送的
    素材编号、模式与理由。m7 的字幕、m9 的逐镜头 QC/身份度量、以及"只重拍一个镜头"
    的降级路径全靠它 —— 没有这份映射，场景片段就是一块不可拆的黑箱。
    """
    from fastvideo3.modules.m4_shots import _last_frame_path
    from fastvideo3.providers.dashscope import actual_resolution_tier, billable_seconds

    frames = frames or {}
    lines = lines or []
    groups = (
        groups if groups is not None
        else group_scenes(script, max_clip_sec=float(MAX_CLIP_SEC))
    )
    char_map = script.character_map()
    out: dict[str, str] = {}
    prev_clip: Path | None = None

    for idx, group in enumerate(groups):
        gid = group.group_id
        dest = storage.path("clips", f"scene_{gid}.mp4")
        meta_path = storage.path("clips", f"scene_{gid}.json")
        rel = f"clips/scene_{gid}.mp4"
        if only is not None and gid not in only:
            if dest.exists():
                out[gid] = str(dest)
            continue
        if dest.exists() and skip_existing and not storage.is_stale(rel):
            out[gid] = str(dest)
            storage.log_event("m4_scene", "scene_reuse", group=gid)
            prev_clip = dest
            continue

        look_cids, scene_sids = _ordered_refs(group, assets)
        gshots = {sh.shot_id: sh for sh in script.all_shots() if sh.shot_id in set(group.shot_ids)}
        speakers = tuple(dict.fromkeys(
            d.character_id for sid in group.shot_ids for d in (gshots[sid].dialogues if sid in gshots else [])
        ))
        # 音色候选顺序：本组有台词的角色优先（他们才需要锁声），其余按出场顺序补
        voice_cids = [c for c in speakers if c in group.characters]
        voice_cids += [c for c in group.characters if c not in voice_cids]
        got_voice, voice_skipped = voiceref.build_voice_refs(storage, lines, group, voice_cids)

        # **先定实际能发什么，再写提示词**：编号（图N/音频N）必须与 media 数组顺序逐位
        # 对应。反过来先写提示词再裁素材，就会出现"提示词说图3、media 里只有两张"，
        # 模型不报错，只会把第三张的锁脸意图套到错误的图上。
        img_cids = look_cids[:10]
        scene_sids = scene_sids[: max(0, 10 - len(img_cids))]
        audio_cids = [c for c in voice_cids if c in got_voice]
        plan = RefPlan(
            images=tuple((f"图{i + 1}", c) for i, c in enumerate(img_cids)),
            scene_images=tuple(
                (f"图{len(img_cids) + i + 1}", s) for i, s in enumerate(scene_sids)
            ),
            audios=tuple((f"音频{i + 1}", c) for i, c in enumerate(audio_cids)),
            dropped_images=tuple(look_cids[10:]),
            dropped_audios=tuple(voice_skipped),
        )

        first_shot, last_shot = group.shot_ids[0], group.shot_ids[-1]
        first_frame = Path(frames[first_shot]) if frames.get(first_shot) else None
        bridging = bool(cfg.scene_bridge and idx > 0 and prev_clip is not None)
        if bridging and first_frame is None and prev_clip is not None:
            tail = storage.path("clips", f"_bridge_{gid}.png")
            first_frame = _tail_frame(prev_clip, tail)
        # 首帧"在盘"不等于"可信"：上游立绘/剧本改过之后血缘会把它标 stale，
        # 拿一张旧脸画的首帧去锁起止，等于把旧脸焊进新片段。
        first_stale = bool(frames.get(first_shot)) and storage.is_stale(
            f"storyboard/shot_{first_shot}.png"
        )
        mode, why = choose_input_mode(
            group,
            has_look_cards=bool(img_cids),
            has_first_frames=first_frame is not None,
            has_dialogue=bool(speakers),
            first_frame_stale=first_stale,
            bridging=bridging,
            forced=cfg.scene_clip_mode,
        )
        prompt = scene_prompt(
            cfg, group, script, char_map,
            plan if mode == MODE_REFERENCE else None,
            bridging=(mode == MODE_FIRST_LAST and bridging),
        )
        duration = scene_duration_sec(group)
        params: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
            "resolution": cfg.i2v_resolution,
        }
        submit_image: Path | None = None
        if mode == MODE_REFERENCE:
            params["references"] = [Path(str(assets[f"char_{c}"].uri)) for c in img_cids] + [
                Path(str(assets[f"scene_{s}"].uri)) for s in scene_sids
            ]
            params["voice_refs"] = [Path(got_voice[c]) for c in audio_cids]
        else:
            submit_image = first_frame
            params["last_frame"] = (
                _last_frame_path(storage, last_shot)
                if (cfg.use_last_frames and not bridging)
                else None
            )

        job_id = f"scene_{gid}"
        existing = jobs.get(job_id)
        if existing is not None and existing.status == JobStatus.SUCCEEDED and dest.exists():
            out[gid] = str(dest)
            prev_clip = dest
            continue
        try:
            ext_id = await provider.submit(dest=dest, image=submit_image, audio=None, video=None, **params)
            jobs.update(job_id, status=JobStatus.RUNNING, output_uri=ext_id)
            status = await poll_until_done(
                provider.poll,
                ext_id,
                timeout_sec=cfg.i2v_poll_timeout_min * 60,
                fast_interval=2 if cfg.fake else 5,
                slow_interval=5 if cfg.fake else 15,
            )
        except AccountGateError:
            jobs.update(job_id, status=JobStatus.FAILED, error="account gate")
            raise  # 欠费/未开通：重试无意义，交上层处理（与 m6 同一口径）
        except CapabilityMismatchError as e:
            # 撞官方硬规格（素材超上限/互斥混发）：重试结果相同，立即止损并留痕
            jobs.update(job_id, status=JobStatus.FAILED, error=str(e)[:200])
            storage.log_event(
                "m4_scene", "scene_capability_mismatch", group=gid, mode=mode,
                error=str(e)[:200],
                hint="参考素材与首尾帧官方互斥；超上限应在 plan_references 里裁掉",
            )
            storage.log_degradation(
                "scene", "scene_clip_capability", str(e)[:200],
                impact=f"场景 {gid} 未生成（素材形态不被接受），该组镜头退回逐镜头路径",
            )
            continue
        except ProviderError as e:
            jobs.update(job_id, status=JobStatus.FAILED, error=str(e)[:200])
            if is_protocol_error(str(e)):
                storage.log_event(
                    "m4_scene", "scene_protocol_mismatch", group=gid, error=str(e)[:200]
                )
                storage.log_degradation(
                    "scene", "scene_clip_protocol", str(e)[:200],
                    impact=f"场景 {gid} 的 media 协议被拒，退回逐镜头路径",
                )
                continue
            raise
        if status.status != "succeeded" or not dest.exists():
            jobs.update(job_id, status=JobStatus.FAILED, error=str(status.error or "")[:200])
            storage.log_event(
                "m4_scene", "scene_failed", group=gid, mode=mode,
                error=str(status.error or "poll failed")[:200],
            )
            storage.log_degradation(
                "scene", "scene_clip_failed", str(status.error or "poll failed")[:200],
                impact=f"场景 {gid} 未出片，该组镜头退回逐镜头路径",
            )
            continue

        secs, qsrc = billable_seconds(status.usage, float(duration))
        billed_res = actual_resolution_tier(status.usage) or str(cfg.i2v_resolution)
        has_dialogue = bool(speakers)
        ledger.record(
            "talk" if has_dialogue else "i2v",
            provider.name,
            price_key_for(provider, billed_res),
            quantity=secs,
            note=f"scene:{gid}",
            meta={
                "quantity_source": qsrc,
                "billed_seconds": round(secs, 3),
                "requested_duration": duration,
                "resolution_requested": cfg.i2v_resolution,
                "resolution_billed": billed_res,
                "usage": dict(status.usage or {}),
                "mode": mode,
                "shots": len(group.shot_ids),
                "n_ref_images": len(img_cids) + len(scene_sids),
                "n_ref_audios": len(audio_cids),
            },
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "group_id": gid,
                    "scene_ids": list(group.scene_ids),
                    "shot_ids": list(group.shot_ids),
                    "windows": {k: list(v) for k, v in group.windows.items()},
                    "mode": mode,
                    "mode_reason": why,
                    "duration_sec": duration,
                    "billed_seconds": round(secs, 3),
                    "resolution_billed": billed_res,
                    "ref_images": [f"图{i + 1}={c}" for i, c in enumerate(img_cids)]
                    + [f"图{len(img_cids) + i + 1}=scene:{s}" for i, s in enumerate(scene_sids)],
                    "ref_audios": [f"音频{i + 1}={c}" for i, c in enumerate(audio_cids)],
                    "dropped": {
                        "images": list(plan.dropped_images),
                        "audios": list(plan.dropped_audios),
                    },
                    "prompt_chars": len(prompt),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        jobs.update(job_id, status=JobStatus.SUCCEEDED, output_uri=str(dest))
        storage.record_regeneration(
            rel,
            [f"assets/char_{c}.png" for c in img_cids]
            + [f"assets/scene_{s}.png" for s in scene_sids]
            + [f"audio/_voiceref_{c}.wav" for c in audio_cids]
            + ["script.json"],
            reason="场景片段重生成",
        )
        storage.log_event(
            "m4_scene", "scene_clip_gen", group=gid, mode=mode, why=why,
            shots=len(group.shot_ids), duration=duration, billed=round(secs, 3),
            res=billed_res, refs=len(img_cids) + len(scene_sids), voices=len(audio_cids),
        )
        out[gid] = str(dest)
        prev_clip = dest
    return out
