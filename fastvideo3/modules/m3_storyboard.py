"""m3 分镜生成（教学第 2 讲延伸：首帧锚定）。

每镜头生成首帧图（含角色固化外貌注入）；可选 TEXT 蒸馏（元提示词 → 纯画面描述）
与 VLM 一致性抽查（参考图 vs 首帧双图对比）。

一致性数据链路（2026-10 改造）：
- 参考图条件按能力位路由，降级必落 `ref_condition_dropped` / `fallback_used{dropped}`；
- 种子作用域可配（`keyframe_seed_scope`），默认 cast = 同角色跨场景同种子；
- `shot.visual_prompt` 只存纯画面文本，"必须与参考图一致"的锁定语料只进本次
  t2i 请求（此前写进 visual_prompt 会被 m6 原样传给口型模型，属无效指令）；
- 每次首帧生成落 `keyframe_gen` 事件 + 台账 meta，事后只看盘就能重建请求内容。
"""

from __future__ import annotations

import random
from pathlib import Path

from fastvideo3.core import pause
from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger, llm_meta
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import Asset, Character, Script, Shot, effective_cast
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import (
    ProviderError,
    T2IProvider,
    TextProvider,
    VLMProvider,
    filter_kwargs_by_capability,
    ref_paths,
)
from fastvideo3.providers.deepseek import strip_code_fence

_MAX_RETRY = 2

_DISTILL_SYS = (
    "你是文生图提示词蒸馏器。把用户给出的镜头需求改写为一段 150 字以内的纯画面描述："
    "只描写可见画面（主体及其外貌逐字保留、景别、机位角度、主光源方向与色温、色调、"
    "真人实拍电影剧照质感），禁止动漫/卡通/插画/3D 渲染风格，画面内无任何文字水印。"
    "输出不要出现「要求」「你是」「提示词」等指令性文字，不分点、不解释、不加引号。"
)


async def build_storyboard(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    assets: dict[str, Asset],
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider | None = None,
    text: TextProvider | None = None,
    fallback: T2IProvider | None = None,
    only: set[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, str]:
    """返回 shot_id → 首帧图路径。

    text 非空时先经 LLM 蒸馏：visual_prompt.v1 是写给 LLM 的元提示词（"你是绘画
    提示词专家…要求：1-6…"），此前渲染结果**未经蒸馏直塞 t2i/i2v/s2v**——cinema 档
    prompt_extend=False 下模型收到指令全文。蒸馏失败退化原文（不阻塞）。

    分步生成参数：`only` = 只重做这些 shot；`skip_existing` = 在盘且未被血缘标 stale
    的首帧直接复用（跳过判断在蒸馏**之前**，否则被跳过的镜头仍烧 TEXT token）。
    """
    from fastvideo3.core.schema import script_fingerprint
    from fastvideo3.modules.m2_assets import size_for_aspect

    # 硬门控下沉到模块层：把"参考图锁脸"当交付承诺的档（high/cinema），主供应商
    # 不支持参考图就必须在**花钱之前**报错。放在这里而不是图节点里，是因为
    # cli stage / step / API 三条入口都会调本函数 —— 门控在节点层就会被绕过。
    if cfg.use_reference_keyframes and cfg.require_reference_keyframes:
        if not getattr(t2i, "supports_reference_image", False):
            raise ProviderError(
                f"require_reference_keyframes 已开启，但 t2i 主供应商 {t2i.name}"
                f"({t2i.model}) 不支持参考图。请把 configs 的 providers.t2i.default "
                f"改为 dashscope_wan27_image，或关闭 require_reference_keyframes。"
            )

    char_map = script.character_map()
    scene_map = {s.scene_id: s for s in script.scenes}
    frames: dict[str, str] = {}
    size = size_for_aspect(cfg.aspect_ratio, hi_res=cfg.resolution[0] >= 1080)
    fp = f"script#{script_fingerprint(script)}"
    uploaded: dict = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}

    char_shots = [s for s in script.all_shots() if s.characters]
    # 一致性抽查驳回时要按**该镜头自己的**参考图/种子重画，所以生成时必须把上下文留下
    # （旧实现复用生成循环结束后的循环变量 → 抽查重画拿的是"最后一个镜头"的参考图与种子，
    #  等于把别人的脸塞进被驳回的镜头）。
    gen_ctx: dict[str, dict] = {}
    if char_shots and not cfg.use_reference_keyframes:
        # 整档未开锁脸：记一条 run 级事件（逐镜 spam 无意义），economy 教学档可见但不吵
        storage.log_event(
            "m3_storyboard",
            "ref_condition_dropped",
            reason="flag_off",
            shots=len(char_shots),
            hint="该档首帧不带角色参考图，一致性仅靠文本复述+种子",
        )

    for shot in script.all_shots():
        scene = scene_map.get(shot.scene_id)
        dest = storage.path("storyboard", f"shot_{shot.shot_id}.png")
        rel = f"storyboard/shot_{shot.shot_id}.png"
        in_only = only is None or shot.shot_id in only
        if dest.exists() and in_only:
            is_upload = any(k == f"shot_{shot.shot_id}.png" or k.endswith(rel) for k in uploaded)
            if is_upload and not storage.is_stale(rel):
                frames[shot.shot_id] = str(dest)  # 人工上传的首帧不覆盖
                storage.log_event("m3_storyboard", "reuse_uploaded", shot=shot.shot_id)
                continue
            if skip_existing and not storage.is_stale(rel):
                frames[shot.shot_id] = str(dest)
                storage.log_event("m3_storyboard", "reuse_existing", shot=shot.shot_id)
                continue
            if is_upload:
                storage.log_event(
                    "m3_storyboard", "overwrite_uploaded", shot=shot.shot_id,
                    hint="上游（定妆照/场景图/剧本）已变更，人工首帧被血缘标 stale 后重做",
                )
        if not in_only:
            if dest.exists():
                frames[shot.shot_id] = str(dest)
            continue  # 不在本轮补齐名单：不重画也不报错
        # 画面里有谁：`characters` 漏标时按台词说话人/画面点名补齐（见 effective_cast）
        cast, implied = effective_cast(script, shot)
        if pause.stop_for_budget(storage, ledger, node="m3_storyboard", unit=shot.shot_id):
            break  # 成本到线：剩余首帧不画了，等人工确认（已画的帧一个不动）
        prompt = _shot_prompt(cfg, script, scene, shot, char_map, assets, cast)
        tokens = _canonical_tokens(char_map, shot, cast)
        prompt = await _distill_prompt(text, prompt, shot.shot_id, storage, ledger, tokens)
        # 硬锁脸：首帧以角色立绘为参考图条件生成（下游 i2v/口型继承同一张脸）
        refs, ref_cids = _char_refs(cfg, t2i, cast, assets, shot, storage)
        if implied:
            # 漏标必须可见：它决定"这张脸是谁给的"。槽位不够时也说清谁没锚上。
            storage.log_event(
                "m3_storyboard",
                "cast_implied",
                shot=shot.shot_id,
                declared=list(shot.characters),
                implied=[char_map[c].name for c in implied if c in char_map],
                anchored=[char_map[c].name for c in ref_cids if c in char_map],
                unanchored=[
                    char_map[c].name for c in cast if c not in ref_cids and c in char_map
                ],
                hint="在场漏标但台词/画面点名 → 已按优先级补进提示词与参考图槽位；"
                "未锚上的人只有文字外貌可依（放开 ref_max_chars 需多图协议探单通过）",
            )
        scene_ref = _scene_ref(cfg, cast, shot, assets, storage)
        all_refs = refs + ([scene_ref] if scene_ref else [])
        lockface = _lockface_prefix(ref_cids, scene_ref, char_map)
        shot.visual_prompt = prompt  # 纯画面文本：锁定语料不写回，避免泄漏进 m6 口型 prompt
        seed = _stable_seed(shot, cfg.keyframe_seed_scope, cast)
        # 同场景+同角色固定种子：脸/服装/色调跨镜头一致的关键工程手段
        used_name, used_model, meta = await _retry_gen(
            t2i,
            lockface + prompt,
            dest,
            ledger,
            f"sb:{shot.shot_id}",
            size=size,
            negative_prompt=cfg.negative_prompt,
            seed=seed,
            ref_image=all_refs or None,
            fallback=fallback,
            storage=storage,
        )
        frames[shot.shot_id] = str(dest)
        ref_ids = [f"char_{c}" for c in ref_cids]  # 与 refs 一一对应（不是"声明的前 N 个"）
        scene_rel = f"assets/scene_{shot.scene_id}.png" if scene_ref else None
        gen_ctx[shot.shot_id] = {
            "refs": refs,
            "ref_cids": ref_cids,
            "cast": cast,
            "scene_ref": scene_ref,
            "all_refs": all_refs,
            "seed": seed,
        }
        ledger.record(
            "t2i",
            used_name,
            used_model,
            note=f"storyboard:{shot.shot_id}",
            meta={
                "route": "keyframe",
                "seed": seed,
                "seed_scope": cfg.keyframe_seed_scope,
                "ref_count": len(all_refs),
                "ref_ids": ref_ids,
                "scene_ref": scene_rel,
                **meta,
            },
        )
        storage.log_event(
            "m3_storyboard",
            "keyframe_gen",
            shot=shot.shot_id,
            provider=used_name,
            model=used_model,
            seed=seed,
            seed_scope=cfg.keyframe_seed_scope,
            ref_count=len(all_refs),
            ref_ids=ref_ids,
            scene_ref=scene_rel,
            scene_ref_mode=cfg.scene_ref_mode,
        )
        # 血缘：帧 ← 角色立绘×N + 场景图 + 剧本指纹。改其中任何一个 → 只作废消费它的镜头
        # 上游用**实际送出去的那几张图**（ref_cids），不是"声明的前 N 个"：漏标的角色
        # 画进了参考图，换掉它就必须作废这一帧，否则新女王配旧脸。
        storage.record_regeneration(
            rel,
            [f"assets/char_{c}.png" for c in ref_cids]
            + ([scene_rel] if scene_rel else [])
            + [fp],
            reason="首帧重生成",
        )

    # ── 尾帧（首尾帧生视频用）：给"要按首尾帧出片"的镜头产 ──────────────────
    # 判据是 `talkmode.shots_need_last_frame`（= 需要运动片段的镜头 ∪ C 链的对白镜头）：
    # C 链的对白镜头**不出**运动片段（出了会被 m7 弃用，白烧钱），但它的画面正是口型
    # 通道一次调用直出的，所以尾帧对它同样有用。A 链（吃图片+音频）不发 last_frame，
    # 为它的对白镜头产尾帧才是白烧 —— 判据里按通道能力区分，不是一刀切。
    # 尾帧失败/缺失一律退回单首帧并留痕，绝不阻断交付。
    if cfg.use_last_frames:
        await _build_last_frames(
            cfg, t2i, script, assets, storage, ledger, frames,
            fallback=fallback, size=size, fp=fp, only=only, uploaded=uploaded,
        )

    # VLM 一致性抽查（含人物的镜头按比例抽）
    if vlm is not None and cfg.storyboard_sample_rate > 0:
        random.seed(42)  # 教学可复现
        sampled = (
            random.sample(char_shots, k=max(1, int(len(char_shots) * cfg.storyboard_sample_rate)))
            if char_shots
            else []
        )
        def _ctx_for(sh: Shot) -> dict:
            """该镜头生成时真正用了哪几张参考图、哪个种子（复用帧则安静地重算一遍）。"""
            got = gen_ctx.get(sh.shot_id)
            if got:
                return got
            cast, _imp = effective_cast(script, sh)
            refs, cids = _char_refs(cfg, t2i, cast, assets, sh)
            sref = _scene_ref(cfg, cast, sh, assets)
            return {
                "ref_cids": cids,
                "cast": cast,
                "scene_ref": sref,
                "all_refs": refs + ([sref] if sref else []),
                "seed": _stable_seed(sh, cfg.keyframe_seed_scope, cast),
            }

        for shot in sampled:
            g = _ctx_for(shot)
            # 评审的对照对象 = 有效在场里第一个有立绘的人。锁脸关着的档（economy）
            # ref_cids 是空的，但抽查照样要做 —— 它查的是"画出来的人像不像定妆照"，
            # 与"生成时有没有喂参考图"是两件事，绝不能因为前者关掉就让后者静默消失。
            anchor = next((c for c in g["cast"] if f"char_{c}" in assets), None)
            char = char_map.get(anchor) if anchor else None
            ref_asset = assets.get(f"char_{anchor}") if anchor else None
            if char and ref_asset:
                rubric = render("image_critic.v1", appearance=char.appearance)
                frame_path = storage.path("storyboard", f"shot_{shot.shot_id}.png")
                # rubric 承诺"对比【角色参考图】与【镜头首帧图】"——此前只送首帧一张，
                # 图图对比形同虚设；参考图在前、首帧在后（VLM 最多收 4 张）
                ref_path = Path(ref_asset.uri)
                media = [ref_path, frame_path] if ref_path.exists() else [frame_path]
                if len(media) == 1:
                    storage.log_event("m3_storyboard", "ref_missing_for_check", shot=shot.shot_id)
                try:
                    result = await vlm.review(media, rubric=rubric)
                except ProviderError as e:
                    # 裁判抽风（ReviewParseError：回 JSON 缺字段 / 限流 / 审核）**不得炸掉
                    # 已经付费的步骤**：实测一次 qwen-vl-max 只回 passed+feedback 缺 score，
                    # 整个 storyboard 步骤在已出帧之后 abort，用户再点一次又付一遍。
                    # 抽查是质量增益不是前置条件：跳过它、留痕、交终审与人工。
                    storage.log_degradation(
                        "storyboard", "consistency_check_unavailable", str(e),
                        impact=f"镜头 {shot.shot_id} 首帧一致性抽查跳过（交终审/人工看片）",
                    )
                    continue
                ledger.record(
                    "vlm",
                    vlm.name,
                    vlm.model,
                    note=f"sb_check:{shot.shot_id}",
                    meta=llm_meta(vlm, purpose="consistency_check", n_media=len(media),
                                  verified=result.verified),
                )
                storage.log_event(
                    "m3_storyboard",
                    "consistency_check",
                    shot=shot.shot_id,
                    passed=result.passed,
                    verified=result.verified,
                )
                if not result.passed:
                    # 驳回 → 重生成该首帧（预算内，同种子保证与过审帧同族）
                    retry_name, retry_model, _ = await _retry_gen(
                        t2i,
                        _lockface_prefix(g["ref_cids"], g["scene_ref"], char_map)
                        + shot.visual_prompt
                        + "。严格保持角色外貌一致。",
                        frame_path,
                        ledger,
                        f"sb_retry:{shot.shot_id}",
                        size=size,
                        negative_prompt=cfg.negative_prompt,
                        seed=g["seed"],
                        ref_image=g["all_refs"] or None,
                        fallback=fallback,
                        storage=storage,
                    )
                    ledger.record(
                        "t2i", retry_name, retry_model, note=f"sb_retry:{shot.shot_id}"
                    )
                    # 复检一次：仍不过只记事件交人工/VLM 终审兜底，不再循环（防成本雪球）
                    try:
                        recheck = await vlm.review(media, rubric=rubric)
                    except ProviderError as e:
                        # 复检失败同理：重拍已经花了钱，裁判再抽风不该让整步 abort；
                        # 留痕交终审，不循环重试（防成本雪球）。
                        storage.log_degradation(
                            "storyboard", "consistency_recheck_unavailable", str(e),
                            impact=f"镜头 {shot.shot_id} 重拍后复检跳过（交终审/人工看片）",
                        )
                        continue
                    ledger.record(
                        "vlm",
                        vlm.name,
                        vlm.model,
                        note=f"sb_recheck:{shot.shot_id}",
                        meta=llm_meta(vlm, purpose="consistency_recheck", n_media=len(media),
                                      verified=recheck.verified),
                    )
                    storage.log_event(
                        "m3_storyboard",
                        "consistency_recheck",
                        shot=shot.shot_id,
                        passed=recheck.passed,
                        verified=recheck.verified,
                    )

    storage.save_json("script.json", script.model_dump(mode="json"))  # 回写 visual_prompt
    storage.log_event("m3_storyboard", "done", frames=len(frames))
    return frames


def _canonical_tokens(
    char_map: dict[str, Character], shot: Shot, cast: list[str] | None = None
) -> list[str]:
    """该镜头角色必须逐字保留的词表（来自 m1 结构化外貌卡；老剧本为空即不校验）。

    cast = 有效在场（含漏标补进来的角色）：他们的关键外貌词同样不许被蒸馏蒸发。
    """
    out: list[str] = []
    for cid in cast if cast is not None else shot.characters:
        ch = char_map.get(cid)
        if ch is not None:
            out.extend(ch.canonical_tokens)
    return out


def _scene_ref(
    cfg: AppConfig,
    cast: list[str],
    shot: Shot,
    assets: dict[str, Asset],
    storage: RunStorage | None = None,
) -> Path | None:
    """场景概念图作为第二路参考（场景连续性）。

    `scene_ref_mode`：off = 不用；scenery_only（默认）= 只给**无角色**的空镜/转场用
    （零身份风险，且正是场景连续性最有价值的地方）；all = 含角色的镜头也带。
    把场景图与身份参考混进同一次请求有稀释人脸条件的真实风险，所以默认保守，
    且 `all` 必须由 m9 度量证明人脸达标率不降后才允许开（见 P0-B 第 4 条）。

    判"有没有人"用**有效在场**（`cast`）而不是 `shot.characters`：漏标但被台词/画面
    点名的人也算人 —— 否则他的立绘与场景图会同时挤进一次请求，人脸条件被稀释成两张
    参考图抢权重。
    """
    if cfg.scene_ref_mode == "off":
        return None
    if cfg.scene_ref_mode == "scenery_only" and cast:
        return None
    asset = assets.get(f"scene_{shot.scene_id}")
    if asset is None or not Path(asset.uri).exists():
        if storage is not None:
            storage.log_event(
                "m3_storyboard", "scene_ref_missing", shot=shot.shot_id,
                hint="场景概念图缺失，该镜头少一路场景参考（不静默）",
            )
        return None
    return Path(asset.uri)


def _lockface_prefix(
    ref_cids: list[str],
    scene_ref: Path | None,
    char_map: dict[str, Character],
) -> str:
    """参考图锁定语料：只进本次 t2i 请求，不写回 shot.visual_prompt。

    图序与 `messages.content` 的 append 顺序严格一致：先人物（1..N）后场景（N+1）。
    场景图只约束背景/光线/空间，**身份以人物参考图为准** —— 这条必须写进语料，
    否则模型可能把场景图当主体依据、稀释人脸条件。

    入参是**实际送出去的那几张图的 character_id**（与 `refs` 成对），不是"声明的前 N 个"：
    漏标的角色补进参考图后，"图1是谁"必须跟着变，否则等于把张三的脸标注成李四。
    """
    names = [char_map[c].name for c in ref_cids if c in char_map]
    if not names and scene_ref is None:
        return ""
    if not names:
        return "背景、色温、光源方向与空间纵深必须与场景参考图一致。"
    if len(names) == 1:
        head = f"画面中{names[0]}的脸、发型、服装必须与图1完全一致。"
    else:
        head = "、".join(f"图{i + 1}是{n}的脸/发型/服装" for i, n in enumerate(names)) + \
            "，必须与对应参考图完全一致。"
    if scene_ref is None:
        return head
    return (
        head
        + f"图{len(names) + 1}是场景与光线依据：人物身份以图1"
        + ("、图2" if len(names) > 1 else "")
        + "为准，只让背景、色温、光源方向、空间纵深与场景图一致。"
    )


def _char_refs(
    cfg: AppConfig,
    t2i: T2IProvider,
    cast: list[str],
    assets: dict[str, Asset],
    shot: Shot,
    storage: RunStorage | None = None,
) -> tuple[list[Path], list[str]]:
    """角色首帧参考图（硬锁脸）。开关开启且供应商支持时按 ref_max_chars 取多张。

    返回 `(图片, 对应的 character_id)`，**两个列表严格一一对应**。旧实现只回图片，
    调用方按 `shot.characters[:len(refs)]` 反推"图1是谁"：某角色资产缺失被 `continue`
    跳过后，refs[0] 其实是第二个角色的立绘，而锁定语料与血缘仍标成第一个角色 ——
    图与名字张冠李戴，换定妆照时作废的也是不相干的那个文件。

    开关开了但条件没凑齐 = 一致性静默失效的高发区，必须按原因落事件。
    """
    if not cast or not cfg.use_reference_keyframes:
        return [], []
    if not getattr(t2i, "supports_reference_image", False):
        if storage is not None:
            storage.log_event(
                "m3_storyboard",
                "ref_condition_dropped",
                reason="provider_no_capability",
                shot=shot.shot_id,
                provider=t2i.name,
                hint="改用 providers.t2i.default: dashscope_wan27_image",
            )
        return [], []
    refs: list[Path] = []
    ids: list[str] = []
    for cid in cast:
        if len(refs) >= max(1, cfg.ref_max_chars):
            break
        asset = assets.get(f"char_{cid}")
        if asset is None:
            if storage is not None:
                storage.log_event(
                    "m3_storyboard",
                    "ref_condition_dropped",
                    reason="asset_missing",
                    shot=shot.shot_id,
                    char=cid,
                )
            continue
        refs.append(Path(asset.uri))
        ids.append(cid)
    return refs, ids


def _stable_seed(shot: Shot, scope: str = "cast", cast: list[str] | None = None) -> int:
    """确定性种子，作用域可配。

    - cast：仅角色组合 → 同角色跨场景同种子（锁脸）。旧实现把 scene_id 编进 key，
      角色一换场景就换采样起点，实测主角 4 个场景出 4 张脸。
    - scene_cast：旧行为，保留作消融对照臂。
    - shot：每镜独立种子，作"构图趋同"的对照臂。

    cast 传**有效在场**：漏标补进来的人同样参与分组，否则两个"characters 都是空"的
    镜头（一个是女王、一个是四人全景）会拿到同一个种子。
    """
    import hashlib

    chars = ",".join(sorted(cast if cast is not None else shot.characters))
    if scope == "scene_cast":
        key = f"{shot.scene_id}|{chars}"
    elif scope == "shot":
        key = f"{shot.scene_id}|{chars}|{shot.shot_id}"
    else:
        key = chars
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


async def _distill_prompt(
    text: TextProvider | None,
    rendered: str,
    shot_id: str,
    storage: RunStorage,
    ledger: CostLedger,
    tokens: list[str] | None = None,
) -> str:
    """元提示词 → 纯画面描述（≤150 字）。失败/为空退化原文，绝不阻塞管线。

    tokens 非空时做出参校验：关键外貌词被蒸馏蒸发（实测"黄色头盔"消失）则带补丁
    重蒸一次，仍缺则字符串追加回 prompt 并落事件 —— 响亮 + 自愈。
    """
    if text is None:
        return rendered
    tokens = tokens or []
    try:
        distilled = await _distill_once(text, rendered)
        missing = [t for t in tokens if t and t not in distilled]
        if missing:
            patch = (
                rendered
                + f"\n以下词必须原样出现在输出中，不得改写或省略：{'、'.join(missing)}"
            )
            try:
                second = await _distill_once(text, patch)
                if all(t in second for t in missing):
                    distilled = second
                    missing = []
                else:
                    missing = [t for t in missing if t not in second]
                    distilled = second
            except Exception:  # noqa: BLE001  # 二次蒸馏失败用一次的结果继续
                pass
        if missing:
            distilled = distilled + "。" + "、".join(missing)
            storage.log_event(
                "m3_storyboard", "distill_attr_lost", shot=shot_id, missing=missing
            )
        ledger.record(
            "text", text.name, text.model, note=f"distill:{shot_id}", meta=llm_meta(text)
        )
        return distilled
    except Exception as e:  # noqa: BLE001
        storage.log_event(
            "m3_storyboard", "distill_fallback", shot=shot_id, error=str(e)[:200]
        )
        return rendered


async def _distill_once(text: TextProvider, rendered: str) -> str:
    raw = await text.chat(
        [
            {"role": "system", "content": _DISTILL_SYS},
            {"role": "user", "content": rendered},
        ],
        temperature=0.2,
    )
    distilled = strip_code_fence(raw).strip().strip('"“”')
    if not distilled:
        raise ValueError("空蒸馏结果")
    return distilled


def _shot_prompt(
    cfg, script, scene, shot: Shot, char_map, assets, cast: list[str] | None = None
) -> str:
    """首帧元提示词。cast = 有效在场（默认退回声明的 `characters`）。

    外貌逐字注入是"文本侧的锁脸"，与参考图互补：漏标的人只要出现在 action 里，
    他的外貌就该出现在提示词里 —— 少一个人描述，模型就替他自由发挥一次。
    """
    chars_desc = (
        "\n".join(
            f"- {char_map[c].name}: {char_map[c].appearance}"
            for c in (cast if cast is not None else shot.characters)
            if c in char_map
        )
        or "- 无人物（空镜）"
    )
    location = scene.location if scene else "未知场景"
    tod = scene.time_of_day if scene else "日"
    return render(
        "visual_prompt.v1",
        style=script.base_info.style,
        style_suffix=cfg.style_suffix,
        location=location,
        time_of_day=tod,
        characters=chars_desc,
        action=shot.action,
        camera=shot.camera or "中景固定镜头",
    )


def _last_frame_prompt(cfg: AppConfig, shot: Shot) -> str:
    """尾帧提示词：**同一场景/人物/服装/光线/画幅**，只把动作推进到结束状态。

    以首帧**已蒸馏好**的 `shot.visual_prompt` 为底，不再走一次 LLM 蒸馏 —— 同一段描述
    蒸馏两次会得到两份措辞，等于亲手给模型换脸换衣服的理由（首尾帧必须是同一个人、
    同一个场景，否则 i2v 会在两张不相干的图之间硬插值）。前面只钉一句"结束状态"。
    """
    base = (shot.visual_prompt or "").strip()
    head = (
        "【本镜头结束时的定格画面】与首帧同一场景、同一人物、同一服装配色、同一光线与画幅；"
        "动作已推进到结束状态：人物停在这段动作完成后的姿态与位置，表情是动作结束时的自然神情，"
        "构图延续首帧、只按运镜自然变化。不要出现新人物，不要改变场景与时间。"
    )
    tail = f" {cfg.style_suffix}".strip() if cfg.style_suffix else ""
    return f"{head}{base}{tail}".strip()


async def _build_last_frames(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    assets: dict[str, Asset],
    storage: RunStorage,
    ledger: CostLedger,
    frames: dict[str, str],
    *,
    fallback: T2IProvider | None = None,
    size: str = "1024*1024",
    fp: str = "",
    only: set[str] | None = None,
    uploaded: dict | None = None,
) -> dict[str, str]:
    """给需要运动片段的镜头生成「动作结束态」尾帧（`storyboard/shot_{id}_last.png`）。

    与首帧**同一套上下文**：同 size（官方要求首尾帧宽高比一致，1:4~4:1）、同参考图
    （锁脸）、同种子（同族采样起点）。任何一步失败都只记事件、不影响首帧与交付 ——
    m4 拿不到尾帧就退回单首帧，那条路是改造前的既有行为。

    为什么放在首帧循环**之后**单独一趟：首帧循环里有三条 `continue`（人工上传复用、
    在盘复用、不在本轮名单），把尾帧塞进去就要在三处各写一遍；单独一趟还能覆盖
    "首帧是复用的、但尾帧还没生成过"这种续跑场景。
    """
    from fastvideo3.core.talkmode import shots_need_last_frame

    uploaded = uploaded or {}
    char_map = script.character_map()
    # 工作集 = 需要运动片段的镜头 ∪ C 链里"画面由口型通道自己出"的对白镜头。
    # 只用 shots_need_i2v 会漏掉后者，而短剧里对白镜头是大头 —— 于是"首尾帧改造"
    # 看着没生效：真正该锁结束态的镜头一个尾帧都没有。
    need = {s.shot_id for s in shots_need_last_frame(script, cfg.talk_mode, cfg.scene_clip_mode)}
    out: dict[str, str] = {}
    # 尾帧的"该不该做"由**尾帧自己**的状态决定，不能被 `only` 卡住。
    # `only` 是 frames 步按**首帧**缺口算出来的续跑名单：一旦某镜头的首帧在早前某次
    # 尝试里已经落盘（而那次尝试在尾帧这趟之前就因欠费/审核中止了），它就再也不会
    # 出现在 only 里 → 尾帧永远补不上，m4 静默退回单首帧。2026-10-12 那条 run 就是
    # 这个形状：8 个首帧在盘、尾帧只有 1 个，`last_frames_done count=1` 连着报了 4 次，
    # 全是复用计数，用户看到的就是"还是只按首帧生视频"。
    skipped_no_first: list[str] = []
    for shot in script.all_shots():
        if shot.shot_id not in need:
            continue
        dest = storage.path("storyboard", f"shot_{shot.shot_id}_last.png")
        rel = f"storyboard/shot_{shot.shot_id}_last.png"
        want_it = not dest.exists() or storage.is_stale(rel)
        if only is not None and shot.shot_id not in only and not want_it:
            out[shot.shot_id] = str(dest)
            continue
        first = frames.get(shot.shot_id)
        if not first or not Path(first).exists():
            # 首帧都没有就谈不上尾帧；首帧缺失本身已是 frames 步的缺口，不在这里重复报
            skipped_no_first.append(shot.shot_id)
            continue
        is_upload = any(
            k == f"shot_{shot.shot_id}_last.png" or k.endswith(rel) for k in uploaded
        )
        if dest.exists() and not storage.is_stale(rel):
            out[shot.shot_id] = str(dest)
            storage.log_event(
                "m3_storyboard",
                "reuse_uploaded" if is_upload else "reuse_existing",
                shot=shot.shot_id,
                artifact="last_frame",
            )
            continue
        if pause.stop_for_budget(storage, ledger, node="m3_storyboard", unit=f"{shot.shot_id}:last"):
            break  # 与首帧同一口径：到线就停手问人，不再逐条静默跳过
        try:
            cast, _implied = effective_cast(script, shot)
            refs, ref_cids = _char_refs(cfg, t2i, cast, assets, shot, storage)
            scene_ref = _scene_ref(cfg, cast, shot, assets, storage)
            all_refs = refs + ([scene_ref] if scene_ref else [])
            lockface = _lockface_prefix(ref_cids, scene_ref, char_map)
            seed = _stable_seed(shot, cfg.keyframe_seed_scope, cast)
            prompt = _last_frame_prompt(cfg, shot)
            used_name, used_model, meta = await _retry_gen(
                t2i,
                lockface + prompt,
                dest,
                ledger,
                f"sbl:{shot.shot_id}",
                size=size,
                negative_prompt=cfg.negative_prompt,
                seed=seed,
                ref_image=all_refs or None,
                fallback=fallback,
                storage=storage,
            )
        except Exception as e:  # noqa: BLE001  # 尾帧是增强项，失败退回单首帧，不阻断交付
            storage.log_event(
                "m3_storyboard", "last_frame_failed", shot=shot.shot_id, error=str(e)[:200],
                hint="退回单首帧生视频（改造前的既有行为），不影响交付",
            )
            continue
        if not dest.exists():
            continue
        out[shot.shot_id] = str(dest)
        scene_rel = f"assets/scene_{shot.scene_id}.png" if scene_ref else None
        ledger.record(
            "t2i", used_name, used_model,
            note=f"storyboard_last:{shot.shot_id}",
            meta={
                "route": "last_frame",
                "seed": seed,
                "ref_count": len(all_refs),
                "ref_ids": [f"char_{c}" for c in ref_cids],
                **meta,
            },
        )
        storage.log_event(
            "m3_storyboard", "last_frame_gen", shot=shot.shot_id,
            provider=used_name, model=used_model, seed=seed,
            ref_count=len(all_refs),
            hint="首尾帧生视频：锁住镜头的结束状态，避免末尾姿态/构图乱飘",
        )
        # 血缘：尾帧与首帧同源（立绘/场景图/剧本指纹）→ 换其中任何一个，两张帧都该重做
        storage.record_regeneration(
            rel,
            [f"assets/char_{c}.png" for c in ref_cids]
            + ([scene_rel] if scene_rel else [])
            + [fp],
            reason="尾帧重生成",
        )
    # ── 覆盖率必须报出来：尾帧缺失会让 m4 静默退回单首帧，而"单首帧"正是改造前
    #    那条质量不高的老路。用户看不出区别，只会觉得"改了没用"。
    missing = sorted(need - set(out) - set(skipped_no_first))
    storage.log_event(
        "m3_storyboard",
        "last_frames_coverage",
        need=len(need),
        have=len(out),
        missing=missing,
        no_first_frame=sorted(skipped_no_first),
        hint="have/need = 能按首尾帧生视频的镜头数；缺的会退回单首帧",
    )
    if missing:
        storage.log_degradation(
            "storyboard",
            "last_frame_missing",
            f"缺尾帧的镜头：{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}",
            impact=(
                f"{len(missing)}/{len(need)} 个镜头将退回**单首帧**生视频"
                "（末尾姿态/构图容易漂）。补法：重跑「分镜首帧」这一步 —— "
                "尾帧按自己的缺口补，不需要重出首帧。"
            ),
        )
    if out:
        storage.log_event("m3_storyboard", "last_frames_done", count=len(out))
    return out


async def _retry_gen(
    t2i: T2IProvider,
    prompt: str,
    dest,
    ledger: CostLedger,
    tag: str,
    size: str = "1024*1024",
    negative_prompt: str = "",
    seed: int | None = None,
    ref_image=None,
    fallback: T2IProvider | None = None,
    storage: RunStorage | None = None,
) -> tuple[str, str, dict]:
    """生成首帧（主供应商重试耗尽后走备选）。返回 (provider, model, meta)。

    fallback 分支按备选供应商的能力位过滤 seed/ref_image，被过滤项进事件与 meta ——
    此前一律丢弃且无痕迹，"锁脸失效"因此不可见。
    """
    kwargs: dict = {
        "size": size,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "ref_image": ref_image,
    }
    last: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await t2i.generate(prompt, dest=dest, **kwargs)
            return t2i.name, t2i.model, {"dropped": []}
        except Exception as e:  # noqa: BLE001
            last = e
    if fallback is not None:
        fb_kwargs, dropped = filter_kwargs_by_capability(fallback, kwargs)
        try:
            await fallback.generate(prompt, dest=dest, **fb_kwargs)
            if storage is not None:
                storage.log_event(
                    "m3_storyboard",
                    "fallback_used",
                    tag=tag,
                    provider=fallback.name,
                    dropped=dropped,
                    seed_value=seed,
                    ref_count=len(ref_paths(ref_image)),
                )
            return fallback.name, fallback.model, {"dropped": dropped, "fallback": True}
        except Exception as e:  # noqa: BLE001
            last = e
    raise ProviderError(f"分镜生成失败: {tag}: {last}")
