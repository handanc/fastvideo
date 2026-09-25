"""m3 分镜生成（教学第 2 讲延伸：首帧锚定）。

每镜头生成首帧图（含角色固化外貌注入）；可选 TEXT 蒸馏（元提示词 → 纯画面描述）
与 VLM 一致性抽查（参考图 vs 首帧双图对比）。
"""

from __future__ import annotations

import random
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import Asset, Script, Shot
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ProviderError, T2IProvider, TextProvider, VLMProvider
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
) -> dict[str, str]:
    """返回 shot_id → 首帧图路径。

    text 非空时先经 LLM 蒸馏：visual_prompt.v1 是写给 LLM 的元提示词（"你是绘画
    提示词专家…要求：1-6…"），此前渲染结果**未经蒸馏直塞 t2i/i2v/s2v**——cinema 档
    prompt_extend=False 下模型收到指令全文。蒸馏失败退化原文（不阻塞）。
    """
    from fastvideo3.modules.m2_assets import size_for_aspect

    char_map = script.character_map()
    scene_map = {s.scene_id: s for s in script.scenes}
    frames: dict[str, str] = {}
    size = size_for_aspect(cfg.aspect_ratio, hi_res=cfg.resolution[0] >= 1080)

    for shot in script.all_shots():
        scene = scene_map.get(shot.scene_id)
        prompt = _shot_prompt(cfg, script, scene, shot, char_map, assets)
        prompt = await _distill_prompt(text, prompt, shot.shot_id, storage, ledger)
        # 硬锁脸：首帧以角色立绘为参考图条件生成（下游 i2v/口型继承同一张脸）
        ref = _char_ref(cfg, t2i, shot, assets)
        if ref is not None and shot.characters[0] in char_map:
            prompt = (
                f"画面中{char_map[shot.characters[0]].name}的脸、发型、服装必须与参考图完全一致。"
                + prompt
            )
        shot.visual_prompt = prompt
        dest = storage.path("storyboard", f"shot_{shot.shot_id}.png")
        # 同场景+同角色固定种子：脸/服装/色调跨镜头一致的关键工程手段
        used_name, used_model = await _retry_gen(
            t2i,
            prompt,
            dest,
            ledger,
            f"sb:{shot.shot_id}",
            size=size,
            negative_prompt=cfg.negative_prompt,
            seed=_stable_seed(shot),
            ref_image=ref,
            fallback=fallback,
            storage=storage,
        )
        frames[shot.shot_id] = str(dest)
        ledger.record("t2i", used_name, used_model, note=f"storyboard:{shot.shot_id}")

    # VLM 一致性抽查（含人物的镜头按比例抽）
    if vlm is not None and cfg.storyboard_sample_rate > 0:
        candidates = [s for s in script.all_shots() if s.characters]
        random.seed(42)  # 教学可复现
        sampled = (
            random.sample(candidates, k=max(1, int(len(candidates) * cfg.storyboard_sample_rate)))
            if candidates
            else []
        )
        for shot in sampled:
            char = char_map.get(shot.characters[0])
            ref_asset = assets.get(f"char_{shot.characters[0]}") if char else None
            if char and ref_asset:
                rubric = render("image_critic.v1", appearance=char.appearance)
                frame_path = storage.path("storyboard", f"shot_{shot.shot_id}.png")
                # rubric 承诺"对比【角色参考图】与【镜头首帧图】"——此前只送首帧一张，
                # 图图对比形同虚设；参考图在前、首帧在后（VLM 最多收 4 张）
                ref_path = Path(ref_asset.uri)
                media = [ref_path, frame_path] if ref_path.exists() else [frame_path]
                if len(media) == 1:
                    storage.log_event("m3_storyboard", "ref_missing_for_check", shot=shot.shot_id)
                result = await vlm.review(media, rubric=rubric)
                ledger.record("vlm", vlm.name, vlm.model, note=f"sb_check:{shot.shot_id}")
                storage.log_event(
                    "m3_storyboard",
                    "consistency_check",
                    shot=shot.shot_id,
                    passed=result.passed,
                )
                if not result.passed:
                    # 驳回 → 重生成该首帧（预算内，同种子保证与过审帧同族）
                    # size 此前漏传 → 落默认 1024² 方图，9:16 构图被 cover 裁毁且下游全继承
                    retry_name, retry_model = await _retry_gen(
                        t2i,
                        shot.visual_prompt + "。严格保持角色外貌一致。",
                        frame_path,
                        ledger,
                        f"sb_retry:{shot.shot_id}",
                        size=size,
                        negative_prompt=cfg.negative_prompt,
                        seed=_stable_seed(shot),
                        ref_image=_char_ref(cfg, t2i, shot, assets),
                        fallback=fallback,
                        storage=storage,
                    )
                    ledger.record("t2i", retry_name, retry_model, note=f"sb_retry:{shot.shot_id}")
                    # 复检一次：仍不过只记事件交人工/VLM 终审兜底，不再循环（防成本雪球）
                    recheck = await vlm.review(media, rubric=rubric)
                    ledger.record("vlm", vlm.name, vlm.model, note=f"sb_recheck:{shot.shot_id}")
                    storage.log_event(
                        "m3_storyboard",
                        "consistency_recheck",
                        shot=shot.shot_id,
                        passed=recheck.passed,
                    )

    storage.save_json("script.json", script.model_dump(mode="json"))  # 回写 visual_prompt
    storage.log_event("m3_storyboard", "done", frames=len(frames))
    return frames


def _char_ref(cfg, t2i, shot: Shot, assets) -> Path | None:
    """角色首帧参考图（硬锁脸开关开启且供应商支持参考图时）。"""
    if not (cfg.use_reference_keyframes and getattr(t2i, "supports_reference_image", False)):
        return None
    if not shot.characters:
        return None
    asset = assets.get(f"char_{shot.characters[0]}")
    return Path(asset.uri) if asset else None


def _stable_seed(shot: Shot) -> int:
    """场景+角色组合 → 确定性种子。同组合跨镜头复用，脸/色调保持一致。"""
    import hashlib

    key = f"{shot.scene_id}|{','.join(sorted(shot.characters))}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


async def _distill_prompt(
    text: TextProvider | None,
    rendered: str,
    shot_id: str,
    storage: RunStorage,
    ledger: CostLedger,
) -> str:
    """元提示词 → 纯画面描述（≤150 字）。失败/为空退化原文，绝不阻塞管线。"""
    if text is None:
        return rendered
    try:
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
        ledger.record("text", text.name, text.model, note=f"distill:{shot_id}")
        return distilled
    except Exception as e:  # noqa: BLE001
        storage.log_event(
            "m3_storyboard", "distill_fallback", shot=shot_id, error=str(e)[:200]
        )
        return rendered


def _shot_prompt(cfg, script, scene, shot: Shot, char_map, assets) -> str:
    chars_desc = (
        "\n".join(
            f"- {char_map[c].name}: {char_map[c].appearance}"
            for c in shot.characters
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
) -> tuple[str, str]:
    """生成首帧（主供应商重试耗尽后走备选）。返回实际用的 (provider_name, model) 供记账。"""
    last: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await t2i.generate(
                prompt,
                dest=dest,
                size=size,
                negative_prompt=negative_prompt,
                seed=seed,
                ref_image=ref_image,
            )
            return t2i.name, t2i.model
        except Exception as e:  # noqa: BLE001
            last = e
    if fallback is not None:
        # 备选供应商保交付（最小公共参数集：FLUX 无 seed/ref_image，多传的 kwargs 会被吸收）
        try:
            await fallback.generate(prompt, dest=dest, size=size, negative_prompt=negative_prompt)
            if storage is not None:
                storage.log_event("m3_storyboard", "fallback_used", tag=tag, provider=fallback.name)
            return fallback.name, fallback.model
        except Exception as e:  # noqa: BLE001
            last = e
    raise ProviderError(f"分镜生成失败: {tag}: {last}")
