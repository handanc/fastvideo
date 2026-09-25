"""m3 分镜生成（教学第 2 讲延伸：首帧锚定）。

每镜头生成首帧图（含角色固化外貌注入）；可选 VLM 一致性抽查。
"""

from __future__ import annotations

import random
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import Asset, Script, Shot
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ProviderError, T2IProvider, VLMProvider

_MAX_RETRY = 2


async def build_storyboard(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    assets: dict[str, Asset],
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider | None = None,
) -> dict[str, str]:
    """返回 shot_id → 首帧图路径。"""
    from fastvideo3.modules.m2_assets import size_for_aspect

    char_map = script.character_map()
    scene_map = {s.scene_id: s for s in script.scenes}
    frames: dict[str, str] = {}
    size = size_for_aspect(cfg.aspect_ratio)

    for shot in script.all_shots():
        scene = scene_map.get(shot.scene_id)
        prompt = _shot_prompt(cfg, script, scene, shot, char_map, assets)
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
        await _retry_gen(
            t2i,
            prompt,
            dest,
            ledger,
            f"sb:{shot.shot_id}",
            size=size,
            negative_prompt=cfg.negative_prompt,
            seed=_stable_seed(shot),
            ref_image=ref,
        )
        frames[shot.shot_id] = str(dest)
        ledger.record("t2i", t2i.name, t2i.model, note=f"storyboard:{shot.shot_id}")

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
                result = await vlm.review(
                    [storage.path("storyboard", f"shot_{shot.shot_id}.png")], rubric=rubric
                )
                ledger.record("vlm", vlm.name, vlm.model, note=f"sb_check:{shot.shot_id}")
                storage.log_event(
                    "m3_storyboard",
                    "consistency_check",
                    shot=shot.shot_id,
                    passed=result.passed,
                )
                if not result.passed:
                    # 驳回 → 重生成该首帧（预算内，同种子保证与过审帧同族）
                    await _retry_gen(
                        t2i,
                        shot.visual_prompt + "。严格保持角色外貌一致。",
                        storage.path("storyboard", f"shot_{shot.shot_id}.png"),
                        ledger,
                        f"sb_retry:{shot.shot_id}",
                        negative_prompt=cfg.negative_prompt,
                        seed=_stable_seed(shot),
                        ref_image=_char_ref(cfg, t2i, shot, assets),
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
) -> None:
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
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise ProviderError(f"分镜生成失败: {tag}: {last}")
