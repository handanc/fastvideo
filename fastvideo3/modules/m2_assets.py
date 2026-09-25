"""m2 资产生成（教学第 2 讲：角色一致性与资产锚定）。

角色立绘 + 场景概念图 → 资产库（全片唯一来源）。资产是后续一切生成的锚点，
失败阻断（不做静默降级）。
"""

from __future__ import annotations

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.schema import Asset, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ModerationError, ProviderError, T2IProvider

_MAX_RETRY = 2

# 画幅 → 万相文生图 size 参数（9:16 竖屏短剧用 720*1280）
ASPECT_SIZES = {"9:16": "720*1280", "16:9": "1280*720", "1:1": "1024*1024"}


def size_for_aspect(aspect_ratio: str) -> str:
    return ASPECT_SIZES.get(aspect_ratio, "1024*1024")


async def build_assets(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
) -> dict[str, Asset]:
    registry: dict[str, Asset] = {}
    size = size_for_aspect(cfg.aspect_ratio)

    # 角色立绘（style_suffix 全片风格锚，防止每格随机画风）
    for ch in script.characters:
        prompt = (
            f"短剧角色立绘，竖屏全身像，电影感光影。"
            f"角色：{ch.name}。固化外貌：{ch.appearance}。"
            f"风格：{script.base_info.style}。{cfg.style_suffix}纯色背景，正面朝向镜头。"
        )
        dest = storage.path("assets", f"char_{ch.character_id}.png")
        await _gen_with_retry(
            t2i,
            prompt,
            dest,
            ledger,
            f"char:{ch.character_id}",
            size=size,
            negative_prompt=cfg.negative_prompt,
        )
        asset = Asset(
            asset_id=f"char_{ch.character_id}",
            kind="character",
            owner_id=ch.character_id,
            uri=str(dest),
            meta={"appearance": ch.appearance},
        )
        registry[asset.asset_id] = asset

    # 场景概念图
    for scene in script.scenes:
        prompt = (
            f"短剧场景概念图，竖屏，电影感。地点：{scene.location}，"
            f"时间：{scene.time_of_day}。氛围：{scene.summary or scene.title}。"
            f"风格：{script.base_info.style}。{cfg.style_suffix}无人物。"
        )
        dest = storage.path("assets", f"scene_{scene.scene_id}.png")
        await _gen_with_retry(
            t2i,
            prompt,
            dest,
            ledger,
            f"scene:{scene.scene_id}",
            size=size,
            negative_prompt=cfg.negative_prompt,
        )
        asset = Asset(
            asset_id=f"scene_{scene.scene_id}",
            kind="scene",
            owner_id=scene.scene_id,
            uri=str(dest),
            meta={"location": scene.location},
        )
        registry[asset.asset_id] = asset

    storage.save_json("asset_registry.json", {k: a.model_dump() for k, a in registry.items()})
    storage.log_event("m2_assets", "done", count=len(registry))
    return registry


async def _gen_with_retry(
    t2i: T2IProvider,
    prompt: str,
    dest,
    ledger: CostLedger,
    tag: str,
    size: str = "1024*1024",
    negative_prompt: str = "",
) -> None:
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await t2i.generate(prompt, dest=dest, size=size, negative_prompt=negative_prompt)
            ledger.record("t2i", t2i.name, t2i.model, note=tag)
            return
        except ModerationError:
            prompt = prompt + "。画面积极健康，无任何敏感元素。"
            last_err = ProviderError(f"{tag}: moderation rejected, retry {attempt}")
        except (ProviderError, Exception) as e:  # noqa: BLE001
            last_err = e
    raise ProviderError(f"资产生成失败（已重试{_MAX_RETRY}次）: {tag}: {last_err}")
