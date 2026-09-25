"""m2 资产生成（教学第 2 讲：角色一致性与资产锚定）。

角色立绘 + 场景概念图 → 资产库（全片唯一来源）。资产是后续一切生成的锚点，
失败阻断（不做静默降级）。cinema 档角色立绘走 best-of-N + VLM 选优
（asset_candidates>1）：立绘崩会被 m3 参考图/r2v/双人首帧全链路继承放大，源头把关。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.schema import Asset, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ModerationError, ProviderError, T2IProvider, VLMProvider

_MAX_RETRY = 2

# 画幅 → 万相文生图 size 参数。单边上限 1440（2026-09-25 真实探单：
# 1080*1920 被拒 "width or height should be between 512 and 1440"）。
# hi_res 供 high/cinema 的 1080P 时间线取最大档，减少上采样细节损失。
ASPECT_SIZES = {"9:16": "720*1280", "16:9": "1280*720", "1:1": "1024*1024"}
ASPECT_SIZES_HI = {"9:16": "810*1440", "16:9": "1440*810", "1:1": "1440*1440"}


def size_for_aspect(aspect_ratio: str, hi_res: bool = False) -> str:
    table = ASPECT_SIZES_HI if hi_res else ASPECT_SIZES
    return table.get(aspect_ratio, "1440*1440" if hi_res else "1024*1024")


_CHAR_QC_RUBRIC = (
    "评估这张短剧角色立绘：①角色外貌与文末固化描述一致（发型/服装/年龄/关键特征）"
    "②五官无畸变、脸部自然 ③肢体完整、手指数量正常 ④真人写实质感（非动漫/3D/插画）"
    "⑤背景干净。只输出 JSON：{\"passed\": true/false, \"score\": 0到1的小数}"
)


async def build_assets(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider | None = None,
    fallback: T2IProvider | None = None,
) -> dict[str, Asset]:
    registry: dict[str, Asset] = {}
    size = size_for_aspect(cfg.aspect_ratio, hi_res=cfg.resolution[0] >= 1080)
    n_cand = max(1, cfg.asset_candidates) if vlm is not None else 1

    # 角色立绘（style_suffix 全片风格锚，防止每格随机画风）
    for ch in script.characters:
        prompt = (
            f"短剧角色立绘，竖屏全身像，电影感光影。"
            f"角色：{ch.name}。固化外貌：{ch.appearance}。"
            f"风格：{script.base_info.style}。{cfg.style_suffix}纯色背景，正面朝向镜头。"
        )
        dest = storage.path("assets", f"char_{ch.character_id}.png")
        if n_cand > 1:
            await _gen_best_char(
                cfg, t2i, vlm, prompt, ch, dest, storage, ledger, size, n_cand,  # type: ignore[arg-type]
                fallback=fallback,
            )
        else:
            await _gen_with_retry(
                t2i,
                prompt,
                dest,
                ledger,
                f"char:{ch.character_id}",
                size=size,
                negative_prompt=cfg.negative_prompt,
                fallback=fallback,
                storage=storage,
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
            fallback=fallback,
            storage=storage,
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
    fallback: T2IProvider | None = None,
    storage: RunStorage | None = None,
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
    if fallback is not None:
        # 主供应商耗尽 → 备选保交付（yaml 的 fallback 此前是死配置，从未接线）。
        # 跨供应商可能画风跳变 —— 事件+台账标注，供人工复核。
        try:
            await fallback.generate(prompt, dest=dest, size=size, negative_prompt=negative_prompt)
            ledger.record("t2i", fallback.name, fallback.model, note=f"{tag}#fallback")
            if storage is not None:
                storage.log_event("m2_assets", "fallback_used", tag=tag, provider=fallback.name)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ProviderError(f"资产生成失败（已重试{_MAX_RETRY}次）: {tag}: {last_err}")


async def _gen_best_char(
    cfg: AppConfig,
    t2i: T2IProvider,
    vlm: VLMProvider,
    prompt: str,
    ch,
    dest: Path,
    storage: RunStorage,
    ledger: CostLedger,
    size: str,
    n_cand: int,
    fallback: T2IProvider | None = None,
) -> None:
    """角色立绘 best-of-N + VLM 选优（<¥1/片，防立绘崩被全链路继承放大）。

    落选候选保留在 assets/_cand_*.png 供人工排查。
    """
    cands: list[Path] = []
    for i in range(n_cand):
        cdest = storage.path("assets", f"_cand_char_{ch.character_id}_{i}.png")
        try:
            await _gen_with_retry(
                t2i,
                prompt,
                cdest,
                ledger,
                f"char:{ch.character_id}#c{i}",
                size=size,
                negative_prompt=cfg.negative_prompt,
                fallback=fallback,
                storage=storage,
            )
            cands.append(cdest)
        except ProviderError as e:
            storage.log_event(
                "m2_assets", "candidate_failed", char=ch.character_id, cand=i, error=str(e)[:200]
            )
    if not cands:
        raise ProviderError(f"角色立绘候选全部失败: char:{ch.character_id}")

    rubric = f"{_CHAR_QC_RUBRIC}\n\n角色固化描述：{ch.appearance}"
    best, best_score = cands[0], -1.0
    for c in cands:
        try:
            r = await vlm.review([c], rubric=rubric)
            score = float(r.score)
            ledger.record("vlm", vlm.name, vlm.model, note=f"char_qc:{ch.character_id}")
        except Exception:  # noqa: BLE001  # 评审失败不阻塞：保底 0.5 分
            score = 0.5
        if score > best_score:
            best, best_score = c, score
    pick_name = best.name
    best.replace(dest)  # 最优候选 → 正式路径（registry/下游 m3 引用点不变）
    storage.log_event(
        "m2_assets",
        "char_best_of_n",
        char=ch.character_id,
        n=len(cands),
        pick=pick_name,
        score=round(best_score, 3),
    )
