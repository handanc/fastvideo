"""角色一致性消融实验（TDD.md:774/788 承诺的兑现）。

用法（充值后）：
    python scripts/run_ablation.py --topic "除夕夜的外卖员" --arms consistency_off,ref_only,ref_plus_r2v,full_cinema
    python scripts/run_ablation.py --with-shots   # 额外跑 m4，让 r2v_route_ratio 可测
离线拓扑自检（零成本）：
    python scripts/run_ablation.py --fake --with-shots

设计要点：
- 同一 topic 只生成**一次**剧本，四臂共用同一份 script.json —— 否则 LLM 的剧本
  差异会污染一致性对比（消融要测的是机制，不是编剧抖动）。
- 每臂只跑 m2 资产 + m3 首帧 + m9 度量（--keyframes-only 语义，本脚本默认如此）：
  跨镜头换脸在首帧层面就完全可测（Tier A 属性位 + Tier B 色相都在 png 上算），
  单臂约 ¥6-9；跑完整视频约 ¥120/臂 而一致性结论不会更硬。
- 产出 output/ablation_<ts>/{table.md, results.json}，compare-consistency 可直接复用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from fastvideo3.core.config import OUTPUT_DIR, load_config
from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.stages import build_context
from fastvideo3.core.storage import RunStorage
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.modules.m9_metrics import compare_table
from fastvideo3.providers.base import Capability
from fastvideo3.providers.factory import make_fallback

for _stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 → UTF-8（与 CLI 同款容错）
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, pragma: no cover
        pass

DEFAULT_ARMS = ["consistency_off", "ref_only", "ref_plus_r2v", "full_cinema"]


async def _gen_script(cfg, topic: str, storage: RunStorage) -> dict:
    from fastvideo3.modules.m1_scripting import generate_script

    ctx = build_context(cfg, storage)
    script = await generate_script(
        cfg,
        ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
        BaseInfo(topic=topic, target_duration_sec=30),
        storage,
        ctx.ledger,
    )
    return script.model_dump(mode="json")


async def _run_arm(
    arm: str, run_id: str, topic: str, script_json: dict, fake: bool, with_shots: bool = False
) -> dict:
    # `quality=` 这个参数已经随档位层一起删除（2026-10-17 前后），现在唯一的旋钮是分辨率；
    # 四臂各自 yaml 里就写着 `i2v_resolution: 480P`（"只比一致性，不比分辨率"），
    # 所以这里**不传 resolution**，让臂自己说话。
    cfg = load_config(ablation=arm)
    cfg.fake = fake
    storage = RunStorage(run_id)
    storage.save_json("run_meta.json", {"run_id": run_id,
                                        "resolution_tier": cfg.i2v_resolution,
                                        "fake": fake, "ablation": arm,
                                        "base_info": BaseInfo(topic=topic).model_dump()})
    storage.save_json("script.json", script_json)
    ctx = build_context(cfg, storage)
    script = Script.model_validate(script_json)

    vlm_assets = ctx.provider(Capability.VLM) if cfg.asset_candidates > 1 else None
    assets = await build_assets(
        cfg,
        ctx.provider(Capability.T2I),  # type: ignore[arg-type]
        script,
        storage,
        ctx.ledger,
        vlm=vlm_assets,  # type: ignore[arg-type]
        fallback=make_fallback(Capability.T2I, cfg),  # type: ignore[arg-type]
    )
    vlm_sb = ctx.provider(Capability.VLM) if cfg.storyboard_sample_rate > 0 else None
    frames = await build_storyboard(
        cfg,
        ctx.provider(Capability.T2I),  # type: ignore[arg-type]
        script,
        assets,  # type: ignore[arg-type]
        storage,
        ctx.ledger,
        vlm=vlm_sb,  # type: ignore[arg-type]
        text=ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
        fallback=make_fallback(Capability.T2I, cfg),  # type: ignore[arg-type]
    )
    if with_shots:
        # 只有跑到 m4 才会有 shot_gen{route} 事件 —— r2v_ratio 列在 keyframes-only
        # 形态下是 None（未测，不是 0）。真实消融建议开：i2v 480P×短时长成本可控。
        from fastvideo3.modules.m4_shots import generate_shots
        from fastvideo3.providers.factory import make_r2v_provider

        await generate_shots(
            cfg,
            ctx.provider(Capability.I2V),  # type: ignore[arg-type]
            script,
            frames,
            storage,
            ctx.ledger,
            ctx.jobs,
            vlm=ctx.provider(Capability.VLM) if cfg.candidates_per_shot > 1 else None,  # type: ignore[arg-type]
            r2v=make_r2v_provider(cfg),  # type: ignore[arg-type]
            assets=assets,
            fallback=make_fallback(Capability.I2V, cfg),  # type: ignore[arg-type]
        )
    vlm_m = ctx.provider(Capability.VLM) if cfg.consistency_metric != "off" else None
    report = await _measure(cfg, script, assets, frames, storage, ctx, vlm_m)
    return {
        "arm": arm,
        "run_id": run_id,
        "cost_cny": ctx.ledger.total(),
        "report": report,
    }


async def _measure(cfg, script, assets, frames, storage, ctx, vlm) -> dict:
    from fastvideo3.modules.m9_metrics import measure_consistency

    return await measure_consistency(
        cfg, script, assets, frames, storage, ctx.ledger, vlm=vlm  # type: ignore[arg-type]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="角色一致性消融（首帧层面，省钱形态）")
    ap.add_argument("--topic", default="除夕夜的外卖员被面馆老板善待")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--fake", action="store_true", help="零成本拓扑自检（只验证事件/路由正确）")
    ap.add_argument(
        "--with-shots",
        action="store_true",
        help="额外跑 m4（让 r2v_route_ratio 可测；默认只跑到首帧，省钱形态）",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def go() -> list[dict]:
        base_id = f"abl_{ts}_script"
        base_cfg = load_config(quality="cinema", ablation=arms[0])
        base_cfg.fake = args.fake
        script_json = await _gen_script(base_cfg, args.topic, RunStorage(base_id))
        results = []
        for arm in arms:
            rid = f"abl_{ts}_{arm}"
            print(f"▶ arm={arm} run_id={rid}")
            results.append(
                await _run_arm(arm, rid, args.topic, script_json, args.fake, args.with_shots)
            )
        return results

    results = asyncio.run(go())

    rows = [(r["arm"], r["report"]) for r in results if r["report"]]
    table = compare_table(rows)
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "arm": r["arm"],
                    "run_id": r["run_id"],
                    "cost_cny": r["cost_cny"],
                    "coverage": (r["report"] or {}).get("coverage"),
                    "summary": (r["report"] or {}).get("summary"),
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(table)
    print(f"✔ 产物: {out_dir}")
    if not args.fake:
        off = next((r for r in results if r["arm"] == "consistency_off"), None)
        if off and (off["report"] or {}).get("coverage", {}).get("ref_attached_ratio"):
            print("⚠ consistency_off 臂的 ref_attached_ratio 非 0 —— 开关未生效，结论无效")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
