"""wan2.7-image 多参考图探单（风险 R1 的"先验证再改"步骤，约 ¥1）。

回答三个问题，决定 cinema 档 ref_max_chars 能否从 1 放开到 2：
1. messages.content 里放 2 张 {"image": ...} 是否返回 200；
2. 同种子下 带参考图 / 不带参考图 的 A/B 是否有**可度量**差异（参考条件是否真被消费，
   而不是"看起来通电、实际被服务端忽略"）；
3. 多图时是否只取最后一张（目视产物与两张 ref 的归属）。

用法（充值后）：
    python scripts/probe_multi_ref.py --ref-a output/<run>/assets/char_C1.png --ref-b output/<run>/assets/char_C2.png
不传 ref 时自动取最近一个真实 run 的前两张角色立绘。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from fastvideo3.core.config import OUTPUT_DIR, load_config
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import Capability, ProviderError
from fastvideo3.providers.factory import make_provider
from fastvideo3.utils import quality as q

for _stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 → UTF-8
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, pragma: no cover
        pass

SEED = 20261001
PROMPT = (
    "真人写实电影剧照，竖屏。图1的人物站在街角面馆门口，图2的人物在门内招呼他。"
    "电影级三点布光，自然皮肤纹理，浅景深。"
)


def _default_refs() -> list[Path]:
    runs = sorted((p for p in OUTPUT_DIR.iterdir() if p.is_dir()), reverse=True)
    for r in runs:
        chars = sorted((r / "assets").glob("char_*.png")) if (r / "assets").exists() else []
        chars = [c for c in chars if not c.name.startswith("_cand") and c.stat().st_size > 50_000]
        if len(chars) >= 2:
            return chars[:2]
    return []


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-a", default="")
    ap.add_argument("--ref-b", default="")
    args = ap.parse_args()

    cfg = load_config(quality="cinema")
    if cfg.fake or not cfg.dashscope_api_key:
        print("需要真实 DASHSCOPE_API_KEY 且非 fake 模式（本脚本就是为真实探单而生）")
        return 1
    refs = [Path(args.ref_a)] if args.ref_a else []
    if args.ref_b:
        refs.append(Path(args.ref_b))
    if not refs:
        refs = _default_refs()
    if not refs:
        print("找不到角色立绘：请用 --ref-a/--ref-b 指定")
        return 1
    for r in refs:
        if not r.exists():
            print(f"参考图不存在: {r}")
            return 1

    storage = RunStorage(f"probe_multiref_{SEED}")
    ledger = CostLedger(storage, 20.0)
    t2i = make_provider(Capability.T2I, cfg)
    print(f"provider={t2i.name} model={t2i.model} refs={[str(r) for r in refs]}")

    cases = [("multi", refs), ("single", refs[:1]), ("noref", [])]
    outs: dict[str, Path] = {}
    for label, ref_list in cases:
        dest = storage.path("review", f"probe_{label}.png")
        try:
            res = await t2i.generate(  # type: ignore[attr-defined]
                PROMPT, dest=dest, size="720*1280", seed=SEED, ref_image=ref_list or None
            )
            ledger.record(
                "t2i", t2i.name, t2i.model, note=f"probe:{label}",
                meta={"ref_count": len(ref_list), "seed": SEED},
            )
            outs[label] = dest
            print(f"  ✔ {label}: ref_count={len(ref_list)} meta={getattr(res, 'meta', {})}")
        except ProviderError as e:
            print(f"  ✘ {label}: {str(e)[:300]}")
            if label == "multi":
                print("→ 多图请求被拒：ref_max_chars 保持 1，记 MultiRefUnsupported")
                return 2

    # 同种子 A/B：参考条件是否真被消费（色相/直方图与 ref-a 的距离应显著小于 noref）
    print("\n与 ref-a 的客观距离（越小越像参考图）：")
    for label in ("multi", "single", "noref"):
        if label not in outs:
            continue
        hue = q.hue_distance(q.dominant_hue(refs[0], q.OUTFIT_BOX),
                             q.dominant_hue(outs[label], q.OUTFIT_BOX))
        hist = q.hist_distance(refs[0], outs[label], q.FACE_BOX)
        print(f"  {label:7s} hue_dist={hue} face_hist={round(hist, 3)}")
    print(f"\n成本 ¥{ledger.total()}  产物目录 {storage.root}")
    print("判读：multi/single 的距离应明显小于 noref；若三者接近 → 参考条件未被消费，止损。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
