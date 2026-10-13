"""口型通道实测：哪条通道真的对口型（离线扫描 + 可选小额探单）。

为什么需要它（2026-10-03 的教训）：判断"wan2.7-i2v+driving_audio 到底有没有口型"时，
我用的是写死的比例框，在真实分镜下它量不到嘴（同素材三区运动幅度 嘴0.8/脸4.4/背景4.2），
于是得到一串 -0.47~0.24 的"相关系数"并险些据此下结论。尺子坏了，结论就作废。
本脚本用修好的两把尺子测量，并把**不可信的情形明说为不可信**：

1. `scan`（免费，本地）：对已有 talk/clip 片段做滑窗定位 + 乱序零分布对照；
2. `probe`（花钱，需 --spend）：同一定妆照 + 同一段混音，逐通道各出 1 条短片再测。

用法：
    python scripts/probe_lipsync.py scan output/run_20260830_211509
    python scripts/probe_lipsync.py probe --quality cinema --frame 首帧.png --wav 混音.wav \
        --channels wan2.7-i2v,wan2.2-s2v --spend

判读规则（重要）：
- `synced` 且 corr 明显高于 null → 该通道确实按语音动嘴；
- `inconclusive` → **不能**据此判通道好坏（素材/度量不支持），要换素材或用 VLM 定位；
- 三条通道全 `unsynced` 而对照能测出真口型 → 才可以说"这些通道不做口型"。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastvideo3.core.config import load_config  # noqa: E402
from fastvideo3.utils.lipsync import (  # noqa: E402
    MOUTH_BOX as ls_MOUTH_BOX,
)
from fastvideo3.utils.lipsync import (
    lipsync_diag,
    locate_boxes,
    scan_lipsync,
    series_from_frames,
)


def _clip_audio(clip: Path, workdir: Path) -> Path | None:
    """从 mp4 里取出自带音轨（16k 单声道 wav）；无音轨返回 None。"""
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    out = workdir / f"{clip.stem}_audio.wav"
    if out.exists():
        return out
    try:
        run_ffmpeg(["-i", str(clip), "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                    "-ac", "1", "-y", str(out)])
    except Exception as e:  # noqa: BLE001
        print(f"  ! 取音轨失败 {clip.name}: {e}")
        return None
    return out if out.exists() else None


def _find_mix(run_dir: Path, shot_id: str) -> Path | None:
    for name in (f"shot_{shot_id}_mix.wav", f"shot_{shot_id}_duo_mix.wav"):
        p = run_dir / "audio" / name
        if p.exists():
            return p
    return None


def cmd_scan(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    work = run_dir / "_lipsync_probe"
    work.mkdir(exist_ok=True)
    clips = sorted((run_dir / "talk").glob("shot_*.mp4")) if (run_dir / "talk").exists() else []
    if not clips:
        clips = sorted((run_dir / "clips").glob("shot_*.mp4"))
    if not clips:
        print(f"{run_dir} 下没有 talk/clips 片段可测")
        return 1
    print(f"# {run_dir.name}  片段 {len(clips)} 条  阈值 {args.min_corr}")
    print("| clip | 音频来源 | scan corr | null | 最强框 | scan 判定 | VLM 定位判定 |")
    print("|---|---|---|---|---|---|---|")
    vlm = None
    if args.with_vlm:
        from fastvideo3.providers.base import Capability
        from fastvideo3.providers.factory import make_provider

        try:
            vlm = make_provider(Capability.VLM, load_config(quality=args.quality))
        except Exception as e:  # noqa: BLE001
            print(f"! VLM 不可用（跳过精确定位）: {e}")
    for clip in clips:
        shot_id = clip.stem[len("shot_"):] if clip.stem.startswith("shot_") else clip.stem
        wav = _find_mix(run_dir, shot_id) or _clip_audio(clip, work)
        src = "run 混音" if _find_mix(run_dir, shot_id) else ("片段自带音轨" if wav else "-")
        if wav is None:
            print(f"| {clip.name} | 无音轨 | - | - | - | 无法测 | 无法测 |")
            continue
        s = scan_lipsync(clip, wav, work / clip.stem, min_corr=args.min_corr)
        v = "-"
        if vlm is not None:
            boxes = asyncio.run(locate_boxes(clip, vlm, work / clip.stem))
            d = lipsync_diag(clip, wav, work / clip.stem, boxes=boxes, min_corr=args.min_corr)
            v = f"{d['verdict']} corr={d['corr']} 眼={d['eyes_corr']} 背景={d['bg_corr']}"
        print(
            f"| {clip.name} | {src} | {s['corr']} | {s['null_corr']} | {s['best_box']} "
            f"| **{s['verdict']}** | {v} |"
        )
        if args.verbose:
            print(f"    ↳ {s['reason']}（候选 {s.get('n_candidates')}/有效 {s.get('n_eligible')}）")
    print(
        "\n判读：synced=确有随语音运动的区域；inconclusive=证据不足（画面运动/压缩噪声"
        "即可产生同等相关），**不得**据此判通道优劣；unsynced=定位到动嘴区但耦合弱。"
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """同素材逐通道各出 1 条短片并测量 —— 花真钱，必须 --spend 显式确认。"""
    if not args.spend:
        print("probe 会产生真实计费（每条 5 秒 × 通道单价）。确认后再加 --spend。")
        for ch in args.channels.split(","):
            print(f"  拟探测通道: {ch.strip()}")
        return 2
    import asyncio

    from fastvideo3.core.cost import price_of
    from fastvideo3.providers.base import Capability, poll_until_done
    cfg = load_config(quality=args.quality)
    cfg.max_video_cost_per_sec = args.cap  # 用户红线：超过就不该考虑这条通道
    frame, wav = Path(args.frame), Path(args.wav)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    budget_left = args.budget

    async def one(model: str) -> dict:
        from fastvideo3.providers.base import price_key_for
        from fastvideo3.providers.factory import _make_real
        from fastvideo3.providers.protocols import infer_protocol, provider_for

        # 按模型名推断协议再装配：绕开 .env 覆盖，探的就是"这个模型本身"行不行
        route = provider_for("talk", infer_protocol(model, "talk")) or "dashscope_wan27_audio"
        provider = _make_real(route, Capability.TALK, cfg, overrides={"talk": model})
        print(f"   协议路由：{route} / {type(provider).__name__}")
        dest = out_dir / f"probe_{model}.mp4"
        dur = max(2, int(round(args.seconds)))
        # 计价走 provider 的计价键：分档模型（fal `端点@480p/@720p`）的模型名不在
        # pricing.models 里，直接 price_of(model) 会静默回落到能力默认价 ¥0.5，
        # 于是红线检查形同虚设、探单前看到的"便宜"是假的。
        price = price_of(price_key_for(provider, cfg.i2v_resolution), "talk", 1.0)
        if price > cfg.max_video_cost_per_sec:
            return {"model": model, "price_cny_per_sec": price,
                    "error": f"单价 ¥{price:.2f}/s 超红线 ¥{args.cap}/s"}
        if price * dur > budget_left:
            return {"model": model, "price_cny_per_sec": price,
                    "error": f"剩余预算 ¥{budget_left:.2f} 不够 ¥{price*dur:.2f}"}
        from fastvideo3.providers.base import TalkProvider

        assert isinstance(provider, TalkProvider), f"{model} 装配到的不是口型通道"
        print(f"→ {model}（¥{price:.2f}/s × {dur}s ≈ ¥{price * dur:.2f}）")
        ext = await provider.submit(
            dest=dest, image=frame, audio=wav, video=None, prompt="",
            resolution=cfg.i2v_resolution, duration=dur,
        )
        st = await poll_until_done(provider.poll, ext, timeout_sec=args.timeout * 60,
                                   fast_interval=5, slow_interval=15)
        if st.status != "succeeded" or not dest.exists():
            return {"model": model, "error": str(st.error or "失败")[:160]}
        s = scan_lipsync(dest, wav, work / model, min_corr=args.min_corr)
        return {"model": model, "price_cny_per_sec": round(price, 4),
                "cost": round(price * dur, 2), **s}

    results = []
    for model in [c.strip() for c in args.channels.split(",") if c.strip()]:
        try:
            results.append(asyncio.run(one(model)))
        except Exception as e:  # noqa: BLE001
            results.append({"model": model, "error": f"{type(e).__name__}: {str(e)[:160]}"})

    print("\n| 通道 | 单价 | 花费 | corr | null | 判定 | 备注 |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r.get('model')} | {price_of(r.get('model', ''), 'talk', 1.0)} "
            f"| {r.get('cost', '-')} | {r.get('corr', '-')} | {r.get('null_corr', '-')} "
            f"| **{r.get('verdict') or r.get('error') or '-'}** "
            f"| {str(r.get('reason') or r.get('error') or '')[:70]} |"
        )
    spent = sum(r.get("cost") or 0 for r in results)
    print(f"\n本次探单实际计费约 ¥{spent:.2f}（正式台账仍由各 run 的 cost.json 记录）")
    print("把这张表贴回对话即可定档；全 inconclusive 时先换素材（特写正脸镜头）再判。")
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    """出"嘴部接片 + 逐帧响度条"：把『到底同不同步』变成 10 秒的人眼判断。

    为什么需要这个而不是自动度量：自动特征在真实素材上会**假阴性**（人眼确认同步的
    videoretalk 产物只给 corr 0.163 / 偏相关 0.113）。度量不可靠时，正确做法不是
    拿它自动重做（那是反复烧钱重抽合格镜头），而是把判断材料摆到人面前。
    """
    import math

    from PIL import Image, ImageDraw

    from fastvideo3.utils.ffmpeg import extract_frames
    from fastvideo3.utils.lipsync import audio_rms_series, locate_boxes, pearson

    run_dir = Path(args.run_dir)
    shot = args.shot
    clip = Path(args.clip) if args.clip else run_dir / "talk" / f"shot_{shot}.mp4"
    if not clip.exists():
        print(f"找不到片段：{clip}")
        return 1
    wav = Path(args.wav) if args.wav else run_dir / "audio" / f"shot_{shot}_mix.wav"
    if not wav.exists():
        print(f"找不到驱动音频：{wav}")
        return 1

    work = run_dir / "_lipsync_probe"
    work.mkdir(exist_ok=True)
    boxes = None
    if args.with_vlm:
        from fastvideo3.providers.base import Capability
        from fastvideo3.providers.factory import make_provider

        try:
            vlm = make_provider(Capability.VLM, load_config(quality=args.quality))
            boxes = asyncio.run(locate_boxes(clip, vlm, work / f"_loc_{clip.stem}"))
        except Exception as e:  # noqa: BLE001
            print(f"! 视觉模型不可用，退回经验框（画面可能框不住嘴）：{e}")
    box: tuple[float, float, float, float]
    if args.box:
        parts = [float(x) for x in args.box.split(",")]
        if len(parts) != 4:
            print("--box 需要 4 个数：x0,y0,x1,y1（归一化）")
            return 1
        box = (parts[0], parts[1], parts[2], parts[3])
    elif boxes is not None:
        box = tuple(boxes.lips)
    else:
        box = ls_MOUTH_BOX
        print("! 未做嘴部定位，用的是经验框 —— 中远景/偏构图下大概率框在衣服上"
              "（加 --with-vlm 或 --box x0,y0,x1,y1）")
    # 外扩一圈，好看清嘴与下巴/鼻的上下文
    pad = (max(0.0, box[0] - 0.07), max(0.0, box[1] - 0.05),
           min(1.0, box[2] + 0.07), min(1.0, box[3] + 0.07))

    frames = extract_frames(clip, work / f"_sheet_{clip.stem}", count=args.frames)[: args.frames]
    rms = audio_rms_series(wav, len(frames))
    peak = max(rms) if rms and max(rms) > 0 else 1.0
    cells = []
    for i, f in enumerate(frames):
        with Image.open(f) as im:
            w, h = im.size
            crop = im.crop((int(w * pad[0]), int(h * pad[1]), int(w * pad[2]), int(h * pad[3])))
        c = crop.convert("RGB").resize((110, 110))
        dr = ImageDraw.Draw(c)
        v = (rms[i] / peak) if i < len(rms) else 0.0
        dr.rectangle([0, 0, int(110 * v), 6], fill=(255, 90, 0))  # 该帧响度条
        dr.text((4, 96), f"#{i}  {v:.2f}", fill=(255, 255, 0))
        cells.append(c)
    cols = 5
    rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * 114, rows * 118), (14, 14, 14))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * 114 + 2, (i // cols) * 118 + 2))
    out = work / f"sheet_{clip.stem}.png"
    sheet.save(out)
    # 顺手给个数字参考（**不作为闸门**，理由见 m6/_lipsync_check 的注释）
    series = series_from_frames(frames, box)
    mm = pearson(series, audio_rms_series(wav, len(series)))
    print(f"接片：{out}  （每格上方橙条=该帧响度，右下角=帧号与相对响度）")
    print(f"参考数字 corr={None if mm is None else round(mm, 3)} —— 只作参考，判定看片。")
    print("看什么：响度条长的帧嘴是否张开、响度条为 0 的帧嘴是否闭合；顺带看有没有第二张嘴/僵脸。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-corr", type=float, default=0.25)
    ap.add_argument("--quality", default="cinema")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("scan", help="免费：测已有片段（离线，不调 API）")
    a1.add_argument("run_dir")
    a1.add_argument("--with-vlm", action="store_true", help="再用视觉模型定位精测（每片段一次 VLM 调用）")
    a1.set_defaults(func=cmd_scan)
    a2 = sub.add_parser("probe", help="花钱：逐通道出 1 条短片再测")
    a2.add_argument("--frame", required=True, help="首帧/定妆照 png")
    a2.add_argument("--wav", required=True, help="驱动音频 wav（对白混音）")
    a2.add_argument(
        "--channels",
        default="wan2.2-s2v,fal-ai/infinitalk,videoretalk",
        help="按模型名/端点ID 逐个探；协议由 protocols.py 推断",
    )
    a2.add_argument("--seconds", type=float, default=5.0)
    a2.add_argument("--cap", type=float, default=5.0, help="单价红线 元/秒")
    a2.add_argument("--budget", type=float, default=20.0)
    a2.add_argument("--out", default="output/_lipsync_probe")
    a2.add_argument("--timeout", type=int, default=25)
    a2.add_argument("--spend", action="store_true")
    a2.set_defaults(func=cmd_probe)
    a3 = sub.add_parser("sheet", help="出带响度条的嘴部接片，交人眼定案（免费）")
    a3.add_argument("run_dir")
    a3.add_argument("--shot", required=True, help="镜头 id，如 shot_1_2")
    a3.add_argument("--clip", default="", help="指定片段文件（默认取 talk/shot_<id>.mp4）")
    a3.add_argument("--wav", default="", help="指定驱动音频（默认取 audio/shot_<id>_mix.wav）")
    a3.add_argument("--frames", type=int, default=20)
    a3.add_argument("--with-vlm", action="store_true")
    a3.add_argument("--box", default="", help="手工嘴框 x0,y0,x1,y1（归一化），例：0.51,0.32,0.58,0.36")
    a3.set_defaults(func=cmd_sheet)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
