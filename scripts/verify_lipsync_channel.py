"""一条对白镜头的口型通道实测：i2v 运动片段 → 口型模型重对口型 → 当场量给你看。

为什么要它：2026-10-04 把口型主通道定为 videoretalk 的依据是**架构与规格**（不重画画面、
输出分辨率与输入一致、¥0.08/秒），不是实测。"它到底动不动嘴"必须由这条镜头的产物来回答，
而产物由修好的尺子（`utils/lipsync.py`：定位框 + 眼/背景对照 + 乱序零分布）与你的眼睛共同判定。

成本控制（写死在这里，不靠人记得住）：
- `--cap`（默认 ¥5/秒，用户的硬红线）落进 `cfg.max_video_cost_per_sec`：路由哪天解析到
  比这更贵的通道，**装配阶段直接报错**，不会先烧钱再告诉你贵了。
- `--spend` 闸门：预估费用 > ¥0.5 时**不加 --spend 就不提交、不加锁**（主通道换成按秒计价的
  fal InfiniteTalk 后，这条脚本从"≈¥0 的验证"变成了"会花真钱的操作"，所以默认只打印计划）。
  想零成本验证链路 → `--channel cheap`（钉回百炼 videoretalk 兜底档）。
- 不传 fallback provider：主通道挂了就是挂了，绝不静默换到更贵的通道。
- 全程过全局单任务锁（`output/.task.lock`）：与网页/CLI 的任何生成本质上是同一件事，
  不能并发，防止同时点两处烧双份钱。

用法（2026-10-04 口型主通道换成 fal InfiniteTalk 后，**默认会花真钱**，故加 --spend 闸门）：
    # 只看计划与费用（零成本，不提交、不加锁）
    python scripts/verify_lipsync_channel.py output/run_20261002_235904 --shot shot_1_2
    # 确认费用后再真跑
    python scripts/verify_lipsync_channel.py output/run_20261002_235904 --shot shot_1_2 --spend
    # 想零成本验证"链路通不通"（i2v→口型→度量→血缘），钉回百炼兜底通道：
    python scripts/verify_lipsync_channel.py <run> --shot shot_1_2 --channel cheap
    加 --with-vlm 再用视觉模型定位精测（每片段一次 VLM 调用，几分钱）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastvideo3.core.config import ProviderRouting, load_config  # noqa: E402
from fastvideo3.core.cost import price_of, price_of_route  # noqa: E402
from fastvideo3.core.tasklock import acquire as lock_acquire  # noqa: E402
from fastvideo3.core.tasklock import heartbeat as lock_heartbeat  # noqa: E402
from fastvideo3.core.tasklock import release as lock_release  # noqa: E402
from fastvideo3.providers.base import (  # noqa: E402
    Capability,
    I2VProvider,
    TalkProvider,
    poll_until_done,
)
from fastvideo3.providers.factory import make_provider  # noqa: E402
from fastvideo3.utils.audio import wav_duration  # noqa: E402
from fastvideo3.utils.lipsync import lipsync_diag, locate_boxes, scan_lipsync  # noqa: E402

# 单条镜头预估费用低于这个数就视为"近乎免费"，可以直接跑；高于它必须 --spend 显式确认。
FREE_LIMIT_CNY = 0.5


def _load_shot(run_dir: Path, shot_id: str) -> tuple[str, str]:
    import json

    script = json.loads((run_dir / "script.json").read_text(encoding="utf-8"))
    for sc in script["scenes"]:
        for sh in sc["shots"]:
            if sh["shot_id"] == shot_id:
                return sh.get("visual_prompt") or "", sc.get("scene_id", "")
    raise SystemExit(f"剧本里找不到镜头 {shot_id}")


def _measure(tag: str, clip: Path, wav: Path, work: Path, vlm, min_corr: float) -> dict:
    s = scan_lipsync(clip, wav, work / f"scan_{tag}", min_corr=min_corr)
    print(f"\n[{tag}] {clip.name}")
    print(
        f"  扫描：verdict={s['verdict']} corr={s['corr']} 乱序对照={s['null_corr']} "
        f"最强框={s['best_box']} 运动幅度={s['mouth_amp']}"
    )
    print(f"        {s['reason']}")
    out = {"scan": s}
    if vlm is not None:
        boxes = asyncio.run(locate_boxes(clip, vlm, work / f"vlm_{tag}"))
        d = lipsync_diag(clip, wav, work / f"diag_{tag}", boxes=boxes, min_corr=min_corr)
        out["vlm"] = d
        print(
            f"  视觉模型定位：verdict={d['verdict']} 嘴={d['corr']} 眼对照={d['eyes_corr']} "
            f"背景对照={d['bg_corr']} located={d['located']}"
        )
        print(f"        {d['reason']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--shot", required=True, help="对白镜头 id，如 shot_1_2")
    ap.add_argument("--quality", default="cinema")
    ap.add_argument("--with-vlm", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=20)
    ap.add_argument("--cap", type=float, default=5.0, help="单价红线 元/秒（用户硬约束）")
    ap.add_argument(
        "--channel",
        choices=("main", "cheap"),
        default="main",
        help="main=按当前路由走（现=fal InfiniteTalk，收费）；cheap=钉回百炼 videoretalk 兜底档",
    )
    ap.add_argument("--spend", action="store_true", help="确认可以产生真实计费")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    shot_id = args.shot
    frame = run_dir / "storyboard" / f"shot_{shot_id}.png"
    wav = run_dir / "audio" / f"shot_{shot_id}_mix.wav"
    wav16 = run_dir / "audio" / f"shot_{shot_id}_mix_16k.wav"
    for p, hint in ((frame, "首帧（storyboard）"), (wav, "对白混音（audio）")):
        if not p.exists():
            print(f"缺 {hint}：{p}")
            return 1
    audio_for_drive = wav16 if wav16.exists() else wav

    cfg = load_config(quality=args.quality)
    # talk_mode 必须与**被装配出来的通道**成对：只吃"图片+音频"的通道（wan2.2-s2v）
    # 拿不到视频输入，这里若硬塞 v2v_overlay，脚本测的就不是管线真实会走的那条路。
    # （具体 mode 在装配出 talk provider 之后按 supports_video_input 定，见下面。）
    cfg.max_video_cost_per_sec = args.cap
    if args.channel == "cheap":
        # 零成本验证链路：绕过"主通道"，直接打兜底的 videoretalk（¥0.08/秒 + 免费额度）
        cfg.routing["talk"] = ProviderRouting(default="dashscope_videoretalk")
    prompt, scene_id = _load_shot(run_dir, shot_id)
    dur = max(int(cfg.shot_duration_sec), int(round(wav_duration(wav))) + 1)

    # 预估按"**我们实际会发的那一档**"取价（分档模型按分辨率换档），不是笼统模型价
    talk_rate = price_of_route(cfg.routing["talk"].default, "talk", cfg.i2v_resolution)
    i2v_model = cfg.model_for("i2v", cfg.routing["i2v"].default, "")
    i2v_rate = price_of(i2v_model, "i2v", 1.0)
    est_cost = (i2v_rate + talk_rate) * dur
    print(f"预估本条花费 ≈ ¥{est_cost:.2f}（i2v ¥{i2v_rate}/s + 口型 ¥{talk_rate}/s × {dur}s）")
    if est_cost > FREE_LIMIT_CNY and not args.spend:
        print("未加 --spend：不提交、不加锁、不计费。确认费用后加 --spend 再跑；"
              "只想验证链路请加 --channel cheap")
        return 2

    print(f"# {run_dir.name} / {shot_id}（场景 {scene_id}）")
    print(f"  首帧 {frame.name}  驱动音频 {audio_for_drive.name} {wav_duration(wav):.2f}s  "
          f"计划时长 {dur}s")
    print(
        f"  通道：i2v={i2v_model}(¥{i2v_rate}/s) → "
        f"talk={cfg.routing['talk'].default}(¥{talk_rate}/s，按档位 {cfg.i2v_resolution} 计价)"
    )
    print(f"  单价红线 ¥{args.cap}/s（比这贵的通道在装配阶段就会报错，不会先烧钱）")

    out_dir = run_dir / "_lipsync_verify"
    out_dir.mkdir(exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    # 全局单任务锁：与网页/CLI 的生成共用一把锁，占用中就退出（不并发烧钱）
    out_root = run_dir.parent
    got, state = lock_acquire(out_root, f"{run_dir.name}#verify:{shot_id}", "verify")
    if not got:
        print("\n任务锁被占用，按「一次只跑一个生成任务」的约定直接退出（不并发烧钱）：")
        print(f"  {state}")
        return 3
    print(f"  已持有全局任务锁：{run_dir.name}#verify:{shot_id}")

    try:
        i2v = make_provider(Capability.I2V, cfg)
        talk = make_provider(Capability.TALK, cfg)
        # 装配层返回基类；这里必须是可提交的视频通道，否则说明路由被改到了别的东西上
        assert isinstance(i2v, I2VProvider), f"i2v 路由异常：{type(i2v).__name__}"
        assert isinstance(talk, TalkProvider), f"talk 路由异常：{type(talk).__name__}"
        use_video = bool(getattr(talk, "supports_video_input", False))
        cfg.talk_mode = "v2v_overlay" if use_video else "s2v_direct"
        print(f"  通道能力：supports_video_input={use_video} → talk_mode={cfg.talk_mode}")
        clip = out_dir / f"clip_{shot_id}.mp4"
        print(f"\n→ 1/2 i2v 运动片段：{i2v.name}/{i2v.model}")
        ext = asyncio.run(
            i2v.submit(
                dest=clip, image=frame, prompt=prompt,
                resolution=cfg.i2v_resolution, duration=dur,
            )
        )
        st = asyncio.run(
            poll_until_done(i2v.poll, ext, timeout_sec=args.timeout_min * 60,
                            fast_interval=5, slow_interval=15)
        )
        if st.status != "succeeded" or not clip.exists():
            print(f"  i2v 失败：{st.error}  （未产生口型素材，停止）")
            return 2
        print(f"  ok {clip} ({clip.stat().st_size // 1024} KB)")
        lock_heartbeat(out_root)

        retalk_out = out_dir / f"talk_{shot_id}.mp4"
        print(f"\n→ 2/2 口型重对：{talk.name}/{talk.model}")
        ext2 = asyncio.run(
            talk.submit(dest=retalk_out, image=frame, audio=audio_for_drive,
                        video=clip if use_video else None,
                        prompt=prompt, resolution=cfg.i2v_resolution, duration=dur)
        )
        st2 = asyncio.run(
            poll_until_done(talk.poll, ext2, timeout_sec=args.timeout_min * 60,
                            fast_interval=5, slow_interval=15)
        )
        if st2.status != "succeeded" or not retalk_out.exists():
            print(f"  retalk 失败：{str(st2.error)[:200]}")
            return 2
        print(f"  ok {retalk_out} ({retalk_out.stat().st_size // 1024} KB)")

        vlm = None
        if args.with_vlm:
            vlm = make_provider(Capability.VLM, cfg)

        base = _measure("i2v 原片段（对照）", clip, wav, work, vlm, cfg.lipsync_min_corr)
        after = _measure("retalk 之后", retalk_out, wav, work, vlm, cfg.lipsync_min_corr)
        print(
            f"\n结论候选：原片段 {base['scan']['verdict']} → retalk 后 "
            f"{after['scan']['verdict']}"
            f"（corr {base['scan']['corr']} → {after['scan']['corr']}，"
            f"乱序对照 {after['scan']['null_corr']}）"
        )
        print(
            "人眼复核：两个 mp4 都看一遍。尺子只回答「有没有随语音运动的区域」，"
            "嘴形像不像话、有没有多张嘴或僵脸，仍需人工判定。"
        )
        print("\n费用：以实际计费为准；videoretalk 有 1800 秒免费额度，"
              "i2v 走 wan3.0-video 试用额度时本条应≈¥0")
        return 0
    finally:
        lock_heartbeat(out_root)
        lock_release(out_root, f"{run_dir.name}#verify:{shot_id}")


if __name__ == "__main__":
    raise SystemExit(main())
