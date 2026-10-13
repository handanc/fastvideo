"""fal.ai InfiniteTalk 口型通道的体检与实测（`check` 零成本，`shot` 才花钱）。

本期它是口型**兜底**通道：主通道 = 百炼 wan2.2-s2v（见 configs/default.yaml）。
`check` 会打印每一档实际解析到的路由，所以"兜底到底还在不在、价格与规格有没有漂"
用这条命令确认最省事。

为什么需要它（本项目被"规格猜错"坑过两次）：
- wan2.2-s2v 曾被路由到错误端点，服务端只回 `400 url error`，于是整条 ¥0.5-0.9/秒
  的专用口型通道被误判成"不能对口型"；
- videoretalk 的单价在 pricing.yaml 里写错了 6 倍（0.50 vs 真实 0.08），直接扭曲过
  通道选型。
结论：**外部供应商的规格与价格必须实测核对，不能靠记忆**。`check` 就是干这个的，
它一分钱不花，并且会在 fal 改枚举/改价/改上限时当场报红。

用法：
    # 1) 零成本体检（KEY、端点规格、价格、档位路由、30s 成本预估）
    python scripts/probe_fal_infinitalk.py check

    # 2) 真跑一条镜头（默认只打印计划与费用，加 --spend 才真提交）
    python scripts/probe_fal_infinitalk.py shot \\
        --clip output/<run>/clips/shot_1_2.mp4 \\
        --wav  output/<run>/audio/shot_1_2_mix.wav
    python scripts/probe_fal_infinitalk.py shot --frame <首帧png> --wav <混音wav> --spend

    # 3) 出片后人眼定案（免费，带逐帧响度条的嘴部接片）
    python scripts/probe_lipsync.py sheet output/<run> --shot shot_1_2 --clip <上一步产物>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastvideo3.core.config import load_config  # noqa: E402
from fastvideo3.core.cost import (  # noqa: E402
    estimate_run_cost_cny,
    load_pricing,
    price_of,
    price_of_provider,
    price_of_route,
)
from fastvideo3.providers.base import poll_until_done  # noqa: E402
from fastvideo3.providers.fal import (  # noqa: E402
    IMAGE_ENDPOINT,
    MAX_FRAMES_I2V,
    MAX_FRAMES_V2V,
    MODEL_FPS,
    V2V_ENDPOINT,
    FalInfiniteTalkProvider,
    frames_for,
    probe_key,
)

OAS_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={ep}"
FX_CNY_PER_USD = 7.15  # 与 pricing.yaml 的注释同源；改了那里也要改这里


def _fetch_schema(endpoint: str) -> dict | None:
    """拉 fal 官方 OpenAPI（无需 KEY）。失败返回 None 并说明，不猜。"""
    req = urllib.request.Request(
        OAS_URL.format(ep=endpoint), headers={"User-Agent": "fastvideo3-probe/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310  # 固定 https 域名
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  ! 取不到 {endpoint} 的 OpenAPI: {e}")
        return None


def _fetch_billing(endpoint: str) -> tuple[float, str]:
    """单价不在 queue OpenAPI 里，只能从模型页的 RSC 负载取（2026-10-04 实测）。

    页面里是转义过的 JSON（`endpointBilling\":{\"price\":0.3,...}`），所以正则要
    同时容忍 `"` 与 `\\"` 两种写法。取不到就返回 (0, "")，由调用方报"无法核对"，
    **绝不拿记忆里的价格当实测值**。
    """
    url = f"https://fal.ai/models/{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:  # noqa: S310
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"  ! 取不到 {endpoint} 的计费页: {e}")
        return 0.0, ""
    m = re.search(
        r"endpointBilling[\\\"':\s]+.{0,400}?price[\\\"':\s]+([0-9.]+)", html
    )
    unit = re.search(r"billing_unit[\\\"':\s]+([a-z_ ]+)", html)
    msg = re.search(r"billingMessage[\\\"':\s]+([^\"\\]{5,120})", html)
    price = float(m.group(1)) if m else 0.0
    note = (unit.group(1) if unit else "seconds") + (f" | {msg.group(1)}" if msg else "")
    return price, note


def _input_schema(oas: dict, endpoint: str) -> dict:
    ref = (
        oas.get("paths", {})
        .get(f"/{endpoint}", {})
        .get("post", {})
        .get("requestBody", {})
    )
    for content in (ref.get("content") or {}).values():
        s = content.get("schema") or {}
        if "$ref" in s:  # 内联引用形式：#/components/schemas/XxxInput
            name = s["$ref"].rsplit("/", 1)[-1]
            return (oas.get("components", {}).get("schemas") or {}).get(name, {})
        return s
    return {}


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config()
    pricing = load_pricing()
    models = pricing.get("models") or {}
    problems: list[str] = []

    print("=== 1) KEY 与账户（空请求探活，不计费）===")
    if not cfg.fal_api_key:
        print("  · .env 里没有 FAL_KEY —— 兜底通道 fal_infinitalk 不可用；主通道仍是百炼 s2v")
        problems.append("FAL_KEY 未配置（只影响兜底通道）")
        status, detail = "no_key", "跳过探活"
    else:
        status, detail = asyncio.run(probe_key(cfg.fal_api_key))
        print(f"  [{status}] {detail}")
        if status not in ("ok",):
            problems.append(f"fal 探活 {status}: {detail}")

    print("\n=== 2) 三档路由与单价 ===")
    for quality in ("economy", "high", "cinema"):
        c = load_config(quality=quality)
        route = c.routing.get("talk")
        rate = price_of_route(route.default, "talk", c.i2v_resolution) if route else 0.0
        worst = price_of_provider(route.default, "talk") if route else 0.0
        print(
            f"  {quality:8s} 主={route.default if route else '-':16s} "
            f"兜底={route.fallback if route else '-':22s} 档位={c.i2v_resolution} "
            f"talk 实发价=¥{rate:.2f}/s 红线校验价=¥{worst:.2f}/s (上限 ¥{c.max_video_cost_per_sec})"
        )
        if route and worst > c.max_video_cost_per_sec:
            problems.append(f"{quality} 最坏档 ¥{worst}/秒 超红线 → 装配期会被拒")

    print("\n=== 3) fal 官方规格 vs 我们写死的规格 ===")
    for endpoint, want_frames_cap in ((IMAGE_ENDPOINT, MAX_FRAMES_I2V), (V2V_ENDPOINT, MAX_FRAMES_V2V)):
        oas = _fetch_schema(endpoint)
        if not oas:
            problems.append(f"{endpoint} OpenAPI 取不到（无法核对规格）")
            continue
        sch = _input_schema(oas, endpoint)
        props = sch.get("properties") or {}
        req = sch.get("required") or []
        res_enum = (props.get("resolution") or {}).get("enum")
        nf = props.get("num_frames") or {}
        acc = (props.get("acceleration") or {}).get("enum")
        print(f"\n  端点 {endpoint}")
        print(f"    必填      : {req}")
        print(f"    resolution: {res_enum}  default={(props.get('resolution') or {}).get('default')}")
        print(f"    num_frames: min={nf.get('minimum')} max={nf.get('maximum')} default={nf.get('default')}")
        print(f"    acceleration: {acc}")
        price_usd, note = _fetch_billing(endpoint)
        key = f"{endpoint}@480p"
        ours = float((models.get(key) or {}).get("price_cny") or 0)
        if price_usd:
            true_cny = price_usd * FX_CNY_PER_USD
            print(
                f"    官方计费  : {note} → ${price_usd}/秒 = ¥{true_cny:.2f}；"
                f"我们记 {key} = ¥{ours:.2f}"
            )
            if abs(ours - true_cny) > 0.15:
                problems.append(
                    f"{key} 价格漂移：官方约 ¥{true_cny:.2f}/秒，pricing.yaml 记 ¥{ours:.2f}"
                )
            # 720p 翻倍是页面明写的规则，单独核一遍我们记的那一档
            key720 = f"{endpoint}@720p"
            ours720 = float((models.get(key720) or {}).get("price_cny") or 0)
            if abs(ours720 - true_cny * 2) > 0.30:
                problems.append(
                    f"{key720} 应为 480p 的两倍（≈¥{true_cny * 2:.2f}），表里 ¥{ours720}"
                )
        else:
            print("    ! 未能从模型页读到单价（fal 页面结构变了）→ 价格未核对")
            problems.append(f"{endpoint} 单价无法核对（页面结构变化），请人工确认")
        # 与代码常量核对
        if res_enum and set(res_enum) != {"480p", "720p"}:
            problems.append(f"{endpoint} 分辨率枚举变了: {res_enum}")
        if nf.get("maximum") and int(nf["maximum"]) != want_frames_cap:
            print(f"    ! 帧数上限官方={nf['maximum']} 我们写死={want_frames_cap}")
            problems.append(f"{endpoint} num_frames 上限漂移 {want_frames_cap}→{nf['maximum']}")
        for field in req:
            if field not in props:
                problems.append(f"{endpoint} 必填字段 {field} 不在 schema 里")

    print("\n=== 4) 我们的换算（帧数必须是 4n+1）===")
    for sec in (3, 5, 8, 12, 15, 20):
        f_v2v = frames_for(sec, max_frames=MAX_FRAMES_V2V)
        f_i2v = frames_for(sec, max_frames=MAX_FRAMES_I2V)
        print(f"    {sec:>2}s → v2v {f_v2v} 帧 ({f_v2v / MODEL_FPS:.2f}s) / i2v {f_i2v} 帧 ({f_i2v / MODEL_FPS:.2f}s)")

    print("\n=== 5) 成本预估（口型按实发档）===")
    for quality in ("economy", "high", "cinema"):
        c = load_config(quality=quality)
        c.fake = False
        est = estimate_run_cost_cny(c, 30)
        print(
            f"    {quality:8s} 30s 成片 ≈ ¥{est['lo']:.0f}-{est['hi']:.0f}"
            f"（其中口型 ¥{est['talk_rate_per_sec']:.2f}/秒 × 对白约 {est['dialogue_sec']:.0f}s）"
            f"，当前熔断 ¥{c.budget_limit_cny}"
        )
        if est["lo"] > c.budget_limit_cny:
            problems.append(
                f"{quality} 预估下限 ¥{est['lo']:.0f} 已超过熔断 ¥{c.budget_limit_cny}："
                f"口型步骤会跳闸 → 成片没有对口型（请把 FV3_BUDGET_LIMIT_CNY 提到 ≥¥{est['hi'] + 40:.0f}）"
            )

    print("\n=== 结论 ===")
    if problems:
        for p in problems:
            print(f"  ✘ {p}")
        return 1
    print("  ✔ KEY 可用、规格与价格与官方一致、熔断覆盖得住口型开销")
    return 0


def _fps_of(path: Path) -> float:
    """从 `ffmpeg -i` 的 stderr 里取视频流帧率（fal 产物到底几 fps 是本次要回答的问题）。"""
    from fastvideo3.utils.ffmpeg import find_ffmpeg

    proc = subprocess.run(
        [find_ffmpeg(), "-i", str(path)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", proc.stderr)
    return float(m.group(1)) if m else 0.0


def cmd_shot(args: argparse.Namespace) -> int:
    clip = Path(args.clip) if args.clip else None
    frame = Path(args.frame) if args.frame else None
    wav = Path(args.wav)
    for p in (clip, frame, wav):
        if p is not None and not p.exists():
            print(f"✘ 素材不存在: {p}")
            return 2
    if clip is None and frame is None:
        print("✘ 必须给 --clip（走 /video-to-video）或 --frame（走 i2v 直出）")
        return 2

    cfg = load_config(quality=args.quality)
    cfg.fake = False
    provider = FalInfiniteTalkProvider(
        cfg.fal_api_key, IMAGE_ENDPOINT, V2V_ENDPOINT, acceleration=args.acceleration
    )
    from fastvideo3.utils.audio import wav_duration
    from fastvideo3.utils.ffmpeg import probe_duration, probe_size

    dur = wav_duration(wav)
    if clip is not None:
        dur = max(dur, probe_duration(clip))
    frames = frames_for(dur, max_frames=MAX_FRAMES_V2V if clip else MAX_FRAMES_I2V)
    sec = frames / MODEL_FPS
    usd_per_sec = 0.30 if clip else 0.20
    if args.resolution == "720p":
        usd_per_sec *= 2
    cost_usd = usd_per_sec * sec
    tier_key = f"{V2V_ENDPOINT if clip else IMAGE_ENDPOINT}@{args.resolution}"
    print(
        f"计划：{'video-to-video（在已有运动片段上重对口型）' if clip else 'i2v（首帧+音频直出）'}\n"
        f"  驱动音频 {wav_duration(wav):.2f}s{' / 片段 ' + format(probe_duration(clip), '.2f') + 's' if clip else ''}\n"
        f"  num_frames={frames} → 约 {sec:.2f}s，分辨率 {args.resolution}，acceleration={args.acceleration}\n"
        f"  官方价约 ${usd_per_sec}/秒 → 本次约 ${cost_usd:.2f}（≈¥{cost_usd * FX_CNY_PER_USD:.2f}，"
        f"台账键 {tier_key} 记 ¥{price_of(tier_key, 'talk', 1.0):.2f}/秒）"
    )
    if clip is not None:
        w, h = probe_size(clip)
        print(f"  输入片段 {w}x{h} @ {_fps_of(clip):.2f}fps")
    if not args.spend:
        print("\n（未加 --spend：只打印计划，不提交、不计费。确认后加 --spend 再跑）")
        return 0

    from fastvideo3.core import tasklock
    from fastvideo3.core.config import OUTPUT_DIR

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"fal_{'v2v' if clip else 'i2v'}_{wav.stem}.mp4"
    ok, st = tasklock.acquire(OUTPUT_DIR, "fal_probe", "step")
    if not ok:
        print(f"✘ 已有任务在跑（{st.get('run_id')} / {st.get('kind')}），一次只允许一个生成任务")
        return 3
    try:
        ext_id = asyncio.run(_submit(provider, dest, frame, wav, clip, args))
        print(f"  已提交 request_id={ext_id}，轮询中…")
        status = asyncio.run(
            poll_until_done(provider.poll, ext_id, timeout_sec=args.timeout * 60)
        )
        if status.status != "succeeded":
            print(f"✘ 失败: {status.error}")
            return 1
        w, h = probe_size(dest)
        print(
            f"\n✔ 产物 {dest}\n  输出 {w}x{h} @ {_fps_of(dest):.2f}fps / {probe_duration(dest):.2f}s\n"
            f"  竖屏是否被裁：源 {'9:16' if clip and probe_size(clip)[1] > probe_size(clip)[0] else '未知'}"
            f" → 产物 {'9:16' if h > w else ('16:9' if w > h else '1:1')}"
        )
        print("  下一步（免费、人眼定案）：\n"
              f"    python scripts/probe_lipsync.py sheet <run目录> --shot <id> --clip {dest} --wav {wav}")
        return 0
    finally:
        tasklock.release(OUTPUT_DIR, "fal_probe")


async def _submit(provider, dest, frame, wav, clip, args) -> str:
    """分辨率按 --resolution 显式指定（这条通道就是用来试不同档位的，不走档位钳位）。"""
    return await provider.submit(
        dest=dest, image=frame, audio=wav, video=clip,
        prompt=args.prompt, resolution=args.resolution,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="零成本体检：KEY/官方规格/价格/熔断（不提交任何任务）")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("shot", help="真跑一条镜头（默认只打印计划，--spend 才提交计费）")
    s.add_argument("--clip", default="", help="已有运动片段 mp4（走 /video-to-video）")
    s.add_argument("--frame", default="", help="首帧/定妆照 png（走 i2v 直出）")
    s.add_argument("--wav", required=True, help="驱动音频 wav（对白混音）")
    s.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    s.add_argument("--acceleration", default="regular", choices=["none", "regular", "high"])
    s.add_argument("--prompt", default="", help="画面描述（fal 必填；留空用中性描述）")
    s.add_argument("--seconds", type=float, default=0, help="覆盖时长（默认由音频/片段决定）")
    s.add_argument("--quality", default="economy", choices=["economy", "high", "cinema"])
    s.add_argument("--out", default="output/_fal_probe")
    s.add_argument("--timeout", type=int, default=25, help="轮询超时（分钟）")
    s.add_argument("--spend", action="store_true", help="真的提交（否则只打印计划）")
    s.set_defaults(func=cmd_shot)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
