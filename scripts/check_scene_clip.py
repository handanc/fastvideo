"""场景片段事后校验（零 API 成本）：把"手动跑一遍"变成一次结构化验收。

用法：
    python scripts/check_scene_clip.py output/run_20261013_xxxxxx
    python scripts/check_scene_clip.py run_20261013_xxxxxx      # 也接受 run_id

场景级生成成为默认路径后，有三条假设是离线测试验不了的（模型行为，只能真跑）：
1. **时间戳被遵守** —— 提示词写"第2个镜头[4-9秒]"，实际切点是否落在 4s/9s 附近；
2. **台词逐字保真** —— 5-6 句台词塞进一次调用，模型会不会漏句/并句/改字；
3. **音色跨场景稳定** —— 同一角色在两个独立场景片段里是不是同一副嗓子。

其中只有 1 能自动量（ffmpeg 切点检测 vs 落盘的 windows）。2 和 3 要人耳，
所以本脚本把它们变成**可逐条听的切片**：按 windows 把每镜头的音轨切出来、
按角色把各组里的发声段切出来，文件名即对照表，听的时候不用猜哪段是谁。

不做的事：不调任何计费接口、不重跑生成、不改产物（只往 review/voice_check/ 写切片）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from fastvideo3.core.config import PROJECT_ROOT  # noqa: E402
from fastvideo3.utils.ffmpeg import find_ffmpeg  # noqa: E402

CUT_TOL_SEC = 0.7  # 切点容差：模型不是帧精确的，卡太紧会误报


def _ffmpeg() -> str:
    return find_ffmpeg()


def _cuts(clip: Path, threshold: float = 0.3) -> list[float]:
    """ffmpeg 场景切点检测（与人工看片用的同一把尺子）。"""
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(clip),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr))


def _cut_audio(clip: Path, start: float, end: float, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(clip),
         "-vn", "-ac", "1", "-ar", "16000", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 1024


def check_run(run_dir: Path) -> int:
    clips = run_dir / "clips"
    metas = sorted(clips.glob("scene_*.json")) if clips.is_dir() else []
    if not metas:
        print("没有找到 clips/scene_*.json —— 这条 run 没走场景级生成（或还没跑到 videos 步）")
        return 1
    script_data = json.loads((run_dir / "script.json").read_text(encoding="utf-8"))
    shots = {sh["shot_id"]: sh for sc in script_data.get("scenes", []) for sh in sc.get("shots", [])}
    out_dir = run_dir / "review" / "voice_check"
    bad = 0

    for jp in metas:
        meta = json.loads(jp.read_text(encoding="utf-8"))
        gid = meta["group_id"]
        clip = clips / f"scene_{gid}.mp4"
        if not clip.is_file():
            print(f"[{gid}] 有映射 json 但没有片段文件（生成失败或被删）")
            bad += 1
            continue
        windows = {k: (float(v[0]), float(v[1])) for k, v in (meta.get("windows") or {}).items()}
        print(f"\n=== {gid}  {meta.get('mode')}  {meta.get('duration_sec')}s  "
              f"实付 {meta.get('billed_seconds')}s @ {meta.get('resolution_billed')}")
        print(f"    理由: {meta.get('mode_reason')}")
        print(f"    实发参考: {meta.get('ref_images')} {meta.get('ref_audios')}")

        # ── 1) 时间戳是否被遵守 ──
        # 相邻窗口共享边界（上一镜的终点=下一镜的起点），去重后才是"该有切点的位置"
        inner = sorted(set(w for sid, (s, e) in windows.items() for w in (s, e)))
        expected = [w for w in inner if 0.5 < w < meta.get("duration_sec", 0) - 0.5]
        got = _cuts(clip)
        print(f"    提示词窗口边界: {[round(w, 1) for w in expected]}")
        print(f"    实测切点:       {[round(c, 1) for c in got]}")
        for w in expected:
            near = min((abs(c - w) for c in got), default=99)
            mark = "✔" if near <= CUT_TOL_SEC else "✘"
            if near > CUT_TOL_SEC:
                bad += 1
            print(f"      {mark} 窗口边界 {w:5.1f}s 最近切点偏差 {near:.2f}s"
                  + ("" if near <= CUT_TOL_SEC else "  ← 模型没按时间戳切（或切点太软检测不到）"))

        # ── 2) 台词逐字保真：切出每镜头音轨，文件名=镜头，打印=剧本原文 ──
        for sid, (s, e) in windows.items():
            text = " / ".join(d.get("text", "") for d in (shots.get(sid) or {}).get("dialogues", []))
            wav = out_dir / f"{gid}__{sid}.wav"
            ok = _cut_audio(clip, s, e, wav)
            print(f"      听 {wav.name if ok else '(切片失败)'}  应听到: {text or '(无台词)'}")

        # ── 3) 音色跨场景：同角色在不同组里的发声段，文件名带组号，A/B 对比听 ──
        for sid, (s, e) in windows.items():
            for d in (shots.get(sid) or {}).get("dialogues", []):
                cid = d.get("character_id", "?")
                wav = out_dir / f"voice_{cid}__in_{gid}.wav"
                _cut_audio(clip, s, min(e, s + 4.0), wav)
    voices = sorted({p.name.split("__")[0] for p in out_dir.glob("voice_*")}) if out_dir.is_dir() else []
    if voices:
        print("\n音色 A/B 对照（同一角色跨组各听一段）：")
        for v in voices:
            print("   ", v, "→", sorted(p.name for p in out_dir.glob(f"{v}__*")))
    print(f"\n结论：{'全部窗口边界命中' if bad == 0 else f'{bad} 处待人工确认'}；"
          f"台词与音色切片在 {out_dir}")
    return 0 if bad == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run 目录或 run_id")
    args = ap.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = PROJECT_ROOT / "output" / args.run
    if not run_dir.is_dir():
        print(f"run 不存在：{args.run}")
        return 1
    return check_run(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
