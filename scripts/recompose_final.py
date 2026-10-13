"""从盘上已有产物**重新合成成片**（不重新调用任何付费生成通道）。

为什么需要它：字幕时间轴、字幕字体、音轨替换策略这类问题全部发生在 m7（纯本地 ffmpeg），
但按现有入口要重看效果就得重跑 videos 步骤 —— 那是几十块钱的口型/运镜费用。
本脚本直接吃盘上的 `clips/`、`talk/`、`audio/` 产物重跑 m7，**零 API 调用、零计费**。

它还会顺手补齐两件事（都是 2026-10-05 修字幕/音频时发现的）：
- 缺 `audio/shot_{id}_driver.wav` 时，从当时的镜头混音（`shot_{id}_mix.wav` /
  `shot_{id}_duo_mix.wav`）重建 —— 否则 A/B 链的口型片段会继续用模型回传的音轨（发闷、
  句尾被截）。**C 链（`talk_mode=scripted_talk` 且 `voice_source=model`）跳过这一步**：
  那条链的口型是模型对着自己发声画的，补建 driver 会让 m7 换回我们的配音 → 声画错开 0～2.2 秒；
- 覆盖前先备份 `final/final.mp4` → `final/final_prev_<时间戳>.mp4`，随时可回退。

用法：
    python scripts/recompose_final.py output/run_20261005_130208            # 只看会做什么
    python scripts/recompose_final.py output/run_20261005_130208 --write    # 真重合成
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastvideo3.core.config import load_config  # noqa: E402
from fastvideo3.core.schema import AudioLine, Script  # noqa: E402
from fastvideo3.core.storage import RunStorage  # noqa: E402


def _reconstruct_drivers(storage: RunStorage, talk_ids: list[str]) -> list[str]:
    """给缺 driver 的口型镜头补一份"真实驱动音频"（旧 run 没有这个产物）。

    取当时那一份镜头混音即可：单段生成时驱动音频就是它。分段生成（`_g1/_g2`）时
    按段号顺序零间隔拼接 —— 与拼接出的视频时间轴一致，不能用整镜头混音
    （它多插了段间换气，会让后段口型与声音错位）。
    """
    from fastvideo3.utils.audio import concat_wavs

    made: list[str] = []
    for sid in talk_ids:
        out = storage.path("audio", f"shot_{sid}_driver.wav")
        if out.exists():
            continue
        single = storage.path("audio", f"shot_{sid}_mix.wav")
        duo = storage.path("audio", f"shot_{sid}_duo_mix.wav")
        parts: list[Path] = []
        if single.exists():
            parts = [single]
        elif duo.exists():
            parts = [duo]
        else:
            segs = sorted(
                storage.dir("audio").glob(f"shot_{sid}*_g*_mix.wav"),
                key=lambda p: int(p.stem.rsplit("_g", 1)[1]),
            )
            parts = list(segs)
        if not parts:
            continue
        if len(parts) == 1:
            shutil.copyfile(parts[0], out)
        else:
            concat_wavs(parts, out, gap_sec=0.0)
        made.append(sid)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--write", action="store_true", help="真的重合成（默认只报告计划）")
    ap.add_argument("--quality", default="", help="覆盖档位（默认读 run_meta.json）")
    args = ap.parse_args()

    # 绝对路径：concat 列表/血缘台账都按相对路径解析，传相对路径会拼出双重路径
    run_dir = Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    script = Script.model_validate(json.loads((run_dir / "script.json").read_text("utf-8")))
    raw_lines = json.loads((run_dir / "audio_map.json").read_text(encoding="utf-8"))
    lines = [AudioLine(**x) for x in raw_lines]
    run_id = run_dir.name
    storage = RunStorage(run_id, root=run_dir.parent)

    cfg = load_config(quality=args.quality or meta.get("quality") or None)
    cfg.fake = False
    cfg.talk_mode = meta.get("talk_mode") or cfg.talk_mode
    # 音轨来源与 run 级模型选择也要按 meta 恢复 —— 下面判断"该不该保留模型原声"用的就是
    # cfg.voice_source，读成 .env/yaml 的默认值会把这条 run 的听感改错（m7 换不换母带看它）
    cfg.voice_source = meta.get("voice_source") or cfg.voice_source
    from fastvideo3.core.config import apply_run_overrides

    apply_run_overrides(cfg, meta)

    talks = {
        sh.shot_id: str(p)
        for sh in script.all_shots()
        if (p := storage.path("talk", f"shot_{sh.shot_id}.mp4")).exists()
    }
    clips = {
        sh.shot_id: str(p)
        for sh in script.all_shots()
        if (p := storage.path("clips", f"shot_{sh.shot_id}.mp4")).exists()
    }
    missing = [sh.shot_id for sh in script.all_shots()
               if sh.shot_id not in talks and sh.shot_id not in clips]
    silent = [sh.shot_id for sh in script.all_shots()
              if sh.dialogues and sh.shot_id not in talks]

    print(f"run: {run_id}  档位={cfg.quality}  talk_mode={cfg.talk_mode}")
    print(f"  口型片段 {len(talks)} 个 / 运动片段 {len(clips)} 个 / 台词 {len(lines)} 句")
    if missing:
        print(f"  ⚠ 无画面素材的镜头（会被跳过）: {missing}")
    if silent:
        print(f"  ⚠ 有对白但没有口型产物（成片里是「无口型叠音」）: {silent}")
    # C 链 + voice_source=model 的成片**就该用模型原声**：绝不能补建 driver.wav ——
    # 建了 m7 就会把音轨换成我们的配音，而口型是模型对着自己的发声画的，
    # 实测声画会错开 0～2.2 秒且偏移不恒定（测量表见 core/talkmode.py）。
    keep_model_audio = cfg.talk_mode == "scripted_talk" and cfg.voice_source != "master"
    made = []
    if keep_model_audio:
        print("  C 链模型原声：跳过 driver 补建（补了会把音轨换回我们的配音 → 嘴不上）")
    elif args.write:
        made = _reconstruct_drivers(storage, list(talks))
    if made:
        print(f"  补齐 driver 音轨（换回我们的 TTS 母带）: {made}")
    if not args.write:
        print("\n（未加 --write：只报告计划，不改动任何产物）")
        return 0

    old = storage.path("final", "final.mp4")
    if old.exists():
        bak = storage.path("final", f"final_prev_{time.strftime('%H%M%S')}.mp4")
        shutil.copyfile(old, bak)
        print(f"  已备份旧成片 → {bak.name}")

    import asyncio

    from fastvideo3.modules.m7_edit import compose_final, subtitle_sidecar_path

    final = asyncio.run(compose_final(cfg, script, clips, talks, lines, storage))
    srt = subtitle_sidecar_path(storage)
    cues = [b.splitlines() for b in srt.read_text(encoding="utf-8").strip().split(chr(10) * 2)]
    print(f"\n✔ 成片: {final}")
    print(f"  字幕 {len(cues)} 条（文本逐字来自剧本），示例:")
    for b in cues[:4]:
        print(f"    {b[1]}  {b[2][:24]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
