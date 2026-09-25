"""m7 剪辑合成（教学第 5 讲：剪辑与转场）。

规格统一 → 逐对转场合并（xfade）→ SRT 字幕 → BGM 混音 → 成片。
转场失败回退硬切；BGM 失败不阻塞 —— 成片永远可交付。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.schema import AudioLine, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.utils import ffmpeg as ff


def build_srt(
    script: Script,
    audio_lines: list[AudioLine],
    clips_order: list[str],
    clip_durations: dict[str, float],
    transition_out: dict[str, str],
    fade_dur: float,
) -> str:
    """按镜头顺序累积时间轴，导出 SRT。"""
    by_shot: dict[str, list[AudioLine]] = {}
    for ln in audio_lines:
        by_shot.setdefault(ln.shot_id, []).append(ln)

    entries: list[str] = []
    idx = 1
    t = 0.0
    for i, sid in enumerate(clips_order):
        dur = clip_durations.get(sid, 5.0)
        local = 0.3
        for ln in by_shot.get(sid, []):
            start = t + local
            end = start + ln.duration_sec
            entries.append(f"{idx}\n{_fmt(start)} --> {_fmt(end)}\n{ln.text}\n")
            idx += 1
            local += ln.duration_sec
        t += dur
        if i < len(clips_order) - 1 and transition_out.get(sid, "cut") != "cut":
            t -= fade_dur
    return "\n".join(entries)


def _fmt(sec: float) -> str:
    ms = int((sec % 1) * 1000)
    s = int(sec) % 60
    m = int(sec // 60) % 60
    h = int(sec // 3600)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def compose_final(
    cfg: AppConfig,
    script: Script,
    clips: dict[str, str],
    talk_clips: dict[str, str],
    audio_lines: list[AudioLine],
    storage: RunStorage,
    bgm_dir: Path | None = None,
) -> Path:
    """把所有素材合成为 final/final.mp4。"""
    from fastvideo3.core.schema import TransitionType

    shots = script.all_shots()
    size = cfg.resolution
    norm_dir = storage.dir("final") / "_norm"
    norm_dir.mkdir(exist_ok=True)

    # 中转产物用高质量档：代际重编码的累积损伤是"越剪越糊"的主因
    # （economy 旧链 3 次 CRF23/ultrafast 中转；现中转 CRF≤18、preset≥veryfast）
    int_crf = min(cfg.encode_crf, cfg.intermediate_crf)
    int_preset = "veryfast" if cfg.encode_preset == "ultrafast" else cfg.encode_preset

    # 1) 选取每个镜头的最终素材并按需挂音轨
    selected: list[tuple[str, Path]] = []
    for shot in shots:
        from_talk = shot.shot_id in talk_clips
        if from_talk:  # 有口型产物优先（不依赖类型标签）
            src = Path(talk_clips[shot.shot_id])
        elif shot.shot_id in clips:
            src = Path(clips[shot.shot_id])
        else:
            src = None
        if src is None or not src.exists():
            # state 里可能残留已删产物的路径（reshoot 重生成失败/预算熔断）——缺文件不进时间线
            storage.log_event("m7_edit", "missing_shot", shot=shot.shot_id)
            continue
        if not from_talk:
            # 对白音轨直叠：audio_overlay 模式，或对话镜头无口型产物（S2V 降级路径）
            wavs = [Path(ln.wav_uri) for ln in audio_lines if ln.shot_id == shot.shot_id]
            if wavs:
                from fastvideo3.utils.audio import concat_wavs

                merged_wav = (
                    concat_wavs(wavs, storage.path("audio", f"shot_{shot.shot_id}_merged.wav"))
                    if len(wavs) > 1
                    else wavs[0]
                )
                with_audio = norm_dir / f"{shot.shot_id}_a.mp4"
                ff.mux_audio(src, merged_wav, with_audio, preset=int_preset, crf=int_crf)
                src = with_audio
        selected.append((shot.shot_id, src))

    if not selected:
        raise RuntimeError("没有任何可用镜头片段，无法合成")

    # 2) 规格统一（I2V/S2V 混源 → 统一分辨率/帧率/像素格式；cover 裁切防黑边）
    normalized: list[tuple[str, Path]] = []
    for sid, src in selected:
        out = norm_dir / f"{sid}.mp4"
        ff.normalize_clip(src, out, size, fps=cfg.output_fps, preset=int_preset, crf=int_crf)
        normalized.append((sid, out))

    trans_map = {
        s.shot_id: (
            s.transition_out.value
            if isinstance(s.transition_out, TransitionType)
            else s.transition_out
        )
        for s in shots
    }

    # 2.5) 片头/片尾标题卡 —— 完整电影形态的必要组成
    from fastvideo3.utils.image import make_title_card

    title = (script.title or script.base_info.topic).strip()
    if title and normalized:
        cards: list[tuple[str, Path]] = []
        for tag, lines_, dur in (
            ("_open", [title, "FastVideo3 出品"], 2.0),
            ("_end", ["—— 完 ——", title], 2.5),
        ):
            png = storage.path("final", f"{tag}_card.png")
            make_title_card(png, lines_, size=size)
            raw = norm_dir / f"{tag}_raw.mp4"
            ff.image_to_video(
                png,
                raw,
                duration=dur,
                fps=cfg.output_fps,
                fade_in=0.7,
                fade_out=0.7,
                preset=int_preset,
                crf=int_crf,
            )
            nrm = norm_dir / f"{tag}.mp4"
            ff.normalize_clip(raw, nrm, size, fps=cfg.output_fps, preset=int_preset, crf=int_crf)
            cards.append((tag, nrm))
        trans_map["_open"] = "fade"  # 片头→首镜、末镜→片尾用淡入淡出
        trans_map[normalized[-1][0]] = "fade"
        normalized = [cards[0]] + normalized + [cards[1]]

    # 3) 合并 + 转场：单 pass（组内 cut 无损拼接、组间 xfade 链一次编码）
    #    旧逐对链式合并使首片段经历 ≈N+2 代 x264（15-20 镜 → 17-22 代累积损伤）
    fade = 0.5
    durations = {sid: ff.probe_duration(path) for sid, path in normalized}
    merged = norm_dir / "merged_all.mp4"
    try:
        ff.merge_all(
            [(path, trans_map.get(sid, "cut")) for sid, path in normalized],
            merged,
            fade_dur=fade,
            preset=int_preset,
            crf=int_crf,
        )
        storage.log_event("m7_edit", "merged_single_pass", segments=len(normalized))
    except Exception as e:  # noqa: BLE001  # 单 pass 失败 → 回退逐对合并（正确性优先）
        storage.log_event("m7_edit", "merge_all_fallback", error=str(e)[:200])
        merged = normalized[0][1]
        for i in range(1, len(normalized)):
            prev_id = normalized[i - 1][0]
            sid, cur = normalized[i]
            trans = trans_map.get(prev_id, "cut")
            out = norm_dir / f"merge_{i}.mp4"
            ff.merge_pair(
                merged, cur, trans, out, fade_dur=fade, preset=int_preset, crf=int_crf
            )
            merged = out
            storage.log_event("m7_edit", "merged", step=i, transition=trans)

    # 4) SRT 字幕（旁挂产物；烧录失败不阻塞）
    order = [sid for sid, _ in normalized]
    srt_text = build_srt(script, audio_lines, order, durations, trans_map, fade)
    srt_path = storage.path("final", "final.srt")
    srt_path.write_text(srt_text, encoding="utf-8")

    # 短剧字幕样式：大号、白字、黑描边、底部留安全区（对齐主流微短剧观感）。
    # 注意：force_style 用 libass 脚本单位（PlayResY≈288）而非像素——
    # Fontname 不能带空格（Microsoft YaHei 会静默渲染失败），尺寸随分辨率自动缩放。
    style = (
        "Fontname=SimHei,Fontsize=12,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H00000000,BorderStyle=1,"
        "Outline=1,Shadow=1,Alignment=2,Bold=1,MarginV=14"
    )
    bgm = _pick_bgm(bgm_dir)
    final_path = storage.path("final", "final.mp4")
    # 终审母版 preset：economy 的 ultrafast 只用于中转，母版至少 medium（对齐旧 safe_encode 行为）
    master_preset = "medium" if cfg.encode_preset == "ultrafast" else cfg.encode_preset

    # 5) 终审母版单 pass：字幕+调色+BGM(sidechain ducking)+两 pass 线性响度归一
    #    旧链 burn→mix_bgm→safe_encode 三段各一代有损、amix 砍对白 -6dB、单 pass loudnorm 泵感
    try:
        ff.final_master(
            merged,
            final_path,
            srt=srt_path,
            srt_style=style,
            grade=cfg.color_grade,
            bgm=bgm,
            bgm_volume=cfg.bgm_volume,
            loudnorm=cfg.loudnorm,
            crf=cfg.encode_crf,
            preset=master_preset,
        )
        storage.log_event("m7_edit", "final_master_ok", bgm=bool(bgm))
    except Exception as e:  # noqa: BLE001  # 单 pass 失败 → 回退旧三段链（正确性优先）
        storage.log_event("m7_edit", "final_master_fallback", error=str(e)[:200])
        composed = norm_dir / "composed.mp4"
        shutil.copyfile(merged, composed)
        burned = norm_dir / "burned.mp4"
        try:
            # 字幕文件用相对名 + cwd 规避 Windows 盘符冒号在 filtergraph 里的转义地狱
            vf = (ff.grade_filter() + "," if cfg.color_grade else "") + (
                f"subtitles={srt_path.name}:force_style='{style}'"
            )
            ff.run_ffmpeg(
                [
                    "-i",
                    str(composed),
                    "-vf",
                    vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    cfg.encode_preset,
                    "-crf",
                    str(cfg.encode_crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    str(burned),
                ],
                cwd=str(srt_path.parent),
            )
            composed = burned
        except Exception:  # noqa: BLE001
            storage.log_event("m7_edit", "subtitle_burn_skipped")

        pre_final = norm_dir / "pre_final.mp4"
        if bgm:
            ff.mix_bgm(composed, bgm, pre_final, bgm_volume=cfg.bgm_volume)
        else:
            shutil.copyfile(composed, pre_final)
        ff.safe_encode(
            pre_final,
            final_path,
            crf=cfg.encode_crf,
            preset=master_preset,
            audio_filter="loudnorm=I=-16:TP=-1.5:LRA=11" if cfg.loudnorm else "",
        )

    storage.log_event(
        "m7_edit",
        "done",
        final=str(final_path),
        duration=round(ff.probe_duration(final_path), 2),
    )
    return final_path


def _pick_bgm(bgm_dir: Path | None) -> Path | None:
    base = bgm_dir or (Path(__file__).resolve().parents[2] / "assets" / "bgm")
    if not base.exists():
        return None
    for f in sorted(base.iterdir()):
        if f.suffix.lower() in (".mp3", ".wav", ".m4a"):
            return f
    return None
