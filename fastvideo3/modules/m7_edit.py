"""m7 剪辑合成（教学第 5 讲：剪辑与转场）。

规格统一 → 逐对转场合并（xfade）→ SRT 字幕 → BGM 混音 → 成片。
转场失败回退硬切；BGM 失败不阻塞 —— 成片永远可交付。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.schema import AudioLine, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.utils import ffmpeg as ff

# 旁挂字幕的文件名**必须与成片不同基名**。字幕已经烧进画面，而 VLC / PotPlayer / MPC
# 的默认行为是自动挂载与视频同基名的 .srt —— 于是画面上出现两层字幕（2026-10-07
# run_20261007_130730 用户实测：抽帧确认片内只有一层，多出来的是播放器挂的旁挂层）。
# 想单独要字幕轨（上传平台/改字幕）的人仍能从看板下载到这份文件。
SUBTITLE_SIDECAR_NAME = "subtitles.srt"
LEGACY_SUBTITLE_NAME = "final.srt"  # 改名前的旧 run 还叫这个，读取时兼容


def subtitle_sidecar_path(storage: RunStorage) -> Path:
    """旁挂字幕的实际路径：新名优先，旧 run 回退到 `final.srt`。"""
    new = storage.path("final", SUBTITLE_SIDECAR_NAME)
    return new if new.is_file() else storage.root / "final" / LEGACY_SUBTITLE_NAME


def subtitle_font_family() -> str:
    """全片唯一中文字体的 libass 家族名（与标题卡取的是同一份候选表）。"""
    from fastvideo3.utils.image import cjk_font

    return cjk_font()[0]


def subtitle_style(fontsize: int = 12, margin_v: int = 14) -> str:
    """字幕样式（**唯一来源**）：白字、黑描边、加粗、底部安全区、居中。

    为什么要有这个函数：字幕可能被烧两次（终审母版 + 母版失败时的回退链），两处各自
    写一份 force_style 字符串就会漂 —— 一深一浅、一高一低，看起来就像"两层字幕"。
    现在两条路径共用这一份。

    注意：`force_style` 用 libass 的脚本单位（PlayResY≈288）而非像素，尺寸随分辨率
    自动缩放；家族名必须**不带空格**（带空格的 "Microsoft YaHei" 在 libass 里静默
    渲染失败 → 悄悄退回系统默认字体，没人会发现字体变了）。
    """
    family = subtitle_font_family() or "DejaVuSans"
    parts = [
        f"Fontname={family}",
        f"Fontsize={fontsize}",
        "PrimaryColour=&H00FFFFFF",
        "OutlineColour=&H00000000",
        "BackColour=&H00000000",
        "BorderStyle=1",
        "Outline=1",
        "Shadow=1",
        "Alignment=2",
        "Bold=1",
        f"MarginV={margin_v}",
    ]
    return ",".join(parts)


# 字幕首条相对镜头起点的延迟 / 两条字幕之间的最小间隔 / 镜头尾部预留（不压到转场）
SRT_LEAD_IN = 0.3
SRT_MIN_GAP = 0.12
SRT_TAIL = 0.15


def build_srt(
    script: Script,
    audio_lines: list[AudioLine],
    clips_order: list[str],
    clip_durations: dict[str, float],
    transition_out: dict[str, str],
    fade_dur: float,
    *,
    lead_in: float = SRT_LEAD_IN,
    min_gap: float = SRT_MIN_GAP,
    tail: float = SRT_TAIL,
    model_audio_shots: set[str] | None = None,
    scene_windows: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> str:
    """按镜头窗口排字幕，并**保证任意两条字幕不重叠**。

    字幕文本永远直接取剧本台词（`AudioLine.text`）—— 不做语音识别、不让模型"生成"字幕：
    AI 画的字幕会错字、会漏句、字体还不受控。画面里出现的每一个字都由这里决定。

    实测病灶（run_20261005_130208/final.srt，成片上表现为"两层字幕"）：
        1  00:00:08,320 --> 00:00:10,240  色即是空...哎？
        2  00:00:09,519 --> 00:00:14,032  师傅！观音菩萨还在天上看着呢！   ← 重叠 0.72s
    旧实现把"字幕时长 = 音频时长、下一条起点 = 下一镜头起点 + 偏移"直接拼起来，
    而口型模型回传的片段**比它的驱动音频短**（s2v 片段 1.70s vs 音频 1.92s），
    上一条字幕于是溢出进下一个镜头的时间窗，两条同时挂在画面上。

    规则（顺序敏感）：
    1. 每条字幕只在自己的镜头窗口 `[t+lead_in, t+dur-tail]` 内排布；
    2. 窗口装不下时**跨过边界也要保证不重叠** —— 后一条被推迟到前一条结束之后
       （重叠是观感事故，轻微滞后不是）；
    3. 同镜头内相邻两句之间留 min_gap，避免字"跳"在一起。
    """
    by_shot: dict[str, list[AudioLine]] = {}
    for ln in audio_lines:
        by_shot.setdefault(ln.shot_id, []).append(ln)

    model_audio_shots = model_audio_shots or set()
    scene_windows = scene_windows or {}
    cues: list[tuple[float, float, str]] = []
    t = 0.0
    prev_end = 0.0
    for i, sid in enumerate(clips_order):
        dur = float(clip_durations.get(sid, 5.0))
        if sid in scene_windows:
            # 场景片段：一段视频覆盖组内多个镜头，**每镜头的起止是已知的**（生成时落盘的
            # windows），所以字幕可以直接落在真实窗口上 —— 比单镜头 C 链的"按字数比例猜"
            # 更准。窗口内多句仍按字数比例细分（模型发声的句间时刻我们同样拿不到）。
            seg_start = t
            for shot_id, (ws, we) in scene_windows[sid].items():
                w_start, w_end = seg_start + float(ws), seg_start + float(we)
                lines = by_shot.get(shot_id, [])
                if not lines:
                    continue
                span = max(0.5, w_end - w_start)
                weights = [max(1, len((ln.text or "").strip())) for ln in lines]
                total = sum(weights) or 1
                cursor = w_start
                for ln, w in zip(lines, weights):
                    want = span * w / total
                    start = max(cursor, prev_end + min_gap)
                    end = min(start + want, w_end)
                    if end - start < 0.35:
                        end = start + want
                    cues.append((start, end, ln.text))
                    prev_end = end
                    cursor = end + min_gap
            t += dur
            if i < len(clips_order) - 1 and transition_out.get(sid, "cut") != "cut":
                t -= fade_dur
            continue
        window_start = t + lead_in
        window_end = t + max(lead_in + 0.5, dur - tail)  # 镜头太短时不能让窗口塌成负数
        cursor = window_start
        lines = by_shot.get(sid, [])
        if sid in model_audio_shots and lines:
            # C 链（wan3.0-video 自发声）的镜头：音轨是模型的，**我们手上没有它每句的时刻**
            # （做 ASR 对齐是红线之外的新依赖，且会引入识别错误）。但知道台词全文与整段时长，
            # 于是按**字数比例**把这个镜头窗口分给各句：
            # 为什么不能用 TTS 时长 —— 那是另一条音轨的节奏，实测与模型的发声相差 0～2.2 秒，
            # 照它排就是"第二层错位"（语音已经不对嘴，字幕再跟着语音跑，观众会以为字幕也错）。
            # 按比例铺满整个窗口，最坏情况是某句早出现/晚消失半秒，而不会整条串位。
            span = max(0.5, window_end - window_start)
            weights = [max(1, len((ln.text or "").strip())) for ln in lines]
            total = sum(weights) or 1
            for ln, w in zip(lines, weights):
                want = span * w / total
                start = max(cursor, prev_end + min_gap)
                end = min(start + want, window_end)
                if end - start < 0.35:
                    end = start + want  # 与 TTS 分支同一规则：不压成闪现
                cues.append((start, end, ln.text))
                prev_end = end
                cursor = end + min_gap
            t += dur
            if i < len(clips_order) - 1 and transition_out.get(sid, "cut") != "cut":
                t -= fade_dur
            continue
        for ln in lines:
            want = float(ln.duration_sec or 0) or 1.2
            start = max(cursor, prev_end + min_gap)
            end = min(start + want, window_end)
            if end - start < 0.35:
                end = start + want  # 窗口装不下：宁可跨镜头，也不把一句话压成 0.3 秒闪现
            cues.append((start, end, ln.text))
            prev_end = end
            cursor = end + min_gap
        t += dur
        if i < len(clips_order) - 1 and transition_out.get(sid, "cut") != "cut":
            t -= fade_dur

    entries: list[str] = []
    for idx, (start, end, text) in enumerate(cues, start=1):
        entries.append(f"{idx}\n{_fmt(start)} --> {_fmt(end)}\n{text}\n")
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
    # model_audio：成片里**用的是口型模型自带音轨**的那些镜头。字幕时间轴要认这个集合 ——
    # 见 `build_srt`：模型发声的节奏与我们 TTS 母带的节奏不是一回事（实测差 0～2.2 秒）。
    # 只有 C 链 + `voice_source=model` 才是"模型原声进成片"；A/B 链的音轨始终是我们的母带。
    scripted_model_track = cfg.talk_mode == "scripted_talk" and cfg.voice_source == "model"
    model_audio: set[str] = set()
    selected: list[tuple[str, Path]] = []

    # ── 场景片段（一次调用覆盖整组镜头）优先于逐镜头产物 ──
    # 组内镜头的切点是模型自己切的，画面/声音天然连续；把它们拆回逐镜头再拼，
    # 等于亲手把连贯性剪断。所以整组作为**一个**时间线段进成片，字幕按落盘的
    # windows 排（比单镜头 C 链的"按字数比例猜"更准）。
    scene_segs: dict[str, dict] = {}
    shot_to_scene: dict[str, str] = {}
    clips_dir = storage.dir("clips")
    if clips_dir.is_dir():
        for jp in sorted(clips_dir.glob("scene_*.json")):
            try:
                meta_j = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            gid = str(meta_j.get("group_id") or jp.stem[len("scene_"):])
            clip = storage.path("clips", f"scene_{gid}.mp4")
            if not clip.is_file() or storage.is_stale(f"clips/scene_{gid}.mp4"):
                continue
            scene_segs[f"scene:{gid}"] = meta_j
            for sid in meta_j.get("shot_ids") or []:
                shot_to_scene[sid] = f"scene:{gid}"
    scene_windows: dict[str, dict[str, tuple[float, float]]] = {}
    for key, meta_j in scene_segs.items():
        scene_windows[key] = {
            sid: (float(w[0]), float(w[1]))
            for sid, w in (meta_j.get("windows") or {}).items()
        }

    for shot in shots:
        seg_key = shot_to_scene.get(shot.shot_id)
        if seg_key is not None:
            first_of_group = scene_segs[seg_key]["shot_ids"][0] == shot.shot_id
            if not first_of_group:
                continue  # 组内其余镜头随整组进时间线，不单独占位
            seg_clip = storage.path("clips", f"scene_{seg_key.split(':', 1)[1]}.mp4")
            selected.append((seg_key, seg_clip))
            # 场景片段的音轨是模型自己的（参考生视频形态自带发声）→ 字幕走 windows
            model_audio.update(scene_segs[seg_key]["shot_ids"])
            continue
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
        if from_talk:
            # 音轨取舍**以配置为准**（cfg.voice_source），不是"driver.wav 存不存在"。
            # 曾经的写法是先看文件在不在 → 一条用 master 跑过的 run 即使把 voice_source
            # 改成 model，盘上遗留的 driver.wav 仍会压过配置，成片继续声画错位
            # （run_20261011_231024 实测：改了配置、重合成，语音照样领先口型）。
            driver = storage.path("audio", f"shot_{shot.shot_id}_driver.wav")
            if scripted_model_track:
                # C 链 + 模型原声：口型就是对着这条音轨画的，不换（换了必然错位 0～2.2s）
                model_audio.add(shot.shot_id)
                storage.log_event(
                    "m7_edit", "talk_audio_kept_model", shot=shot.shot_id,
                    hint="保留模型原声（voice_source=model）：嘴型与声音同轨；字幕按窗口比例排",
                )
            elif driver.exists():
                # 口型片段**只取它的画面**，音轨换回我们自己的 TTS 母带。
                # 实测原因（run_20261005_130208）：wan2.2-s2v 回传的 mp4 自带一条
                # 44.1kHz/stereo/AAC-101kbps 音轨，那是我们 16k 单声道驱动音频经模型
                # 上变换的产物（高频受损 = 发闷/电话音），而且片段只有 4.30s 而音频是
                # 4.51s —— **句尾被截 0.21s**。嘴型在视频流里，换音轨不影响口型对齐；
                # 音频更长时 mux_audio 会克隆末帧补足（不截句子）。
                swapped = norm_dir / f"{shot.shot_id}_a.mp4"
                ff.mux_audio(src, driver, swapped, preset=int_preset, crf=int_crf)
                src = swapped
                storage.log_event("m7_edit", "talk_audio_replaced", shot=shot.shot_id)
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
    # 场景片段的"出点转场"取**组内最后一个镜头**的 transition_out：切点发生在组边界，
    # 组内部的切点是模型自己切的（那正是场景级生成要保的连贯性），不该被我们再叠一层 xfade
    shot_trans = {
        s.shot_id: (s.transition_out.value if isinstance(s.transition_out, TransitionType)
                    else s.transition_out)
        for s in shots
    }
    for key, meta_j in scene_segs.items():
        ids = meta_j.get("shot_ids") or []
        if ids:
            trans_map[key] = shot_trans.get(ids[-1], "cut")

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
    srt_text = build_srt(
        script, audio_lines, order, durations, trans_map, fade,
        model_audio_shots=model_audio, scene_windows=scene_windows,
    )
    srt_path = storage.path("final", SUBTITLE_SIDECAR_NAME)
    srt_path.write_text(srt_text, encoding="utf-8")

    # 短剧字幕样式：大号、白字、黑描边、底部留安全区（对齐主流微短剧观感）。
    # 字体与标题卡**同源**（utils.image.cjk_font），全片只有一种中文字体。
    style = subtitle_style()
    storage.log_event("m7_edit", "subtitle_font", family=subtitle_font_family(),
                      note="字幕与片头/片尾标题卡共用同一款字体")
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

    # 5) 技术 QC：**确定性数值判据**，零 API 调用（黑帧/静帧/静音/时长偏差/黑边/规格合规）。
    # 为什么不交给 VLM 终审：m8 已经翻过两次车 —— JSON 缺字段时落 0.6 兜底分并 passed=True，
    # 异常被写成"通过"。这类问题不需要模型判断，量到了就是硬事实。
    from fastvideo3.utils import qc

    target_sec = float(script.base_info.target_duration_sec or 0)
    items = [
        qc.check_media(
            final_path,
            storage.dir("final") / "_qc",
            expected_duration=target_sec or None,
            # 成片总时长 vs 用户填的目标 = **意图**偏差（实际由镜头数×每镜时长决定），
            # 只警告不判死；素材级的时长偏差（片段 vs 它自己的驱动音频）仍然是 failed
            strict_duration=False,
            label="成片",
        ),
        qc.check_spec(
            final_path,
            size=size,
            fps=cfg.output_fps,
            aspect=cfg.aspect_ratio,
            target_duration=target_sec or None,
        ),
    ]
    report = qc.qc_report(items)
    storage.save_json("final/qc_report.json", report)
    storage.log_event(
        "m7_edit",
        "tech_qc",
        passed=report["passed"],
        failed=[f["problem"] for f in report["failed"]][:6],
        warnings=[w["problem"] for w in report["warnings"]][:6],
        not_checked=report["not_checked"],
        hint=(
            "确定性判据（黑帧/静帧/静音/时长偏差/黑边/规格）；"
            "not_checked 是「没量」不是「量了没问题」"
        ),
    )
    for f in report["failed"]:
        storage.log_degradation(
            "m7_edit", "tech_qc_failed", f["problem"], impact=f"成片技术不合格：{f['item']}"
        )

    storage.log_event(
        "m7_edit",
        "done",
        final=str(final_path),
        duration=round(ff.probe_duration(final_path), 2),
        tech_qc_passed=report["passed"],
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
