"""音色参考取样：从 m5 的 TTS 母带里给每个角色切一段 ≤4 秒的样本，喂给 wan3.0。

为什么需要它：场景级生成把整段画面交给模型自己发声（C 链语义），于是**跨场景的音色
一致性**成了新问题 —— 每次调用都是独立采样，同一个角色在场景 1 和场景 3 可能换嗓子。
官方给的抓手是 `reference_audio`（"提示词 + 音乐/台词/音色参考音频 → 音轨对齐参考音频"），
而 m5 本来就为每个角色用固定音色合成过台词，母带就是现成的音色样本。

硬预算（官方 API 限制，不是实现偏好）：`reference_audio` **≤5 段且总时长 ≤15 秒**，
所以每人取 ≤4 秒、由 `scenegroup.plan_references` 决定谁能拿到名额（有台词的优先）。

取材规则（确定性、可复现）：
- 只用**本组内**该角色说过的台词（跨组取音会引入别的场景的底噪/情绪）；
- 在同组候选里取**最长**的一句（信息量最大），仍不足 1.5 秒则跳过 —— 太短的样本
  抓不住音色，发上去等于给模型一段噪声参考；
- 超过 4 秒就从头截 4 秒（不是取中间：句首通常没有气口，音色最干净）。

产物落在 `audio/_voiceref_{character_id}.wav`（下划线前缀 = 中间产物，与
`_cand_char_*` 同一套约定），并登记血缘：母带变了样本就该重切。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastvideo3.core.scenegroup import VOICE_REF_SEC
from fastvideo3.utils.audio import wav_duration

#: 样本最短可用长度：再短就抓不住音色，不如不发（发了等于给模型一段噪声参考）
MIN_VOICE_REF_SEC = 1.5


def trim_wav(src: Path, dest: Path, max_sec: float) -> Path | None:
    """把 wav 截到 ≤max_sec（从头截），输出 16k 单声道 —— 参考音频只需要音色。

    失败返回 None 而不是抛错：音色参考是**增强项**，切不出来就退回"不给音色参考"，
    绝不能因为一段样本让整个场景生成失败（那会把已经花掉的钱全废掉）。
    """
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    try:
        src = Path(src)
        if not src.is_file():
            return None
        dur = wav_duration(src)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
        if dur > max_sec:
            args += ["-t", f"{max_sec:.3f}"]
        run_ffmpeg(args + [str(dest)])
        return dest if dest.is_file() and dest.stat().st_size > 1024 else None
    except Exception:  # noqa: BLE001  # 见 docstring：增强项失败不阻断交付
        return None


def character_lines(lines: list[Any]) -> dict[str, list[Any]]:
    """每个角色在**全剧**的台词母带（按 audio_map 顺序），过滤掉文件不在盘的。

    取材范围是全局而不是"本组"，是修一个死循环：样本文件按角色全局存
    （`audio/_voiceref_{cid}.wav`），若源按组取，组1 用第1句、组2 用第3句，
    两组轮流跑就会轮流"源变了"→ 轮流 record_regeneration → 轮流作废对方的场景片段，
    步骤永久 gap（实测每点一次「开始生成」就作废一次自己刚出的片段）。
    全局取源还顺带服务了目标本身：音色参考要锁的是**这个角色全片的嗓子**，
    用哪一句切样本都该是同一个确定性答案。
    """
    out: dict[str, list[Any]] = {}
    for ln in lines or []:
        cid = str(getattr(ln, "character_id", "") or "")
        if not cid:
            continue
        wav = getattr(ln, "wav_uri", "") or ""
        if not wav or not Path(wav).exists():
            continue
        out.setdefault(cid, []).append(ln)
    return out


def pick_lines(lines: list[Any], group_shot_ids: set[str] | None = None) -> dict[str, Any]:
    """每个角色最长的一句台词（确定性取材）。`group_shot_ids` 保留仅为兼容旧调用。"""
    pool = character_lines(lines)
    if group_shot_ids is not None:
        pool = {
            cid: [ln for ln in ls if str(getattr(ln, "shot_id", "")) in group_shot_ids]
            for cid, ls in pool.items()
        }
        pool = {cid: ls for cid, ls in pool.items() if ls}
    return {cid: max(ls, key=lambda x: float(getattr(x, "duration_sec", 0) or 0))
            for cid, ls in pool.items()}


def _dur(ln: Any) -> float:
    try:
        return float(getattr(ln, "duration_sec", 0) or 0) or wav_duration(Path(str(ln.wav_uri)))
    except Exception:  # noqa: BLE001
        return 0.0


def build_voice_refs(
    storage: Any,
    lines: list[Any],
    group: Any,
    wanted: tuple[str, ...] | list[str],
    *,
    per_sec: float = VOICE_REF_SEC,
) -> tuple[dict[str, str], list[str]]:
    """按 `plan_references` 给定的名额切样本。返回 `(character_id → wav 路径, 跳过说明)`。

    返回顺序无关（调用方按 RefPlan.audios 的顺序上传，编号才对得上）；
    拿不到样本的角色进 `skipped`，由调用方落事件/降级 —— 静默少发一段音色参考，
    表现就是"这个角色跨场景换了嗓子"，而台账和看板都看不出来。
    """
    out: dict[str, str] = {}
    skipped: list[str] = []
    by_cid = character_lines(lines)  # 全局取源：见 character_lines 的注释（死循环修复）
    for cid in wanted:
        cands = by_cid.get(cid) or []
        if not cands:
            skipped.append(f"{cid}: 本组内没有可用台词母带")
            continue
        rel = f"audio/_voiceref_{cid}.wav"
        dest = storage.path("audio", f"_voiceref_{cid}.wav")
        # 最长的一句够长就单用；不够就把本组该角色的台词**按顺序拼起来**再截。
        # 短剧台词本来就短（"阿弥陀佛" ≈1.4s），只取最长的一句会让多数角色拿不到样本，
        # 那这个功能就等于没做 —— 而"没锁到音色"的表现是跨场景换嗓子，非常难归因。
        longest = max(cands, key=_dur)
        sources = [longest] if _dur(longest) >= MIN_VOICE_REF_SEC else list(cands)
        src = Path(str(sources[0].wav_uri))
        if len(sources) > 1:
            from fastvideo3.utils.audio import concat_wavs

            joined = storage.path("audio", f"_voiceref_{cid}_src.wav")
            try:
                src = concat_wavs([Path(str(x.wav_uri)) for x in sources], joined, gap_sec=0.1)
            except Exception as e:  # noqa: BLE001  # 拼接失败退回最长那句，别放弃整个角色
                src = Path(str(longest.wav_uri))
                skipped.append(f"{cid}: 拼接样本失败（{str(e)[:60]}），改用最长的一句")
        avail = sum(_dur(x) for x in sources)
        if 0 < avail < MIN_VOICE_REF_SEC:
            skipped.append(f"{cid}: 样本仅 {avail:.1f}s（<{MIN_VOICE_REF_SEC}s，抓不住音色）")
            continue
        # 血缘 inputs 必须是**仓库相对路径**（与全项目其它 record_* 调用一致）：
        # 写绝对路径的话 dependents_of 匹配不上，作废链就断了。
        src_rels: list[str] = []
        for x in sources:
            try:
                src_rels.append(str(Path(str(x.wav_uri)).relative_to(storage.root)))
            except ValueError:
                src_rels.append(str(x.wav_uri))
        # **源没变就复用，绝不 record_regeneration**：每次跑都登记"重切"会把没变的样本
        # 也当成变更 → 依赖它的场景片段被标 stale → 步骤永久 gap（实测死循环：
        # 每点一次「开始生成」就作废一次自己刚出的片段）。
        if (
            dest.exists()
            and not storage.is_stale(rel)
            and list(storage.upstream_of(rel)) == src_rels
        ):
            out[cid] = str(dest)
            continue
        got = trim_wav(src, dest, per_sec)
        if got is None:
            skipped.append(f"{cid}: 切样失败（{src.name}）")
            continue
        out[cid] = str(got)
        # 母带真的换了（改台词/换音色/首次切）→ 登记重生成，作废依赖它的场景片段：
        # 锁的嗓子变了，旧片段里的声音就对不上新样本了
        try:
            storage.record_regeneration(rel, src_rels, reason="音色参考重切")
        except Exception:  # noqa: BLE001  # 血缘登记失败不该影响生成
            pass
    return out, skipped


def total_seconds(paths: dict[str, str]) -> float:
    tot = 0.0
    for p in paths.values():
        try:
            tot += wav_duration(Path(p))
        except Exception:  # noqa: BLE001
            continue
    return round(tot, 3)


def within_budget(paths: dict[str, str], *, max_segments: int = 5, max_total_sec: float = 15.0) -> bool:
    """发之前自查官方预算（≤5 段、总 ≤15 秒）：超了服务端会 400，而报文只说参数非法。"""
    return len(paths) <= max_segments and total_seconds(paths) <= max_total_sec
