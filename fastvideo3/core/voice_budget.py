"""语音预算：**只读的容量估算**，不改任何数字、不拦任何保存。

为什么需要这一层（实测 run_20261015_220939）：人工保存路径上唯一相关的软规则是
"单句 >30 字警告"。而那份稿子 10 个镜头、600 字台词 —— 按中文 4.5 字/秒算，**光是念完
就需要 133 秒**，剧本却只有 90 秒。30 字那条规则一条都没报（每句都不长），报出来的两条
反而是稿子里唯一说得下的两句。绝对值判据在镜头时长可长可短之后就已经失真了：
3 秒镜头塞 25 字必然吞字，9 秒镜头说 35 字却绰绰有余。

于是判据换成**容量恒等式**（这就是"需要秒数 ≥ 字数/语速"的正确形式）：

    片长 ≈ Σ 镜头需要秒，其中  镜头需要 = max(语音秒, 演出秒) (+ 静默内容的额外留白)
            语音秒 = round(字数 / 4.5) + 1        ← 用户裁定的公式，那个 +1 就是每镜呼吸

三项归属不同，混在一起算必错：
- **语音秒**：硬下界。台词删不掉的那部分，只有减字数能降；给少了模型会吞句尾/把语速压成
  念广告，给多了拖出无话可说的空拍。
- **演出秒**：动作/位移/反应要演的时间。与语音**可重叠**（人可以一边说一边做）→ 取 max，不是相加。
- **呼吸与静默**：公式里那个 `+1` 就是每镜的呼吸余量，它同时也是"每多一个镜头就多交一份"
  的固定开销 —— 不含这一项的公式一定算错（35 镜就是 35 秒）。凝视、落泪、末镜接转场
  这类**确实需要更多空白**的内容再额外加，由 `_hold_sec` 自动判断，不能与语音并行。
  这一项是"镜头税"：镜头拆得越多，固定开销越大（1 秒/镜），90 秒的片子里 35 个镜头就要
  交掉 35 秒。任何不含这一项的公式都会算错。

三条边界，都按用户裁定处理：
1. **留白由本模块自动判断**（情绪档 + 画面里的静默信号 + 是否收束镜），剧本稿里**不新增
   人工字段** —— 但估出来的是草稿，不是审美；数字会连着依据一起回显，人看得见才谈得上改。
2. **超出只告警，不做门禁**：保存照旧成功，一个字的时长都不改。真正决定片段内部分配的
   是 wan3.0 自己（`duration` 之外不再细粒度控制，那是模型的自由度，写进提示词也管不住）。
3. 所有系数进 `AppConfig`（`voice_*`），改一处全链路生效 —— 语速口径与 `m6_talk` 共用同一份，
   避免出现"保存页说装得下、配音页说不够"。

这里的数字是**预估**，用来在花钱之前发现"这份稿子拍不出来"，不是对成片的保证。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from fastvideo3.core.scenegroup import (
    CLIP_CAP_SEC,
    MAX_CLIP_SEC,
    MAX_SHOTS_PER_CLIP,
    SHOT_MAX_SEC,
    SHOT_MIN_SEC,
)

#: 触发"这一镜需要更多呼吸"的信号：静默本身是内容（凝视、落泪、愣住），不是空缺。
_HOLD_WORDS = (
    "凝视", "注视", "看着", "对视", "沉默", "静默", "无声", "愣", "呆", "含泪", "泪",
    "低头", "叹气", "深呼吸", "屏息", "回眸", "定住", "僵", "停顿", "不语", "望",
)
#: 这些情绪档说得慢、说完要留反应拍（与 utils/emotion 的档位同名，不做模糊匹配）
_HOLD_EMOTIONS = {"sad", "tender", "solemn", "thoughtful", "shy", "confused", "helpless", "calm"}
#: 收束拍的额外余量：一场最后一镜之后要接转场，不能话落即切
_SCENE_END_HOLD_SEC = 0.4


@dataclass(frozen=True)
class Params:
    """系数包。默认值就是文档里的经验值；全部可被 AppConfig 覆盖（`voice_*`）。"""

    cps: float = 4.5              # 字/秒：中文语音语速（与 m6 同一口径）
    breath: float = 1.0           # 每镜呼吸余量：最小预估 = round(字数/cps) + 这个 1 秒
    beat: float = 1.2             # 一个可拍动作单元的演出秒
    hold: float = 0.6             # 静默内容的**额外**留白（情绪/画面里的凝视落泪，见 _hold_sec）
    # 单次调用的预算线，默认就是分段用的 CLIP_CAP_SEC：预算与分段必须是同一个数，
    # 否则"会拆成 2 段"的提示与实际出几段会各说各话。配置调低只会更早告警（保守方向），
    # 调高则由分段兜住 —— 服务端硬顶是 MAX_CLIP_SEC=30。
    clip_soft: float = CLIP_CAP_SEC
    band_low: float = 0.30        # 非语音占比下限（低于它=念稿）
    band_high: float = 0.45       # 非语音占比上限（高于它=空转）
    shot_min: float = SHOT_MIN_SEC
    shot_max: float = SHOT_MAX_SEC
    tolerance: float = 0.15       # 成片时长偏差验收带（PRD F6）

    @staticmethod
    def from_cfg(cfg: Any) -> "Params":
        """从配置取系数；cfg 为 None 时退回默认值（测试与离线估算用）。"""
        if cfg is None:
            return Params()

        def g(name: str, dflt: float) -> float:
            # 用 `is None` 而不是真值判断：系数可以合法地设成 0
            # （`voice_hold_sec: 0` = 不留白；`clip_soft_cap_sec: 30` = 不设软顶）
            v = getattr(cfg, name, None)
            try:
                return dflt if v is None else float(v)
            except (TypeError, ValueError):
                return dflt

        return Params(
            cps=g("voice_cps", Params.cps), breath=g("voice_breath_sec", Params.breath),
            beat=g("voice_beat_sec", Params.beat), hold=g("voice_hold_sec", Params.hold),
            clip_soft=g("clip_soft_cap_sec", Params.clip_soft),
            band_low=g("voice_nonspeech_band_low", Params.band_low),
            band_high=g("voice_nonspeech_band_high", Params.band_high),
        )


@dataclass
class ShotBudget:
    """一镜的预算。`window` 是人写的时长，`need` 是算出来的需要 —— 两者谁都不覆盖谁。"""

    shot_id: str
    scene_id: str
    lines: int
    chars: int
    speech: float
    stage: float
    hold: float
    hold_why: str
    window: float
    #: 按官方"每分镜 2–5 秒"，这一镜的内容实际需要几个分镜
    shots_needed: int
    #: 这一镜的最小可行时长：max(语音, 演出) + 静默内容的额外留白
    core: float = 0.0

    @property
    def need(self) -> float:
        """这一镜"该有的长度"。窗口比它小就是装不下 —— 预算本身只报，抬时长由
        `apply_duration_floor` 按用户裁定执行。"""
        return round(self.core, 3)

    @property
    def over_by(self) -> float:
        return round(max(0.0, self.need - self.window), 2)

    @property
    def free(self) -> float:
        return round(self.window - self.need, 2)

    @property
    def nonspeech_ratio(self) -> float:
        """非语音占比：内容里有多少是给动作与呼吸的。太低=念稿，太高=空转。"""
        return round((self.core - self.speech) / self.core, 3) if self.core > 0 else 1.0

    def verdict(self, p: Params | None = None) -> str:
        if self.lines and self.need > self.window + 0.05:
            return "说不完"
        if not self.lines and self.need > self.window + 0.05:
            return "演不完"
        if self.window - self.need > max(2.0, 0.5 * self.window):
            return "空转"
        pp = p or Params()
        if self.lines and not (pp.band_low <= self.nonspeech_ratio <= pp.band_high):
            return "偏念稿" if self.nonspeech_ratio < pp.band_low else "偏空"
        return ""


@dataclass
class FilmBudget:
    shots: list[ShotBudget]
    scene_need: dict[str, float] = field(default_factory=dict)
    scene_clips: dict[str, int] = field(default_factory=dict)
    target: float = 0.0
    params: Params | None = None

    # ---- 汇总量 ----
    @property
    def speech_total(self) -> float:
        return round(sum(s.speech for s in self.shots), 1)

    @property
    def need_total(self) -> float:
        return round(sum(s.need for s in self.shots), 1)

    @property
    def window_total(self) -> float:
        return round(sum(s.window for s in self.shots), 1)

    @property
    def shots_needed_total(self) -> int:
        return sum(s.shots_needed for s in self.shots)

    @property
    def clips_total(self) -> int:
        return sum(self.scene_clips.values())

    @property
    def deviation_pct(self) -> float | None:
        """内容需要 vs 目标时长的偏差（PRD F6 的 ±15% 验收带按它判，不是按人写的秒数判）。"""
        if self.target <= 0:
            return None
        return round((self.need_total - self.target) / self.target * 100, 1)

    # ---- 给人看的话 ----
    def warnings(self) -> list[str]:
        """给人的容量判断。只报，不改任何人写的数字 —— 片段内部时长归 wan3.0 分配。"""
        p = self.params or Params()
        out: list[str] = []
        worst = [s for s in self.shots if s.verdict(p) in ("说不完", "演不完")]
        if worst:
            ex = worst[0]
            out.append(
                f"语音预算：{len(worst)}/{len(self.shots)} 个镜头装不下自己的台词或动作"
                f"（如 {ex.shot_id}：{ex.lines} 句 {ex.chars} 字≈念 {ex.speech:.1f}s，"
                f"镜头只写了 {ex.window:g}s）。不改你的数字 —— wan3.0 会自己分配时长，"
                f"但念不完的典型表现是句尾被吞或语速被压快"
            )
        thin = [s for s in self.shots if s.verdict(p) == "空转"]
        if thin:
            out.append(
                f"语音预算：{len(thin)} 个镜头内容比时长少一大截"
                f"（如 {thin[0].shot_id}：内容≈{thin[0].need:.1f}s，写了 {thin[0].window:g}s）"
                "—— 会拖出无话可说的空拍"
            )
        if self.target > 0:
            dev = (self.need_total - self.target) / self.target
            if abs(dev) > p.tolerance:
                if dev > 0:
                    # 每镜都要交一份"呼吸 + 起幅落幅"的固定开销（就是公式里那个 +1），
                    # 所以"把长句拆成更多镜头"本身要花秒数 —— 不含这一项的公式一定算错
                    tax = float(self.shots_needed_total)
                    fit = int(max(0.0, (self.target * (1 + p.tolerance) - tax)) * p.cps)
                    words = sum(s.chars for s in self.shots)
                    out.append(
                        f"语音预算：这份稿子按 {p.cps:g} 字/秒念完需要 {self.speech_total:g}s，"
                        f"算上演出与每镜呼吸余量共约 {self.need_total:g}s，"
                        f"而目标只有 {self.target:g}s（{dev * 100:+.0f}%，验收带 ±{p.tolerance * 100:.0f}%）。"
                        f"两条出路：把台词从 {words} 字压到 ≤{fit} 字，"
                        f"或把目标时长提到约 {self.need_total:.0f}s —— 拆镜头救不了它："
                        f"每多一个镜头就多一份呼吸余量（按 {self.shots_needed_total} 镜算≈{tax:.0f}s）"
                    )
                else:
                    out.append(
                        f"语音预算：内容约 {self.need_total:g}s，目标 {self.target:g}s"
                        f"（{dev * 100:+.0f}%）—— 成片会明显短于目标，"
                        "可以加动作/空镜或减少目标时长"
                    )
        split = {k: v for k, v in self.scene_clips.items() if v > 1}
        if split:
            out.append(
                "片段预算：以下场景的内容超过单次调用软顶 "
                f"{p.clip_soft:g}s，会被拆成多段（同一张场景图、各自提示词）："
                + "、".join(f"{k}→{v}段" for k, v in list(split.items())[:5])
            )
        return out

    def summary(self) -> dict[str, Any]:
        """机器可读的容量表（看板那一行与"这场会被拆成几段"都读它）。"""
        p = self.params or Params()
        return {
            "cps": p.cps,
            "speech_sec": self.speech_total,
            "need_sec": self.need_total,
            "written_sec": self.window_total,
            "target_sec": self.target,
            "deviation_pct": self.deviation_pct,
            "shots": len(self.shots),
            "shots_needed": self.shots_needed_total,
            "clips": self.clips_total,
            "unsayable": sum(
                1 for s in self.shots if s.verdict(p) in ("说不完", "演不完")
            ),
            "per_scene": {
                k: {"need_sec": round(v, 1), "clips": self.scene_clips.get(k, 1)}
                for k, v in self.scene_need.items()
            },
            # 逐镜最小可行时长：前端"手填时长少于最小预估"的告警直接引这个数，
            # **不在前端重算公式** —— 公式只有一份（改了这里，界面提示不会漂）。
            "per_shot": {
                s.shot_id: {
                    "need_sec": round(s.need, 1),
                    "speech_sec": round(s.speech, 1),
                    "stage_sec": round(s.stage, 1),
                    "hold_sec": round(s.hold, 1),
                    "lines": s.lines,
                    "chars": s.chars,
                    "written_sec": round(s.window, 1),
                    "verdict": s.verdict(p),
                }
                for s in self.shots
            },
        }


def _stage_sec(action: str, beat: float) -> float:
    """从画面描述数"要演几个动作"。粗口径：≥4 字的短句各算一个单元，封顶 6 个。"""
    units = [p for p in re.split(r"[，,。；;、\s]+", action or "") if len(p) >= 4]
    return round(beat * max(1, min(6, len(units))), 3)


def _hold_sec(shot: Any, emotion_hit: bool, scene_end: bool, p: Params) -> tuple[float, str]:
    """静默内容的**额外**留白，自动判断（用户裁定：不留人工字段）。

    每镜普通呼吸余量已经计在 `speech` 的 +1 里，这里只加"这一镜确实需要更多空白"的部分：
    静默类情绪、画面里写着的凝视/落泪/愣住、以及本场景末镜（话落要接转场）。
    估的是草稿，依据随数字一起回显。
    """
    if not getattr(shot, "dialogues", None):
        return 0.0, "无台词：整镜本来就是氛围拍，不再另计留白"
    hold = 0.0
    why: list[str] = []
    if emotion_hit:
        hold += 0.6
        why.append(f"情绪偏静（凝视/落泪/迟疑一类）+0.6s → 共 {hold:.1f}s")
    if any(w in (shot.action or "") for w in _HOLD_WORDS):
        hold += 0.4
        why.append(f"画面里写了静默动作 +0.4s → 共 {hold:.1f}s")
    if scene_end:
        hold += _SCENE_END_HOLD_SEC
        why.append(f"本场景末镜（要接转场）+{_SCENE_END_HOLD_SEC:g}s → 共 {hold:.1f}s")
    return round(min(hold, 1.4), 3), ("、".join(why) or "无额外留白（呼吸余量已计在语音里）")


def budget(script: Any, cfg: Any) -> FilmBudget:
    """整份剧本的语音预算。**纯读**：不写回任何字段，不拦任何保存。"""
    p = Params.from_cfg(cfg)
    fb = FilmBudget(shots=[], target=float(getattr(script.base_info, "target_duration_sec", 0) or 0),
                    params=p)
    for scene in script.scenes:
        shots = list(scene.shots or [])
        need_sum = 0.0
        chunks = 1
        cur = 0.0
        cur_n = 0
        for i, shot in enumerate(shots):
            lines = list(shot.dialogues or [])
            chars = sum(len((d.text or "").strip()) for d in lines)
            # 用户裁定的公式：最小预估 = round(字数 / 字每秒) + 1 秒呼吸
            # （37 字 → round(8.22)+1 = 9s。以前还另加句间换气与起幅落幅，同一句会算成 10s）
            speech = round(chars / p.cps) + p.breath if lines else 0.0
            emotion_hit = any((d.emotion or "").strip().lower() in _HOLD_EMOTIONS for d in lines)
            hold, hold_why = _hold_sec(shot, emotion_hit, i == len(shots) - 1, p)
            stage = _stage_sec(shot.action, p.beat)
            core = max(speech, stage) + hold
            need = round(core, 3)
            # 按官方"每分镜 2–5 秒"，这一镜的内容至少要几个分镜
            shots_needed = max(1, math.ceil(need / p.shot_max))
            fb.shots.append(ShotBudget(
                shot_id=shot.shot_id, scene_id=scene.scene_id, lines=len(lines), chars=chars,
                speech=speech, stage=stage, hold=hold, hold_why=hold_why,
                window=float(shot.duration_sec or 0.0), shots_needed=shots_needed,
                core=core,
            ))
            need_sum += need
            # 拆段：软顶或分镜数先到为先，且不切断单个分镜（与 _split_oversized 同口径）
            unit = min(need, p.shot_max * shots_needed)
            if cur and (cur + unit > min(p.clip_soft, MAX_CLIP_SEC) or cur_n + 1 > MAX_SHOTS_PER_CLIP):
                chunks += 1
                cur = 0.0
                cur_n = 0
            cur += unit
            cur_n += 1
        fb.scene_need[scene.scene_id] = round(need_sum, 3)
        fb.scene_clips[scene.scene_id] = chunks
    return fb


def budget_of(script: Any, cfg: Any) -> FilmBudget:
    """`budget` 的别名：给"只想要一次估算"的调用方（保存事务与看板都走这里）。"""
    return budget(script, cfg)


def warnings(script: Any, cfg: Any) -> list[str]:
    """保存/看板共用的软校验出口：**只产告警文本，不改脚本、不拦保存**。

    取代旧的 `line_length_violations` + `duration_warnings` 两条：一条按绝对字数、
    一条按人写的秒数合计，都不看"这份稿子实际念得完吗"。
    """
    return budget(script, cfg).warnings()


def capacity(script: Any, cfg: Any) -> dict[str, Any]:
    """看板用的容量表（`FilmBudget.summary()`）。"""
    return budget(script, cfg).summary()


def apply_duration_floor(script: Any, cfg: Any) -> tuple[Any, list[str]]:
    """把"念不完/演不完"的镜头**抬到最小可行时长**，封顶 `clip_soft_cap_sec`（默认 25s）。

    为什么这一处**要**改用户写的数字（与"语音预算只告警"并不矛盾）：
    镜头时长不是备注字段。它同时决定两件事 —— 提交给 wan3.0 的 `duration` 参数，以及提示词里
    的分镜时间戳（`第1个镜头[0-3秒]`）。留着 3s 的窗口去装 9.9s 的话，结果不是"稍微挤一下"，
    而是**句尾被吞**，被吞的正是台词本身（实测 run_20261015_220939：10 镜里 8 镜装不下自己
    的台词）。抬到下限是让"说得出"成立的最小动作。

    三条边界（用户裁定）：
    - **需要 ≤ 写的 ≤ 25s → 一律不动**：这段区间里时长是人的节奏（留白、氛围、慢拍），
      机器没有意见可提；
    - **写短了才抬**：抬到最小可行时长（max(语音, 演出) + 静默额外留白，且不低于官方
      每分镜 2 秒下限），**封顶 25s**；
    - **写超 25s 才压**：这是唯一允许的向下动作，因为 25s 是单段预算线（超过它一次调用
      也演不完），不是我们对节奏的意见；
    - 抬到 25s 仍不够（需要 > 25s）时说明这一镜话太多：**照旧只告警**，
      拆成"说话者 + 听者反应"两镜由写剧本的人决定。
    """
    p = Params.from_cfg(cfg)
    fb = budget(script, cfg)
    cap = max(p.clip_soft, p.shot_max)
    notes: list[str] = []
    by_id = {s.shot_id: s for s in fb.shots}
    for shot in script.all_shots():
        b = by_id.get(shot.shot_id)
        if b is None:
            continue
        # 官方多镜头提示词要求每个分镜 2–5 秒：1.5 秒的窗口连时间戳都排不出来，也算装不下
        need = max(b.need, p.shot_min)
        if need <= shot.duration_sec <= cap:
            continue  # 落在"最小可行 ~ 25s"之间：这是人的节奏，一处都不动
        if shot.duration_sec > cap:
            # 唯一允许的向下动作：封顶。超过单段预算线的镜头，模型也不可能在一段里演出层次
            notes.append(
                f"{shot.shot_id}: {shot.duration_sec:g}s → {cap:g}s（超过单段预算 {cap:g}s，"
                f"已封顶；这场会被拆成多段）"
            )
            shot.duration_sec = cap
            continue
        raised = min(float(math.ceil(need)), cap)
        if raised <= shot.duration_sec:
            continue
        why = f"{b.lines} 句 {b.chars} 字≈念 {b.speech:.1f}s" if b.lines else f"演出≈{b.stage:.1f}s"
        notes.append(f"{shot.shot_id}: {shot.duration_sec:g}s → {raised:g}s（{why}，需要 {need:.1f}s）")
        shot.duration_sec = raised
        if need > cap + 0.05:
            notes.append(
                f"{shot.shot_id}: 需要约 {need:.1f}s，已封顶 {cap:g}s —— 这一镜的台词/动作仍超出"
                f"单镜容量，建议拆成两镜（说话者 + 听者反应）或删减台词"
            )
    if notes:
        notes.insert(
            0,
            f"镜头时长按语音预算抬到最小可行值（{len(notes)} 处，只升不降、封顶 {cap:g}s）："
            "时长同时决定提交给 wan3.0 的 duration 和提示词里的分镜时间戳，写短了会吞句尾。"
            "你把镜头写长的那些（节奏/留白）一处没动。",
        )
    return script, notes


def remaining_cost_note(script: Any, cfg: Any, gate: dict[str, Any]) -> str:
    """到线时给人看的那一句话：花了多少、上限多少、还差多少内容没出片。"""
    fb = budget(script, cfg)
    return (
        f"已花 ¥{float(gate.get('spent_cny') or 0):.2f} / 上限 ¥{float(gate.get('limit_cny') or 0):.2f}"
        f"；这份稿子预估需要 {fb.need_total:g}s（约 {fb.clips_total} 段），"
        "确认继续会按新的上限接着跑"
    )
