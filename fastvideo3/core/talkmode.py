"""口型「模式 ↔ 通道」的唯一映射表（2026-10-11 五次定档）。

**为什么要有这个文件**（这条 bug 真的烧过两次钱）：本项目把"用哪个模型做口型"抄成了两份 ——
一份藏在 `talk_mode` 的取值里，一份写在 `configs/*.yaml` 的 `providers.talk` 路由里。两边
只要有一次没成对改，后果不是报错而是**静默降级**：对白镜头先过 m4 花一份运动片段的钱，再被
口型通道拒收（或根本拿不到片段），最终成片一条口型都没有，而"生成镜头视频"步骤因缺口永远
不绿、"拼接成片"卡在待处理（`run_20261002_235904`、`run_20261011_164152` 两次实测）。

现在收成一份：**模式的语义就是通道的输入形态**。谁需要知道口型怎么走，都从这里读：

- `m4_shots.shots_need_i2v` → `needs_clip_for_dialogue()`（要不要为对白镜头出运动片段）
- `m6_talk`           → `voices_script()` / `accepts_last_frame()`
- `core/cost`         → `provider_name()`（预估与 ¥/秒红线取**实际那条通道**的单价）
- `core/stages`       → `provider_name()` / `fallback_name()`（装配主通道与兜底）
- `core/progress`     → `lipsync_required()`（没口型到底算缺口还是算完成）
- `app/api`           → `channel()` / `tier_conflict()`（显示语义与冲突告警由后端给，JS 不抄第二份）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TalkMode = Literal["scripted_talk", "audio_overlay"]

#: C 链（`scripted_talk`）的音轨来源。**默认 model（保留模型自己的音轨）** —— 这是实测翻的案。
#: C 链的口型是模型**对着自己发声的节奏**画出来的，把音轨换成我们的 TTS 母带就是声画两张皮。
#: `run_20261011_231024` 逐镜量过（嘴部运动能量 × 音频包络，取相关最大时的平移量）：
#:     镜头        clip    母带   母带要对齐嘴型需延后
#:     shot_1_2   4.04s  2.53s   +0.83s
#:     shot_2_1   3.02s  1.65s   +2.17s
#:     shot_2_2   3.02s  1.65s   +1.50s
#:     shot_3_1   4.04s  2.19s   +0.00s
#:     shot_3_2   4.04s  2.72s   +0.00s
#: 两个结论：① 语音领先口型最多 2.2 秒，观众一定看得出来；② **偏移不是常数**，所以
#: "给母带加一个固定 delay"这条路修不好（而且这把尺子在人眼确认过的素材上本就会假阴性，
#: 见 INTEGRATION"口型度量证伪与重建"，不能拿它做自动对齐的依据）。
#: 选 `master`（换回我们的配音）换来的是**语音一致性**（全片音色统一、跨镜头不漂），
#: 代价就是上面那个错位。要"音色统一 + 嘴型对着这条配音"在现通道下做不到（旧 B 链
#: videoretalk 已于 2026-11-04 移除）；能做的只有用 reference_audio 克隆**音色**。
VoiceSource = Literal["master", "model"]
DEFAULT_VOICE_SOURCE: VoiceSource = "model"


@dataclass(frozen=True)
class TalkChannel:
    """一条口型链路：模式名 + 它实际调用的通道 + 这条链路对上下游的要求。"""

    mode: str
    label: str  # 前端显示的**通道名**（用户看得懂的那个，不是内部模式串）
    provider: str | None  # factory 里的 provider 名；audio_overlay 不需要通道
    model_key: str  # configs `models:` 里可覆盖该通道模型名的键
    default_model: str
    eats: str  # 输入形态（人话，进前端说明与事件 hint）
    needs_clip_for_dialogue: bool  # 对白镜头是否要过 m4 出运动片段
    voices_script: bool = False  # 台词写进提示词、由模型自己发声（C 链的特征）
    accepts_last_frame: bool = False  # 通道支持首尾帧锁起止
    #: 同一条链路上的**其它可选模型**（同协议、同输入形态，只是快慢/档位不同）。
    #: 例：C 链的 `wan3.0-video-prime` 是优速版，与 `wan3.0-video` 走同一个 provider、
    #: 同一套 media 协议，所以它是"同一条链的另一个模型"，**不是**新模式 ——
    #: 为它新开一个 mode 就要把 needs_clip/voices_script 那套语义抄第二份。
    alt_models: tuple[str, ...] = ()
    note: str = ""  # 取舍说明（前端/文档同一份）

    def models(self) -> tuple[str, ...]:
        """本链路可选的全部模型（默认档在前）。"""
        return (self.default_model,) + tuple(self.alt_models) if self.default_model else ()


TALK_CHANNELS: dict[str, TalkChannel] = {
    "scripted_talk": TalkChannel(
        mode="scripted_talk",
        label="wan3.0-video",
        provider="dashscope_wan30_talk",
        model_key="dashscope_wan30_talk",
        default_model="wan3.0-video",
        eats="图片 + 提示词（内含台词）→ 口型+动作一体",
        needs_clip_for_dialogue=False,
        voices_script=True,
        accepts_last_frame=True,
        # 优速版：同协议同输入形态，只是出片更快 → 同一条链的另一个模型，不是新模式
        alt_models=("wan3.0-video-prime",),
        note=(
            "一次调用出**口型 + 动作 + 自带音轨**，原生 1080P、显式 9:16，最快也最像"
            "商业管线。代价在一致性：音色由模型决定（跨镜头可能换嗓子，`voice_source` 可切回"
            "我们的母带）；官方 `first_frame/last_frame` 与 `reference_audio` **互斥** → 用严格"
            "首帧锁脸就不能同时喂音色克隆。"
        ),
    ),
    "audio_overlay": TalkChannel(
        mode="audio_overlay",
        label="不做口型",
        provider=None,
        model_key="",
        default_model="",
        eats="只叠音轨（m6 整步跳过）",
        needs_clip_for_dialogue=True,
        note=(
            "显式放弃口型的**唯一**方式：教学零成本演示与四臂消融都靠它"
            "（`configs/ablations/*.yaml` 统一设成本值，以便只比一致性、不比口型）。"
            "其余模式下「对白镜头没有口型」永远是缺口而不是完成。"
        ),
    ),
}

#: 每条模式**允许**的通道覆盖（yaml 的 `providers.talk.default` 只有形态相容才算有效覆盖）。
#: 例：同一形态内换供应商（如旧 B 链的 videoretalk ↔ 可灵）是合法的；跨形态换成
#: wan2.2-s2v 则会被静默拒收视频 —— 那正是历史上的错配，这里直接判它无效。
#: （fal InfiniteTalk 已于 2026-11-04 移除：账户 403 锁死，且口型主通道早已是 wan3.0 自带。）
_MODE_ALLOWED_PROVIDERS: dict[str, frozenset[str]] = {
    "scripted_talk": frozenset({"dashscope_wan30_talk"}),
    "audio_overlay": frozenset(),
}


def channel(mode: str) -> TalkChannel:
    """按模式取通道规格；**认不出的模式串一律按 C 链处理**（不 500，也不静默变成没口型）。

    为什么不回落到 audio_overlay：那会把"要口型"静默变成"没口型"，正是本文件注释里
    两次烧钱的形状。A/B 链模式串已于 2026-11-04 随通道删除，读到它们的只剩旧 run_meta。
    不抛错的理由不变：看板不该因为一条历史字符串就打不开。
    """
    return TALK_CHANNELS.get(mode or "", TALK_CHANNELS["scripted_talk"])


def shots_need_last_frame(script: Any, talk_mode: str, scene_mode: str = "auto") -> list[Any]:
    """要产尾帧的镜头集合（m3 产尾帧、frames 步缺口、看板覆盖率共用这一个判据）。

    尾帧**只在会被用到的地方产**（每张 ≈¥0.2，产了不用就是白烧）：

    - 场景级生成开着（scripted_talk + scene_mode != off）：只有走**首尾帧形态**的组
      需要尾帧（规则3：≤1 角色且无对白），取每组的**末镜**（尾帧锁的是结束态）。
      走参考形态的组由模型自己画中间帧，不需要尾帧。
    - 场景级生成关着（economy / A 链 / B 链 / 手动 off）：回到逐镜头路径，
      需要运动片段的镜头都要尾帧（m4 发 last_frame）；C 链对白镜头的画面由口型
      通道直出且该通道吃尾帧，也要。
    """
    from fastvideo3.core.scenegroup import MODE_FIRST_LAST, choose_input_mode, group_scenes
    from fastvideo3.modules.m4_shots import shots_need_i2v

    want: set[str] = set()
    scene_active = talk_mode == "scripted_talk" and scene_mode != "off"
    if scene_active:
        shots_by_id = {sh.shot_id: sh for sh in script.all_shots()}
        for g in group_scenes(script):
            dlg = any(bool(shots_by_id[sid].dialogues) for sid in g.shot_ids if sid in shots_by_id)
            mode, _why = choose_input_mode(
                g, has_dialogue=dlg, has_first_frames=True, forced=scene_mode
            )
            if mode == MODE_FIRST_LAST and g.shot_ids:
                want.add(g.shot_ids[-1])
    else:
        want |= {s.shot_id for s in shots_need_i2v(script, talk_mode)}
        if accepts_last_frame(talk_mode):
            want |= {s.shot_id for s in script.all_shots() if s.dialogues}
    return [s for s in script.all_shots() if s.shot_id in want]


def needs_clip_for_dialogue(mode: str) -> bool:
    """对白镜头是否需要 m4 的运动片段（决定这份钱花不花）。"""
    return channel(mode).needs_clip_for_dialogue


def voices_script(mode: str) -> bool:
    """台词是否进提示词由模型发声（C 链）。这类产物自带音轨，音轨来源要显式决定。"""
    return channel(mode).voices_script


def accepts_last_frame(mode: str) -> bool:
    return channel(mode).accepts_last_frame


def lipsync_required(mode: str) -> bool:
    """本模式是否要求对白镜头出口型产物 —— `progress.lipsync_coverage` 的唯一判据。"""
    return channel(mode).provider is not None


def provider_name(cfg: Any, mode: str) -> str | None:
    """本模式实际应装配的 provider 名。

    优先用 yaml `providers.talk.default`，但**仅当它与本模式的输入形态相容**——
    这样 configs 仍是"换供应商"的正规入口（某家账户被锁时可以改路由换供应商），
    而跨形态错配（模式要图片、通道只吃视频）不再有可能发生。
    """
    ch = channel(mode)
    if ch.provider is None:
        return None
    route = (getattr(cfg, "routing", None) or {}).get("talk")
    want = getattr(route, "default", "") if route else ""
    if want and want in _MODE_ALLOWED_PROVIDERS.get(ch.mode, frozenset()):
        return want
    return ch.provider


def fallback_name(cfg: Any, mode: str) -> str | None:
    """兜底通道名：形态必须同样相容，否则"兜底"等于第二次失败。"""
    ch = channel(mode)
    if ch.provider is None:
        return None
    route = (getattr(cfg, "routing", None) or {}).get("talk")
    fb = getattr(route, "fallback", None) if route else None
    if fb and fb in _MODE_ALLOWED_PROVIDERS.get(ch.mode, frozenset()) and fb != provider_name(cfg, mode):
        return fb
    return None


def tier_conflict(
    mode: str, i2v_resolution: str = "", voice_source: str = ""
) -> str:
    """模式 × 分辨率 × 音轨来源的**真实硬冲突**，返回人话警告（无冲突返回空串）。

    参数里不再有"画质档"：它与分辨率一对一，两个旋钮表达同一件事就必然出现
    "cinema + 480P"这种自相矛盾的组合 —— 本函数过去正是为那个组合写了特例。

    这条判据必须留在后端而不是写进前端 JS：它一半来自供应商规格（旧 B 链要求输入
    视频边长 ≥640），一半来自实测的声画关系（C 链换母带会错位 0～2.2s）——
    抄一份到 JS 里就会在下次改动时漂。警告不阻断：用户可能就是想试一条镜头，
    让他先看见代价，比让他花完钱再发现"成片嘴不上"好。
    """
    ch = channel(mode)
    # （`i2v_resolution` 参数保留：调用方一直在传，而现剩下的唯一硬冲突只与音轨来源有关
    #   —— 原先靠分辨率判的那条是旧 B 链的"运动片段边长 ≥640"，2026-11-04 随通道删除。）
    # 文案按"外行一眼看懂"写：只说**后果 + 怎么办**，不解释协议名与内部机制
    # （细节留在本模块注释与 INTEGRATION；界面不抄第二份，抄了就会漂）。
    if ch.mode == "scripted_talk":
        parts: list[str] = []
        # 这里删掉过两句长期挂在面板上的告警，两句都是"判据改了、文案没跟着改"：
        #   ① "480P 不锁脸，人物长相只靠文字描述"（2026-10-17，随 480P 开锁脸删除）；
        #   ② "口型与声音都由模型生成…音色由模型定，同一角色跨镜头可能不同"
        #      （2026-10-18，按用户要求去掉）。②已经不成立：每个有对白的场景组都把本组
        #      说话人的音色样本发进 `reference_audio`（`core/voiceref.py`，官方预算
        #      ≤5 段且总 ≤15 秒），跨场景锁声就是这条机制的目标。**真要有人没拿到名额**
        #      （说话人多到超预算）那是逐次的实况，写在降级台账与卡片上，而不是每天对着
        #      所有任务复读一句必然让人以为"声音一定会飘"的话。
        # 留在这里的只有 `voice_source=master` 那条：它是实测的硬冲突（语音领先口型
        # 0～2.2 秒且每镜不同），不是泛泛的风险提示。
        if (voice_source or DEFAULT_VOICE_SOURCE) == "master":
            parts.append(
                "已选「我们的配音」：实测语音会领先口型 0～2.2 秒且每镜不同，修不齐。"
                "要声画对齐就回到「模型原声」；旧 B 链（对着母带重画嘴）已于 2026-11-04 移除。"
            )
        return "".join(parts)
    return ""
