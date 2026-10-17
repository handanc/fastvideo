"""口型「模式 ↔ 通道」的唯一映射表（2026-10-11 五次定档）。

**为什么要有这个文件**（这条 bug 真的烧过两次钱）：本项目把"用哪个模型做口型"抄成了两份 ——
一份藏在 `talk_mode` 的取值里，一份写在 `configs/*.yaml` 的 `providers.talk` 路由里。两边
只要有一次没成对改，后果不是报错而是**静默降级**：对白镜头先过 m4 花一份运动片段的钱，再被
口型通道拒收（或根本拿不到片段），最终成片一条口型都没有，而"生成镜头视频"步骤因缺口永远
不绿、"拼接成片"卡在待处理（`run_20261002_235904`、`run_20261011_164152` 两次实测）。

现在收成一份：**模式的语义就是通道的输入形态**。谁需要知道口型怎么走，都从这里读：

- `m4_shots.shots_need_i2v` → `needs_clip_for_dialogue()`（要不要为对白镜头出运动片段）
- `m6_talk`           → `needs_audio()` / `voices_script()` / `accepts_last_frame()`
- `core/cost`         → `provider_name()`（预估与 ¥/秒红线取**实际那条通道**的单价）
- `core/stages`       → `provider_name()` / `fallback_name()`（装配主通道与兜底）
- `core/progress`     → `lipsync_required()`（没口型到底算缺口还是算完成）
- `app/api`           → `ui_channels()`（前端三个选项由后端给，JS 里不再抄一份价目表/通道名）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TalkMode = Literal["s2v_direct", "v2v_overlay", "scripted_talk", "audio_overlay"]

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
#: 代价就是上面那个错位。同时要"音色统一 + 嘴型对着配音"，请走 B 链 `videoretalk`：
#: 它先出运动片段、再对着**我们的母带**重画嘴，声/嘴/字幕天然同源。
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
    needs_audio: bool  # 通道是否吃我们的 TTS 音频
    voices_script: bool = False  # 台词写进提示词、由模型自己发声（C 链的特征）
    accepts_last_frame: bool = False  # 通道支持首尾帧锁起止
    #: 同一条链路上的**其它可选模型**（同协议、同输入形态，只是快慢/档位不同）。
    #: 例：C 链的 `wan3.0-video-prime` 是优速版，与 `wan3.0-video` 走同一个 provider、
    #: 同一套 media 协议，所以它是"同一条链的另一个模型"，**不是**新模式 ——
    #: 为它新开一个 mode 就要把 needs_clip/needs_audio/voices_script 那套语义抄第二份。
    alt_models: tuple[str, ...] = ()
    note: str = ""  # 取舍说明（前端/文档同一份）

    def models(self) -> tuple[str, ...]:
        """本链路可选的全部模型（默认档在前）。"""
        return (self.default_model,) + tuple(self.alt_models) if self.default_model else ()


TALK_CHANNELS: dict[str, TalkChannel] = {
    "s2v_direct": TalkChannel(
        mode="s2v_direct",
        label="wan2.2-s2v",
        provider="dashscope_wan22_s2v",
        model_key="dashscope_s2v",
        default_model="wan2.2-s2v",
        eats="图片 + 语音 + 提示词 → 口型视频",
        needs_clip_for_dialogue=False,
        needs_audio=True,
        note=(
            "优点：吃**我们的** TTS 母带 → 语音一致性由 m5 的音色池保证；口型对着真实台词时长。"
            "缺点：单人同屏（无多人各自口型）、画面由模型从一张静帧重新发明 → 不出动作镜头、"
            "只到 720P（1080P 成片会被钳位）。"
        ),
    ),
    "v2v_overlay": TalkChannel(
        mode="v2v_overlay",
        label="videoretalk",
        provider="dashscope_videoretalk",
        model_key="dashscope_retalk",
        default_model="videoretalk",
        eats="wan3.0 运动片段 + 语音 → 重加口型",
        needs_clip_for_dialogue=True,
        needs_audio=True,
        note=(
            "两段链：m4 先出动作视频（首帧/首尾帧），videoretalk 只重绘嘴部 → 不重画画面、"
            "输出分辨率与帧率=输入（1080P 不降级）、语音仍来自我们的母带。"
            "缺点：**链路最长最慢**（两次付费异步任务），且要求运动片段边长 ≥640 → 480P 用不了。"
        ),
    ),
    "scripted_talk": TalkChannel(
        mode="scripted_talk",
        label="wan3.0-video",
        provider="dashscope_wan30_talk",
        model_key="dashscope_wan30_talk",
        default_model="wan3.0-video",
        eats="图片 + 提示词（内含台词）→ 口型+动作一体",
        needs_clip_for_dialogue=False,
        needs_audio=False,
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
        needs_audio=False,
        note=(
            "显式放弃口型的**唯一**方式：教学零成本演示与四臂消融都靠它"
            "（`configs/ablations/*.yaml` 统一设成本值，以便只比一致性、不比口型）。"
            "其余模式下「对白镜头没有口型」永远是缺口而不是完成。"
        ),
    ),
}

#: 每条模式**允许**的通道覆盖（yaml 的 `providers.talk.default` 只有形态相容才算有效覆盖）。
#: 例：把 B 链的 videoretalk 换成 fal 的 /video-to-video 是合法的（同样吃视频+音频）；
#: 换成 wan2.2-s2v 则会被静默拒收视频 —— 那正是历史上的错配，这里直接判它无效。
_MODE_ALLOWED_PROVIDERS: dict[str, frozenset[str]] = {
    "s2v_direct": frozenset({"dashscope_wan22_s2v", "fal_infinitalk", "dashscope_wan27_audio"}),
    "v2v_overlay": frozenset({"dashscope_videoretalk", "fal_infinitalk", "kling_lipsync"}),
    "scripted_talk": frozenset({"dashscope_wan30_talk"}),
    "audio_overlay": frozenset(),
}


def channel(mode: str) -> TalkChannel:
    """按模式取通道规格；未知模式回落到 s2v_direct（**不 500、不猜**）。

    回落而不抛错：run_meta.json 里的 talk_mode 是建任务那一刻定死的历史值，读旧 run
    （或手改坏的配置）时应该照常出片，而不是让整个看板打不开。
    """
    return TALK_CHANNELS.get(mode or "", TALK_CHANNELS["s2v_direct"])


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


def needs_audio(mode: str) -> bool:
    """通道是否吃音频输入（决定 m6 要不要为它混音与检查 <20s 上限）。"""
    return channel(mode).needs_audio


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
    这样 configs 仍是"换供应商"的正规入口（fal 账户被锁时可以改路由换供应商），
    而错配（v2v_overlay 配 s2v、scripted_talk 配 retalk）不再有可能发生。
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

    这条判据必须留在后端而不是写进前端 JS：它一半来自供应商规格（videoretalk 要求输入
    视频边长 ≥640），一半来自实测的声画关系（C 链换母带会错位 0～2.2s）——
    抄一份到 JS 里就会在下次改动时漂。警告不阻断：用户可能就是想试一条镜头，
    让他先看见代价，比让他花完钱再发现"成片嘴不上"好。
    """
    ch = channel(mode)
    res = (i2v_resolution or "").strip().upper()
    # 文案按"外行一眼看懂"写：只说**后果 + 怎么办**，不解释协议名与内部机制
    # （细节留在本模块注释与 INTEGRATION；界面不抄第二份，抄了就会漂）。
    if ch.mode == "v2v_overlay" and res.startswith("48"):
        return (
            "videoretalk 要求运动片段边长 ≥640 像素，480P 会被拒 → 对白镜头没有口型。"
            "请改用 720P / 1080P，或口型选 wan2.2-s2v。"
        )
    if ch.mode == "scripted_talk":
        parts: list[str] = []
        # 这里以前有一条"480P 不锁脸，人物长相只靠文字描述" —— 已随 480P 开锁脸一起删除
        # （2026-10-17）。留着它就等于对着一个不存在的前提劝用户加钱：文案也是判据，
        # 判据改了要回来改文案。
        parts.append("口型与声音都由模型生成（默认用模型原声）；音色由模型定，同一角色跨镜头可能不同。")
        if (voice_source or DEFAULT_VOICE_SOURCE) == "master":
            parts.append(
                "已选「我们的配音」：实测语音会领先口型 0～2.2 秒且每镜不同，修不齐。"
                "要音色统一又要口型对得上，口型请选 videoretalk。"
            )
        return "".join(parts)
    return ""


def ui_models() -> list[dict[str, Any]]:
    """给「模型选择」组件用的口型行：显示**模型名**，落盘的是 `mode`。

    为什么不让用户直接写模型名进 `model_overrides["talk"]`：口型通道与它的输入形态是
    绑死的（s2v 只吃图片+音频、videoretalk 只吃视频、wan3.0 只吃图片+提示词），
    而"要不要为对白镜头出运动片段""成片音轨用谁的""字幕按谁的时间轴排"全都由模式决定。
    让模型名和模式各存一份，就是本项目反复出问题的那个形状（两次实测：对白镜头先烧一份
    运动片段的钱、再 100% 降级成无口型）。所以这里只做**展示层反查**：
    用户选模型 → 前端提交对应的 mode → 后端由 mode 解析通道，一条链，不分叉。
    """
    rows = []
    for ch in TALK_CHANNELS.values():
        # 一条链路可能有多个可选模型（如 C 链的 wan3.0-video 与优速版 -prime）：
        # 每个模型出一行，但**共用同一个 mode** —— 它们同协议同输入形态，
        # "要不要出运动片段/音轨用谁的/字幕按谁排"这些语义完全一样。
        for model in (ch.models() or ("",)):
            rows.append(
                {
                    "value": model,  # 前端 select 的值：模型名；audio_overlay 为空串
                    "model": model,
                    "label": model or ch.label,  # "不做口型"没有模型名，用通道标签
                    "mode": ch.mode,  # **真正落盘的字段**（run_meta.talk_mode）
                    "eats": ch.eats,
                    "note": ch.note,
                    "needs_clip_for_dialogue": ch.needs_clip_for_dialogue,
                    "voices_script": ch.voices_script,
                }
            )
    return rows


#: 协议 → 模式。用于**目录之外**的口型模型名（例如 .env 写了 `fal-ai/infinitalk`）：
#: 判据是"这个模型的请求形态吃什么"，与通道表同一套事实（`providers/protocols.py`），
#: 不再抄一份模型名清单。
_PROTOCOL_TO_MODE = {
    "s2v": "s2v_direct",              # 图片 + 音频
    "i2v_driving_audio": "s2v_direct",  # 首帧 + driving_audio，同样不吃视频
    "wan30_scripted_talk": "scripted_talk",  # 图片 + 提示词，模型自发声
    "videoretalk": "v2v_overlay",     # 视频 + 音频
    "fal_infinitalk_v2v": "v2v_overlay",
    "fal_infinitalk_image": "s2v_direct",
}

#: 协议 → **相容的模式集合**。fal 两个端点由同一个 provider 承接、按输入形态自动选端点，
#: 所以两种模式都相容；其余协议只吃一种形态，模式错了就是提交必败。
_PROTOCOL_COMPATIBLE_MODES = {
    "s2v": frozenset({"s2v_direct"}),
    "i2v_driving_audio": frozenset({"s2v_direct"}),
    "wan30_scripted_talk": frozenset({"scripted_talk"}),
    "videoretalk": frozenset({"v2v_overlay"}),
    "fal_infinitalk_v2v": frozenset({"v2v_overlay", "s2v_direct"}),
    "fal_infinitalk_image": frozenset({"s2v_direct", "v2v_overlay"}),
}


def mode_for_model(model: str) -> str | None:
    """模型名 → 模式（找不到返回 None，调用方报错而不是猜）。

    两级：① 通道表里的模型名（含各链路的备选模型，如 wan3.0-video-prime）；
    ② 表外的名字按**协议**反推（`.env` 可以写任何模型名，例如 fal 的端点 ID）。
    第 ② 级是必须的：面板的默认值来自 `.env`，若表外名字推不出模式，就会
    "界面显示着它、一保存就 400"。
    """
    want = (model or "").strip()
    if not want:
        return None
    exact = _exact_mode_for_model(want)
    if exact:
        return exact
    from fastvideo3.providers import protocols

    proto = protocols.infer_protocol(want, "talk")
    return _PROTOCOL_TO_MODE.get(proto or "")


def _exact_mode_for_model(model: str) -> str | None:
    """只查通道表（含 alt_models），不做协议推断。"""
    for ch in TALK_CHANNELS.values():
        if model in ch.models() or model == ch.label:
            return ch.mode
    return None


def reconcile_mode(model: str, current_mode: str) -> str:
    """按 `.env` 点名的口型模型**校准**模式：相容就保持不动，不相容才改。

    为什么不直接用 `mode_for_model` 覆盖：`fal-ai/infinitalk` 两个端点由同一个 provider
    承接（按输入形态自动选），`s2v_direct` 与 `v2v_overlay` **都相容**。若一律按协议反推，
    cinema 档 yaml 里的 `v2v_overlay` 会被悄悄改成 `s2v_direct` —— 对白镜头不再出运动片段，
    成本与画面都变了，而用户只是换了个供应商。
    "相容就别动、不相容才纠正"既保住了 yaml/用户的显式选择，又堵住了那个真会静默降级的错配
    （通道只吃提示词、模式却要喂音频 → 每个对白镜头 CapabilityMismatch → 全部无口型）。
    """
    want = (model or "").strip()
    if not want:
        return current_mode
    exact = _exact_mode_for_model(want)
    if exact:
        return exact  # 通道表内的名字与模式一一对应，没有歧义
    from fastvideo3.providers import protocols

    proto = protocols.infer_protocol(want, "talk")
    if not proto:
        return current_mode  # 认不出来就不猜（保持原模式，交给装配层报错）
    compatible = _PROTOCOL_COMPATIBLE_MODES.get(proto)
    if compatible is None or current_mode in compatible:
        return current_mode
    return _PROTOCOL_TO_MODE.get(proto, current_mode)


def resolve_talk_choice(value: str) -> tuple[str, str]:
    """前端提交的口型选择 → `(talk_mode, 模型覆盖或空串)`。

    - 传**模型名**（wan2.2-s2v / videoretalk / wan3.0-video / wan3.0-video-prime）：
      由模型反查模式，并把模型名作为覆盖（写进 run_meta.model_overrides['talk'] 与
      `.env` 的 FV3_TALK_MODEL）—— 同一条链有多个模型时，只有这样才能区分选了哪个。
    - 传**模式名**（s2v_direct 等）：仍接受（CLI/老前端/存量脚本），用该链的默认模型。
    - 空串 = 不做口型（audio_overlay），不设模型覆盖。
    认不出来返回 ("", "")，由调用方报 400 并列出可选值。
    """
    want = (value or "").strip()
    if not want:
        return "audio_overlay", ""
    if want in TALK_CHANNELS:
        ch = TALK_CHANNELS[want]
        return ch.mode, (ch.default_model if ch.default_model else "")
    mode = mode_for_model(want)
    if mode:
        return mode, want
    return "", ""


def ui_channels() -> list[dict[str, Any]]:
    """给前端的口型模式清单（选项、标签、输入形态、取舍）—— 单一事实来源在本表。

    此前前端 JS 里抄了一份通道名与单价表，一漂就等于让用户按假数字填熔断
    （熔断中途跳闸的表现恰恰是"成片没有对口型"）。
    """
    return [
        {
            "mode": ch.mode,
            "label": ch.label,
            "eats": ch.eats,
            "note": ch.note,
            "needs_clip_for_dialogue": ch.needs_clip_for_dialogue,
        }
        for ch in TALK_CHANNELS.values()
    ]


def default_voice_source() -> VoiceSource:
    return DEFAULT_VOICE_SOURCE
