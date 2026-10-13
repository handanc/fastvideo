"""可选模型目录：前端模型选择器 + 建任务校验的**唯一来源**（2026-10-11）。

**为什么要把模型选择做成 run 级**（本轮排查出的真实问题）：`.env` 的 `FV3_TEXT_MODEL` /
`FV3_T2I_MODEL` / `FV3_I2V_MODEL` 是**全局**开关 —— 改一次就把 economy/high/cinema 三个档位、
所有新 run 一起换掉，而且没人记得改回来。实测后果就是"档位表名不副实"：三份 yaml 写着
wanx2.1-t2i-turbo / wan2.2-i2v-plus / wan2.7-i2v，实际跑出来的却是 qwen-image-3.0 与
wan3.0-video（10-06、10-07 两条 run 的 `cost.json` 可查）。这里把选择权收到**每次生成**上，
落进 `run_meta.json`，续跑与单步都按原样恢复（见 `config.apply_run_overrides`）。

**为什么目录里的名字必须能被 `providers/protocols.py` 解析出协议**：模型名与 HTTP 端点/请求体
协议是绑定的（`models:` 的 provider 键同时决定了打哪个端点）。收了协议解析不出来的名字，
结果就是打到错误端点直接 400 —— 所以 `validate()` 对 t2i/i2v 强制要求能解析出协议，
解析不出来就**不收**，宁可让用户换个名字也不让他花一次钱才知道。

文本能力（text）没有这种绑定：`dashscope_qwen` 走 OpenAI 兼容端点，模型名只是路由参数，
所以不做协议校验。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from fastvideo3.core.cost import load_pricing

#: 允许手动选择的能力。口型（talk）也在其中 —— 但它的可选项来自 `core/talkmode.py`，
#: 且选中后**同时**决定 `talk_mode`（模型与模式一一对应，见 `resolve_talk_choice`）；
#: tts/vlm/r2v 不开放（改动它们要连带改提示词/度量口径，不是一次点选能说清的）。
SELECTABLE_CAPS: tuple[str, ...] = ("text", "t2i", "i2v")

#: 走"通用模型行"渲染的能力（口型单独一行：它还要带模式与音轨来源）
UI_CAPS: tuple[str, ...] = ("text", "t2i", "i2v")

FOLLOW_PROFILE = ""  # 空值 = 跟随画质档位（configs/*.yaml 的 models:），也就是不覆盖


@dataclass(frozen=True)
class ModelChoice:
    model: str
    label: str
    note: str  # 一句话：它实际改变什么观感/成本

    def to_dict(self) -> dict:
        return {"model": self.model, "label": self.label, "note": self.note}


CHOICES: dict[str, list[ModelChoice]] = {
    # ── 编剧 / 评审文本（走百炼模型市场专属端点，见 .env 的 DASHSCOPE_TEXT_BASE_URL）──
    "text": [
        ModelChoice("qwen3.7-flash", "qwen3.7-flash", "默认档：剧本与分镜够用，最省"),
        ModelChoice("qwen3.7-flash-2026-07-15", "qwen3.7-flash-2026-07-15", "锁定日期的快照版（可复现旧 run）"),
        ModelChoice("qwen3.7-plus", "qwen3.7-plus", "yaml 出厂值：创意与结构更稳，单价高于 flash"),
        ModelChoice("qwen3.7-plus-2026-05-26", "qwen3.7-plus-2026-05-26", "plus 的快照版"),
        ModelChoice("qwen3.7-max", "qwen3.7-max", "最强剧本档，贵；只在剧本反复不过审时值得"),
        ModelChoice("qwen3.7-max-2026-06-08", "qwen3.7-max-2026-06-08", "max 的快照版"),
        ModelChoice("qwen-plus", "qwen-plus", "通用老档：新剧本任务不建议"),
        ModelChoice("qwen3.8-flash", "qwen3.8-flash", "新一代 flash"),
        ModelChoice("qwen3.8-max", "qwen3.8-max", "新一代 max"),
        ModelChoice("qwen3.8-max-0902", "qwen3.8-max-0902", "3.8 max 的快照版"),
        ModelChoice("qwen3.8-27b", "qwen3.8-27b", "开源权重档：便宜，结构化输出稳定性需自行验证"),
    ],
    # ── 文生图（角色立绘 / 场景图 / 分镜首帧）──
    # t2i 的 provider 键同时决定端点协议：qwen-image* 与 wan2.7-image* 走
    # multimodal-generation **同步** messages 协议（`dashscope_wan27_image`），这一族才支持
    # 参考图（720P/1080P 的"硬锁脸"依赖它）；wanx2.1* 走异步 text2image 端点、不支持参考图，
    # 选它 + 720P/1080P 会在建任务时就被 `ref_image_warning` 拦下（不会等到 m3 花钱前才报错）。
    # ⚠ 清单必须**涵盖 configs 会默认用到的每一个模型**：面板去掉了"跟随默认"这个空选项，
    # 于是"当前生效值"必须本身就是可选项，否则 480P（默认 wanx2.1-t2i-turbo）无值可选。
    "t2i": [
        ModelChoice("qwen-image-2.0", "qwen-image-2.0", "支持参考图；写实质感与提示词遵循度好"),
        ModelChoice("qwen-image-2.0-pro", "qwen-image-2.0-pro", "2.0 高配档"),
        ModelChoice("qwen-image-3.0", "qwen-image-3.0", "支持参考图；立绘/首帧主力"),
        ModelChoice("qwen-image-3.0-pro", "qwen-image-3.0-pro", "3.0 高配档：细节更足"),
        ModelChoice("wan2.7-image", "wan2.7-image", "720P/1080P 出厂值：参考图锁脸 + 单边 1440 上限"),
        ModelChoice("wan2.7-image-pro", "wan2.7-image-pro", "2.7 高配档"),
        ModelChoice("wanx2.1-t2i-turbo", "wanx2.1-t2i-turbo", "480P 出厂值：最省；**不支持参考图**（不锁脸）"),
        ModelChoice("wanx2.1-t2i-plus", "wanx2.1-t2i-plus", "构图/质感优于 turbo；同样不支持参考图"),
    ],
    # ── 图生视频（动作镜头 / 空镜 / B 链的运动片段）──
    # wan3.0/wan2.7 走 `input.media` 新协议（→ provider `dashscope_wan27_i2v`），wan2.2 系走
    # `img_url` 旧协议（→ `dashscope_wan22_i2v`）；协议由 `providers/protocols.py` 按名字自动
    # 改道，所以两族可以并列在同一个下拉里。旧协议**不支持首尾帧**（那是另一个模型 kf2v 系），
    # 选它时 m4 会退回单首帧并记事件。
    # 只保留 wan3.0 系：场景级多镜头生成、首尾帧、参考素材、原生台词口型都只有这一族有。
    # wan2.7/wan2.2 系下线（用户裁定：image2video 只保留 wan3.0，先做最优质量再谈兼容）。
    "i2v": [
        ModelChoice("wan3.0-video", "wan3.0-video", "全能档：参考生视频/首尾帧，最长 30s、原生 1080P、自带台词口型"),
        ModelChoice("wan3.0-video-prime", "wan3.0-video-prime", "优速版：更快出片，规格同 3.0（单价 ×1.5）"),
    ],
}

#: 口型不再是可选能力：wan3.0 生成场景片段时原生带台词口型，独立口型通道已下线。
#: `talk` 从 SELECTABLE_CAPS 移除；FV3_TALK_MODEL 仍可在 wan3.0-video / -prime 间选模型。


def _pricing_known(model: str) -> bool:
    """模型名是否在 `pricing.yaml` 的 models 表里（决定台账与预估是不是真实价）。

    不在表里时 `price_of()` 会回落到该能力的**默认单价** —— 台账与成本上限都会失真。
    这条检查用于把"单价数据缺失"如实标出来（`priced=false`），而不是假装知道价格。
    """
    return model in ((load_pricing().get("models") or {}))


def is_known(capability: str, model: str) -> bool:
    return capability in CHOICES and any(c.model == model for c in CHOICES[capability])


def choices_for(capability: str) -> list[dict]:
    """给前端的选项列表。**不再有"跟随默认"这一空项**：面板必须始终显示一个具体模型，
    所以 configs 会默认用到的模型全部收进了 CHOICES（否则 480P 的 wanx2.1-t2i-turbo
    这类"当前生效值"根本不在下拉里，用户无从选起）。

    `FOLLOW_PROFILE`（空串）仍然保留为**内部语义**：run_meta 里没有该项 = 跟随配置，
    存量 run 与 CLI 不传该参数时都走这条路。
    """
    if capability not in CHOICES:
        return []
    return [c.to_dict() for c in CHOICES[capability]]


def labels_for(capability: str) -> list[str]:
    return [FOLLOW_PROFILE] + [c.model for c in CHOICES.get(capability, [])]


def note_for(capability: str, model: str) -> str:
    for c in CHOICES.get(capability, []):
        if c.model == model:
            return c.note
    return ""


def validate(capability: str, model: str, extra_allowed: Iterable[str] = ()) -> str | None:
    """返回错误说明（None = 通过）。建任务时**逐个**调用，别等花钱后才知道名字打错。

    `extra_allowed` = 目录之外但**当前配置里正在用**的模型名（`.env` 的 FV3_*_MODEL、
    或 yaml `models:` 里写的）。必须放行它们：面板的默认值就是从配置读出来的，
    若目录里恰好没收录这个名字，就会出现"界面显示着它、用户一点保存就 400"——
    那种自相矛盾比不收这个模型更糟。
    """
    if capability not in SELECTABLE_CAPS:
        return (
            f"不支持手动选择 {capability} 的模型（可选：{', '.join(SELECTABLE_CAPS)}）；"
            "tts/vlm/r2v 请改 configs 或 .env —— 换它们要连带改提示词与度量口径"
        )
    if model == FOLLOW_PROFILE:
        return None
    if not is_known(capability, model) and model not in set(extra_allowed):
        return (
            f"{capability} 不支持模型 {model!r}；可选："
            + " | ".join([c.model for c in CHOICES.get(capability, [])])
            + "（或留空＝跟随画质档位）"
        )
    return None


#: 能力 → yaml `models:` 里对应的键（用于算出"当前配置在用哪个模型"）。
#: 口型的键随模式而变（`talkmode.TalkChannel.model_key`），所以单独处理。
_YAML_MODEL_KEY = {
    "text": ("dashscope_text", "qwen3.7-flash"),
    "t2i": ("dashscope_t2i", "wanx2.1-t2i-turbo"),
    "i2v": ("dashscope_i2v", "wan3.0-video"),
}


def configured_models(cfg: Any) -> dict[str, str]:
    """当前配置下每个能力**实际会用**的模型名（.env 覆盖 > yaml > 内置默认）。

    这是面板"默认 X"的唯一来源，也是 `validate` 的 `extra_allowed`：
    目录没收录、但配置正在用的名字，必须既显示得出来、也保存得下去。
    """
    out: dict[str, str] = {}
    for cap, (key, builtin) in _YAML_MODEL_KEY.items():
        out[cap] = str(cfg.model_for(cap, key, builtin) or "")
    # 口型：模式决定通道，通道再决定模型（.env 的 FV3_TALK_MODEL 是最高优先级覆盖）
    from fastvideo3.core.talkmode import channel, provider_name

    prov = provider_name(cfg, getattr(cfg, "talk_mode", "")) or ""
    ch = channel(getattr(cfg, "talk_mode", ""))
    talk_ov = str((getattr(cfg, "model_overrides", None) or {}).get("talk") or "")
    out["talk"] = talk_ov or (
        str((getattr(cfg, "models", None) or {}).get(ch.model_key) or "")
        if prov == ch.provider and ch.model_key
        else ""
    ) or ch.default_model
    return {k: v for k, v in out.items() if v}


def protocol_error(capability: str, model: str) -> str | None:
    """模型名解析不出调用协议 = 打不对端点 = 提交必 400，所以不收。

    只校验 t2i/i2v：这两类的 provider 键同时决定 HTTP 端点与请求体协议
    （`providers/protocols.py`），文本类没有这种绑定。
    """
    if capability not in ("t2i", "i2v") or not model:
        return None
    from fastvideo3.providers import protocols

    proto = protocols.infer_protocol(model, capability)
    if not proto:
        return f"模型 {model} 无法解析调用协议（会被打到错误的端点）"
    want = protocols.provider_for(capability, proto)
    if not want:
        return f"模型 {model} 的协议 {proto} 没有对应的 provider 实现"
    return None


def ref_image_warning(capability: str, model: str, quality: str) -> str:
    """选了不支持参考图的 t2i，而该档位要求硬锁脸 → 花钱前就会报错，提前说出来。

    判据用协议而不是模型名表：`multimodal_sync`（messages 端点）这一族才带 `ref_image`
    （`DashScopeWan27ImageProvider.supports_reference_image=True`），异步 text2image 端点
    不支持 —— 与 m3 的 `require_reference_keyframes` 硬门控是同一条事实，两处各写必然漂。
    """
    if capability != "t2i" or not model or quality not in ("high", "cinema"):
        return ""
    from fastvideo3.providers import protocols

    if protocols.infer_protocol(model, "t2i") != protocols.T2I_MULTIMODAL:
        return (
            f"{quality} 档强制「参考图锁脸」（require_reference_keyframes），"
            f"而 {model} 不走 multimodal-generation 协议、不支持参考图 → m3 会在花钱前报错。"
        )
    return ""


def ui_catalog(effective: dict[str, str], sources: dict[str, str]) -> dict:
    """给 `GET /api/model-choices`：每个能力的选项 + 当前生效值 + 它是谁定的 + 价格状态。

    "当前生效值"必须一起显示：用户看不到"现在实际在用哪个模型"，选择器就会变成猜谜。

    **生效值不在目录里时，把它作为一项补进去**（标注来源）：目录是人工维护的清单，
    `.env` 却可以写任何名字。不补的话会出现"默认值显示着 X、下拉里却没有 X"，
    用户一保存就 400 —— 自相矛盾的界面比清单不全更糟。
    """
    out = []
    # 只渲染通用三行；口型由 API 单独给一块（它还要带 mode / 音轨来源 / 换模型代价说明）
    for cap in UI_CAPS:
        rows: list[dict[str, Any]] = []
        eff = effective.get(cap, "")
        if eff and eff not in {row["model"] for row in choices_for(cap)}:
            rows.append(
                {
                    "model": eff,
                    "label": f"{eff}（配置指定）",
                    "note": "来自 .env / configs 的当前配置，未收录进可选清单；仍可继续使用",
                    # 补项不走下面的循环，priced/current 得在这儿一次给齐
                    "priced": _pricing_known(eff),
                    "current": True,
                }
            )
        for row in choices_for(cap):
            m = row["model"]
            rows.append(
                {
                    **row,
                    "priced": bool(m) and _pricing_known(m),
                    "current": eff == m,
                }
            )
        out.append(
            {
                "capability": cap,
                "choices": rows,
                "effective_model": effective.get(cap, ""),
                "source": sources.get(cap, "yaml"),  # run | .env | yaml
                "current_priced": _pricing_known(effective.get(cap, "")),
            }
        )
    return {"capabilities": out}
