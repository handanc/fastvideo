"""Provider 工厂：按配置路由装配实例；Fake 模式整体替换（TDD §6.1）。"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core.config import AppConfig, ProviderRouting
from fastvideo3.core.schema import BaseInfo
from fastvideo3.providers.base import Capability, Provider, ProviderError


def _talk_route(cfg: AppConfig, route: ProviderRouting) -> ProviderRouting:
    """把**口型模式**解析成的通道套到路由上（模式就是通道的选择器）。

    为什么解析在工厂而不是各调用方：`make_provider` / `make_fallback` 是 m6、CLI、doctor、
    成本预估共同经过的**唯一装配口**，在这里解析一次，所有读取方自动一致。
    历史上"用哪个模型做口型"被抄成两份（`talk_mode` 一份、`providers.talk` 路由一份），
    漂一次的后果不是报错而是静默降级：对白镜头先烧一份运动片段的钱、再 100% 出无口型叠音
    （run_20261002_235904、run_20261011_164152 两次实测）。现在结构上不可能再错配。

    映射表与形态相容规则见 `core/talkmode.py`：yaml 路由仍然可以**换供应商**（例如把 B 链的
    videoretalk 换成 fal 的 /video-to-video），但只在该供应商吃同样输入形态时生效。
    """
    from fastvideo3.core import talkmode

    # **例外**：`.env` 里显式写了 `FV3_TALK_MODEL=<模型名>` 时按它装配，本表不否决。
    # 理由：那是专家级的主动覆盖（试新模型/换端点），而且 `_auto_route` 已把 provider 改道到
    # 该模型真正的协议；真喂错输入时通道会**响亮拒绝**（wan30 通道收到音频即
    # CapabilityMismatchError），不是静默降级 —— 与本表要消灭的那个失效模式正好相反。
    if (getattr(cfg, "model_overrides", None) or {}).get("talk"):
        return route

    name = talkmode.provider_name(cfg, cfg.talk_mode)
    if name is None:  # audio_overlay：本模式不需要口型通道
        return route
    fb = talkmode.fallback_name(cfg, cfg.talk_mode)
    if (route.default, route.fallback) == (name, fb):
        return route
    return ProviderRouting(default=name, fallback=fb)


#: provider 名 → (yaml `models:` 键, 内置默认)。与 `_build_real` 里每个分支的 `mm(...)`
#: **一一对应**，并由 `tests/test_model_in_effect.py` 逐条与真实装配结果比对钉住。
#:
#: 为什么必须有这张表：想知道"这次会打哪个模型"以前只能猜 provider 键。`api`、`doctor`、
#: `model_catalog` 三处都硬写了 `dashscope_t2i`，于是 t2i 路由换成 `dashscope_wan27_image`
#: 之后，**面板显示的模型与真正调用的模型是两个东西**（实测：三档都路由到 wan2.7-image，
#: 面板"默认"仍显示 wanx2.1-t2i-turbo / plus —— 那两个只是兜底路由）。用户在错的基线上
#: 做选择，比没有默认值更糟。
_PROVIDER_MODEL_KEY: dict[str, tuple[str, str]] = {
    "deepseek": ("deepseek_text", "deepseek-chat"),
    "dashscope_qwen": ("dashscope_text", "qwen3.7-plus"),
    "dashscope_t2i": ("dashscope_t2i", "wanx2.1-t2i-turbo"),
    "dashscope_wan27_image": ("dashscope_t2i27", "wan2.7-image"),
    "siliconflow_flux": ("siliconflow_t2i", "black-forest-labs/FLUX.1-dev"),
    "dashscope_wan22_i2v": ("dashscope_i2v", "wan2.2-i2v-flash"),
    "dashscope_wan27_i2v": ("dashscope_i2v", "wan3.0-video"),
    "dashscope_wan27_audio": ("dashscope_i2v27", "wan2.7-i2v"),
    "dashscope_wan30_talk": ("dashscope_wan30_talk", "wan3.0-video"),
    "siliconflow_wan22": ("siliconflow_i2v", "Wan-AI/Wan2.2-I2V-A14B"),
    "dashscope_wan27_r2v": ("dashscope_r2v", "wan2.7-r2v"),
    "dashscope_wan22_s2v": ("dashscope_s2v", "wan2.2-s2v"),
    "dashscope_videoretalk": ("dashscope_retalk", "videoretalk"),
    "kling_lipsync": ("kling_talk", "kling-v2-master"),
    "fal_infinitalk": ("fal_talk", "fal-ai/infinitalk"),
    "dashscope_cosyvoice": ("dashscope_tts", "cosyvoice-v2"),
    "edge_tts": ("", ""),
    "dashscope_vlm": ("dashscope_vlm", "qwen-vl-plus"),
}

#: 外部平台**不吃** `.env`/run 的能力级模型覆盖（`qwen-image-3.0`、`wan3.0-video` 在
#: SiliconFlow/可灵平台上不存在，漏进兜底就是 404 —— 见 `_build_real` 里那段注释）。
#: `edge_tts` 同理：它压根没有"模型名"这个概念。
#: 显示也必须按同一条规则算，否则 `doctor` 会报出一个"根本不会发出去"的名字。
_NO_OVERRIDE_PROVIDERS = ("siliconflow_", "kling_", "edge_tts")


def provider_for(cfg: AppConfig, capability: Capability | str) -> str:
    """这一份配置会走**哪个 provider**（含协议自动改道、口型通道解析），不装配、不读 KEY。

    路由解析与 `make_provider` 逐步同序（`_talk_route` → `_auto_route`），因为这两件事必须
    给出同一个答案：面板显示的那个 provider 就是等一下真正被调用的那个。
    """
    cap = capability.value if isinstance(capability, Capability) else str(capability)
    route = cfg.routing.get(cap)
    if route is None:
        return ""
    if cap == Capability.TALK.value:
        route = _talk_route(cfg, route)
    return _auto_route(route.default, cap, cfg.model_overrides or {})


def model_in_effect(cfg: AppConfig, capability: Capability | str) -> str:
    """这一份配置**实际会调用**的模型名（run 选择 > `.env` 覆盖 > 路由对应的 yaml 键 > 内置默认）。

    判据与 `mm()` 完全一致，但**不需要 API KEY、不构造 provider**，所以没配 KEY 也能显示。
    """
    cap = capability.value if isinstance(capability, Capability) else str(capability)
    name = provider_for(cfg, cap)
    key, builtin = _PROVIDER_MODEL_KEY.get(name, ("", ""))
    if not key:
        return ""  # 认不出的路由（或 edge_tts 这种没有模型名的通道）就说不清：留空，别编一个
    if cap == "talk" and name == "dashscope_wan30_talk":
        return str(_wan30_talk_model(cfg))  # C 链口型有自己的"跟随 i2v 同族"规则
    picked = str((cfg.model_overrides or {}).get(cap) or "")
    if picked and not name.startswith(_NO_OVERRIDE_PROVIDERS):
        return picked
    return str(cfg.models.get(key, builtin) or "")


def make_provider(capability: Capability, cfg: AppConfig, work_dir: Path | None = None) -> Provider:
    """capability → 具体 Provider 实例。

    Fake 模式下返回对应 Fake 实现（零成本离线）；
    真实模式按 configs 路由选择默认供应商，**口型能力另由 talk_mode 决定实际通道**。
    """
    if cfg.fake:
        return _make_fake(capability, cfg, work_dir)

    route = cfg.routing.get(capability.value)
    if route is None:
        raise ProviderError(f"配置缺少 {capability.value} 的供应商路由")
    if capability is Capability.TALK:
        route = _talk_route(cfg, route)
    return _make_real(route.default, capability, cfg)


def make_r2v_provider(cfg: AppConfig) -> Provider | None:
    """角色一致性参考生视频供应商（cinema 档配置 r2v 路由时启用）。"""
    if cfg.fake:
        return _make_fake(Capability.I2V, cfg, None)
    route = cfg.routing.get("r2v")
    if route is None:
        return None
    try:
        # cap_key="r2v"：r2v 是独立能力，不能被 FV3_I2V_MODEL 的覆盖串台
        return _make_real(route.default, Capability.I2V, cfg, cap_key="r2v")
    except ProviderError as e:
        # 此前静默返回 None：cinema 的角色一致性引擎挂了无任何告警
        import logging

        logging.getLogger(__name__).warning(
            "r2v 供应商构造失败，角色一致性退化为普通 i2v: %s", e
        )
        return None


def make_fallback(
    capability: Capability, cfg: AppConfig, work_dir: Path | None = None
) -> Provider | None:
    """兜底供应商。.env 指定了试用/新模型时，**第一兜底是 yaml 原本指向的付费模型**。

    必要性：economy 的 t2i fallback 是 siliconflow_flux，没配 SILICONFLOW_API_KEY 时
    构造即抛错 → 返回 None。若 .env 把模型换成尚未开通的 qwen-image-3.0，主路径会
    AccountGateError 而兜底为空，整条管线直接失败（用户要的是"额度耗尽自动回落付费"，
    不是"跑不动"）。
    """
    route = cfg.routing.get(capability.value)
    if cfg.fake or route is None:
        return None
    if capability is Capability.TALK:
        # 兜底也必须**吃得下本模式的输入形态**：scripted_talk 的兜底是空（fal 要音频、
        # kling/retalk 要视频，都不吃"纯提示词自发声"），宁可响亮降级也不要第二次失败。
        route = _talk_route(cfg, route)
    if cfg.model_overrides.get(capability.value):
        try:  # 用未覆盖的 yaml 模型名重建同一个 provider
            return _make_real(route.default, capability, cfg, overrides={})
        except ProviderError:
            pass
    if not route.fallback:
        return None
    try:
        # overrides={}：兜底必须用 yaml 里它自己平台的模型。否则 .env 的
        # FV3_TALK_MODEL=wan3.0-video 会同时污染主通道与兜底通道（都变成不认
        # driving_audio 的同一个模型），"换道重试"等于原地重试（实测卡死过）。
        return _make_real(route.fallback, capability, cfg, overrides={})
    except ProviderError:
        return None


def _make_fake(capability: Capability, cfg: AppConfig, work_dir: Path | None) -> Provider:
    from fastvideo3.providers import fake

    if capability is Capability.TEXT:
        return fake.FakeTextProvider(base_info=BaseInfo(topic="fake"))
    if capability is Capability.T2I:
        return fake.FakeT2IProvider()
    if capability is Capability.I2V:
        return fake.FakeI2VProvider()
    if capability is Capability.TALK:
        return fake.FakeTalkProvider()
    if capability is Capability.TTS:
        return fake.FakeTTSProvider()
    if capability is Capability.VLM:
        return fake.FakeVLMProvider()
    raise ProviderError(f"未知能力: {capability}")


def _auto_route(name: str, cap_key: str, overrides: dict[str, str]) -> str:
    """.env 里换模型名 → 顺带把 provider 改道到该模型真正使用的协议。

    必要性：`models:` 的键只决定"用哪个 provider 类"（即哪个端点/请求体协议）。
    把 t2i 模型改成 qwen-image-3.0 而 provider 仍是 dashscope_t2i（万相异步
    text2image 端点）时，请求必然 400。协议推断不出来时**不改道**（不猜）。
    """
    from fastvideo3.providers import protocols

    model = overrides.get(cap_key)
    if not model:
        return name
    want = protocols.provider_for(cap_key, protocols.infer_protocol(model, cap_key))
    if want and want != name:
        import logging

        logging.getLogger(__name__).warning(
            "模型 %s 走 %s 协议，路由 %s 自动改道为 %s",
            model,
            protocols.infer_protocol(model, cap_key),
            name,
            want,
        )
        return want
    return name


def _video_cost_cap(cfg: AppConfig, provider: Provider) -> None:
    """视频类通道单价闸门：超过 cfg.max_video_cost_per_sec 直接报错。

    用户红线"口型视频成本不应超过 ¥5/s"必须落在装配层：否则 .env 改一个模型名
    就可能静默切到贵通道，烧钱直到看账单才发现。

    取 `worst_case_price_key()`（该供应商**可能发出的最高单价**）而不是默认档：
    fal InfiniteTalk 的 480p/720p、i2v/v2v 差到 3 倍，按低档校验等于红线形同虚设。
    """
    from fastvideo3.core.cost import price_of

    key = provider.worst_case_price_key()
    price = price_of(key, provider.capability.value, 1.0)
    if price > cfg.max_video_cost_per_sec:
        raise ProviderError(
            f"模型 {key} 单价 ¥{price}/秒 超过上限 ¥{cfg.max_video_cost_per_sec}/秒。"
            f"两条出路：① 降分辨率档 `i2v_resolution`（wan3.0-video 1080P ¥1.2/秒 → "
            f"720P ¥0.6 → 480P ¥0.3，这条最直接）；② 确实要这一档就调高 "
            f"FV3_MAX_VIDEO_COST_PER_SEC 或 yaml 的 max_video_cost_per_sec。"
            f"（别再列 A/B 链的便宜通道：videoretalk / fal-infinitalk / wan2.2-s2v 已从"
            f"本项目删除，照着换是换不出模型的。）"
        )


def _make_real(
    name: str,
    capability: Capability,
    cfg: AppConfig,
    cap_key: str | None = None,
    overrides: dict[str, str] | None = None,
) -> Provider:
    """装配真实 provider，并对视频类通道做单价闸门（i2v/talk/r2v 全覆盖）。"""
    provider = _build_real(name, capability, cfg, cap_key, overrides)
    if provider.capability in (Capability.I2V, Capability.TALK):
        _video_cost_cap(cfg, provider)
    return provider


def _wan30_talk_model(cfg: AppConfig) -> str:
    """C 链的口型模型：**跟随面板选的 i2v 模型**（同族才跟），否则用目录默认 `wan3.0-video`。

    为什么要跟随：C 链本来就是"用 i2v 家族的模型一次出口型+动作"。用户在面板上把图生视频
    换成 `wan3.0-video-prime`（优速档）时，期望口型也走 prime；不跟随就会出现同一个 run 里
    两种档位混跑、账单上 `i2v=wan3.0-video-prime` 而 `talk=wan3.0-video`
    （`run_20261011_231024` 实测就是这个形状）。

    为什么"同族才跟"：C 链的口型靠**提示词里的台词**（wan3.0 系没有 driving_audio），
    而 `wan2.7-i2v` 的口型靠 driving_audio 音频驱动 —— 跟着换过去就是把请求发成对方不认的
    形态（真实 400，原文存在 tests/test_talk_fallback.py）。所以判据是"该模型在 talk 能力上
    解析出来的协议是不是 `TALK_SCRIPTED`"，不是名字前缀匹配。
    """
    from fastvideo3.providers import protocols

    ov = (cfg.model_overrides or {}).get("talk")
    if ov:
        return ov  # 专家级显式指定口型模型：优先级最高（.env 的 FV3_TALK_MODEL）
    picked = (cfg.model_overrides or {}).get("i2v") or ""
    if picked and protocols.infer_protocol(picked, "talk") == protocols.TALK_SCRIPTED:
        return picked
    return cfg.models.get("dashscope_wan30_talk", "wan3.0-video")


def _build_real(
    name: str,
    capability: Capability,
    cfg: AppConfig,
    cap_key: str | None = None,
    overrides: dict[str, str] | None = None,
) -> Provider:
    from fastvideo3.providers import dashscope, deepseek, edge_tts, fal, kling, siliconflow

    cap_key = cap_key or capability.value
    ov = cfg.model_overrides if overrides is None else overrides
    name = _auto_route(name, cap_key, ov)

    def mm(provider_key: str, default: str) -> str:
        """.env 能力级覆盖 > yaml models[provider_key] > 内置默认。"""
        return ov.get(cap_key) or cfg.models.get(provider_key, default)

    m = cfg.models
    if name == "deepseek":
        return deepseek.DeepSeekTextProvider(
            cfg.deepseek_api_key,
            cfg.deepseek_base_url,
            mm("deepseek_text", "deepseek-chat"),
            timeout_sec=cfg.text_timeout_sec,
        )
    if name == "dashscope_qwen":
        return dashscope.DashScopeTextProvider(
            cfg.dashscope_api_key,
            mm("dashscope_text", "qwen3.7-plus"),
            base_url=cfg.dashscope_text_base_url,
            enable_thinking=cfg.enable_thinking,
            timeout_sec=cfg.text_timeout_sec,
        )
    if name == "dashscope_t2i":
        return dashscope.DashScopeT2IProvider(
            cfg.dashscope_api_key,
            mm("dashscope_t2i", "wanx2.1-t2i-turbo"),
        )
    if name == "dashscope_wan27_image":
        # multimodal-generation 同步协议：wan2.7-image 与 qwen-image 系同族
        return dashscope.DashScopeWan27ImageProvider(
            cfg.dashscope_api_key,
            mm("dashscope_t2i27", "wan2.7-image"),
        )
    if name == "dashscope_wan22_i2v":
        return dashscope.DashScopeI2VProvider(
            cfg.dashscope_api_key,
            mm("dashscope_i2v", "wan2.2-i2v-flash"),
        )
    if name == "dashscope_wan22_s2v":
        # base_url 可指到业务空间专属端点（FV3_S2V_BASE_URL）：s2v 官方文档要求华北2
        # （北京）地域，公共端点不通时换端点就够了，不该逼着改代码。
        return dashscope.DashScopeTalkProvider(
            cfg.dashscope_api_key,
            mm("dashscope_s2v", "wan2.2-s2v"),
            base_url=cfg.dashscope_video_base_url,
        )
    if name == "dashscope_videoretalk":
        return dashscope.DashScopeRetalkProvider(
            cfg.dashscope_api_key,
            mm("dashscope_retalk", "videoretalk"),
        )
    if name == "dashscope_wan27_i2v":
        # media 数组新协议（wan2.7-i2v 与 wan3.0-video 系同族）。模型名读 **dashscope_i2v**
        # —— 与 `model_for("i2v")`、目录校验、台账取价是同一个键。此前它读 dashscope_i2v27，
        # 于是 yaml 里 `dashscope_i2v: wan2.2-i2v-flash` 会被面板显示、被目录拒（400）、
        # 却根本不会发出去：三份事实各漂一次，排查的人只能猜。
        return dashscope.DashScopeI2V27Provider(
            cfg.dashscope_api_key,
            mm("dashscope_i2v", "wan3.0-video"),
            base_url=cfg.dashscope_video_base_url,
        )
    if name == "dashscope_wan27_audio":
        return dashscope.DashScopeI2V27TalkProvider(
            cfg.dashscope_api_key,
            mm("dashscope_i2v27", "wan2.7-i2v"),
        )
    if name == "dashscope_wan30_talk":
        # C 链：图片 + 提示词（台词在里面）→ 口型+动作+音轨一体。
        # base_url 与 s2v 同一个可覆盖项（FV3_S2V_BASE_URL）：万相视频族同样要求华北2/北京，
        # 业务空间专属端点时提交与轮询必须同端点，否则只会得到"任务不存在"。
        return dashscope.DashScopeWan30TalkProvider(
            cfg.dashscope_api_key,
            _wan30_talk_model(cfg),
            base_url=cfg.dashscope_video_base_url,
        )
    if name == "dashscope_wan27_r2v":
        return dashscope.DashScopeR2VProvider(
            cfg.dashscope_api_key,
            mm("dashscope_r2v", "wan2.7-r2v"),
        )
    if name == "dashscope_cosyvoice":
        return dashscope.DashScopeTTSProvider(
            cfg.dashscope_api_key,
            mm("dashscope_tts", "cosyvoice-v2"),
        )
    if name == "dashscope_vlm":
        return dashscope.DashScopeVLMProvider(
            cfg.dashscope_api_key,
            mm("dashscope_vlm", "qwen-vl-plus"),
        )
    # 非百炼供应商：只认 yaml 的 models:，**不吃 .env 的能力级覆盖** —— 用户设的
    # qwen-image-3.0 / wan3.0-video 在 SiliconFlow/可灵上不存在，漏进 fallback 会让
    # 兜底路径直接 404（fallback 必须永远指向它自己平台的模型）。
    if name == "siliconflow_flux":
        return siliconflow.SiliconFlowT2IProvider(
            cfg.siliconflow_api_key, m.get("siliconflow_t2i", "black-forest-labs/FLUX.1-dev")
        )
    if name == "siliconflow_wan22":
        return siliconflow.SiliconFlowI2VProvider(
            cfg.siliconflow_api_key, m.get("siliconflow_i2v", "Wan-AI/Wan2.2-I2V-A14B")
        )
    if name == "kling_lipsync":
        return kling.KlingTalkProvider(
            cfg.kling_access_key, cfg.kling_secret_key, m.get("kling_talk", "kling-v2-master")
        )
    # fal.ai 托管的 InfiniteTalk 口型通道（本期口型主通道，见 providers/fal.py）。
    # 与上面两个外部平台不同，它**吃** .env 的 FV3_TALK_MODEL：把模型名换成
    # `fal-ai/infinitalk/video-to-video` 时 protocols 会把协议解析成 v2v 并仍留在本
    # provider（providers/protocols.py），所以覆盖项在这里是有效入口而不是污染源。
    # v2v 端点名只认 yaml（防止一次覆盖同时改掉两个端点）。
    if name == "fal_infinitalk":
        return fal.FalInfiniteTalkProvider(
            cfg.fal_api_key,
            mm("fal_talk", fal.IMAGE_ENDPOINT),
            m.get("fal_talk_v2v", fal.V2V_ENDPOINT),
            acceleration=cfg.fal_acceleration,
        )
    if name == "edge_tts":
        return edge_tts.EdgeTTSProvider()
    raise ProviderError(f"未知供应商: {name}")
