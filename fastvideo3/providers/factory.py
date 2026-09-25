"""Provider 工厂：按配置路由装配实例；Fake 模式整体替换（TDD §6.1）。"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.schema import BaseInfo
from fastvideo3.providers.base import Capability, Provider, ProviderError


def make_provider(capability: Capability, cfg: AppConfig, work_dir: Path | None = None) -> Provider:
    """capability → 具体 Provider 实例。

    Fake 模式下返回对应 Fake 实现（零成本离线）；
    真实模式按 configs 路由选择默认供应商。
    """
    if cfg.fake:
        return _make_fake(capability, cfg, work_dir)

    route = cfg.routing.get(capability.value)
    if route is None:
        raise ProviderError(f"配置缺少 {capability.value} 的供应商路由")
    return _make_real(route.default, capability, cfg)


def make_r2v_provider(cfg: AppConfig) -> Provider | None:
    """角色一致性参考生视频供应商（cinema 档配置 r2v 路由时启用）。"""
    if cfg.fake:
        return _make_fake(Capability.I2V, cfg, None)
    route = cfg.routing.get("r2v")
    if route is None:
        return None
    try:
        return _make_real(route.default, Capability.I2V, cfg)
    except ProviderError:
        return None


def make_fallback(
    capability: Capability, cfg: AppConfig, work_dir: Path | None = None
) -> Provider | None:
    route = cfg.routing.get(capability.value)
    if cfg.fake or route is None or not route.fallback:
        return None
    try:
        return _make_real(route.fallback, capability, cfg)
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


def _make_real(name: str, capability: Capability, cfg: AppConfig) -> Provider:
    from fastvideo3.providers import dashscope, deepseek, edge_tts, kling, siliconflow

    m = cfg.models
    if name == "deepseek":
        return deepseek.DeepSeekTextProvider(
            cfg.deepseek_api_key, cfg.deepseek_base_url, m.get("deepseek_text", "deepseek-chat")
        )
    if name == "dashscope_qwen":
        return dashscope.DashScopeTextProvider(
            cfg.dashscope_api_key,
            m.get("dashscope_text", "qwen3.7-plus"),
            base_url=cfg.dashscope_text_base_url,
            enable_thinking=cfg.enable_thinking,
        )
    if name == "dashscope_t2i":
        return dashscope.DashScopeT2IProvider(
            cfg.dashscope_api_key, m.get("dashscope_t2i", "wanx2.1-t2i-turbo")
        )
    if name == "dashscope_wan27_image":
        return dashscope.DashScopeWan27ImageProvider(
            cfg.dashscope_api_key, m.get("dashscope_t2i27", "wan2.7-image")
        )
    if name == "dashscope_wan22_i2v":
        return dashscope.DashScopeI2VProvider(
            cfg.dashscope_api_key, m.get("dashscope_i2v", "wan2.2-i2v-flash")
        )
    if name == "dashscope_wan22_s2v":
        return dashscope.DashScopeTalkProvider(
            cfg.dashscope_api_key, m.get("dashscope_s2v", "wan2.2-s2v")
        )
    if name == "dashscope_videoretalk":
        return dashscope.DashScopeRetalkProvider(
            cfg.dashscope_api_key, m.get("dashscope_retalk", "videoretalk")
        )
    if name == "dashscope_wan27_i2v":
        return dashscope.DashScopeI2V27Provider(
            cfg.dashscope_api_key, m.get("dashscope_i2v27", "wan2.7-i2v")
        )
    if name == "dashscope_wan27_audio":
        return dashscope.DashScopeI2V27TalkProvider(
            cfg.dashscope_api_key, m.get("dashscope_i2v27", "wan2.7-i2v")
        )
    if name == "dashscope_wan27_r2v":
        return dashscope.DashScopeR2VProvider(
            cfg.dashscope_api_key, m.get("dashscope_r2v", "wan2.7-r2v")
        )
    if name == "dashscope_cosyvoice":
        return dashscope.DashScopeTTSProvider(
            cfg.dashscope_api_key, m.get("dashscope_tts", "cosyvoice-v2")
        )
    if name == "dashscope_vlm":
        return dashscope.DashScopeVLMProvider(
            cfg.dashscope_api_key, m.get("dashscope_vlm", "qwen-vl-plus")
        )
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
    if name == "edge_tts":
        return edge_tts.EdgeTTSProvider()
    raise ProviderError(f"未知供应商: {name}")
