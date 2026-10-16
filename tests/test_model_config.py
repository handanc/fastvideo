""".env 模型选择 + 协议自动改道的回归测试。

守住的病灶：模型名与调用协议过去绑死在 provider 键上 —— 在 .env 里把 t2i 换成
qwen-image-3.0 而路由仍是万相异步端点时，请求必然 400；且 .env 覆盖若漏进
fallback，会让兜底路径指向它平台上不存在的模型。
"""

from __future__ import annotations

import pytest

from fastvideo3.core.config import ProviderRouting, load_config
from fastvideo3.core.cost import price_of
from fastvideo3.providers import protocols
from fastvideo3.providers.base import Capability
from fastvideo3.providers.factory import make_fallback, make_provider

KEYS = {"DASHSCOPE_API_KEY": "sk-test", "SILICONFLOW_API_KEY": "sk-sf", "FAL_KEY": "fal-test"}
_MODEL_ENVS = (
    "FV3_TEXT_MODEL", "FV3_T2I_MODEL", "FV3_I2V_MODEL", "FV3_TALK_MODEL", "FV3_R2V_MODEL",
    "FV3_VLM_MODEL", "FV3_TTS_MODEL", "FV3_T2I_PROTOCOL", "FV3_I2V_PROTOCOL",
)


def _cfg(monkeypatch, tmp_path, **env):
    """干净环境：先清掉项目 .env 可能带进来的模型覆盖，再按用例注入。"""
    for k in (*KEYS, *_MODEL_ENVS):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**KEYS, **env}.items():
        monkeypatch.setenv(k, v)
    return load_config(dotenv_path=tmp_path / "none.env", resolution="480P")


def test_env_model_override_beats_yaml(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, FV3_T2I_MODEL="qwen-image-3.0")
    assert cfg.model_for("t2i", "dashscope_t2i", "wanx2.1-t2i-turbo") == "qwen-image-3.0"
    # 未覆盖的能力仍取 yaml
    assert cfg.model_for("vlm", "dashscope_vlm", "qwen-vl-plus") == "qwen-vl-plus"


def test_blank_env_does_not_clobber_yaml(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, FV3_T2I_MODEL="")
    assert cfg.model_for("t2i", "dashscope_t2i", "wanx2.1-t2i-turbo") == "wanx2.1-t2i-turbo"


def test_t2i_trial_model_reroutes_to_multimodal_provider(monkeypatch, tmp_path):
    """qwen-image-3.0 走 multimodal-generation，而不是万相异步 text2image 端点。"""
    from fastvideo3.providers.dashscope import DashScopeWan27ImageProvider

    cfg = _cfg(monkeypatch, tmp_path, FV3_T2I_MODEL="qwen-image-3.0")
    p = make_provider(Capability.T2I, cfg)
    assert isinstance(p, DashScopeWan27ImageProvider)
    assert p.model == "qwen-image-3.0"
    assert p.supports_reference_image is True  # 锁脸能力随协议一起到位


def test_i2v_trial_model_reroutes_to_media_array_provider(monkeypatch, tmp_path):
    from fastvideo3.providers.dashscope import DashScopeI2V27Provider

    cfg = _cfg(monkeypatch, tmp_path, FV3_I2V_MODEL="wan3.0-video")
    p = make_provider(Capability.I2V, cfg)
    assert isinstance(p, DashScopeI2V27Provider) and p.model == "wan3.0-video"


def test_talk_trial_model_routes_to_its_real_protocol(monkeypatch, tmp_path):
    """.env 写 `FV3_TALK_MODEL=<模型名>` 时，provider 必须改道到**该模型真正的口型协议**。

    wan3.0-video 与 wan2.7-i2v 都叫"视频大模型做口型"，协议却相反：
    - `wan3.0-video`：media 枚举里**没有 driving_audio** → 台词写进提示词、模型自发声
      （`TALK_SCRIPTED` → `dashscope_wan30_talk`）。曾按 2.7 那套发音频 → 每个对白镜头
      一次真实 400（原文存在 tests/test_talk_fallback.py）。
    - `wan2.7-i2v`：确实有 `media[].type=driving_audio` → 音频驱动（`TALK_MEDIA_AUDIO`）。
    显式覆盖优先于模式表（见 factory._talk_route 的例外分支）：那是专家级的手动换通道，
    喂错输入时通道自己会响亮拒绝，不是静默降级。
    """
    from fastvideo3.providers.dashscope import (
        DashScopeI2V27TalkProvider,
        DashScopeWan30TalkProvider,
    )

    cfg = _cfg(monkeypatch, tmp_path, FV3_TALK_MODEL="wan3.0-video")
    p = make_provider(Capability.TALK, cfg)
    assert isinstance(p, DashScopeWan30TalkProvider) and p.model == "wan3.0-video"
    assert p.accepts_audio is False and p.supports_video_input is False

    cfg2 = _cfg(monkeypatch, tmp_path, FV3_TALK_MODEL="wan2.7-i2v")
    p2 = make_provider(Capability.TALK, cfg2)
    assert isinstance(p2, DashScopeI2V27TalkProvider) and p2.model == "wan2.7-i2v"


def test_fallback_is_the_paid_yaml_model_not_the_trial_one(monkeypatch, tmp_path):
    """主路径是未开通的试用模型时，兜底必须是 yaml 指向的付费模型（否则跑不动）。"""
    from fastvideo3.providers.dashscope import DashScopeT2IProvider

    cfg = _cfg(monkeypatch, tmp_path, FV3_T2I_MODEL="qwen-image-3.0")
    fb = make_fallback(Capability.T2I, cfg)
    assert isinstance(fb, DashScopeT2IProvider)
    assert fb.model == "wanx2.1-t2i-turbo"  # 不吃 .env 覆盖


def test_third_party_fallback_never_eats_dashscope_override(monkeypatch, tmp_path):
    """SiliconFlow/可灵 上没有 qwen-image-3.0，覆盖漏进去会让兜底 404。"""
    from fastvideo3.providers.siliconflow import SiliconFlowT2IProvider

    cfg = _cfg(monkeypatch, tmp_path, FV3_T2I_MODEL="qwen-image-3.0")
    cfg.routing["t2i"] = ProviderRouting(default="dashscope_t2i", fallback="siliconflow_flux")
    from fastvideo3.providers import factory

    p = factory._make_real("siliconflow_flux", Capability.T2I, cfg, overrides={})
    assert isinstance(p, SiliconFlowT2IProvider)
    assert p.model == "black-forest-labs/FLUX.1-dev"


def test_r2v_model_not_clobbered_by_i2v_override(monkeypatch, tmp_path):
    from fastvideo3.providers.dashscope import DashScopeR2VProvider
    from fastvideo3.providers.factory import make_r2v_provider

    cfg = _cfg(monkeypatch, tmp_path, FV3_I2V_MODEL="wan3.0-video")
    cfg.routing["r2v"] = ProviderRouting(default="dashscope_wan27_r2v")
    p = make_r2v_provider(cfg)
    assert isinstance(p, DashScopeR2VProvider)
    assert p.model == "wan2.7-r2v"  # 不是 wan3.0-video


def test_protocol_resolution_priority(tmp_path):
    # 内置命名推断
    assert protocols.infer_protocol("qwen-image-3.0", "t2i") == protocols.T2I_MULTIMODAL
    assert protocols.infer_protocol("wan3.0-video", "i2v") == protocols.I2V_MEDIA
    # 环境变量显式覆盖 > 推断
    assert (
        protocols.infer_protocol("mystery-model", "t2i", env={"FV3_T2I_PROTOCOL": "async_text2image"})
        == protocols.T2I_ASYNC
    )
    # 探测结论文件 > 环境变量
    probe = {"t2i/mystery-model": "multimodal_sync"}
    assert (
        protocols.infer_protocol(
            "mystery-model", "t2i", env={"FV3_T2I_PROTOCOL": "async_text2image"}, probe=probe
        )
        == protocols.T2I_MULTIMODAL
    )
    # 未知 → None（保持原路由，不猜）
    assert protocols.infer_protocol("totally-unknown", "t2i", env={}, probe={}) is None


def test_free_quota_placeholders_are_gone_and_real_prices_are_in():
    """旧表把额度期模型记成 ¥0，额度用尽后账单照收而台账仍记 ¥0 —— 那些占位必须清掉。

    这条断言原来是反过来的（"免费额度模型必须显式列 0.0"）：它把 ¥6.30 那种失真
    当成了正确行为。2026-10-12 官网价目页回填后，口径改成：
    **单价一律记官网原价**（限时折扣随时会停，按折扣记会让熔断晚跳 = 超支），
    ¥0 只允许出现在显式标 `free = true` 的真免费端点上。
    """
    assert price_of("qwen-image-3.0", "t2i") == 0.18, "输出图 0.18 元/张"
    assert price_of("wan3.0-video@720p", "i2v", 5.0) == 3.0, "720P 0.6 元/秒 × 5 秒"
    assert price_of("wan3.0-video@1080p", "i2v", 5.0) == 6.0, "1080P 是 720P 的两倍"
    assert price_of("wan3.0-video-prime@1080p", "talk", 5.0) == 9.0
    # 文本按 token：没有用量就记 0（不许编），有用量就按输入/输出两个价算
    from fastvideo3.core import pricing

    flash = pricing.resolve("qwen3.7-flash", "text")
    assert pricing.amount_of(flash, tokens_in=1_000_000, tokens_out=1_000_000) == 1.0
    assert pricing.amount_of(flash, quantity=1) == 0.0, "没拿到 usage 就记 0，不猜"
    # 真免费的端点：¥0 且**显式声明**，所以算确定价
    zero = pricing.resolve("edge-tts", "tts")
    assert zero.price_cny == 0.0 and zero.certain is True
    # 未登记的模型仍回退默认价（且标明不可信）
    unknown = pricing.resolve("some-unlisted-model", "t2i")
    assert unknown.price_cny > 0 and unknown.certain is False


def test_video_cost_cap_rejects_overpriced_channel(monkeypatch, tmp_path):
    """¥/秒 上限拦的是**实际会被装配出来的那条通道的最贵一档**，报错还要给出出路。

    口型通道只剩 wan3.0（场景片段自带台词口型），所以今天被这条红线拦下来的一定是
    `wan3.0-video@1080p`（¥1.2/秒）或 `wan3.0-video-prime@1080p`（¥1.8/秒）。
    闸门取 `worst_case_price_key()` 而不是"这次实际会发的档位"：用户在建任务之后
    还能把分辨率调到 1080P，而装配期闸门只过一次 —— 按低档校验等于红线形同虚设。

    拦下时必须把**拦的价、上限、以及改哪个变量**写进报错：只说"超了"的报错会让用户
    去翻源码才知道 `FV3_MAX_VIDEO_COST_PER_SEC` 才是那个旋钮。
    """
    from fastvideo3.core.cost import price_of
    from fastvideo3.providers.base import Capability, ProviderError
    from fastvideo3.providers.factory import make_provider

    cfg = _cfg(monkeypatch, tmp_path)
    p = make_provider(Capability.TALK, cfg)
    assert p.name == "dashscope_wan30_talk" and p.model == "wan3.0-video"
    assert p.worst_case_price_key() == "wan3.0-video@1080p"
    assert price_of(p.worst_case_price_key(), "talk", 1.0) == 1.2

    cfg.max_video_cost_per_sec = 1.1  # 红线压到所选通道的最高档之下
    with pytest.raises(ProviderError) as e:
        make_provider(Capability.TALK, cfg)
    msg = str(e.value)
    assert "超过上限" in msg and "wan3.0-video@1080p" in msg, msg
    assert "¥1.2/秒" in msg and "¥1.1/秒" in msg, "拦住的价与上限都要写出来：" + msg
    assert "FV3_MAX_VIDEO_COST_PER_SEC" in msg, "必须给出可行的出路，不许只报超限"

    # 上限抬到最高档单价之上就放行（同一份配置只改红线这一个数）
    cfg.max_video_cost_per_sec = 1.2
    assert make_provider(Capability.TALK, cfg).model == "wan3.0-video"
    # 贵一档的 prime（¥1.8/秒）在同一条红线上必须被拒：闸门跟的是**实际选中的模型**，
    # 不是通道名 —— 否则"换优速版"在成本上看起来是免费的。
    # 口型模型跟随面板选的 i2v（同族才跟），所以这里改 i2v 就等于改口型单价。
    cfg.model_overrides = {"i2v": "wan3.0-video-prime"}
    with pytest.raises(ProviderError, match="超过上限"):
        make_provider(Capability.TALK, cfg)
    cfg.max_video_cost_per_sec = 1.8
    assert make_provider(Capability.TALK, cfg).model == "wan3.0-video-prime"
