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
    return load_config(dotenv_path=tmp_path / "none.env", quality="economy")


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
    """用户红线 ¥5/s 落在装配层：改 .env 切到贵通道会直接报错，而不是静默烧钱。"""
    from fastvideo3.providers.base import Capability, ProviderError
    from fastvideo3.providers.factory import make_provider

    cfg = _cfg(monkeypatch, tmp_path, FV3_TALK_MODEL="wan2.7-i2v")
    cfg.routing["talk"] = ProviderRouting(default="dashscope_wan27_audio")
    assert make_provider(Capability.TALK, cfg).model == "wan2.7-i2v"  # ¥1/s 在红线内
    cfg.max_video_cost_per_sec = 0.4  # 把红线压到 ¥0.4/s
    with pytest.raises(ProviderError, match="超过上限"):
        make_provider(Capability.TALK, cfg)
    # s2v 台账单价按**我们实际发送的 720P 档**记 ¥0.9/s（480P 才是 ¥0.5/s）：
    # 熔断宁可早触发也不能晚触发，所以取高值。红线 ¥0.6/s 时它应被拒。
    cfg.max_video_cost_per_sec = 0.6
    cfg.routing["talk"] = ProviderRouting(default="dashscope_wan22_s2v")
    cfg.model_overrides = {}
    with pytest.raises(ProviderError, match="超过上限"):
        make_provider(Capability.TALK, cfg)
    cfg.max_video_cost_per_sec = 0.95
    assert make_provider(Capability.TALK, cfg).model == "wan2.2-s2v"
    # 兜底通道 videoretalk 只要 ¥0.08/s：红线压到 ¥0.1/s 也应当可用（降级不能被闸门挡住）。
    # 注意必须配上它的模式（v2v_overlay）：模式表会否决"形态不相容"的路由，
    # 在 s2v_direct 下写 videoretalk 不再会被装配出来（这正是防错配的设计）。
    cfg.max_video_cost_per_sec = 0.1
    cfg.talk_mode = "v2v_overlay"
    cfg.routing["talk"] = ProviderRouting(default="dashscope_videoretalk")
    assert make_provider(Capability.TALK, cfg).model == "videoretalk"
    # 同一份路由换个模式：装配的是**模式解析出来的那条通道**（A 链 ¥0.9/s），
    # 于是 ¥0.1/s 的红线把它拒了 —— 拒的依据不再是路由里写的名字，防的就是"看着便宜
    # 其实贵"与"模式与路由错配"这两类静默失效。
    cfg.talk_mode = "s2v_direct"
    with pytest.raises(ProviderError, match="超过上限"):
        make_provider(Capability.TALK, cfg)


def test_talk_channel_per_quality_and_pairing(monkeypatch, tmp_path):
    """口型通道**按档位定档**（2026-10-11 四次定档），且通道与 talk_mode 必须成对。

    A 链 = `economy` / `high`：主 = 百炼 `wan2.2-s2v`（官方 wan-s2v-api），兜底 = fal InfiniteTalk
    - wan2.2-s2v：单张图片 + 音频 → 说话/唱歌/表演视频；480P ¥0.5/秒、720P ¥0.9/秒；
      新账号 100 秒免费额度；失败不计费；并发 1。
    - 与 talk_mode=s2v_direct **必须成对**（s2v 不接受视频输入；provider 现在会在
      收到 video 时直接 CapabilityMismatchError，不再静默丢掉 m4 花钱出的运动片段）。
    - 兜底选 fal_infinitalk 的图像端点：与主通道**同为"图片+音频"**输入形态，所以在
      s2v_direct 下真的能用；videoretalk 只吃视频，在这个 mode 下等于没有兜底。
      fal 需要 .env 的 FAL_KEY，没填时兜底为空 → 降级叠音并落 degradations.json（可见）。
    - **为什么不升级成 B 链**：economy 的 i2v 是 480P（短边 480 < videoretalk 官方下限
      640px）→ 提交必被服务端拒；high 的片段虽然够大，但 A 链的对白镜头**根本不过 i2v**，
      省下一整份运动费 —— 教学档与常规交付档优先保成本与出片率。

    B 链 = `cinema`：m4 的 1080P 运动片段 + 对白音频 → `videoretalk` 重对口型（`v2v_overlay`）
    - retalk **不重画整段画面**，只重绘嘴部 → m3 参考图锁住的脸一路继承；
      官方规格「输出分辨率与帧率=输入」→ 1080P 对白镜头不再被钳到 720P
      （A 链在 cinema 的结构性代价：`resolution_clamped` 每个对白镜头都吃一次）。
    - ¥0.08/秒 + 新账号 1800 秒免费额度；音频上限 120s → 不需要 s2v 那套 <20s 分段拼接。
    """
    from fastvideo3.providers.dashscope import DashScopeRetalkProvider, DashScopeTalkProvider
    from fastvideo3.providers.fal import FalInfiniteTalkProvider

    for quality in ("economy", "high"):
        # _cfg 的副作用才是重点：清掉项目 .env 可能带进来的模型覆盖 + 注入测试 KEY
        _cfg(monkeypatch, tmp_path)
        cfg2 = load_config(dotenv_path=tmp_path / "none.env", quality=quality)
        assert cfg2.routing["talk"].default == "dashscope_wan22_s2v", quality
        assert cfg2.routing["talk"].fallback == "fal_infinitalk", quality
        assert cfg2.talk_mode == "s2v_direct", quality  # 路由与 mode 成对
        p = make_provider(Capability.TALK, cfg2)
        assert isinstance(p, DashScopeTalkProvider) and p.model == "wan2.2-s2v", quality
        assert p.supports_video_input is False and p.max_resolution == "720P"
        fb = make_fallback(Capability.TALK, cfg2)
        assert isinstance(fb, FalInfiniteTalkProvider), quality

    # cinema：B 链要装配成 videoretalk（吃视频、输出规格=输入、¥0.08/秒不分档）
    _cfg(monkeypatch, tmp_path)
    cfgc = load_config(dotenv_path=tmp_path / "none.env", quality="cinema")
    assert cfgc.routing["talk"].default == "dashscope_videoretalk"
    assert cfgc.talk_mode == "v2v_overlay", "B 链必须与 v2v_overlay 成对，否则拿不到运动片段"
    pc = make_provider(Capability.TALK, cfgc)
    assert isinstance(pc, DashScopeRetalkProvider) and pc.model == "videoretalk"
    assert pc.supports_video_input is True and pc.max_audio_sec == 120.0

    # 兜底的 FAL_KEY 缺失不影响主通道：s2v 照样装配成功（用户只需原有 DASHSCOPE KEY）
    monkeypatch.delenv("FAL_KEY", raising=False)
    cfg3 = load_config(dotenv_path=tmp_path / "none.env", quality="high")
    assert isinstance(make_provider(Capability.TALK, cfg3), DashScopeTalkProvider)
    assert make_fallback(Capability.TALK, cfg3) is None  # 没 KEY 就没兜底（降级会记账，不静默）


def test_v2v_overlay_needs_i2v_for_dialogue_shots(monkeypatch, tmp_path):
    """v2v_overlay 下对白镜头必须先有运动片段：这是"在已有片段上重对口型"的前置依赖。

    cinema 出厂即 `v2v_overlay`（B 链：videoretalk 吃 m4 的运动片段），所以这条前置在
    出厂配置下**必须自动成立** —— 本用例同时钉住两件事：
    1. 出厂 mode 与出厂 talk 路由成对（错配的真实代价见 run_20261011_164152：对白镜头
       先烧一份 i2v 的钱、再 100% 降级成无口型，"生成镜头视频"步骤因缺口永远不绿）；
    2. 显式设成 v2v_overlay 时，对白镜头必须进 m4；设成 s2v_direct 时必须**不进**（省那份钱）。
    """
    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.modules.m4_shots import shots_need_i2v
    from fastvideo3.providers.fake import fake_script

    cfg = load_config(dotenv_path=tmp_path / "none.env", quality="cinema")
    assert cfg.talk_mode == "v2v_overlay", "cinema 出厂即 B 链，mode 必须与 videoretalk 成对"
    assert cfg.routing["talk"].default == "dashscope_videoretalk", "mode 与路由一起改才算成对"
    script = fake_script(BaseInfo(topic="t"))
    dialogue = [s for s in script.all_shots() if s.type.value == "dialogue"]
    need = {s.shot_id for s in shots_need_i2v(script, cfg.talk_mode)}
    assert dialogue, "用例前提：fake 剧本要有对白镜头"
    assert all(s.shot_id in need for s in dialogue), "v2v_overlay 时必须为对白镜头出 i2v 片段"
    # 反证：s2v_direct 模式对白镜头不出 i2v（由 s2v 直出），两种模式不能混
    direct = {s.shot_id for s in shots_need_i2v(script, "s2v_direct")}
    assert not (set(s.shot_id for s in dialogue) & direct)
