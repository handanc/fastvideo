"""阶段执行器的装配快照测试：接线不许丢。

为什么要有这个文件：一致性关键装配藏在 `core/stages.py` 的参数里 —— 立绘抽卡的 vlm、
首帧蒸馏的 text、r2v 参考生视频、`require_reference_keyframes` 硬门控、以及三处
`log_degradation`。搬动或改写执行器时**丢任何一处都是静默质量退化**（正是历史上
"wan2.7-image 零计费、锁脸形同虚设"的病灶）。

出片只有一条路径（`cli step` / API `/step` → 本执行器），所以这里锁住的装配
就是用户实际拿到的装配。
"""

from __future__ import annotations

import pytest
from PIL import Image

from fastvideo3.core.schema import Asset, BaseInfo
from fastvideo3.core.stages import (
    build_context,
    stage_assets,
    stage_shots,
    stage_storyboard,
)
from fastvideo3.providers.base import Capability, ProviderError
from fastvideo3.providers.fake import FakeT2IProvider, fake_script


@pytest.fixture
def ctx(cfg, storage, jobs):
    return build_context(cfg, storage)


def _seed_script(storage):
    script = fake_script(BaseInfo(topic="装配快照"))
    storage.save_json("script.json", script.model_dump(mode="json"))
    return script


def _char_assets(storage, script) -> dict[str, Asset]:
    """按剧本造一份定妆照（盘上真有文件，m3 才肯把它当参考图挂上去）。"""
    out: dict[str, Asset] = {}
    for c in script.characters:
        uri = storage.path("assets", f"char_{c.character_id}.png")
        Image.new("RGB", (8, 8), (9, 9, 9)).save(uri)
        out[f"char_{c.character_id}"] = Asset(
            asset_id=f"char_{c.character_id}",
            kind="character",
            owner_id=c.character_id,
            uri=str(uri),
        )
    return out


async def test_assets_wire_vlm_for_best_of_n(cfg, ctx, storage):
    """立绘抽卡必须把 VLM 传下去，否则 best-of-N 选优与写实闸门整体失效。"""
    cfg.asset_candidates = 2
    script = _seed_script(storage)
    out = await stage_assets(ctx, script)
    assert out["assets"]
    # 抽卡事件存在 = vlm 真被调用（vlm=None 时 n_cand 恒为 1，不会有这条事件）
    assert any(e["status"] == "char_best_of_n" for e in storage.read_events()), "立绘抽卡未执行"
    qc_notes = [e.note for e in ctx.ledger.entries if e.capability == "vlm"]
    assert any(n.startswith("char_qc:") for n in qc_notes), "VLM 评分未记账 → vlm 没接上"


async def test_storyboard_hard_gate_raises_when_provider_cannot_ref(cfg, ctx, storage):
    """require_reference_keyframes + 不支持参考图的 provider → 花钱前必须抛错。"""

    class NoRefT2I(FakeT2IProvider):
        supports_reference_image = False

    cfg.use_reference_keyframes = True
    cfg.require_reference_keyframes = True
    script = _seed_script(storage)
    ctx.providers[Capability.T2I.value] = NoRefT2I()
    with pytest.raises(ProviderError, match="require_reference_keyframes"):
        await stage_storyboard(ctx, script, _char_assets(storage, script))


async def test_storyboard_wires_text_distillation(cfg, ctx, storage):
    """storyboard 必须接 TEXT 做蒸馏，否则元提示词全文直塞 t2i。"""
    script = _seed_script(storage)
    out = await stage_storyboard(ctx, script, _char_assets(storage, script))
    assert out["frames"]
    notes = [e.note for e in ctx.ledger.entries if e.capability == "text"]
    assert any(n.startswith("distill:") for n in notes), "蒸馏未接线（text provider 没传）"


async def test_shots_wires_r2v_and_records_route(cfg, ctx, storage):
    """shots 必须接 r2v + assets，否则 cinema 的参考生视频引擎整条关闭。"""
    cfg.use_r2v_for_characters = True
    from fastvideo3.core.config import ProviderRouting

    cfg.routing["r2v"] = ProviderRouting(default="dashscope_wan27_r2v")
    script = _seed_script(storage)
    assets = _char_assets(storage, script)
    frames = (await stage_storyboard(ctx, script, assets))["frames"]
    out = await stage_shots(ctx, script, frames, assets)
    assert out["clips"]
    routes = {e.get("route") for e in storage.read_events() if e.get("status") == "shot_gen"}
    assert "r2v" in routes, "r2v 通道未接通（r2v/assets 没传 → 一致性引擎静默关闭）"


async def test_degradations_are_loud_not_silent(cfg, storage, jobs, monkeypatch):
    """VLM 装配失败必须落降级台账，不能静默变 None（否则"评审 0 条"无人察觉）。"""
    from fastvideo3.providers.factory import make_provider as real_make

    def boom(capability, cfg_, work_dir=None):
        if capability is Capability.VLM:
            raise ProviderError("模拟未开通/欠费")
        return real_make(capability, cfg_, work_dir)

    monkeypatch.setattr("fastvideo3.core.stages.make_provider", boom)
    cfg.asset_candidates = 2
    ctx = build_context(cfg, storage)
    script = _seed_script(storage)
    await stage_assets(ctx, script)
    assert storage.read_degradations(), "VLM 不可用被静默吞掉 → 降级台账为空"
