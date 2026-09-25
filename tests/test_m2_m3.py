"""m2 资产库 + m3 分镜：锚定生成与失败重试。"""

import pytest

from fastvideo3.core.schema import Asset, BaseInfo
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.providers.base import ProviderError
from fastvideo3.providers.fake import FakeT2IProvider, FakeVLMProvider, fake_script


async def test_build_assets_registers_all(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    # 2 角色 + 3 场景
    assert len(assets) == 5
    assert all(a.uri and Asset(**a.model_dump()).uri for a in assets.values())
    assert storage.exists("asset_registry.json")


async def test_asset_retry_then_success(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider(fail_first=1)  # 首个资产失败一次 → 重试成功
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    assert len(assets) == 5


async def test_asset_failure_blocks(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider(fail_first=999)  # 永远失败 → 阻断（资产不可降级）
    with pytest.raises(ProviderError):
        await build_assets(cfg, t2i, script, storage, ledger)


async def test_storyboard_frames_for_all_shots(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(
        cfg, t2i, script, assets, storage, ledger, vlm=FakeVLMProvider()
    )
    assert len(frames) == script.shot_count()
    for shot in script.all_shots():
        assert frames[shot.shot_id].endswith(".png")
        assert shot.visual_prompt  # 提示词已回写


async def test_storyboard_regen_on_vlm_reject(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(
        cfg, t2i, script, assets, storage, ledger, vlm=FakeVLMProvider(always_pass=False)
    )
    assert len(frames) == script.shot_count()  # 驳回后重生成仍交付
