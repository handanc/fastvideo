"""m8 评审：文本评审解析 + 成片终审容错。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m8_review import review_final, review_script
from fastvideo3.providers.fake import FakeTextProvider, FakeVLMProvider, fake_script


async def test_review_script_pass(ledger):
    script = fake_script(BaseInfo(topic="t"))
    result = await review_script(FakeTextProvider(), script, ledger)
    assert result.passed


async def test_review_final_with_video(cfg, storage, ledger, tmp_path):
    from fastvideo3.utils.ffmpeg import image_to_video
    from fastvideo3.utils.image import make_placeholder_image

    img = tmp_path / "f.png"
    make_placeholder_image(img)
    video = tmp_path / "v.mp4"
    image_to_video(img, video, duration=2.0)

    script = fake_script(BaseInfo(topic="t"))
    result = await review_final(FakeVLMProvider(), Path(video), script, storage, ledger)
    assert result.passed
    assert storage.exists("final_review.json")


async def test_review_final_error_tolerated(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))

    class BrokenVLM(FakeVLMProvider):
        async def review(self, media, *, rubric, dest=None):
            raise RuntimeError("boom")

    result = await review_final(BrokenVLM(), Path("不存在.mp4"), script, storage, ledger)
    assert result.passed  # 评审异常 → 放行并告警
