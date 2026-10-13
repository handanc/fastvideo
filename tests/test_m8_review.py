"""m8 评审：成片终审容错 + 终审重拍选镜。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m8_review import pick_reshoot_shots, review_final
from fastvideo3.providers.fake import FakeVLMProvider, fake_script


def test_pick_reshoot_shots_hits_keywords_and_never_picks_dialogue():
    """终审重拍只在动作/空镜镜头里挑：对白镜头重拍要重付口型的钱。"""
    script = fake_script(BaseInfo(topic="t"))
    dialogue_ids = {sh.shot_id for sh in script.all_shots() if sh.dialogues}

    hit = pick_reshoot_shots(script, "阳光洒落 那段空镜太长")
    assert hit == ["s1_01"], "命中意见关键词的镜头应被优先选中"

    fallback = pick_reshoot_shots(script, "整体节奏一般")  # 无命中 → 前 limit 个动作/空镜
    assert fallback == ["s1_01", "s2_02"]
    assert not (set(fallback) & dialogue_ids), "对白镜头不该入选（口型重生成本高）"


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
