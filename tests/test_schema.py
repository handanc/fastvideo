"""数据契约测试（Schema-first 的守门员）。"""

from fastvideo3.core.schema import (
    BaseInfo,
    JobStatus,
    ProjectConfig,
    RunJob,
    Script,
    Shot,
    ShotType,
    TransitionType,
)
from fastvideo3.providers.fake import fake_script


def test_fake_script_is_schema_valid():
    script = fake_script(BaseInfo(topic="测试"))
    assert isinstance(script, Script)
    assert script.shot_count() == 6
    types = {s.type for s in script.all_shots()}
    assert types == {ShotType.DIALOGUE, ShotType.ACTION, ShotType.SCENERY}


def test_script_roundtrip_json():
    script = fake_script(BaseInfo(topic="测试"))
    clone = Script.model_validate_json(script.model_dump_json())
    assert clone.shot_count() == script.shot_count()
    assert clone.character_map()["c1"].name == "小明"


def test_shot_defaults():
    shot = Shot(shot_id="x", scene_id="s", type=ShotType.ACTION, action="跑")
    assert shot.duration_sec == 5.0
    assert shot.transition_out is TransitionType.CUT


def test_project_config_defaults():
    pc = ProjectConfig(run_id="r1", base_info=BaseInfo(topic="t"))
    assert pc.profile == "edu-sqlite"
    # 口型不再是选项：wan3.0 场景片段自带台词口型，音轨恒为模型原声。
    # 默认值散落多处=静默分歧，所以这里钉住 schema 这一份。
    assert pc.talk_mode == "scripted_talk"
    assert pc.voice_source == "model"
    assert pc.ignored_legacy_fields == []
    assert pc.budget_limit_cny == 150.0


def test_aspect_ratio_validation():
    import pytest
    from pydantic import ValidationError

    BaseInfo(topic="t", aspect_ratio="9:16")
    with pytest.raises(ValidationError):
        BaseInfo(topic="t", aspect_ratio="4:3")


def test_runjob_status_enum():
    job = RunJob(job_id="j1", capability="i2v", provider="p")
    assert job.status is JobStatus.PENDING
