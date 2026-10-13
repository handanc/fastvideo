"""台账与血缘的"类防"：让这四类静默失效在结构上不可能再发生。

都是真踩出来的，不是假想防御：

1. `job_id` 形如 `scene_s1`，只在单个 run 内唯一，而它是表的主键 —— 第二条 run 用同样的
   镜头号会撞同一行：要么 `IntegrityError`，要么读到**别的 run** 的任务状态，续跑时把
   别人的 task_id 当自己的轮询、把别人的产物下载进本 run。
2. 只调 `jobs.update()` 不调 `submit()` 的模块（`m4_scene` 之前就是这样）台账永远是空的，
   `get()` 恒 None → "跳过已成功任务""在途任务续跑"两条防重复付费的保护全是死代码。
3. `mark_stale` 对一条既不存在的产物、又没有血缘条目的路径会凭空写一条"已作废"，永远挂在
   lineage 里 —— 本项目反复吃过"遗留状态被当成缺口 → 步骤永久 gap"。
"""

from __future__ import annotations

import pytest

from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import JobStatus, RunJob
from fastvideo3.core.storage import RunStorage


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}"


def test_same_job_id_in_two_runs_does_not_collide(db):
    """跨 run 的同名任务互不可见：B 读不到 A 的状态，也不会把 A 的 task_id 当自己的。"""
    a = JobStore(db, run_id="run_A")
    b = JobStore(db, run_id="run_B")
    a.submit(RunJob(
        job_id="scene_s1", run_id="run_A", capability="talk", provider="dashscope_wan30_talk",
        params={}, status=JobStatus.SUCCEEDED, output_uri="ext-A",
    ))

    assert a.get("scene_s1") is not None and a.get("scene_s1").status == JobStatus.SUCCEEDED
    assert b.get("scene_s1") is None, "B 绝不能读到 A 的任务行"

    b.update("scene_s1", status=JobStatus.RUNNING, output_uri="ext-B")
    assert b.get("scene_s1").output_uri == "ext-B"
    assert a.get("scene_s1").output_uri == "ext-A", "B 的写入不能污染 A"


def test_b_second_submit_does_not_raise_integrity_error(db):
    """同一 run 重跑：插行撞主键时降级为更新，不抛异常（重跑不该炸管线）。"""
    a = JobStore(db, run_id="run_A")
    a.submit(RunJob(job_id="i2v_s1_01_0", run_id="run_A", capability="i2v", provider="p",
                    params={"a": 1}, status=JobStatus.SUCCEEDED, output_uri="ext-1"))
    a.submit(RunJob(job_id="i2v_s1_01_0", run_id="run_A", capability="i2v", provider="p",
                    params={"a": 2}))
    row = a.get("i2v_s1_01_0")
    assert row is not None and row.status == JobStatus.SUCCEEDED, "重跑提交不能把成功状态抹掉"
    assert row.params == {"a": 2}


def test_update_without_prior_submit_creates_row(db):
    """只 update 不 submit 也要留下行 —— 这条保护决定"跳过已提交任务"会不会静默失效。"""
    s = JobStore(db, run_id="run_A")
    assert s.get("scene_s2") is None
    s.update("scene_s2", status=JobStatus.RUNNING, output_uri="ext-existing")
    row = s.get("scene_s2")
    assert row is not None, "update 见不到行就插一条：否则续跑保护永远是死代码"
    assert row.status == JobStatus.RUNNING and row.output_uri == "ext-existing"
    assert row.run_id == "run_A"
    assert s.list_open("run_A") and s.list_open("run_A")[0].job_id == "scene_s2"


def test_store_without_run_id_stays_compatible(db):
    """不带 run_id 的旧用法（runtime dispose、老测试）保持裸键，不改变历史行为。"""
    s = JobStore(db)
    s.update("j1", status=JobStatus.PENDING)
    assert s.get("j1") is not None


def test_mark_stale_refuses_to_invent_missing_products(tmp_path):
    """不存在的产物 + 没有血缘条目 = 不作废（否则 lineage 里永远挂着一条幽灵状态）。"""
    storage = RunStorage("run_guard", root=tmp_path)
    n = storage.mark_stale(["clips/scene_scene_1.mp4"], "组名打错造成的不存在的相对路径")
    assert n == 0
    assert storage.stale_set() == set(), f"凭空写出的作废条目：{sorted(storage.stale_set())}"

    # 盘上真有文件的仍然照作标 stale（这是正常语义，不能被上面的守卫顺手关掉）
    live = storage.root / "clips" / "scene_s1.mp4"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_bytes(b"x")
    assert storage.mark_stale(["clips/scene_s1.mp4"], "上游变了") == 1
    assert "clips/scene_s1.mp4" in storage.stale_set()
