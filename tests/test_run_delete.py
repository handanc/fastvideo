"""运行历史里的「删除任务」：连盘上目录一起删，且只在安全的时候删。

删除是这一页唯一的破坏性动作，所以判据全在服务端（前端只是勾选框 + 确认框）：
- 正在生成的任务不许删（子进程还握着这个目录，删到一半会留半截目录 + 一把指向空气的锁）；
- 别的任务持有全局锁时**可以**删自己（锁是 per-run 语义，不能因为它就删不动）；
- 删完必须把锁释放掉 —— 锁文件在 OUTPUT_DIR 层、不在 run 目录里，删目录带不走它；
- run_id 必须过名字校验（路径穿越一律 404）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import tasklock

RUN = "run_20260101_000095"


@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


@pytest.fixture
def run(outdir):
    root, _ = _mk_run(outdir, RUN, with_final=True)
    return root


def test_delete_removes_the_whole_run_directory(client, run, outdir):
    assert run.is_dir() and (run / "clips").is_dir()
    r = client.delete(f"/api/runs/{RUN}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["removed_files"] > 5 and body["freed_mb"] >= 0
    assert not run.exists(), "盘上目录必须一起删掉（用户要的就是这个）"
    listed = client.get("/api/runs").json()
    rows = listed if isinstance(listed, list) else listed["runs"]
    assert RUN not in [x["run_id"] for x in rows], "列表里不该再出现被删掉的任务"


def test_delete_refused_while_that_run_is_generating(client, run, outdir):
    assert tasklock.acquire(outdir, RUN, "step")[0]
    r = client.delete(f"/api/runs/{RUN}")
    assert r.status_code == 409, r.text
    assert "正在生成" in r.text
    assert run.is_dir(), "被拒绝就不许动盘上一个字节"
    tasklock.release(outdir, RUN)


def test_delete_allowed_when_another_run_holds_the_lock(client, run, outdir):
    """全局锁是"一次只跑一个任务"，不是"有人在跑就谁都不许删"。"""
    assert tasklock.acquire(outdir, "run_20260101_000096", "step")[0]
    r = client.delete(f"/api/runs/{RUN}")
    assert r.status_code == 200, r.text
    assert not run.exists()
    # 别人的锁必须还在（release 只删与自己 run_id 相符的那把）
    st = tasklock.status(outdir)
    assert st.get("busy") and st.get("run_id") == "run_20260101_000096", st
    tasklock.release(outdir, "run_20260101_000096")


def test_delete_releases_a_stale_lock_of_that_run(client, run, outdir):
    """锁文件在 OUTPUT_DIR 层：删目录带不走它，不释放就会留下一把指向已消失任务的死锁。"""
    tasklock.acquire(outdir, RUN, "step")
    tasklock.release(outdir, RUN)          # 模拟已结束但锁文件仍在的常见状态
    assert client.delete(f"/api/runs/{RUN}").status_code == 200
    assert not run.exists()
    assert tasklock.status(outdir).get("busy") is not True


def test_delete_unknown_run_is_404(client, outdir):
    assert client.delete("/api/runs/run_20991231_999999").status_code == 404


def test_delete_rejects_path_traversal(client, run, outdir):
    for bad in ("..", "../..", "a/b", "..%2F.."):
        # 405 = 框架层就把它当成别的路由了，压根到不了处理函数；404/400 = 名字校验拦下
        assert client.delete(f"/api/runs/{bad}").status_code in (404, 400, 405), bad
    assert run.is_dir(), "穿越尝试不许碰到任何真实目录"


def test_delete_run_keeps_other_runs_intact(client, run, outdir):
    other, _ = _mk_run(outdir, "run_20260101_000097", with_final=False)
    assert client.delete(f"/api/runs/{RUN}").status_code == 200
    assert other.is_dir() and (other / "script.json").is_file()
    assert json.loads((other / "run_meta.json").read_text(encoding="utf-8"))["run_id"] == (
        "run_20260101_000097"
    )
