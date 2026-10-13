"""Web API（FastAPI）离线单测：TestClient + 伪 _spawn，不起真实子进程、不联网。"""

from __future__ import annotations

import json
import os
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fastvideo3.app.api as api_mod


@pytest.fixture
def client(outdir, monkeypatch):
    """TestClient + 隔离输出目录 + 记录 _spawn 调用（替代真实子进程）。"""
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[attr-defined]
        yield c


def _make_run(
    outdir,
    run_id: str = "run_20260101_000000",
    *,
    topic: str = "测试主题",
    done: bool = True,
    cost: list | None = None,
    age_sec: int = 0,
):
    """手工搭建一个符合目录契约的 run（不经过 CLI）。"""
    root = outdir / run_id
    for stage in ("assets", "storyboard", "clips", "talk", "final"):
        (root / stage).mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "base_info": {
            "topic": topic,
            "style": "都市轻喜剧",
            "target_duration_sec": 30,
            "aspect_ratio": "9:16",
        },
        "fake": True,
        "quality": "economy",
    }
    (root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (root / "events.jsonl").write_text(
        json.dumps({"ts": "2026-01-01 00:00:00", "node": "m1", "status": "done"}) + "\n",
        encoding="utf-8",
    )
    if cost is not None:
        (root / "cost.json").write_text(json.dumps(cost), encoding="utf-8")
    if done:
        (root / "final" / "final.mp4").write_bytes(b"fake-mp4")
    if age_sec:  # 回拨 mtime，让启发式状态判定不再视为"运行中"
        old = time.time() - age_sec
        for p in root.rglob("*"):
            os.utime(p, (old, old))
    return root


# ───────────────────────── 基础 ─────────────────────────
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["app"] == "fastvideo3"


def test_index_reachable(client):
    """/ 要么托管构建后的前端，要么返回构建提示 —— 都应 200。"""
    assert client.get("/").status_code == 200


# ───────────────────────── 新建运行 ─────────────────────────
def test_runs_empty(client):
    assert client.get("/api/runs").json() == []


def test_create_run_only_writes_meta_and_never_spawns(client, outdir):
    """新建任务只落 run_meta.json：不启动任何生成，也不占全局任务锁（建任务不花钱）。"""
    r = client.post("/api/runs", json={"topic": "校园喜剧", "duration": 45, "fake": True})
    assert r.status_code == 201
    run_id = r.json()["run_id"]
    assert run_id.startswith("run_")
    assert client.spawned == [], "建任务不该 spawn 任何子进程（生成由 /step 一步一步来）"
    # run_meta.json 先行落盘 → 运行列表立即可见
    meta = json.loads((outdir / run_id / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["base_info"]["topic"] == "校园喜剧"
    assert meta["base_info"]["target_duration_sec"] == 45
    assert meta["fake"] is True


def test_create_run_defaults_to_real_mode(client, outdir):
    """不带 fake 字段的请求必须默认真实生成（此前默认 True，Web 运行静默走占位演示）。"""
    from fastvideo3.app.api import RunCreate

    assert RunCreate(topic="x").fake is False
    r = client.post("/api/runs", json={"topic": "默认值验证"})
    assert r.status_code == 201
    run_id = r.json()["run_id"]
    meta = json.loads((outdir / run_id / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["fake"] is False
    # budget=0 → meta 落 .env 实际生效值（落 0.0 会让看板熔断显示 ¥0、第一步就跳闸）
    assert meta["budget_limit_cny"] > 0


def test_create_run_validation(client):
    assert client.post("/api/runs", json={}).status_code == 422  # 缺 topic
    bad = client.post("/api/runs", json={"topic": "x", "aspect": "4:3"})  # 非法画幅
    assert bad.status_code == 422


# ───────────────────────── 列表与状态启发式 ─────────────────────────
def test_list_statuses(client, outdir):
    _make_run(outdir, "run_done", topic="完成剧")
    _make_run(outdir, "run_idle", topic="停摆剧", done=False, age_sec=600)
    runs = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert runs["run_done"]["status"] == "done"
    assert runs["run_done"]["topic"] == "完成剧"
    assert runs["run_idle"]["status"] == "idle"
    assert "review_mode" not in runs["run_done"], "审片模式已随端到端一起删除"


def test_status_running_and_failed(client, outdir):
    root = _make_run(outdir, "run_live", done=False)  # mtime 新鲜 → running
    assert client.get("/api/runs/run_live").json()["status"] == "running"
    with open(root / "events.jsonl", "a", encoding="utf-8") as f:  # 末条事件失败 → failed
        f.write(json.dumps({"ts": "t", "node": "m4", "status": "failed"}) + "\n")
    assert client.get("/api/runs/run_live").json()["status"] == "failed"


# ───────────────────────── 详情 ─────────────────────────
def test_detail(client, outdir):
    _make_run(
        outdir,
        "run_d",
        cost=[
            {"capability": "t2i", "provider": "dashscope", "model": "wanx2.1", "amount_cny": 0.36},
            {"capability": "i2v", "provider": "dashscope", "model": "wan2.2", "amount_cny": 1.1},
        ],
    )
    (outdir / "run_d" / "assets" / "role_a.png").write_bytes(b"png")
    d = client.get("/api/runs/run_d").json()
    assert d["status"] == "done"
    assert d["cost"]["total_cny"] == pytest.approx(1.46)
    assert len(d["cost"]["items"]) == 2
    assert d["images"]["assets"] == ["/api/files/run_d/assets/role_a.png"]
    assert d["final_video"] == "/api/files/run_d/final/final.mp4"
    assert d["events"][0]["node"] == "m1"
    assert d["meta"]["base_info"]["topic"] == "测试主题"


def test_detail_404(client, outdir):
    _make_run(outdir, "run_ok")
    assert client.get("/api/runs/run_nope").status_code == 404
    assert client.get("/api/runs/run_ok/../../etc").status_code in (400, 404)


# ───────────────────────── 产物文件与路径穿越防护 ─────────────────────────
def test_files(client, outdir):
    root = _make_run(outdir, "run_f")
    (root / "assets" / "a.png").write_bytes(b"img")
    r = client.get("/api/files/run_f/assets/a.png")
    assert r.status_code == 200 and r.content == b"img"
    assert client.get("/api/files/run_f/assets/missing.png").status_code == 404
    assert client.get("/api/files/run_f/secret/a.png").status_code == 404  # stage 白名单外


def test_files_traversal_blocked(client, outdir):
    """函数级直测穿越防护（绕过 HTTP 客户端的 URL 规范化）。"""
    _make_run(outdir, "run_t")
    for bad_name in ("../run_meta.json", "..", "a.png..%2f..%2fsecret"):
        with pytest.raises(HTTPException):
            api_mod.get_artifact_file("run_t", "assets", bad_name)
    with pytest.raises(HTTPException):
        api_mod.get_artifact_file("../other_run", "assets", "a.png")


# ───────────────────────── 环境自检 ─────────────────────────
def test_doctor(client):
    d = client.get("/api/doctor").json()
    assert isinstance(d["all_ok"], bool)
    items = {c["item"]: c for c in d["checks"]}
    assert "Python >= 3.11" in items
    assert items["Python >= 3.11"]["ok"] is True
    assert all(set(c) == {"item", "ok", "detail"} for c in d["checks"])
