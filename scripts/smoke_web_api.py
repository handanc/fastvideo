"""Web API 全链路冒烟脚本（需先启动 fastvideo3 serve --port 8765）。

流程：health → POST 新建 Fake 任务 → 循环「开始生成」补完六步 → 校验详情/产物文件 → 输出结果。
用途：本地或 CI 环境验证 "FastAPI + Vue 面板" 全链路（零成本，Fake 模式）。
"""

from __future__ import annotations

import sys
import time

import httpx

for _stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 → UTF-8（与 CLI 同款容错）
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, pragma: no cover
        pass

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
STEP_TIMEOUT_SEC = 600  # 单步上限：Fake 档一步几秒，真实档给足余量


def _wait_idle(c: httpx.Client, rid: str) -> dict:
    """等这一步的子进程跑完（tasklock 释放），返回最后的 /steps 视图。"""
    deadline = time.time() + STEP_TIMEOUT_SEC
    steps = c.get(f"/api/runs/{rid}/steps").json()
    while time.time() < deadline:
        steps = c.get(f"/api/runs/{rid}/steps").json()
        if not steps["running"] and not c.get("/api/task").json().get("busy"):
            return steps
        time.sleep(2)
    raise AssertionError(f"步骤超时未完成: {[s['id'] for s in steps['steps']]}")


def main() -> int:
    with httpx.Client(base_url=BASE, timeout=30) as c:
        h = c.get("/api/health").json()
        print(f"[1] health: {h}")
        assert h["status"] == "ok"

        idx = c.get("/")
        print(f"[2] GET /: {idx.status_code}, html={'<!doctype html>' in idx.text.lower()}")
        assert idx.status_code == 200

        r = c.post(
            "/api/runs",
            json={"topic": "Web面板CI自检", "duration": 15, "fake": True, "quality": "economy"},
        )
        print(f"[3] POST /api/runs: {r.status_code} {r.json()}")
        assert r.status_code == 201
        rid = r.json()["run_id"]

        # [4] 一次点击 = 一步，点到没有下一步为止（与页面上的「开始生成」同一端点）
        for i in range(12):
            steps = c.get(f"/api/runs/{rid}/steps").json()
            nxt = steps["next"]
            if nxt is None:
                print(f"[4.{i}] 六步全部完成，累计 ¥{steps['spent_cny']}")
                break
            print(
                f"[4.{i}] ▶ 开始生成 · {nxt['label']}"
                f"（缺口 {len(nxt['missing'])} / 待重做 {len(nxt['stale'])}）"
            )
            sr = c.post(f"/api/runs/{rid}/step")
            assert sr.status_code == 202, f"/step 失败: {sr.status_code} {sr.text}"
            time.sleep(1)  # 给子进程一点时间认领任务锁，否则 running 还是 False
            done = _wait_idle(c, rid)
            step = next(s for s in done["steps"] if s["id"] == nxt["id"])
            assert step["state"] != "failed", f"步骤 {nxt['id']} 失败: {step.get('error')}"
        else:
            raise AssertionError("点了 12 次仍未完成，进度状态机可能卡住")

        d = c.get(f"/api/runs/{rid}").json()
        print(f"[5] status={d['status']} events={len(d['events'])} cost=¥{d['cost']['total_cny']}")
        assert d["status"] == "done", f"运行未成功: {d['status']}"
        assert d["final_video"], "缺少成片 URL"

        f = c.get(d["final_video"])
        print(f"[6] 成片 {d['final_video']}: {f.status_code}, {len(f.content)} bytes")
        assert f.status_code == 200 and len(f.content) > 0

        imgs = d["images"]["assets"] + d["images"]["storyboard"]
        print(
            f"[7] 图片产物 {len(imgs)} 张; 视频片段 "
            f"clips={len(d['videos']['clips'])} talk={len(d['videos']['talk'])}"
        )
        if imgs:
            g = c.get(imgs[0])
            assert g.status_code == 200, f"产物文件不可达: {imgs[0]}"

        bad = c.get(f"/api/files/{rid}/secret/x.png")
        print(f"[8] 越权 stage 访问: {bad.status_code} (期望 404)")
        assert bad.status_code == 404

        runs = c.get("/api/runs").json()
        row = next(x for x in runs if x["run_id"] == rid)
        print(f"[9] 列表可见: {row['status']} topic={row['topic']}")
        assert row["status"] == "done"

        doc = c.get("/api/doctor").json()
        print(f"[10] doctor: all_ok={doc['all_ok']} checks={len(doc['checks'])}")

    print("\n✔ Web API 全链路冒烟通过（建任务 → 六步 → 成片）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
