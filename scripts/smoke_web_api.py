"""Web API 端到端冒烟脚本（需先启动 fastvideo3 serve --port 8765）。

流程：health → POST 新建 Fake 运行 → 轮询到完成 → 校验详情/产物文件 → 输出结果。
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
            json={
                "topic": "Web面板CI自检",
                "duration": 15,
                "fake": True,
                "review": "auto",
                "quality": "economy",
            },
        )
        print(f"[3] POST /api/runs: {r.status_code} {r.json()}")
        assert r.status_code == 201
        rid = r.json()["run_id"]

        for i in range(100):
            d = c.get(f"/api/runs/{rid}").json()
            print(
                f"[4.{i}] status={d['status']} events={len(d['events'])} "
                f"cost=¥{d['cost']['total_cny']}"
            )
            if d["status"] in ("done", "failed"):
                break
            time.sleep(3)
        assert d["status"] == "done", f"运行未成功: {d['status']}"

        assert d["final_video"], "缺少成片 URL"
        f = c.get(d["final_video"])
        print(f"[5] 成片 {d['final_video']}: {f.status_code}, {len(f.content)} bytes")
        assert f.status_code == 200 and len(f.content) > 0

        imgs = d["images"]["assets"] + d["images"]["storyboard"]
        print(
            f"[6] 图片产物 {len(imgs)} 张; 视频片段 "
            f"clips={len(d['videos']['clips'])} talk={len(d['videos']['talk'])}"
        )
        if imgs:
            g = c.get(imgs[0])
            assert g.status_code == 200, f"产物文件不可达: {imgs[0]}"

        bad = c.get(f"/api/files/{rid}/secret/x.png")
        print(f"[7] 越权 stage 访问: {bad.status_code} (期望 404)")
        assert bad.status_code == 404

        runs = c.get("/api/runs").json()
        row = next(x for x in runs if x["run_id"] == rid)
        print(f"[8] 列表可见: {row['status']} topic={row['topic']}")
        assert row["status"] == "done"

        doc = c.get("/api/doctor").json()
        print(f"[9] doctor: all_ok={doc['all_ok']} checks={len(doc['checks'])}")

    print("\n✔ Web API 端到端冒烟全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
