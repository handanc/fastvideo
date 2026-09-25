"""FastVideo3 Web API（FastAPI）：为 Vue 前端提供 REST 接口并托管前端构建产物。

启动：fastvideo3 serve  （或 uvicorn fastvideo3.app.api:app --port 8000）

接口一览（对齐参考项目 FastVideo 的 "FastAPI 服务 + Web UI" 架构）：
  GET  /api/health                            存活探针
  GET  /api/doctor                            环境自检（JSON 版 doctor）
  GET  /api/runs                              历史运行列表（含状态摘要）
  POST /api/runs                              新建运行（后台子进程跑 CLI，与 Streamlit 面板同方案）
  GET  /api/runs/{run_id}                     运行详情：meta / 事件流 / 成本 / 产物清单
  POST /api/runs/{run_id}/resume              人工审片回复（approve | reject:<意见>）并续跑
  GET  /api/files/{run_id}/{stage}/{name}     产物文件（图片/视频，带路径穿越防护）
  GET  /*                                     web/dist 静态前端（存在时挂载）
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastvideo3.core import config as config_mod
from fastvideo3.core.config import PROJECT_ROOT, load_config
from fastvideo3.core.schema import BaseInfo, ProjectConfig
from fastvideo3.core.storage import STAGES, RunStorage, list_runs

app = FastAPI(title="FastVideo3 API", description="多智能体短剧视频生成 Web 服务", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_NAME_RE = re.compile(r"^[\w.-]+$")  # run_id / 文件名白名单（拒绝路径分隔符）
_ALLOWED_STAGES = frozenset(STAGES)
_RECENT_SEC = 120  # 目录最近 2 分钟有写入 → 视为运行中


# ───────────────────────── 请求模型 ─────────────────────────
class RunCreate(BaseModel):
    """新建运行请求（字段与 CLI run 命令一一对应）。"""

    topic: str = Field(min_length=1, max_length=200)
    style: str = Field(default="都市轻喜剧", max_length=100)
    duration: int = Field(default=30, ge=5, le=600)
    # fake 默认 False（真实生成）：此前默认 True，Web 面板发起的运行在未察觉时
    # 全走占位色块演示模式（.env 的 FV3_FAKE=0 被请求体覆盖）
    fake: bool = False
    review: Literal["auto", "gated"] = "auto"
    talk_mode: Literal["s2v_direct", "v2v_overlay", "audio_overlay"] = "s2v_direct"
    quality: Literal["economy", "high", "cinema"] = "economy"
    aspect: Literal["9:16", "16:9", "1:1"] = "9:16"
    budget: float = Field(default=0.0, ge=0.0, le=100000.0)


class ResumeRequest(BaseModel):
    """审片回复：approve 或 reject:<意见>。"""

    decision: str = Field(default="approve", pattern=r"^(approve|reject(:[\s\S]{0,500})?)$")


# ───────────────────────── 内部工具 ─────────────────────────
def _out() -> Path:
    """动态读取输出目录（测试中 monkeypatch config.OUTPUT_DIR 即生效）。"""
    return config_mod.OUTPUT_DIR


def _new_run_id() -> str:
    base = time.strftime("run_%Y%m%d_%H%M%S")
    rid, n = base, 1
    while (_out() / rid).exists():
        rid = f"{base}_{n}"
        n += 1
    return rid


def _spawn(cmd: list[str], log_path: Path | None = None) -> None:
    """后台启动 CLI 子进程；输出追加到 log_path 便于排障。"""
    kwargs: dict[str, Any] = {"cwd": str(PROJECT_ROOT)}
    if sys.platform == "win32":  # 服务化运行时不弹控制台窗口
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    fh = open(log_path, "ab") if log_path is not None else None
    try:
        if fh is not None:
            kwargs.update(stdout=fh, stderr=fh)
        subprocess.Popen(cmd, **kwargs)
    finally:
        if fh is not None:
            fh.close()  # 子进程已继承句柄


def _load_meta(root: Path) -> dict | None:
    p = root / "run_meta.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_events(root: Path, limit: int = 100) -> list[dict]:
    p = root / "events.jsonl"
    if not p.is_file():
        return []
    events: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _read_cost(root: Path) -> dict:
    p = root / "cost.json"
    items: list = []
    if p.is_file():
        try:
            items = json.loads(p.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            items = []
    total = sum(float(e.get("amount_cny", 0)) for e in items if isinstance(e, dict))
    return {"total_cny": round(total, 4), "items": items}


def _mtime(root: Path) -> float:
    latest = 0.0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


def _run_status(root: Path) -> str:
    """启发式状态：done(有成片) > failed(末条事件失败) > running(近期有写入) > paused/idle。"""
    if (root / "final" / "final.mp4").is_file():
        return "done"
    events = _read_events(root, limit=5)
    last = str(events[-1].get("status", "")).lower() if events else ""
    if last in ("failed", "error"):
        return "failed"
    if time.time() - _mtime(root) < _RECENT_SEC:
        return "running"
    meta = _load_meta(root) or {}
    return "paused" if meta.get("review_mode") == "gated" else "idle"


def _list_files(d: Path, exts: tuple[str, ...]) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in exts)


def _safe_run_root(run_id: str) -> Path:
    if not _NAME_RE.match(run_id):
        raise HTTPException(status_code=404, detail="run 不存在")
    root = _out() / run_id
    if not (root / "run_meta.json").is_file():
        raise HTTPException(status_code=404, detail="run 不存在")
    return root


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("fastvideo3")
    except Exception:  # noqa: BLE001  # 未安装(源码直跑)时兜底
        return "0.1.0"


# ───────────────────────── API 路由 ─────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": "fastvideo3", "version": _version()}


@app.get("/api/doctor")
def doctor(probe: int = 0) -> dict:
    """环境自检（与 CLI doctor 同源逻辑，返回 JSON）。?probe=1 时真实 KEY 探活。"""
    checks: list[dict] = []

    def row(item: str, ok: bool, detail: Any = "") -> None:
        checks.append({"item": item, "ok": bool(ok), "detail": str(detail)})

    row("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    missing = []
    for dep in ("pydantic", "httpx", "langgraph", "typer", "PIL", "sqlalchemy", "yaml"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    row("核心依赖", not missing, "全部就绪" if not missing else f"缺失: {missing}")

    from fastvideo3.utils.ffmpeg import find_ffmpeg

    try:
        row("FFmpeg", True, find_ffmpeg())
    except Exception as e:  # noqa: BLE001
        row("FFmpeg", False, e)

    cfg = load_config()
    row(
        "DEEPSEEK_API_KEY",
        bool(cfg.deepseek_api_key) or cfg.fake,
        "已配置" if cfg.deepseek_api_key else "未配置(Fake 模式可跳过)",
    )
    row(
        "DASHSCOPE_API_KEY",
        bool(cfg.dashscope_api_key) or cfg.fake,
        "已配置" if cfg.dashscope_api_key else "未配置",
    )

    if probe and cfg.dashscope_api_key:
        import asyncio

        from fastvideo3.providers.dashscope import probe_account

        status, detail = asyncio.run(
            probe_account(cfg.dashscope_api_key, cfg.model_name("dashscope_text", "qwen3.7-plus"))
        )
        row("DASHSCOPE 账户探活", status == "ok", f"[{status}] {detail}")
    elif probe:
        row("DASHSCOPE 账户探活", False, "未配置 KEY，跳过")

    from fastvideo3.core.config import MODELS_DIR
    from fastvideo3.utils.yolo import WEIGHT_NAME

    row(
        "yolov8n-seg.pt(双人同屏)",
        (MODELS_DIR / WEIGHT_NAME).exists(),
        "就位" if (MODELS_DIR / WEIGHT_NAME).exists() else "缺失(自动降级拼接)",
    )
    try:
        from fastvideo3.core.jobs import JobStore

        JobStore(cfg.db_url).dispose()
        row("数据库", True, cfg.db_url.split("://")[0])
    except Exception as e:  # noqa: BLE001
        row("数据库", False, str(e)[:80])

    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


@app.get("/api/runs")
def runs_list() -> list[dict]:
    """历史运行摘要列表（倒序，最新在前）。"""
    out = _out()
    summaries = []
    for rid in list_runs(root=out):
        root = out / rid
        meta = _load_meta(root) or {}
        base = meta.get("base_info") or {}
        mtime = _mtime(root)
        summaries.append(
            {
                "run_id": rid,
                "topic": base.get("topic", ""),
                "style": base.get("style", ""),
                "fake": bool(meta.get("fake")),
                "quality": meta.get("quality", "economy"),
                "review_mode": meta.get("review_mode", "auto"),
                "status": _run_status(root),
                "cost_total_cny": _read_cost(root)["total_cny"],
                "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                if mtime
                else "",
            }
        )
    return summaries


@app.post("/api/runs", status_code=201)
def create_run(req: RunCreate) -> dict:
    """新建运行：先落 run_meta.json（列表立即可见），再后台启动 CLI 子进程。"""
    cfg = load_config(quality=req.quality)
    run_id = _new_run_id()
    base = BaseInfo(
        topic=req.topic,
        style=req.style,
        target_duration_sec=req.duration,
        aspect_ratio=req.aspect,
    )
    project = ProjectConfig(
        run_id=run_id,
        base_info=base,
        profile=cfg.profile,  # type: ignore[arg-type]
        review_mode=req.review,
        talk_mode=req.talk_mode,
        # budget=0 表示"跟随 .env"：meta 落实际生效值（此前落 0.0 → resume 时熔断恒触发）
        budget_limit_cny=req.budget if req.budget > 0 else cfg.budget_limit_cny,
        fake=req.fake,
        quality=req.quality,
    )
    storage = RunStorage(run_id, root=_out())
    storage.save_json("run_meta.json", project.model_dump(mode="json"))

    cmd = [
        sys.executable,
        "-m",
        "fastvideo3.app.cli",
        "run",
        "--topic",
        req.topic,
        "--style",
        req.style,
        "--duration",
        str(req.duration),
        "--review",
        req.review,
        "--talk-mode",
        req.talk_mode,
        "--quality",
        req.quality,
        "--aspect",
        req.aspect,
        "--run-id",
        run_id,
    ]
    if req.fake:
        cmd.append("--fake")
    if req.budget > 0:
        cmd += ["--budget", str(req.budget)]
    _spawn(cmd, log_path=storage.root / "server_spawn.log")
    return {"run_id": run_id, "started": True}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    """运行详情：meta + 事件流(末100条) + 成本 + 产物 URL 清单。"""
    root = _safe_run_root(run_id)
    meta = _load_meta(root) or {}
    images = {
        stage: [
            f"/api/files/{run_id}/{stage}/{n}"
            for n in _list_files(root / stage, (".png", ".jpg", ".jpeg"))
        ]
        for stage in ("assets", "storyboard")
    }
    videos = {
        stage: [f"/api/files/{run_id}/{stage}/{n}" for n in _list_files(root / stage, (".mp4",))]
        for stage in ("clips", "talk")
    }
    final = root / "final" / "final.mp4"
    review: dict | None = None
    rp = root / "final_review.json"
    if rp.is_file():
        try:
            review = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            review = None
    return {
        "run_id": run_id,
        "status": _run_status(root),
        "meta": meta,
        "events": _read_events(root),
        "cost": _read_cost(root),
        "images": images,
        "videos": videos,
        "review": review,  # 终审结果（含 review_error 标识：异常放行不再静默）
        "final_video": f"/api/files/{run_id}/final/final.mp4" if final.is_file() else None,
    }


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, req: ResumeRequest) -> dict:
    """人工审片回复 / 崩溃续跑：后台启动 CLI resume 子进程。"""
    root = _safe_run_root(run_id)
    cmd = [
        sys.executable,
        "-m",
        "fastvideo3.app.cli",
        "resume",
        run_id,
        "--decision",
        req.decision,
    ]
    _spawn(cmd, log_path=root / "server_spawn.log")
    return {"run_id": run_id, "decision": req.decision, "started": True}


@app.get("/api/files/{run_id}/{stage}/{name}")
def get_artifact_file(run_id: str, stage: str, name: str) -> FileResponse:
    """产物文件访问：stage 白名单 + 名称正则 + resolve 越界检查（三重防护）。"""
    if not _NAME_RE.match(run_id) or stage not in _ALLOWED_STAGES or not _NAME_RE.match(name):
        raise HTTPException(status_code=404, detail="文件不存在")
    base = (_out() / run_id / stage).resolve()
    p = (base / name).resolve()
    if not p.is_file() or not p.is_relative_to(base):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(p)


# ───────────────────────── 前端静态托管 ─────────────────────────
def _web_dist() -> Path:
    return PROJECT_ROOT / "web" / "dist"


if _web_dist().is_dir():
    app.mount("/", StaticFiles(directory=_web_dist(), html=True), name="web")
else:  # 未构建时给出可操作提示（开发模式走 vite dev server + /api 代理）

    @app.get("/", include_in_schema=False)
    def index_hint() -> dict:
        return {
            "hint": "前端未构建：在 web/ 目录执行 npm install && npm run build 后重启服务；"
            "开发模式用 npm run dev (http://127.0.0.1:5173)",
            "docs": "/docs",
        }


def main() -> None:
    """模块直跑入口：python -m fastvideo3.app.api"""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
