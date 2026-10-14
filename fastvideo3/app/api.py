"""FastVideo3 Web API（FastAPI）：为 Vue 前端提供 REST 接口并托管前端构建产物。

启动：fastvideo3 serve  （或 uvicorn fastvideo3.app.api:app --port 8000）

接口一览（对齐参考项目 FastVideo 的 "FastAPI 服务 + Web UI" 架构）：
  GET  /api/health                            存活探针
  GET  /api/doctor                            环境自检（JSON 版 doctor）
  GET  /api/runs                              历史运行列表（含状态摘要）
  POST /api/runs                              新建任务（只落 run_meta.json，不启动生成）
  GET  /api/runs/{run_id}                     运行详情：meta / 事件流 / 成本 / 产物清单
  GET  /api/runs/{run_id}/board               制作看板：剧本+角色图+场景图+逐镜头(首帧/片段/度量)+成片
  GET  /api/runs/{run_id}/steps               六步进度：done/gap/active/failed + 缺口清单 + 花费
  POST /api/runs/{run_id}/step                开始生成：只跑最靠前的未完成步骤（后台子进程）
  PUT  /api/runs/{run_id}/artifacts/{stage}/{name}  上传替换角色图/场景图/分镜首帧（按画幅归一化）
  DELETE /api/runs/{run_id}/artifacts/{stage}/{name}  删产物并沿血缘精确作废下游
  GET  /api/runs/{run_id}/uploads             人工上传清单
  GET/PUT /api/runs/{run_id}/script           剧本 JSON 读取 / 编辑保存（结构校验 + 血缘作废）
  DELETE /api/runs/{run_id}/script             删除剧本（归档 + 全量作废下游 → 下一步回到「生成剧本」）
  GET/PUT /api/runs/{run_id}/script/text       剧本稿（可读文本）读取 / 本地解析保存
  POST /api/runs/{run_id}/script/regenerate    按 prompt 重写剧本（revise 保留剧情 / rewrite 推倒重写）
  POST /api/runs/{run_id}/characters/{id}/portrait  人工指定定妆照（候选 → 正式，旧图自动备份）
  GET  /api/task · POST /api/task/unlock       全局单任务锁状态 / 人工解锁
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastvideo3.core import config as config_mod
from fastvideo3.core.config import (
    DURATION_MAX,
    DURATION_MIN,
    PROJECT_ROOT,
    load_config,
    resolution_for_quality,
)
from fastvideo3.core.schema import BaseInfo, ProjectConfig, Script
from fastvideo3.core.storage import STAGES, RunStorage, list_runs, safe_promote

app = FastAPI(title="FastVideo3 API", description="多智能体短剧视频生成 Web 服务", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_NAME_RE = re.compile(r"^[\w.-]+$")  # run_id / 文件名白名单（拒绝路径分隔符）
_ALLOWED_STAGES = frozenset(STAGES)
_RECENT_SEC = 120  # 目录最近 2 分钟有写入 → 视为运行中


# ───────────────────────── 请求模型 ─────────────────────────
class RunCreate(BaseModel):
    """新建任务请求（字段与 CLI `new` 命令一一对应）。"""

    topic: str = Field(min_length=1, max_length=200)
    style: str = Field(default="都市轻喜剧", max_length=100)
    # 1~3600 秒（边界与 CLI、前端表单同一来源：config.DURATION_MIN/MAX）。
    # 下限 1 而不是 0：0 会让剧本没有目标长度、成本预估归零，且熔断判据（`exceeded()`
    # 用 `>=`）在第一步就跳闸。
    duration: int = Field(default=30, ge=DURATION_MIN, le=DURATION_MAX)
    # fake 默认 False（真实生成）：此前默认 True，Web 面板发起的运行在未察觉时
    # 全走占位色块演示模式（.env 的 FV3_FAKE=0 被请求体覆盖）
    fake: bool = False
    # 口型**不再是选项**：wan3.0 生成场景片段时原生带台词口型（音频与嘴型同轨），
    # 独立口型通道与"不做口型"都已下线。老前端/老脚本若仍提交这三个字段，接受但忽略，
    # 并在 run_meta 里记一笔 ignored_legacy_fields —— 静默忽略会让老调用方以为生效了。
    talk_mode: str | None = None
    talk_model: str | None = None
    voice_source: str | None = None
    # 面板上的主选择是**分辨率**；给了它就由它映射出内部档位（config.RESOLUTION_TO_QUALITY）。
    # quality 仍保留：CLI/测试/老前端可能只给档位，两者都给时以 resolution 为准。
    resolution: Literal["480P", "720P", "1080P", ""] = ""
    quality: Literal["economy", "high", "cinema"] = "economy"
    aspect: Literal["9:16", "16:9", "1:1"] = "9:16"
    budget: float = Field(default=0.0, ge=0.0, le=100000.0)
    ablation: str = Field(default="", max_length=64)  # 消融配置名（configs/ablations/ 内）
    # run 级模型选择：能力 → 模型名（text / t2i / i2v），空串＝跟随画质档位。
    # 可选清单与校验规则都在 `core/model_catalog.py`（前端选项由 GET /api/model-choices 生成）。
    # 落进 run_meta.json 并由 `config.apply_run_overrides` 原样恢复 —— 每步必须用同一个模型，
    # 否则换过一次模型后续跑就会"资产是新模型的、首帧是旧模型的"。
    model_overrides: dict[str, str] = Field(default_factory=dict, max_length=8)


class ScriptTextRequest(BaseModel):
    """剧本稿全文（可读文本，不是 JSON）。上限按一部 10 分钟短剧的稿子留余量。"""

    text: str = Field(min_length=1, max_length=200_000)


class RegenerateRequest(BaseModel):
    """按人写的 prompt 重新生成剧本。

    - `revise`：把**当前剧本稿**连同意见交给模型，剧情以稿子为准，只改被点名的部分
      （走 `generate_script(existing_script_text=...)` 的解析模式）→ 字段级作废，省钱。
    - `rewrite`：不要旧剧情，用主题 + 你的要求重新出一版 → 旧剧本归档、下游全量作废。
    """

    prompt: str = Field(min_length=1, max_length=2000)
    mode: Literal["revise", "rewrite"] = "revise"


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
    rows = [e for e in items if isinstance(e, dict)]
    total = sum(float(e.get("amount_cny", 0)) for e in rows)
    # 单价**不可信**（模型不在价目表里，只能按能力默认价估算）的那部分金额。
    # 成本上限是按 total 判的，所以这部分必须单独说出来：真实账单可能明显更高，
    # "还没到上限"这句话的可信度取决于它。
    uncertain = sum(
        float(e.get("amount_cny", 0))
        for e in rows
        if (e.get("meta") or {}).get("price_certain") is False
    )
    sources: dict[str, int] = {}
    for e in rows:
        src = str((e.get("meta") or {}).get("price_source") or "unknown")
        sources[src] = sources.get(src, 0) + 1
    # 计费量来源：api_usage = 服务端回传的真实产出；requested = 只能按我们请求的量估算
    qsources: dict[str, int] = {}
    for e in rows:
        qs = str((e.get("meta") or {}).get("quantity_source") or "")
        if qs:
            qsources[qs] = qsources.get(qs, 0) + 1
    # 真实账单口径（上次对账的结果）。有它就必须一起显示：台账是"单价×用量"的估算，
    # 而账单是实际扣了多少钱 —— 两者对不上说明单价表失真，这是本项目唯一的外部校验。
    from fastvideo3.core import billing

    bill = billing.load_reconciliation(root)
    return {
        "total_cny": round(total, 4),
        "bill": bill,
        "uncertain_cny": round(uncertain, 4),
        "price_sources": sources,
        "quantity_sources": qsources,
        "items": items,
    }


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
    """启发式状态：done(有成片) > failed(末条事件失败) > running(近期有写入) > idle。"""
    if (root / "final" / "final.mp4").is_file():
        return "done"
    events = _read_events(root, limit=5)
    last = str(events[-1].get("status", "")).lower() if events else ""
    # 本项目失败一律写成 `step_failed` / `xxx_failed`，不是裸 "failed"：只认死字符串的话，
    # 一次 9 分钟的读超时在面板上只显示"已停止"，用户完全不知道发生了什么（2026-10-14 实测）。
    if "fail" in last or "error" in last or "timeout" in last:
        return "failed"
    if time.time() - _mtime(root) < _RECENT_SEC:
        return "running"
    return "idle"


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


@app.get("/api/cost-estimate")
def cost_estimate(quality: str = "economy", duration: float = 30, talk_mode: str = "") -> dict:
    """新建任务前的成本预估 —— **单价的唯一来源是 pricing.yaml + 供应商路由**。

    存在理由：口型通道从 videoretalk(¥0.08/秒) 换成 fal InfiniteTalk(¥2.15-4.34/秒)
    涨了 25~50 倍。这种量级变化绝不能靠前端 JS 里那份手抄表反映（抄一份就漂一份），
    所以表单直接问后端。talk_mode=audio_overlay 时口型项归零（用户明确不要口型）。
    """
    from fastvideo3.core.cost import estimate_run_cost_cny
    from fastvideo3.core.talkmode import channel, provider_name, tier_conflict

    # 未知档位名：不 overlay 任何 profile，等于按 default.yaml 估（不 500、不猜）
    cfg = load_config(quality=quality if quality in ("economy", "high", "cinema") else None)
    cfg.talk_mode = talk_mode or cfg.talk_mode
    est = estimate_run_cost_cny(cfg, duration)
    ch = channel(cfg.talk_mode)
    return {
        **est,
        "budget_default_cny": cfg.budget_limit_cny,
        # 通道取**模式解析出来的那一条**（estimate_run_cost_cny 也按它计价），
        # 不再是 yaml 路由名 —— 否则前端显示的通道和真正花钱的通道可能不是一条
        "talk_route": provider_name(cfg, cfg.talk_mode) or "",
        "talk_channel": ch.label,
        "talk_eats": ch.eats,
        "talk_note": ch.note,
        "tier_warning": tier_conflict(
            cfg.talk_mode, cfg.quality, cfg.i2v_resolution, cfg.voice_source
        ),
        "talk_mode": cfg.talk_mode,
        "quality": cfg.quality,
    }


@app.get("/api/doctor")
def doctor(probe: int = 0) -> dict:
    """环境自检（与 CLI doctor 同源逻辑，返回 JSON）。?probe=1 时真实 KEY 探活。"""
    checks: list[dict] = []

    def row(item: str, ok: bool, detail: Any = "") -> None:
        checks.append({"item": item, "ok": bool(ok), "detail": str(detail)})

    # 检查项只有一份（core/runtime.py），CLI 与这里共用；下面只加 API 独有的探活项。
    from fastvideo3.core import runtime

    cfg = load_config()
    checks.extend(runtime.checks(cfg))

    if probe and cfg.dashscope_api_key:
        import asyncio

        from fastvideo3.providers.dashscope import probe_account

        status, detail = asyncio.run(
            probe_account(cfg.dashscope_api_key, cfg.model_name("dashscope_text", "qwen3.7-plus"))
        )
        row("DASHSCOPE 账户探活", status == "ok", f"[{status}] {detail}")
    elif probe:
        row("DASHSCOPE 账户探活", False, "未配置 KEY，跳过")

    if probe:
        import asyncio

        from fastvideo3.providers.dashscope import probe_model_activation

        # 缺必填字段的 s2v 请求：校验阶段就被拒，官方明写失败不计费、不消耗免费额度
        status, detail = asyncio.run(
            probe_model_activation(
                cfg.dashscope_api_key,
                cfg.model_name("dashscope_s2v", "wan2.2-s2v"),
                base_url=cfg.dashscope_video_base_url,
            )
        )
        row("百炼口型通道探活(¥0)", status == "ok", f"[{status}] {detail}")

        if cfg.fal_api_key:
            from fastvideo3.providers.fal import probe_key

            status, detail = asyncio.run(probe_key(cfg.fal_api_key))
            row("FAL 账户探活(¥0)", status == "ok", f"[{status}] {detail}")
        else:
            row("FAL 账户探活(¥0)", True, "未配置 FAL_KEY(可选兜底通道)")

    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}


@app.post("/api/doctor/provision")
def doctor_provision(force: int = 0, lock: int = 0) -> dict:
    """把运行时资产收进项目目录（ffmpeg → `bin/`、权重 → `models/`、缺 .env 时复制模板）。

    只动项目内的文件，**从不覆盖已存在的 `.env`**，也不返回任何配置内容 ——
    自检面板的文本会原样显示在浏览器里，密钥一旦进 detail 就等于泄露。
    """
    from fastvideo3.core import runtime

    done = runtime.provision_all(force=bool(force), with_lock=bool(lock))
    checks = runtime.checks(load_config())
    return {
        "done": done,
        "all_ok": all(c["ok"] for c in checks),
        "checks": checks,
        "remaining": [c["item"] for c in checks if not c["ok"]],
    }


@app.get("/api/model-choices")
def model_choices(quality: str = "cinema", run_id: str = "") -> dict:
    """模型选择器的唯一来源：可选清单 + **当前实际生效的模型** + 是谁定的（run/.env/yaml）。

    "当前生效值"必须一起给：用户看不见"现在到底在用哪个模型"，选择器就变成猜谜 ——
    本轮排查出的真实问题正是 `.env` 的全局 `FV3_*_MODEL` 把三个档位的 t2i/i2v 都换掉了，
    而 yaml 的档位表看起来还是原样。
    """
    from fastvideo3.core.config import apply_run_overrides

    # 通用三项在这里算生效值；口型（talk）单独一块给 —— 它还要带模式与音轨来源，
    # 而且它的"模型名"不在下面这张 provider 键映射表里
    from fastvideo3.core.model_catalog import UI_CAPS, ui_catalog

    q = quality if quality in ("economy", "high", "cinema") else "cinema"
    base_cfg = load_config(quality=q)
    meta = _load_meta(_safe_run_root(run_id)) if run_id else None
    cfg = apply_run_overrides(load_config(quality=q), meta) if meta else base_cfg
    run_chosen = (meta or {}).get("model_overrides") or {}

    # 能力 → (yaml models 键, 内置默认)：用于没配 KEY 时也能显示档位默认模型
    keys = {"text": ("dashscope_text", "qwen3.7-plus"),
            "t2i": ("dashscope_t2i", "wanx2.1-t2i-turbo"),
            "i2v": ("dashscope_i2v", "wan2.2-i2v-flash")}
    effective: dict[str, str] = {}
    sources: dict[str, str] = {}
    for cap in UI_CAPS:
        if cap in run_chosen:
            sources[cap] = "run"
        elif cap in base_cfg.model_overrides:
            sources[cap] = ".env"
        else:
            sources[cap] = "yaml"
        key, default = keys[cap]
        model = ""
        try:  # 装配真实 provider 读它的 model —— 这才是"等一下会打哪个模型"的真相
            from fastvideo3.providers.base import Capability
            from fastvideo3.providers.factory import make_provider

            model = str(getattr(make_provider(getattr(Capability, cap.upper()), cfg), "model", ""))
        except Exception:  # noqa: BLE001  # 没配 KEY 时装配不了，退回按配置解析（不影响选择器可用）
            model = cfg.model_for(cap, key, default)
        effective[cap] = model or cfg.model_for(cap, key, default)
    # 口型不再是可选能力：wan3.0 场景片段自带台词口型。这里只回一个说明块，
    # 让老前端拿到 200 而不是 404，并明确告诉它"没有口型下拉可渲染"。
    return {
        "quality": q,
        "run_id": run_id or "",
        "talk": {
            "removed": True,
            "note": "口型由 wan3.0 场景生成自带，不再是可选项；音轨恒为模型原声",
        },
        # 这条 run 已点名的模型（详情页的选择器要用它做初始值，否则会显示成"跟随档位"，
        # 让人以为之前的选择没生效）
        "run_overrides": {k: v for k, v in run_chosen.items() if v},
        **ui_catalog(effective, sources),
    }


@app.get("/api/resolutions")
def resolutions() -> dict:
    """面板的分辨率选项（480P/720P/1080P）与它映射到的内部档位。

    映射只有一份（`config.RESOLUTION_TO_QUALITY`）：档位继续在后端驱动"一致性深度 /
    单价量级 / 是否自动重拍"，用户不必逐条记。三类闸门（结构性、技术 QC、身份客观层）
    **与分辨率无关，任何档都跑** —— 这是 2026-10-12 的裁定，写死在 configs/default.yaml。
    """
    from fastvideo3.core.config import RESOLUTION_TO_QUALITY, RESOLUTIONS, apply_resolution
    from fastvideo3.core.cost import estimate_run_cost_cny

    rows = []
    for res in RESOLUTIONS:
        q = RESOLUTION_TO_QUALITY[res]
        # 必须 apply_resolution：档位 yaml 里的 resolution 是它自己的默认（profile_high 写着
        # 1080×1920），不覆盖就会在面板上显示"720P → 1080×1920"这种自相矛盾的规格
        cfg = apply_resolution(load_config(quality=q), res)
        est = estimate_run_cost_cny(cfg, 30)
        rows.append(
            {
                "resolution": res,
                "quality": q,
                "output_size": list(cfg.resolution),
                "i2v_resolution": cfg.i2v_resolution,
                "cost_30s": [est["lo"], est["hi"]],
                "auto_reshoot": cfg.consistency_gate == "reshoot",
                "note": (
                    "教学/试跑：量一致性但不自动重拍（重拍要花钱）"
                    if q == "economy"
                    else "交付：参考图锁脸 + 度量 + 不达标自动重拍 ≤2 轮"
                ),
            }
        )
    return {"resolutions": rows}


@app.patch("/api/runs/{run_id}/models")
async def patch_run_models(run_id: str, request: Request) -> dict:
    """**中途改模型**：改 run_meta 的模型选择，之后的步骤按新模型跑。

    为什么允许中途改：一条 run 要跑六步、几分钟到几十分钟，跑到一半发现选错了模型
    （或免费额度用尽想换档），只能删掉重建 = 前面几步的钱白烧。
    两个必须守住的边界：
    1. **持锁期间不许改**（`_require_idle`）：正在生成时改配置会让同一步的前后两个镜头
       用不同模型，产物不同源，而且台账上看不出是哪一步换的；
    2. 改的是"**之后的步骤**"，已经落盘的产物不动 —— 想让它生效就删掉对应产物再点
       「开始生成」（详情页本来就有删除单产物的入口，会沿血缘精确作废下游）。
    """
    root = _safe_run_root(run_id)
    _require_idle("改模型")
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"请求体不是合法 JSON: {e}") from e
    meta = _load_meta(root) or {}
    quality = str(meta.get("quality") or "economy")
    raw = body.get("model_overrides") if isinstance(body, dict) else None
    if raw is not None and not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="model_overrides 必须是对象（能力→模型名）")
    chosen = _clean_model_overrides(raw or {}, quality)  # 非法名字/协议 → 400，不当场烧钱
    # 显式传空串 = **撤回**这一项点名，回到"跟随档位/.env"。
    # 必须在合并前单独算出来：`_clean_model_overrides` 会把空值直接丢掉（建任务时
    # 空值＝没选，不该落进 meta），撤回的意图只有这里能识别。
    cleared = {k for k, v in (raw or {}).items() if not str(v or "").strip()}

    merged = {**(meta.get("model_overrides") or {}), **chosen}
    merged = {k: v for k, v in merged.items() if k not in cleared and v}
    # **改 i2v 就是改场景生成/口型模型**：一次 wan3.0 调用同时出画面与口型，只有一个杠杆。
    # 不同步就会"面板选了 prime、跑的是 .env 里的 video"（实测装配优先级 talk>.env i2v，
    # 旧的 talk 覆盖永远赢过新选的 i2v）。撤回 i2v 点名时 talk 一起撤回，回到跟随档位。
    if "i2v" in chosen or "i2v" in cleared:
        if merged.get("i2v"):
            merged["talk"] = merged["i2v"]
        else:
            merged.pop("talk", None)
    meta["model_overrides"] = merged

    # ── 口型模型也可以中途换：它落盘在 talk_mode（不是 model_overrides，理由见 /api/model-choices）──
    from fastvideo3.core.talkmode import (
        provider_name,
    )

    # 面板提交模型名（`talk_model`），CLI/老前端可能提交模式名（`talk_mode`）—— 两者都收，
    # 但都走 `resolve_talk_choice` 这一个解析口，避免"模式与模型各存一份"重新分叉。
    # 口型**模型**跟着 `i2v_model` 走（见上），不再是单独一项；**口型模式/音轨来源**
    # 彻底不再是可改项（wan3.0 场景片段自带口型、音轨恒为模型原声）。老请求体提交
    # talk_mode/voice_source/talk_model 时记入忽略清单，并说明该改哪个字段。
    talk_changed = ""
    ignored: list[str] = []
    # **口型模型 = 场景生成模型 = `i2v_model`，同一条调用、同一个键。**
    # 口型不再是一条单独链路（A/B 链已删），场景视频就是 wan3.0 一次调用出的带口型片段。
    # 此前 `talk_model` 在 POST 被"收下并忽略"、在 PATCH 却写进 model_overrides["talk"]
    # 并回写 .env —— 同一个字段两个端点两种语义，老前端按 PATCH 的语义就会以为建任务时
    # 也能单独指定口型模型。现在两边一致：`i2v_model` 决定它，`talk_model` 一律进忽略清单。
    want_talk = body.get("talk_model") if isinstance(body, dict) else None
    if want_talk not in (None, ""):
        ignored.append("talk_model")
        talk_changed = "talk_model 已忽略：口型与场景视频是同一次 wan3.0 调用，换它请改 i2v_model"
    for legacy in ("talk_mode", "voice_source"):
        if isinstance(body, dict) and body.get(legacy) not in (None, ""):
            ignored.append(legacy)
    if ignored:
        meta["ignored_legacy_fields"] = sorted(
            set((meta.get("ignored_legacy_fields") or []) + ignored)
        )
    env_written: list[str] = []
    env_error = ""
    if not meta.get("fake"):
        from fastvideo3.core.envfile import update_env_models

        env_written, err = update_env_models(chosen)
        env_error = err or ""
        if env_written:
            meta["model_env_written"] = sorted(
                set((meta.get("model_env_written") or []) + env_written)
            )
        if env_error:
            meta["model_env_error"] = env_error
    RunStorage(run_id, root=root.parent).save_json("run_meta.json", meta)
    RunStorage(run_id, root=root.parent).log_event(
        "models", "changed", overrides=merged, env_written=env_written, env_error=env_error,
        talk_mode=meta.get("talk_mode"), voice_source=meta.get("voice_source"),
        hint="只对**之后**的步骤生效；已落盘产物不变，要重出请删对应产物再点开始生成",
    )
    cfg = _cfg_for_run(root)
    return {
        "run_id": run_id,
        "model_overrides": merged,
        "env_written": env_written,
        "env_error": env_error,
        "effective": {
            cap: cfg.model_overrides.get(cap, "") for cap in ("text", "t2i", "i2v")
        },
        # 口型恒为 wan3.0 场景生成自带；这里只回当前生效的口型**模型**供界面显示
        "talk_mode": cfg.talk_mode,
        "talk_provider": provider_name(cfg, cfg.talk_mode) or "",
        "voice_source": cfg.voice_source,
        "ignored_legacy_fields": ignored,
        "talk_warning": talk_changed,
        "hint": "已生效于之后的步骤；已生成的产物不会自动重做",
    }


@app.delete("/api/runs/{run_id}/final")
def delete_final(run_id: str) -> dict:
    """删除成片（`final/final.mp4` + 旁挂字幕），让「拼接成片」重新变成缺口。

    为什么单独一个端点而不是复用 artifacts 删除：成片的文件名（`final.mp4` /
    `subtitles.srt`）不匹配那条路由的白名单（只收 char_*/scene_*/shot_*），
    而且删成片要连带把 QC 报告与终审结论一起清掉 —— 留着旧报告会让人以为
    "这条（已经不存在的）成片是合格的"。
    """
    from fastvideo3.modules.m7_edit import SUBTITLE_SIDECAR_NAME

    root = _safe_run_root(run_id)
    _require_idle("删除成片")
    storage = RunStorage(run_id, root=root.parent)
    removed: list[str] = []
    for rel in ("final/final.mp4", f"final/{SUBTITLE_SIDECAR_NAME}", "final/qc_report.json"):
        p = root / rel
        if p.exists():
            from fastvideo3.core.storage import safe_unlink

            if safe_unlink(p):
                removed.append(rel)
    if not removed:
        raise HTTPException(status_code=404, detail="这条 run 还没有成片")
    storage.log_event(
        "m7_edit", "final_deleted", removed=removed,
        hint="删除后「拼接成片」回到缺口，点开始生成即可零 API 费用重合成（素材都在盘上）",
    )
    return {"run_id": run_id, "removed": removed}


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
                # 面板只认分辨率；老 run 没有 resolution_tier 字段 → 由档位反推，
                # 免得列表里新旧 run 显示口径不一致
                "resolution": meta.get("resolution_tier")
                or resolution_for_quality(str(meta.get("quality", "economy"))),
                "status": _run_status(root),
                "cost_total_cny": _read_cost(root)["total_cny"],
                "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                if mtime
                else "",
            }
        )
    return summaries


def _clean_model_overrides(
    raw: dict[str, str], quality: str, *, fake: bool = False
) -> dict[str, str]:
    """校验 run 级模型选择，非法就 400（列清可选值）—— 不等到提交 API 才 400 才发现打错名。

    三道检查，都对应真实踩过的坑：
    1. 能力是否在允许清单内（口型通道**不在这里**选：模式即通道，见 `core/talkmode.py`）；
    2. 模型名是否在目录内（`core/model_catalog.py`）；
    3. 模型名能否解析出调用协议 —— t2i/i2v 的 provider 键同时决定端点与请求体，
       名字对但协议解析不出来 = 打到错误端点直接 400（`.env` 换模型时踩过的坑）；
    4. high/cinema 强制「参考图锁脸」，选了不支持参考图的 t2i 会在 m3 花钱前报错，
       那就在这里先说。
    """
    from fastvideo3.core.model_catalog import (
        SELECTABLE_CAPS,
        configured_models,
        protocol_error,
        ref_image_warning,
        validate,
    )

    cfg_now = load_config(quality=quality)
    # 目录之外、但**当前配置正在用**的模型名也要放行：面板的默认值就是从配置读的，
    # 若目录恰好没收录它，就会"界面显示着、一保存就 400"（自相矛盾比清单不全更糟）。
    extra = set(configured_models(cfg_now).values())

    errors: list[str] = []
    clean: dict[str, str] = {}
    for cap, model in (raw or {}).items():
        name = (model or "").strip()
        if cap not in SELECTABLE_CAPS:
            errors.append(validate(cap, name) or f"不支持的能力 {cap}")
            continue
        err = (
            validate(cap, name, extra_allowed=extra)
            or protocol_error(cap, name, cfg_now, check_keys=not fake)
            or ref_image_warning(cap, name, quality)
        )
        if err:
            errors.append(err)
            continue
        if name:
            clean[cap] = name
    if errors:
        raise HTTPException(status_code=400, detail="；".join(errors))
    return clean


@app.post("/api/runs", status_code=201)
def create_run(req: RunCreate) -> dict:
    """新建任务：只落 run_meta.json（列表立即可见），**不启动任何生成**。

    生成由详情页的「开始生成」一步一步推进（`POST /step`）—— 每一步之间人都在场，
    可以改剧本、换定妆照、删图重生成，不必为"一把跑完"付一次不可回头的钱。
    因此这里也不取全局任务锁：建任务不花钱，锁只该拦真正烧钱的 `/step`。
    """
    # 面板选的是**分辨率**；给了它就由它决定内部档位（单一映射在 config.RESOLUTION_TO_QUALITY）。
    # 两者都给时以分辨率为准 —— 否则会出现"面板显示 720P、后端按 economy 跑"这种对不上。
    from fastvideo3.core.config import (
        apply_resolution,
        quality_for_resolution,
        resolution_for_quality,
    )

    quality = quality_for_resolution(req.resolution, req.quality) if req.resolution else req.quality
    resolution_tier = req.resolution or resolution_for_quality(quality)
    try:
        cfg = load_config(quality=quality, ablation=req.ablation or None)
    except ValueError as e:  # 消融路径越界/不存在 → 400 而非 500
        raise HTTPException(status_code=400, detail=str(e)) from e
    # 分辨率标签必须真的决定成片尺寸与 I2V 档位（档位 yaml 只管一致性深度/抽卡/单价量级）
    cfg.aspect_ratio = req.aspect
    apply_resolution(cfg, resolution_tier)
    # run 级模型选择：校验后**立刻**盖到 cfg（预估、以及之后每一步都从 run_meta 恢复同一份）
    chosen = _clean_model_overrides(req.model_overrides, quality, fake=req.fake)
    # 口型不再是选项：wan3.0 场景片段自带台词口型。老请求体里的 talk_mode/talk_model/
    # voice_source 接受但忽略（字段上注明），忽略要留痕 —— 静默丢弃会让老调用方以为生效了。
    ignored_legacy = [
        f for f in ("talk_mode", "talk_model", "voice_source")
        if getattr(req, f, None) not in (None, "")
    ]
    cfg.talk_mode = "scripted_talk"
    if chosen:
        # 同 PATCH：面板点名的 i2v 就是场景生成/口型模型，两边一起落，别让 .env 的旧
        # FV3_TALK_MODEL 赢过用户刚选的那一档
        if "i2v" in chosen:
            chosen = {**chosen, "talk": chosen["i2v"]}
        cfg.model_overrides = {**cfg.model_overrides, **chosen}
    # 并把选择**写回 .env**，成为之后新建任务的默认 —— 多数用户只用面板、不碰 .env，
    # 不写回就会出现"面板选了 A、面板下面跑的是 B"（`.env` 优先级高于 yaml）。
    # Fake 模式不写：那条路一个模型都不调用，写回只会污染真实配置。
    env_written: list[str] = []
    env_error = ""
    if chosen and not req.fake:
        from fastvideo3.core.envfile import update_env_models

        env_written, err = update_env_models(chosen)
        env_error = err or ""
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
        # 口型恒为 wan3.0 场景生成自带（scripted_talk）；音轨恒为模型原声（嘴型与声音同轨）
        talk_mode="scripted_talk",  # type: ignore[arg-type]
        voice_source="model",
        ignored_legacy_fields=ignored_legacy,
        # budget=0 表示"跟随 .env"：meta 落**实际生效**值。落 0 会让看板的熔断显示 ¥0、
        # 成本红幅永不触发（`exceeded()` 用 `>=`，0 预算还会让第一步就跳闸）
        budget_limit_cny=req.budget if req.budget > 0 else cfg.budget_limit_cny,
        fake=req.fake,
        quality=quality,
        resolution_tier=resolution_tier,
        ablation=cfg.ablation,
        # 只落**用户点名**的那些能力（空＝跟随画质档位）；续跑由 config.apply_run_overrides
        # 原样恢复 —— 每步必须用同一个模型，否则资产是新模型出的、首帧是旧模型出的
        model_overrides=chosen,
        # 写回 .env 的留痕（改的是机器上的全局配置，必须可追溯）
        model_env_written=env_written,
        model_env_error=env_error,
    )
    RunStorage(run_id, root=_out()).save_json(
        "run_meta.json", project.model_dump(mode="json")
    )
    return {
        "run_id": run_id,
        "hint": "任务已建，点「开始生成」按顺序推进",
        # 面板选的模型同时写回了 .env（成为之后新建任务的默认）；写不回也不影响本次任务
        "env_written": env_written,
        "env_error": env_error,
    }


def _url(run_id: str, stage: str, name: str) -> str:
    return f"/api/files/{run_id}/{stage}/{name}"


def _artifact(root: Path, run_id: str, stage: str, name: str) -> str | None:
    """存在才给 URL：run 可能停在任意阶段，前端要能渲染半成品。

    URL 带文件 mtime 版本号：换定妆照/上传替换后路径不变，浏览器会拿缓存的旧图，
    表现为"点了设为定妆照但左边没变"。mtime 进 query 后每次替换都强制重新拉取。
    """
    p = root / stage / name
    if not p.is_file():
        return None
    try:
        v = int(p.stat().st_mtime)
    except OSError:
        v = 0
    return f"{_url(run_id, stage, name)}?v={v}"


@app.post("/api/runs/{run_id}/reconcile-cost")
def reconcile_run_cost(run_id: str) -> dict:
    """用**真实账单**校正这条 run 的台账（账单口径优先，台账只负责解释钱花在哪个能力上）。

    台账金额 = 单价 × 用量。用量已优先取 API 回传的 usage，但单价那一侧永远可能失真
    —— 2026-10-12 那条 run 就是极端例子：合计 ¥6.30，而 i2v/talk/text 全按 ¥0 记，
    账单却在正常扣。只有 BSS 账单能揭穿这种失真。

    边界（必须回给前端一起显示，不能装作是精确值）：账单粒度是**账户×月×产品**，
    本 run 的金额是按同月台账占比**分摊**出来的；未配 RAM AK/SK 时直接返回原因，
    绝不返回 ¥0 冒充"这个月没花钱"。
    """
    from fastvideo3.core import billing

    root = _safe_run_root(run_id)
    if not (root / "cost.json").is_file():
        raise HTTPException(status_code=404, detail="这条 run 还没有成本台账")
    data = billing.reconcile_run(root, output_dir=_out())
    billing.save_reconciliation(root, data)
    return data


def _scene_prompt_preview(cfg: Any, group: Any, script: Any, ov: dict) -> str:
    """参考形态下"将要发送"的提示词预览（编号与实发素材同序）。"""
    from fastvideo3.core.scenegroup import plan_references
    from fastvideo3.modules.m4_scene import scene_prompt

    speakers = tuple(dict.fromkeys(
        d.character_id for sh in script.all_shots()
        if sh.shot_id in set(group.shot_ids) for d in sh.dialogues
    ))
    plan = plan_references(group, speakers=speakers, scene_ids=tuple(group.scene_ids))
    return scene_prompt(cfg, group, script, script.character_map(), plan)


def _scene_groups_payload(run_id: str, root: Path) -> dict:
    """分镜视频界面的数据源：每个场景组的模式、可选素材、首尾帧提示词与产物状态。

    素材清单只列**盘上已存在**的图（角色页/场景页的产物，含落选候选）—— 界面只允许
    "选已有的图"，不允许在这里上传或改图：改图是角色页的职责，在这里开放编辑会出现
    "这条片段锁的脸和角色页显示的脸不是同一张"的分裂。
    """
    from fastvideo3.core.assets import refs_on_disk
    from fastvideo3.core.scenegroup import choose_input_mode, group_scenes
    from fastvideo3.modules.m4_scene import _first_last_prompts

    meta = _load_meta(root) or {}
    cfg = _cfg_for_run(root)
    script = _script_or_none(root)
    if script is None:
        return {"groups": []}
    overrides = meta.get("scene_overrides") or {}
    assets_dir = root / "assets"
    char_imgs: dict[str, list[Any]] = {}
    for p in sorted(assets_dir.glob("char_*.png")) if assets_dir.is_dir() else []:
        if p.name.endswith(".bak.png"):
            continue
        cid = p.stem[len("char_"):]
        char_imgs.setdefault(cid, []).append(_url(run_id, "assets", p.name))
    cand_imgs: dict[str, list[Any]] = {}
    cand_rel: dict[str, list[str]] = {}
    for p in sorted(assets_dir.glob("_cand_char_*.png")) if assets_dir.is_dir() else []:
        rest = p.stem[len("_cand_char_"):]
        cid = rest.rsplit("_", 1)[0]
        cand_imgs.setdefault(cid, []).append(_url(run_id, "assets", p.name))
        cand_rel.setdefault(cid, []).append(f"assets/{p.name}")
    scene_imgs: dict[str, str] = {}
    for p in sorted(assets_dir.glob("scene_*.png")) if assets_dir.is_dir() else []:
        scene_imgs[p.stem[len("scene_"):]] = _url(run_id, "assets", p.name)
    frames_dir = root / "storyboard"
    scene_by_id = {s.scene_id: s for s in script.scenes}
    shot_by_id = {sh.shot_id: sh for sh in script.all_shots()}
    storage = RunStorage(run_id, root=root.parent)
    groups = []
    for g in group_scenes(script, max_clip_sec=30.0):
        ov = overrides.get(g.group_id) or {}
        speakers = {
            d.character_id
            for sh in script.all_shots()
            if sh.shot_id in set(g.shot_ids)
            for d in sh.dialogues
        }
        first_shot, last_shot = g.shot_ids[0], g.shot_ids[-1]
        first_rel = f"storyboard/shot_{first_shot}.png"
        last_rel = f"storyboard/shot_{last_shot}_last.png"
        prompts = _first_last_prompts(cfg, g, script)
        # 参考素材的物理前提：**盘上真有图**。只看 asset_registry.json 会把"用户直接上传
        # 的角色图/场景图"看成不存在 —— 于是界面显示参考形态、生成时却发 0 张素材，
        # 请求被本地规格守卫拦下，表现就是"点了生成毫无反应"（实测事故）。
        has_refs = bool(
            refs_on_disk(storage, characters=g.characters, scene_ids=g.scene_ids, overrides=ov)
        )
        mode, why = choose_input_mode(
            g,
            has_dialogue=bool(speakers),
            has_first_frames=(frames_dir / f"shot_{first_shot}.png").is_file(),
            has_references=has_refs,
            forced=str(ov.get("mode") or cfg.scene_clip_mode or "auto"),
        )
        # 场景标题/地点/氛围挪进场景卡片：这一段戏"是谁在哪干什么"只该出现在它真正的
        # 生成单元上，逐镜头卡片已经把这件事割裂成"每镜复述一遍场景名"。
        labels = [
            f"{scene_by_id[s].title} · {scene_by_id[s].location}"
            for s in g.scene_ids if s in scene_by_id
        ]
        summaries = [
            scene_by_id[s].summary for s in g.scene_ids
            if s in scene_by_id and scene_by_id[s].summary
        ]
        groups.append(
            {
                "group_id": g.group_id,
                "scene_ids": list(g.scene_ids),
                "shot_ids": list(g.shot_ids),
                "windows": {k: list(v) for k, v in g.windows.items()},
                "duration": g.duration,
                "characters": list(g.characters),
                "has_dialogue": bool(speakers),
                "has_references": has_refs,
                "scene_label": " ／ ".join(labels),
                "scene_summary": " ".join(summaries),
                "location": g.location,
                "time_of_day": g.time_of_day,
                "shots": [
                    {
                        "shot_id": sid,
                        "type": shot_by_id[sid].type.value if hasattr(shot_by_id[sid].type, "value") else str(shot_by_id[sid].type),
                        "duration_sec": shot_by_id[sid].duration_sec,
                        "characters": list(shot_by_id[sid].characters),
                        "action": shot_by_id[sid].action,
                        "camera": shot_by_id[sid].camera,
                        "dialogues": [
                            {"who": d.character_id, "text": d.text, "emotion": d.emotion}
                            for d in shot_by_id[sid].dialogues
                        ],
                        "window": list(g.windows.get(sid) or ()),
                    }
                    for sid in g.shot_ids if sid in shot_by_id
                ],
                "auto_mode": mode if not ov.get("mode") else choose_input_mode(
                    g, has_dialogue=bool(speakers),
                    has_first_frames=(frames_dir / f"shot_{first_shot}.png").is_file(),
                    has_references=has_refs,
                )[0],
                "mode": mode,
                # **下拉框要显示"用户选的那个"，不是"系统算出来的那个"**：
                # 之前 select 绑的是 `mode`（受物理前提影响算出的生效形态）。用户手工选了
                # "首尾帧"、但本组还没有首帧 → 后端算成 reference 并在盘上按 first_last 计划
                # 需求（frame_needs 假设帧可以画），界面却显示"角色+场景生成" —— 于是出现
                # "我明明选了角色+场景，为什么还要先画首帧"的死循环式困惑（2026-10-14 实测）。
                "override_mode": str(ov.get("mode") or "auto"),
                "mode_reason": why,
                "clip": _url(run_id, "clips", f"scene_{g.group_id}.mp4")
                if (root / "clips" / f"scene_{g.group_id}.mp4").is_file() else None,
                "clip_stale": bool((root / "clips" / f"scene_{g.group_id}.mp4").is_file())
                and RunStorage(run_id, root=root.parent).is_stale(f"clips/scene_{g.group_id}.mp4"),
                "first_frame": _url(run_id, "storyboard", f"shot_{first_shot}.png")
                if (root / first_rel).is_file() else None,
                "last_frame": _url(run_id, "storyboard", f"shot_{last_shot}_last.png")
                if (root / last_rel).is_file() else None,
                "prompt_first": str(ov.get("prompt_first") or prompts[0]),
                "prompt_last": str(ov.get("prompt_last") or prompts[1]),
                # 参考形态的整段提示词也给人看、给人改：机器拼的模板是默认值不是上限
                "prompt": str(ov.get("prompt") or _scene_prompt_preview(cfg, g, script, ov)),
                "char_options": {
                    cid: {
                        "portrait": (char_imgs.get(cid) or [None])[0],
                        "portrait_rel": f"assets/char_{cid}.png",
                        "candidates": cand_imgs.get(cid, []),
                        "candidates_rel": cand_rel.get(cid, []),
                    }
                    for cid in g.characters
                },
                "scene_options_rel": {sid: f"assets/scene_{sid}.png" for sid in g.scene_ids},
                "char_refs": list(ov.get("char_refs") or g.characters),
                "char_images": dict(ov.get("char_images") or {}),
                "scene_options": {sid: scene_imgs.get(sid) for sid in g.scene_ids},
                "scene_refs": list(ov.get("scene_refs") or g.scene_ids),
            }
        )
    return {"groups": groups}


@app.post("/api/runs/{run_id}/scene-groups/{group_id}/generate", status_code=202)
def generate_scene_group(run_id: str, group_id: str) -> dict:
    """只重出**这一个**场景组（界面「生成」按钮）。

    走子进程 + 全局任务锁，与「开始生成」同一套防重复付费机制：场景片段一次调用
    就是 ¥6–36（720P/1080P × 10–30s），连点两下烧双份钱是不可接受的。
    不整步重跑：整步会把别的组也按缺口补一遍，而用户此刻只想看他改过的这一组。
    """
    root = _safe_run_root(run_id)
    _require_idle("生成场景片段")
    from fastvideo3.core import tasklock

    tasklock.acquire(_out(), run_id, "step")
    cmd = [sys.executable, "-m", "fastvideo3.app.cli", "scene-gen", run_id, "--group", group_id]
    _spawn(cmd, root / "server_spawn.log")
    _events_append(root, "web_api", "scene_group_generate", group=group_id)
    return {"ok": True, "group": group_id, "spawned": cmd[-2:]}


@app.get("/api/runs/{run_id}/scene-groups")
def scene_groups(run_id: str) -> dict:
    """分镜视频界面：场景组清单 + 每组的模式/素材/首尾帧提示词/产物状态。"""
    root = _safe_run_root(run_id)
    return _scene_groups_payload(run_id, root)


class SceneGroupPatch(BaseModel):
    mode: Literal["", "auto", "reference", "first_last"] = ""
    char_refs: list[str] = Field(default_factory=list, max_length=10)
    scene_refs: list[str] = Field(default_factory=list, max_length=10)
    char_images: dict[str, str] = Field(default_factory=dict, max_length=10)
    prompt_first: str = ""
    prompt_last: str = ""
    prompt: str = ""


@app.patch("/api/runs/{run_id}/scene-groups/{group_id}")
def patch_scene_group(run_id: str, group_id: str, req: SceneGroupPatch) -> dict:
    """改某一组的 image2video 形态与素材选择，存进 run_meta.scene_overrides。

    校验只做"物理上可能"的事：素材必须是本组角色/场景盘上已有的图（路径穿越与
    别的角色的图都拒），图片总数 ≤10（官方上限），首尾帧形态的提示词不能为空
    （空提示词 + 首尾帧 = 服务端 400，官方明写首尾帧模式提示词必填）。
    """
    import re as _re

    from fastvideo3.core.scenegroup import MAX_REF_IMAGES, group_scenes

    root = _safe_run_root(run_id)
    meta = _load_meta(root) or {}
    script = _script_or_none(root)
    if script is None:
        raise HTTPException(status_code=404, detail="剧本尚未生成")
    group = next((g for g in group_scenes(script) if g.group_id == group_id), None)
    if group is None:
        raise HTTPException(status_code=404, detail=f"场景组不存在：{group_id}")
    ov = dict((meta.get("scene_overrides") or {}).get(group_id) or {})
    if req.mode:
        ov["mode"] = req.mode
    if req.char_refs:
        bad = [c for c in req.char_refs if c not in group.characters]
        if bad:
            raise HTTPException(status_code=400, detail=f"这些角色不在本组：{bad}")
        ov["char_refs"] = req.char_refs
    if req.scene_refs:
        bad = [s for s in req.scene_refs if s not in group.scene_ids]
        if bad:
            raise HTTPException(status_code=400, detail=f"这些场景不在本组：{bad}")
        ov["scene_refs"] = req.scene_refs
    if req.char_images:
        for cid, rel in req.char_images.items():
            if cid not in group.characters:
                raise HTTPException(status_code=400, detail=f"角色不在本组：{cid}")
            if not _re.fullmatch(r"assets/(char_|_cand_char_)[A-Za-z0-9_\-]+\.png", rel):
                raise HTTPException(status_code=400, detail=f"非法素材路径：{rel}")
            if not (root / rel).is_file():
                raise HTTPException(status_code=404, detail=f"素材不在盘上：{rel}")
        ov["char_images"] = req.char_images
    if req.prompt_first.strip():
        ov["prompt_first"] = req.prompt_first.strip()
    if req.prompt_last.strip():
        ov["prompt_last"] = req.prompt_last.strip()
    if req.prompt.strip():
        ov["prompt"] = req.prompt.strip()
    n_imgs = len(ov.get("char_refs") or group.characters) + len(
        ov.get("scene_refs") or group.scene_ids
    )
    if n_imgs > MAX_REF_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"参考图共 {n_imgs} 张，超过官方上限 {MAX_REF_IMAGES} 张",
        )
    overrides = dict(meta.get("scene_overrides") or {})
    overrides[group_id] = ov
    meta["scene_overrides"] = overrides
    RunStorage(run_id, root=root.parent).save_json("run_meta.json", meta)
    _events_append(root, "web_api", "scene_group_patched", group=group_id,
                   mode=ov.get("mode", ""), n_refs=n_imgs)
    return _scene_groups_payload(run_id, root)


@app.post("/api/runs/{run_id}/scene-groups/{group_id}/frames")
async def generate_scene_frames(run_id: str, group_id: str) -> dict:
    """按（可编辑的）提示词生成首帧/尾帧图 —— 首尾帧形态的前置步骤。

    产物落在 storyboard/ 的既有约定位置（`shot_{首镜}.png` / `shot_{末镜}_last.png`），
    与 m3 的逐镜头产物同位同血缘：换立绘会连带作废它们，重跑 frames 步也会补它们。
    提示词以 run_meta.scene_overrides 里编辑过的版本为准（没编辑就用自动生成的）。
    """
    from fastvideo3.core.cost import CostLedger
    from fastvideo3.core.scenegroup import group_scenes
    from fastvideo3.modules import m4_scene
    from fastvideo3.providers.base import Capability
    from fastvideo3.providers.factory import make_provider

    root = _safe_run_root(run_id)
    _require_idle("生成首尾帧")
    meta = _load_meta(root) or {}
    cfg = _cfg_for_run(root)
    script = _script_or_none(root)
    if script is None:
        raise HTTPException(status_code=404, detail="剧本尚未生成")
    group = next((g for g in group_scenes(script) if g.group_id == group_id), None)
    if group is None:
        raise HTTPException(status_code=404, detail=f"场景组不存在：{group_id}")
    ov = (meta.get("scene_overrides") or {}).get(group_id) or {}
    prompts = m4_scene._first_last_prompts(cfg, group, script)
    p_first = str(ov.get("prompt_first") or prompts[0]).strip()
    p_last = str(ov.get("prompt_last") or prompts[1]).strip()
    if not p_first or not p_last:
        raise HTTPException(status_code=400, detail="首帧/尾帧提示词不能为空（官方：首尾帧模式提示词必填）")
    try:
        t2i = make_provider(Capability.T2I, cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"文生图供应商不可用：{e}") from e
    storage = RunStorage(run_id, root=root.parent)
    ledger = CostLedger(storage, budget_limit_cny=float(meta.get("budget_limit_cny") or 0) or 150.0)
    made = await m4_scene.generate_group_frames(
        cfg, t2i, group, script, storage, ledger, p_first, p_last
    )
    return {"group_id": group_id, "made": made,
            "first_frame": _url(run_id, "storyboard", f"shot_{group.shot_ids[0]}.png"),
            "last_frame": _url(run_id, "storyboard", f"shot_{group.shot_ids[-1]}_last.png")}


@app.get("/api/runs/{run_id}/board")
def production_board(run_id: str) -> dict:
    """制作看板：剧本 + 角色定妆照 + 场景概念图 + 逐镜头(首帧/片段/口型/一致性) + 成片。

    "产物 ↔ 镜头/角色"的关联全部在服务端按目录命名约定 join
    （storyboard/shot_{id}.png、clips|talk/shot_{id}.mp4、assets/char_{cid}.png）。
    此前接口只回一串 URL 列表，前端画廊看不出"哪张图属于哪个镜头、角色说了什么"，
    等于把剧本/分镜的信息全丢了。
    """
    from fastvideo3.modules.m2_assets import scene_background_prompt

    root = _safe_run_root(run_id)
    meta = _load_meta(root) or {}
    consistency: dict = {}
    if (root / "consistency_report.json").is_file():
        try:
            consistency = json.loads((root / "consistency_report.json").read_text("utf-8"))
        except json.JSONDecodeError:
            consistency = {}
    per_shot_metric = {r.get("shot"): r for r in consistency.get("per_shot", [])}
    per_char_metric = consistency.get("per_char") or {}

    uploads = _read_json(root / "uploads.json") or {}
    board: dict[str, Any] = {
        "run_id": run_id,
        "meta": meta,
        "title": "",
        "available": False,
        "characters": [],
        "scenes": [],
        "shot_count": 0,
        "final_video": _artifact(root, run_id, "final", "final.mp4"),
        # 旁挂字幕与成片**不同基名**（同名会被播放器自动再挂一层 → 两层字幕）
        "subtitle": _artifact(root, run_id, "final", "subtitles.srt")
        or _artifact(root, run_id, "final", "final.srt"),  # 旧 run 兼容
        "consistency": consistency,
        "degradations": _read_json(root / "degradations.json") or [],
        "uploads": uploads,  # 人工上传台账：前端据此标记"这张是用户上传的"
        "quality_verdict": (_read_json(root / "run_summary.json") or {}).get("quality_verdict"),
        # 技术 QC（m7 落盘的 final/qc_report.json）：黑帧/静帧/静音/时长偏差/黑边/规格合规。
        # 放在**初始 dict** 里（不在"有剧本才走到"的分支）：没有剧本的 run 也该看得到
        # 上一轮成片为什么被判不合格。这一层是确定性数值，与 VLM 终审分开显示 ——
        # m8 解析失败时会落 0.6 兜底分并 passed=True，把异常写成"通过"，谁也别替谁说话。
        "tech_qc": _read_json(root / "final" / "qc_report.json"),
    }
    script_json = _read_json(root / "script.json")
    if not script_json:  # 只跑了 intake/尚未出剧本
        base = meta.get("base_info") or {}
        board["title"] = base.get("topic", "")
        return board
    try:
        script = Script.model_validate(script_json)
    except Exception:  # noqa: BLE001  # 老/坏剧本降级为"只给原始 JSON"，不让整个面板 500
        board["raw_script"] = script_json
        return board

    char_names = {c.character_id: c.name for c in script.characters}
    board.update({
        "available": True,
        "title": script.title or script.base_info.topic,
        # 剧本版本号：前端据此判断"剧本在别处变了"（如按 prompt 重写完成），
        # 该重新载入正开着的剧本稿编辑器，而不是继续显示旧内容
        "script_version": script.version,
    })
    # 成本预估（**按剧本实况的对白秒数**，不是经验占比）：口型通道换成 fal InfiniteTalk
    # 后单价涨了 25~50 倍，"熔断够用"不再能靠旧量级直觉判断。预估低于熔断时前端给红条，
    # 免得用户像 2026-10-04 那次一样：钱花在运动片段上、口型步骤跳闸、成片依旧没口型。
    from fastvideo3.core.cost import estimate_run_cost_cny

    cfg_run = _cfg_for_run(root)
    total_sec = sum(float(sh.duration_sec or 0) for sh in script.all_shots())
    talk_sec = sum(float(sh.duration_sec or 0) for sh in script.all_shots() if sh.dialogues)
    board["cost_estimate"] = estimate_run_cost_cny(cfg_run, total_sec, dialogue_sec=talk_sec)
    # meta 里落 0 的老 run（历史 bug：`run` 直接写 `--budget` 默认值）要回落到实际生效值，
    # 否则看板显示"熔断 ¥0"、红条永不触发，而真正会跳闸的是 .env 里的 ¥120
    budget_cny = float(meta.get("budget_limit_cny") or 0) or float(cfg_run.budget_limit_cny)
    board["cost_estimate"]["budget_cny"] = budget_cny
    board["cost_estimate"]["too_low"] = bool(
        budget_cny and board["cost_estimate"]["lo"] > budget_cny
    )
    # 口型覆盖率**按盘上实况算**（不读 run_summary 里的旧结论）：合成前后都能看到
    # "还差哪几个对白镜头没口型"，「开始生成」的缺口与这里的数字必然一致（同一个函数）。
    from fastvideo3.core.progress import last_frame_coverage, lipsync_coverage

    _st = RunStorage(run_id, root=root.parent)  # RunStorage 会再拼一层 run_id
    _cfg_run = _cfg_for_run(root)
    _mode = str(meta.get("talk_mode") or _cfg_run.talk_mode)
    # cfg 必须传：口型现在多半在**场景片段**里（wan3.0 自带），只认 talk 产物会把
    # 一部口型正常的成片报成"缺 N 个口型"
    board["lipsync_coverage"] = lipsync_coverage(_st, script, _mode, _cfg_run)
    # 尾帧覆盖率：缺尾帧时 m4 会退回单首帧，而产物照样出得来 —— 完成度判据看不见它，
    # 所以必须在看板上单独说（否则用户的观感就是"首尾帧改造没生效"）。
    board["last_frame_coverage"] = last_frame_coverage(_st, script, _mode, _cfg_for_run(root))
    # 技术 QC 见上面初始 dict 的 "tech_qc"（不依赖剧本，所以不在这里补）
    # 被任何镜头（在场或台词）引用过的角色 id：只在清单里出现、没进任何镜头的角色
    # 照样会花一次定妆照的钱，剧本页签必须把这件事说出来
    used_ids = {
        cid for sh in script.all_shots()
        for cid in set(sh.characters) | {d.character_id for d in sh.dialogues}
    }
    for c in script.characters:
        cid = c.character_id
        candidates = sorted(
            _url(run_id, "assets", p.name)
            for p in (root / "assets").glob(f"_cand_char_{cid}_*.png")
        ) if (root / "assets").is_dir() else []
        board["characters"].append(
            {
                "id": cid,
                "name": c.name,
                "gender": c.gender,
                "age_band": c.age_band,
                "used": cid in used_ids,
                "appearance": c.appearance,
                "voice_id": c.voice_id,
                "look": c.look.model_dump(exclude_defaults=True) if c.look else None,
                "canonical_tokens": c.canonical_tokens,
                "portrait": _artifact(root, run_id, "assets", f"char_{cid}.png"),
                "candidates": candidates,
                "metrics": per_char_metric.get(cid),
                "uploaded": f"assets/char_{cid}.png" in uploads,
            }
        )
    for scene in script.scenes:
        shots = []
        for sh in scene.shots:
            shots.append(
                {
                    "shot_id": sh.shot_id,
                    "type": sh.type.value,
                    "action": sh.action,
                    "camera": sh.camera,
                    "duration_sec": sh.duration_sec,
                    "transition_out": sh.transition_out.value,
                    "characters": [char_names.get(x, x) for x in sh.characters],
                    "dialogues": [
                        {"who": char_names.get(d.character_id, d.character_id),
                         "text": d.text, "emotion": d.emotion}
                        for d in sh.dialogues
                    ],
                    "visual_prompt": sh.visual_prompt,
                    "frame": _artifact(root, run_id, "storyboard", f"shot_{sh.shot_id}.png"),
                    "frame_uploaded": f"storyboard/shot_{sh.shot_id}.png" in uploads,
                    "clip": _artifact(root, run_id, "clips", f"shot_{sh.shot_id}.mp4"),
                    "talk": _artifact(root, run_id, "talk", f"shot_{sh.shot_id}.mp4"),
                    "metric": per_shot_metric.get(sh.shot_id),
                }
            )
        board["scenes"].append(
            {
                "id": scene.scene_id,
                "title": scene.title,
                "location": scene.location,
                "time_of_day": scene.time_of_day,
                "summary": scene.summary,
                "concept": _artifact(root, run_id, "assets", f"scene_{scene.scene_id}.png"),
                "uploaded": f"assets/scene_{scene.scene_id}.png" in uploads,
                # **场景概念图实际会发出的提示词**。面板原本只摆 `summary`（那是剧情梗概，
                # 必然带人名），于是用户据此以为"背景图要画角色"——把真正发出去的提示词
                # 摆出来，空景过滤是对的还是漏了，一眼可核，不需要信任何人的口头保证。
                "bg_prompt": scene_background_prompt(cfg_run, script, scene),
                "shots": shots,
            }
        )
    board["shot_count"] = len(script.all_shots())
    return board


def _read_json(p: Path) -> Any:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ── 人工上传替换 ──────────────────────────────────────────────
_UPLOAD_STAGES = frozenset({"assets", "storyboard"})
# 可删除的阶段：比上传多 clips/talk —— 用于"强制重出单镜头"，典型场景是把
# 无口型的降级片段（m6 failed_fallback → I2V 叠音）删掉重做真口型。
# 对白镜头不在 shots_need_i2v 里，删它的 clip 不会触发 m4 重跑（不重复烧 i2v 的钱）。
_DELETE_STAGES = frozenset({"assets", "storyboard", "clips", "talk"})
_UPLOAD_NAME_RE = re.compile(r"^(?:char_[\w.-]+|scene_[\w.-]+|shot_[\w.-]+|_cand_char_[\w.-]+)\.png$")
# 删除比上传多收 mp4（clips/talk 强制重出单镜头用）
_DELETE_NAME_RE = re.compile(
    r"^(?:char_[\w.-]+|scene_[\w.-]+|shot_[\w.-]+|_cand_char_[\w.-]+)\.(png|mp4)$"
)
_UPLOAD_MAX_BYTES = 12 * 1024 * 1024


def _load_script_dict(root: Path) -> dict:
    try:
        return json.loads((root / "script.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  # 没剧本时不做 id 校验（仍受名称正则约束）
        return {}


def _script_or_none(root: Path):
    """血缘回退需要剧本的引用关系；剧本缺失/损坏时返回 None（只靠血缘）。"""
    data = _load_script_dict(root)
    if not data:
        return None
    try:
        return Script.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


@app.put("/api/runs/{run_id}/artifacts/{stage}/{name}")
async def put_artifact(run_id: str, stage: str, name: str, request: Request) -> dict:
    """上传替换角色图 / 场景图 / 分镜首帧（裸二进制 PUT，不引入 multipart 依赖）。

    做了三件事：① 校验镜像素与目标位（`char_{cid}` 的 cid 必须是剧本里的角色）；
    ② 按本片画幅 cover 裁切归一化 —— 直接塞横图会被下游 9:16 cover 二次裁切毁构图；
    ③ 记 `uploads.json` 台账，让 m2 重跑时**跳过**人工资产而非覆盖它。
    """
    root = _safe_run_root(run_id)
    if stage not in _UPLOAD_STAGES:
        raise HTTPException(status_code=400, detail=f"stage 只能是 {sorted(_UPLOAD_STAGES)}")
    if not _UPLOAD_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="文件名必须是 char_*/scene_*/shot_*.png")
    body = await request.body()
    if not 0 < len(body) <= _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"文件大小需 1-{_UPLOAD_MAX_BYTES // 1024 // 1024}MB")

    import io

    from PIL import Image

    from fastvideo3.modules.m2_assets import size_for_aspect

    try:
        src = Image.open(io.BytesIO(body))
        src.load()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"不是可读图片: {e}") from e

    meta = _load_meta(root) or {}
    cfg = load_config(quality=meta.get("quality") or None)
    # 目标位校验：角色/场景 id 必须存在于剧本
    script = _load_script_dict(root)
    ids = {
        "char_": {f"char_{c.get('character_id')}.png" for c in script.get("characters", [])},
        "scene_": {f"scene_{s.get('scene_id')}.png" for s in script.get("scenes", [])},
        "shot_": {
            f"shot_{sh.get('shot_id')}.png"
            for s in script.get("scenes", [])
            for sh in s.get("shots", [])
        },
    }
    # 候选图（_cand_char_c1_0.png）不属于剧本清单，但角色 id 必须对得上
    prefix = next((p for p in ids if name.startswith(p)), None)
    if prefix and script and name not in ids[prefix]:
        if not (
            prefix == "char_"
            and any(name.startswith(f"_cand_char_{c.get('character_id')}_")
                    for c in script.get("characters", []))
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{name} 不在剧本产物清单里（可上传："
                f"{sorted(ids[prefix])[:5]}{'…' if len(ids[prefix]) > 5 else ''}）",
            )

    want = size_for_aspect(
        (meta.get("base_info") or {}).get("aspect_ratio", cfg.aspect_ratio),
        hi_res=(cfg.resolution[0] >= 1080),
    )
    tw, th = (int(x) for x in want.split("*"))
    img = src.convert("RGB")
    scale = max(tw / img.width, th / img.height)  # cover：填满目标框后居中裁切
    img = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (img.width - tw) // 2
    top = (img.height - th) // 2
    img = img.crop((left, top, left + tw, top + th))

    base = (root / stage).resolve()
    dest = (base / name).resolve()
    if not dest.is_relative_to(base):
        raise HTTPException(status_code=400, detail="非法路径")
    if dest.is_file():
        # 备份用复制而非 rename：Windows 上该文件可能正被 /api/files 请求或杀软握住
        if not safe_promote(dest, base / (dest.stem + ".bak.png")):
            raise HTTPException(
                status_code=409,
                detail="旧图正被占用，无法备份；请关闭预览该图的页面标签后重试",
            )
    base.mkdir(parents=True, exist_ok=True)
    try:
        img.save(dest, format="PNG")
    except OSError as e:
        raise HTTPException(status_code=409, detail=f"目标文件被占用，写入失败：{e}") from e

    manifest = _read_json(root / "uploads.json") or {}
    manifest[f"{stage}/{name}"] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_size": [src.width, src.height],
        "normalized_to": want,
        "backed_up": (base / (dest.stem + ".bak.png")).is_file(),
    }
    storage_save = root / "uploads.json"
    storage_save.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _events_append(root, "web_api", "artifact_uploaded", stage=stage, name=name, size=want)
    # 沿血缘精确作废下游：换定妆照 → 只重做消费过它的镜头；换场景图 → 只重做该场景。
    # 新上传的文件本身不是 stale（它就是新真相），include_self=False。
    from fastvideo3.core import progress as prog

    storage = RunStorage(run_id, root=_out())
    invalidated = prog.invalidate_artifact(
        storage, f"{stage}/{name}", reason="artifact_replaced",
        include_self=False, script=_script_or_none(root),
    )
    return {
        "ok": True,
        "url": _url(run_id, stage, name),
        "normalized_to": want,
        "backed_up": manifest[f"{stage}/{name}"]["backed_up"],
        "invalidated": invalidated,
        "stale_stages": sorted({i.split("/")[0] for i in invalidated}),
        "hint": "点「开始生成」只重做被作废的产物；其他镜头/场景不会重复花钱",
    }


@app.get("/api/runs/{run_id}/script")
def get_script_raw(run_id: str) -> Any:
    """剧本原文（供前端编辑器预填）。board 只回 join 后的展示字段，编辑需要全量。"""
    root = _safe_run_root(run_id)
    data = _read_json(root / "script.json")
    if data is None:
        raise HTTPException(status_code=404, detail="剧本尚未生成")
    return data


@app.get("/api/runs/{run_id}/uploads")
def list_uploads(run_id: str) -> dict:
    """人工上传清单（前端据此显示"这张是用户上传的"标记）。"""
    root = _safe_run_root(run_id)
    return {"uploads": _read_json(root / "uploads.json") or {}}


class PortraitPick(BaseModel):
    """把某张落选候选立绘设为该角色的正式定妆照（参考项目的 ✔ 人工挑选）。"""

    candidate: str = Field(min_length=1, max_length=120)


@app.post("/api/runs/{run_id}/characters/{char_id}/portrait", status_code=200)
def pick_portrait(run_id: str, char_id: str, req: PortraitPick) -> dict:
    """人工指定定妆照：立绘是全片锚点，崩一张会被 m3 参考图/r2v 继承放大。

    只允许 `assets/_cand_char_{char_id}_*.png` → `assets/char_{char_id}.png`，
    旧图先备份为 `.bak.png`（可反复改选）。改完需重跑 storyboard 及下游才生效。
    """
    root = _safe_run_root(run_id)
    if not _NAME_RE.match(char_id) or not _NAME_RE.match(req.candidate):
        raise HTTPException(status_code=400, detail="非法名称")
    expected_prefix = f"_cand_char_{char_id}_"
    if not req.candidate.startswith(expected_prefix) or not req.candidate.endswith(".png"):
        raise HTTPException(status_code=400, detail=f"候选名必须是 {expected_prefix}N.png")
    base = (root / "assets").resolve()
    src = (base / req.candidate).resolve()
    dest = (base / f"char_{char_id}.png").resolve()
    if not src.is_file() or not src.is_relative_to(base):
        raise HTTPException(status_code=404, detail="候选图不存在")
    if dest.is_file() and not safe_promote(dest, base / f"char_{char_id}.bak.png"):
        raise HTTPException(status_code=409, detail="旧定妆照正被占用，无法备份；请关闭预览页后重试")
    # 复制而非 rename：候选图留在盘上当"落选候选"可回看，且 rename 在 Windows
    # 上会被 /api/files 请求或杀软瞬时握住建死（实测打死过整条 run）
    if not safe_promote(src, dest):
        raise HTTPException(status_code=409, detail="候选图正被占用，复制失败；请稍后重试")
    # 换锚点必须作废下游：否则旧定妆照画出的首帧/片段会被"文件在盘即跳过"保留，
    # 成片里混着旧脸（与上传替换同一套血缘逻辑）
    from fastvideo3.core import progress as prog

    storage = RunStorage(run_id, root=_out())
    rel = f"assets/{dest.name}"
    invalidated = prog.invalidate_artifact(
        storage, rel, reason="portrait_picked", include_self=False,
        script=_script_or_none(root),
    )
    _events_append(root, "web_api", "portrait_picked", char=char_id, pick=req.candidate,
                   invalidated=len(invalidated))
    return {
        "ok": True,
        "char_id": char_id,
        "portrait": _artifact(root, run_id, "assets", dest.name),
        "invalidated": invalidated,
        "hint": "已作废下游产物；点「开始生成」只重做消费过旧脸的镜头",
    }


# ───────────────────────── 开始生成（一次一步） ─────────────────────────
@app.get("/api/runs/{run_id}/steps")
def get_steps(run_id: str) -> dict:
    """五步进度视图：每步 done/gap/pending/active/failed + 缺口清单 + 花费。"""
    from fastvideo3.core import progress as prog
    from fastvideo3.core import tasklock

    root = _safe_run_root(run_id)
    meta = _load_meta(root) or {}
    cfg = load_config(quality=meta.get("quality"), ablation=meta.get("ablation") or None)
    lock = tasklock.status(_out())
    running = bool(lock.get("busy")) and lock.get("run_id") == run_id
    return prog.compute_progress(RunStorage(run_id, root=_out()), meta, cfg, running=running).to_dict()


@app.post("/api/runs/{run_id}/step", status_code=202)
def run_step(run_id: str) -> dict:
    """点一次「开始生成」：只跑最靠前的未完成步骤（后台子进程，一次一步）。"""
    from fastvideo3.core import progress as prog
    from fastvideo3.core import tasklock

    root = _safe_run_root(run_id)
    # API 侧先持锁再 spawn：spawn 到子进程建锁之间有约 1 秒空窗，不先持锁则
    # 连点两次会 spawn 两个子进程、同阶段跑两遍烧双份钱。子进程会认领这把锁。
    ok, st = tasklock.acquire(_out(), run_id, "step")
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"任务 {st.get('run_id')} 正在生成中（{st.get('kind')}）。"
            f"一次只允许一个生成任务，完成前不能开始新的一步。",
        )
    meta = _load_meta(root) or {}
    cfg = load_config(quality=meta.get("quality"), ablation=meta.get("ablation") or None)
    try:
        report = prog.compute_progress(RunStorage(run_id, root=_out()), meta, cfg)
        if report.next is None:
            tasklock.release(_out(), run_id)  # 无事可做，立即还锁
            return {"run_id": run_id, "started": False, "reason": "all_done"}
        _spawn(
            [sys.executable, "-m", "fastvideo3.app.cli", "step", run_id],
            log_path=root / "server_spawn.log",
        )
    except Exception:
        # 持锁后、子进程接管前的任何异常都必须还锁，否则锁泄漏到进程死亡，
        # 全系统被一把死锁卡住（实测发生过：compute_progress 抛 IndexError 后 500）
        tasklock.release(_out(), run_id)
        raise
    return {
        "run_id": run_id,
        "started": True,
        "step": report.next.id,
        "label": report.next.label,
        "missing": report.next.missing,
        "stale": report.next.stale,
    }


async def _commit_script(
    storage: RunStorage,
    run_id: str,
    new_script: Script,
    old_script: Script | None,
    source: str,
) -> dict:
    """剧本落盘的唯一入口（事务实现见 `core/scriptio.commit_script`）。

    这里只负责在提交**前**把旧版原文捕获下来：`generate_script` 自己会写 `script.json`，
    等到事务里再读盘就读到新版了，版本号与作废范围都会算错。
    """
    from fastvideo3.core import scriptio

    root = storage.root
    p = root / "script.json"
    had = p.is_file()
    try:
        prev_dict = json.loads(p.read_text(encoding="utf-8")) if had else None
    except Exception:  # noqa: BLE001  # 坏 JSON：没有可备份/可比对的结构，但仍要全量作废
        prev_dict = None
    return await scriptio.commit_script(
        storage,
        _cfg_for_run(root),
        new_script,
        old_script=old_script,
        prev_dict=prev_dict,
        source=source,
        replaced_unparsable=bool(had and old_script is None),
    )


def _require_idle(action: str) -> None:
    """全局单任务锁：生成中不许改剧本/删剧本/重跑剧本步。"""
    from fastvideo3.core import tasklock

    st = tasklock.status(_out())
    if st.get("busy"):
        raise HTTPException(
            status_code=409,
            detail=f"任务 {st.get('run_id')} 正在生成中，先等它结束再{action}。",
        )


def _load_old_script_or_404(root: Path) -> Script:
    data = _load_script_dict(root)
    if not data:
        raise HTTPException(status_code=404, detail="还没有剧本（script.json 不存在）")
    try:
        return Script.model_validate(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=409, detail=f"现有剧本结构不合法，无法继承：{e}") from e


@app.get("/api/runs/{run_id}/script/text")
def get_script_text(run_id: str) -> dict:
    """剧本稿（可读文本）：让人改剧本本身，而不是改 JSON。"""
    from fastvideo3.core.script_text import render_script_text

    root = _safe_run_root(run_id)
    script = _load_old_script_or_404(root)
    return {"text": render_script_text(script), "version": script.version}


@app.put("/api/runs/{run_id}/script/text")
async def put_script_text(run_id: str, req: ScriptTextRequest) -> dict:
    """保存剧本稿：本地确定性解析 → 三层校验 → 与 JSON 编辑同一条落盘路径。

    三层：① 解析器报格式问题（带行号）② `_script_cross_check` 报语义问题（角色引用
    不存在、空场景）③ pydantic 报 schema 硬上限（台词 ≤60 字）。软规则（≤30 字、时长
    偏差 15%）不阻断，作为 `warnings` 返回。
    """
    from fastvideo3.core.script_text import parse_script_text

    root = _safe_run_root(run_id)
    _require_idle("改剧本")
    old = _load_old_script_or_404(root)
    try:
        script, problems = parse_script_text(req.text, old)
    except Exception as e:  # noqa: BLE001  # 解析器没预料到的 schema 错误也当成稿子问题回 422
        raise HTTPException(status_code=422, detail=f"剧本稿无法解析：{str(e)[:300]}") from e
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    semantic = _script_cross_check(script)
    if semantic:
        raise HTTPException(status_code=422, detail=semantic)
    try:
        result = await _commit_script(
            RunStorage(run_id, root=_out()), run_id, script, old, source="text_edited"
        )
    except Exception as e:  # noqa: BLE001  # pydantic 硬约束（台词 >60 字等）
        raise HTTPException(status_code=422, detail=f"剧本不合法：{e}") from e
    return result


@app.put("/api/runs/{run_id}/script")
async def put_script(run_id: str, request: Request) -> dict:
    """手动编辑剧本 JSON：服务端权威校验 + 沿血缘作废下游（不物理删除）。"""
    from fastvideo3.core.schema import Script as ScriptSchema

    root = _safe_run_root(run_id)
    _require_idle("改剧本")
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"请求体不是合法 JSON: {e}") from e
    try:
        new_script = ScriptSchema.model_validate(body)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"剧本结构不合法: {e}") from e
    problems = _script_cross_check(new_script)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    return await _commit_script(
        RunStorage(run_id, root=_out()), run_id, new_script, _script_or_none(root),
        source="edited",
    )


@app.delete("/api/runs/{run_id}/script")
def delete_script(run_id: str) -> dict:
    """删除剧本：归档后移除 `script.json`，下游产物全量作废。

    效果就是"回到第一步"：`compute_progress` 见不到剧本 → 下一步变成
    `script`（标签「生成剧本」），前端那颗主按钮自动从「开始生成 · ××」变回
    「开始生成 · 生成剧本」。下游产物只标 stale 不物理删，防止新剧本沿用同一批
    id 时把旧角色的脸当成新角色的定妆照。
    """
    from fastvideo3.core import scriptio

    root = _safe_run_root(run_id)
    _require_idle("删剧本")
    storage = RunStorage(run_id, root=_out())
    try:
        archive, cleared, invalidated = scriptio.discard_script(storage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="没有剧本可删（script.json 不存在）") from None
    except OSError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    _events_append(
        root, "web_api", "script_deleted", archived=archive, invalidated=len(invalidated)
    )
    return {
        "ok": True,
        "removed": "script.json",
        "archived": archive,
        "cleared": cleared,
        "count": len(invalidated),
        "items": invalidated[:50],
        "hint": "「开始生成」已变回「生成剧本」；新剧本会重做所有下游产物（你上传过的图会保留）",
    }


@app.post("/api/runs/{run_id}/script/regenerate", status_code=202)
def regenerate_script(run_id: str, req: RegenerateRequest) -> dict:
    """按人写的 prompt 重新生成剧本 —— 替代原来的 AI critic 自动重写。

    两种语义的代价完全不同，所以必须让人选：
    - `revise`（保留剧情，按意见改）：`script.json` 先留着，子进程把「渲染出来的剧本稿 +
      意见」交给 `generate_script` 的解析模式，回来后走字段级 diff 作废 —— 没被点名的
      镜头不重画。
    - `rewrite`（推倒重写）：先归档删除 `script.json` 并全量作废下游（与「删除剧本」同
      一步），意见写进 `script_prompt.json` 供子进程（以及它万一失败后的「开始生成」）
      读取。
    """
    import sys

    from fastvideo3.core import scriptio, tasklock

    root = _safe_run_root(run_id)
    storage = RunStorage(run_id, root=_out())
    if not (root / "script.json").is_file() and req.mode == "revise":
        raise HTTPException(
            status_code=404, detail="没有现有剧本可改写；要出新剧本请点「开始生成」或选「推倒重写」"
        )
    ok, st = tasklock.acquire(_out(), run_id, "script_regen")
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"任务 {st.get('run_id')} 正在生成中（{st.get('kind')}）。"
            f"一次只允许一个生成任务，完成前不能重写剧本。",
        )
    try:
        if req.mode == "rewrite":
            try:
                scriptio.discard_script(storage, reason="script_rewritten")
            except FileNotFoundError:
                pass  # 本来就还没出剧本：直接生成就是"推倒重写"
            storage.save_json(
                "script_prompt.json", {"prompt": req.prompt, "mode": req.mode}
            )
        cmd = [
            sys.executable, "-m", "fastvideo3.app.cli", "script-regen", run_id,
            "--prompt", req.prompt, "--mode", req.mode,
        ]
        _spawn(cmd, log_path=root / "server_spawn.log")
    except Exception:
        tasklock.release(_out(), run_id)  # 持锁后异常必须还锁，否则死锁卡住全系统
        raise
    _events_append(root, "web_api", "script_regenerate", mode=req.mode, prompt=req.prompt[:200])
    return {
        "run_id": run_id,
        "mode": req.mode,
        "started": True,
        "hint": "revise 只作废被意见点到的镜头；rewrite 会重做全部下游产物",
    }


def _cfg_for_run(root: Path) -> Any:
    meta = _load_meta(root) or {}
    from fastvideo3.core.config import apply_resolution, apply_run_overrides, resolution_for_quality

    # 与 CLI 的 `_cfg_from_meta` 共用同一套恢复口径：run 级模型选择、档位、消融配置、
    # 分辨率、**口型模式与音轨来源**都要按 meta 盖回来。少恢复一项就会出现"网页和命令行
    # 跑不同规格"——这一项此前就漏了：cinema 档 yaml 写着 v2v_overlay，于是一条
    # talk_mode=s2v_direct 的 run 在看板上按 videoretalk 的单价预估（¥0.08 vs ¥0.9，差 11 倍）。
    cfg = apply_run_overrides(
        load_config(quality=meta.get("quality"), ablation=meta.get("ablation") or None), meta
    )
    if meta.get("talk_mode"):
        cfg.talk_mode = str(meta["talk_mode"])
    if meta.get("voice_source"):
        cfg.voice_source = str(meta["voice_source"])
    return apply_resolution(cfg, meta.get("resolution_tier") or resolution_for_quality(cfg.quality))


# （已移到 core/scriptio.py）refresh_looks_for_changed_chars 与 cross_check：剧本事务要用，
# CLI 子进程也要用，留在 API 层会变成"子进程再抄一遍"。

def _script_cross_check(script: Script) -> list[str]:
    """引用完整性检查的 API 入口（实现见 `core/scriptio.cross_check`，只有一份）。"""
    from fastvideo3.core import scriptio

    return scriptio.cross_check(script)


@app.delete("/api/runs/{run_id}/artifacts/{stage}/{name}")
def delete_artifact(run_id: str, stage: str, name: str) -> dict:
    """删除单个产物（角色图/场景图/首帧）：沿血缘精确作废下游，其他场景不动。"""
    from fastvideo3.core import progress as prog
    from fastvideo3.core import tasklock
    from fastvideo3.core.storage import safe_unlink

    root = _safe_run_root(run_id)
    if stage not in _DELETE_STAGES:
        raise HTTPException(status_code=400, detail=f"stage 只能是 {sorted(_DELETE_STAGES)}")
    want_ext = ".mp4" if stage in ("clips", "talk") else ".png"
    if not _DELETE_NAME_RE.match(name) or not name.endswith(want_ext):
        raise HTTPException(
            status_code=400, detail=f"{stage} 阶段文件名必须是 char_*/scene_*/shot_*{want_ext}"
        )
    st = tasklock.status(_out())
    if st.get("busy"):
        raise HTTPException(status_code=409, detail=f"任务 {st.get('run_id')} 正在生成中，先等它结束再删产物。")
    base = (root / stage).resolve()
    target = (base / name).resolve()
    if not target.is_file() or not target.is_relative_to(base):
        raise HTTPException(status_code=404, detail="文件不存在")

    storage = RunStorage(run_id, root=_out())
    rel = f"{stage}/{name}"
    removed_extra: list[str] = []
    if name.startswith("char_") and not name.startswith("_cand_"):
        cid = name[len("char_"):-len(".png")]
        # 候选图**保留**（用户要删了主图后仍能看/选候选）；只清上传台账条目。
        # 另加 take 计数：确定性种子下"删除→重新生成"会拿到同一张脸，
        # take+1 让重生成真的换一张（否则用户以为删除没生效）。
        uploads = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}
        if rel in uploads:
            uploads.pop(rel, None)
            storage.save_json("uploads.json", uploads)
        takes = storage.load_json("takes.json") if storage.exists("takes.json") else {}
        takes[cid] = int(takes.get(cid, 0)) + 1
        storage.save_json("takes.json", takes)
        removed_extra.append(f"takes.json:{cid}={takes[cid]}")
    if not safe_unlink(target):
        raise HTTPException(status_code=409, detail="文件正被占用（预览/杀软），请关闭后重试")
    invalidated = prog.invalidate_artifact(
        storage, rel, reason="artifact_deleted", script=_script_or_none(root)
    )
    _events_append(root, "web_api", "artifact_deleted", stage=stage, name=name,
                   invalidated=len(invalidated), extra=removed_extra)
    return {"ok": True, "removed": [name, *removed_extra], "invalidated": invalidated}


@app.get("/api/task")
def task_status() -> dict:
    """全局任务占用情况：前端据此禁用所有生成按钮（一次只跑一个任务）。"""
    from fastvideo3.core import tasklock

    return tasklock.status(_out())


@app.post("/api/task/unlock")
def task_unlock() -> dict:
    """人工解锁：进程被外部杀死后锁会残留，这里提供显式解锁通道（留痕）。"""
    from fastvideo3.core import tasklock

    data = tasklock.force_release(_out())
    if data:
        rid = str(data.get("run_id") or "")
        if (_out() / rid).is_dir():
            _events_append(_out() / rid, "web_api", "task_lock_released", by="manual")
    return {"ok": True, "released": data}


def _events_append(root: Path, node: str, status: str, **data: Any) -> None:
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "node": node, "status": status}
    entry.update(data)
    with open(root / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


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
    consistency: dict | None = None
    cp = root / "consistency_report.json"
    if cp.is_file():
        try:
            consistency = json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            consistency = None
    degradations: list = []
    dp = root / "degradations.json"
    if dp.is_file():
        try:
            degradations = json.loads(dp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            degradations = []
    return {
        "run_id": run_id,
        "status": _run_status(root),
        "meta": meta,
        "events": _read_events(root),
        "cost": _read_cost(root),
        "images": images,
        "videos": videos,
        "review": review,  # 终审结果（含 review_error/verified/gate_error 标识）
        "consistency": consistency,  # 人物一致性度量（m9）：summary/per_shot/per_char
        "degradations": degradations,  # 降级台账：保交付但降级的路径全部在此
        "final_video": f"/api/files/{run_id}/final/final.mp4" if final.is_file() else None,
    }


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
