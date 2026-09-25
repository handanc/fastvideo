"""FastVideo3 CLI（Typer）：run / stage / resume / doctor / cost / ls-runs。"""

from __future__ import annotations

import asyncio
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from fastvideo3.core.config import OUTPUT_DIR, PROJECT_ROOT, AppConfig, load_config


def _fix_console_encoding() -> None:
    """Windows GBK 控制台无法输出 ✔ 等符号 → 统一 UTF-8（容错）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, pragma: no cover
            pass


_fix_console_encoding()

app = typer.Typer(help="FastVideo3 —— 多智能体短剧视频生成（全 API 化）", no_args_is_help=True)
console = Console()


def _new_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def _open_checkpointer(cfg: AppConfig):
    """按 profile 打开 Checkpointer（edu-sqlite 零配置 / prod-postgres 走 Docker）。"""
    if cfg.profile == "prod-postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as e:
            raise RuntimeError(
                'prod-postgres 模式需要: pip install "fastvideo3[postgres]" 并 docker compose up -d'
            ) from e
        dsn = (
            f"postgresql://{cfg.pg_user}:{cfg.pg_password}@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_db}"
        )
        return AsyncPostgresSaver.from_conn_string(dsn)
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(OUTPUT_DIR / "checkpoint.db"))


async def _invoke(cfg: AppConfig, run_id: str, initial_state: dict | None, resume_value=None):
    from langgraph.types import Command

    from fastvideo3.core.storage import RunStorage
    from fastvideo3.graph.builder import build_graph
    from fastvideo3.graph.nodes import build_context

    storage = RunStorage(run_id)
    ctx = build_context(cfg, storage)
    async with _open_checkpointer(cfg) as saver:
        graph = build_graph(ctx, saver)
        config = {"configurable": {"thread_id": run_id}}
        if resume_value is not None:
            # 区分两种续跑：停在审片门(有 interrupt) → Command(resume)；崩溃恢复 → 无输入继续
            snap_before = await graph.aget_state(config)
            has_interrupt = any(getattr(t, "interrupts", None) for t in (snap_before.tasks or []))
            if has_interrupt:
                result = await graph.ainvoke(Command(resume=resume_value), config)
            else:
                result = await graph.ainvoke(None, config)
        elif initial_state is not None:
            result = await graph.ainvoke(initial_state, config)
        else:  # 崩溃续跑：无新输入，从检查点继续
            result = await graph.ainvoke(None, config)
        snap = await graph.aget_state(config)
        return result, snap, ctx


@app.command()
def run(
    topic: str = typer.Option(..., help="短剧主题"),
    style: str = typer.Option("都市轻喜剧", help="风格"),
    duration: int = typer.Option(30, help="目标时长(秒)"),
    fake: bool = typer.Option(False, "--fake", help="Fake 模式：零成本离线跑通全流程"),
    review: str = typer.Option("auto", help="auto=全自动 | gated=人工审片"),
    talk_mode: str = typer.Option("s2v_direct", help="s2v_direct | v2v_overlay | audio_overlay"),
    budget: float = typer.Option(
        0.0, help="成本熔断阈值(元); 0=跟随 .env 的 FV3_BUDGET_LIMIT_CNY(默认50)"
    ),
    aspect: str = typer.Option("9:16", help="画幅: 9:16 | 16:9 | 1:1"),
    quality: str = typer.Option(
        "economy", help="画质档位: economy(教学) | high(交付) | cinema(商业级, 约¥1/秒)"
    ),
    run_id: str = typer.Option("", "--run-id", help="指定 run_id（Web API 用，默认按时间戳生成）"),
):
    """端到端生成：主题 → 完整短剧视频。"""
    cfg = load_config(quality=quality)
    cfg.fake = fake or cfg.fake
    cfg.review_mode = review
    cfg.talk_mode = talk_mode
    if budget > 0:  # 未显式传 --budget 时尊重 .env 熔断值（用户成本红线）
        cfg.budget_limit_cny = budget
    cfg.aspect_ratio = aspect
    if cfg.quality == "cinema" and not cfg.fake:
        console.print(
            f"[yellow]⚠ cinema 档多模态生成约 ¥1/秒，本次熔断 ¥{cfg.budget_limit_cny}。"
            "教学/演示请用 economy 或 --fake。[/]"
        )

    run_id = run_id or _new_run_id()
    from fastvideo3.core.schema import BaseInfo, ProjectConfig
    from fastvideo3.core.storage import RunStorage

    base = BaseInfo(topic=topic, style=style, target_duration_sec=duration, aspect_ratio=aspect)  # type: ignore[arg-type]
    storage = RunStorage(run_id)
    project = ProjectConfig(
        run_id=run_id,
        base_info=base,
        profile=cfg.profile,  # type: ignore[arg-type]
        review_mode=review,  # type: ignore[arg-type]
        talk_mode=talk_mode,  # type: ignore[arg-type]
        budget_limit_cny=budget,
        fake=cfg.fake,
        quality=cfg.quality,
    )
    storage.save_json("run_meta.json", project.model_dump(mode="json"))

    console.print(f"[bold]▶ FastVideo3 开始生成[/]  run_id={run_id}  fake={cfg.fake}")
    initial = {"base_info": base.model_dump(mode="json")}
    result, snap, ctx = asyncio.run(_invoke(cfg, run_id, initial))

    if snap.next:  # 停在人工审片门
        console.print(f"[yellow]⏸ 已暂停在人工审片节点: {snap.next}[/]")
        console.print(f"继续: fastvideo3 resume {run_id} --decision approve")
        raise typer.Exit(code=3)

    console.print(f"[green]✔ 生成完成[/]  成片: {result.get('final_video')}")
    console.print(f"  成本: ¥{ctx.ledger.total()}  明细: {ctx.ledger.summary()}")


@app.command()
def resume(
    run_id: str = typer.Argument(...),
    decision: str = typer.Option("approve", help="approve 或 reject:<意见>（对暂停的审片门）"),
):
    """续跑：人工审片回复，或从中断处恢复。"""
    from fastvideo3.core.storage import RunStorage

    meta = RunStorage(run_id).load_json("run_meta.json")
    cfg = load_config(quality=meta.get("quality"))  # 按原档位恢复模型栈/规格
    cfg.fake = meta.get("fake", cfg.fake)
    cfg.review_mode = meta.get("review_mode", cfg.review_mode)
    cfg.talk_mode = meta.get("talk_mode", cfg.talk_mode)
    cfg.budget_limit_cny = meta.get("budget_limit_cny", cfg.budget_limit_cny)

    result, snap, ctx = asyncio.run(_invoke(cfg, run_id, None, resume_value=decision))
    if snap.next:
        console.print(f"[yellow]⏸ 仍暂停在: {snap.next}[/]")
        raise typer.Exit(code=3)
    console.print(f"[green]✔ 续跑完成[/]  成片: {result.get('final_video')}")


@app.command()
def stage(
    name: str = typer.Argument(
        ..., help="script | assets | storyboard | shots | audio | talk | edit"
    ),
    topic: str = typer.Option("演示主题", help="（script 阶段用）主题"),
    run_id: str = typer.Option("", help="复用已有 run 目录（默认新建）"),
    fake: bool = typer.Option(True, "--fake/--real", help="默认 Fake 模式"),
    quality: str = typer.Option("economy", help="画质档位: economy | high | cinema"),
):
    """单阶段运行（教学用）：独立跑某一环节并查看中间产物。"""
    cfg = load_config(quality=quality)
    cfg.fake = fake
    rid = run_id or _new_run_id()
    asyncio.run(_run_stage(cfg, name, rid, topic))
    console.print(f"[green]✔ stage '{name}' 完成[/]  产物目录: {OUTPUT_DIR / rid}")


async def _run_stage(cfg: AppConfig, name: str, run_id: str, topic: str) -> None:
    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.core.storage import RunStorage
    from fastvideo3.graph.nodes import build_context
    from fastvideo3.providers.base import Capability

    storage = RunStorage(run_id)
    ctx = build_context(cfg, storage)
    storage.log_event("cli", "stage", name=name)

    def _load_script():
        from fastvideo3.core.schema import Script

        return Script.model_validate(storage.load_json("script.json"))

    if name == "script":
        from fastvideo3.modules.m1_scripting import generate_script

        await generate_script(
            cfg,
            ctx.provider(Capability.TEXT),
            BaseInfo(topic=topic),  # type: ignore[arg-type]
            storage,
            ctx.ledger,
        )
    elif name == "assets":
        from fastvideo3.modules.m2_assets import build_assets

        await build_assets(cfg, ctx.provider(Capability.T2I), _load_script(), storage, ctx.ledger)  # type: ignore[arg-type]
    elif name == "storyboard":
        from fastvideo3.core.schema import Asset
        from fastvideo3.modules.m3_storyboard import build_storyboard

        registry = storage.load_json("asset_registry.json")
        await build_storyboard(
            cfg,
            ctx.provider(Capability.T2I),
            _load_script(),  # type: ignore[arg-type]
            {k: Asset(**v) for k, v in registry.items()},
            storage,
            ctx.ledger,
        )
    elif name == "shots":
        from fastvideo3.modules.m4_shots import generate_shots

        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
        }
        await generate_shots(
            cfg,
            ctx.provider(Capability.I2V),
            _load_script(),
            frames,  # type: ignore[arg-type]
            storage,
            ctx.ledger,
            ctx.jobs,
        )
    elif name == "audio":
        from fastvideo3.modules.m5_audio import synthesize_dialogues

        await synthesize_dialogues(
            cfg,
            ctx.provider(Capability.TTS),
            _load_script(),
            storage,
            ctx.ledger,  # type: ignore[arg-type]
        )
    elif name == "talk":
        from fastvideo3.core.schema import Asset, AudioLine
        from fastvideo3.modules.m6_talk import make_talk_clips

        registry = storage.load_json("asset_registry.json")
        lines = [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
        }
        talk_provider = ctx.provider(Capability.TALK) if cfg.talk_mode != "audio_overlay" else None  # type: ignore[assignment]
        talks, fallbacks = await make_talk_clips(
            cfg,
            talk_provider,
            ctx.provider(Capability.I2V),
            _load_script(),  # type: ignore[arg-type]
            frames,
            {k: Asset(**v) for k, v in registry.items()},
            lines,
            storage,
            ctx.ledger,
            ctx.jobs,
        )
        console.print(f"  口型片段 {len(talks)} | 降级片段 {len(fallbacks)}")
    elif name == "edit":
        from fastvideo3.core.schema import AudioLine
        from fastvideo3.modules.m7_edit import compose_final

        lines = [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
        clips = {
            p.stem.replace("shot_", ""): str(p) for p in storage.dir("clips").glob("shot_*.mp4")
        }
        talks = {
            p.stem.replace("shot_", ""): str(p) for p in storage.dir("talk").glob("shot_*.mp4")
        }
        await compose_final(cfg, _load_script(), clips, talks, lines, storage)
    else:
        raise typer.BadParameter(f"未知阶段: {name}")


@app.command()
def doctor():
    """环境自检：依赖 / FFmpeg / KEY / 模型 / 数据库。"""
    table = Table(title="FastVideo3 doctor")
    table.add_column("检查项")
    table.add_column("结果")

    def row(item: str, ok: bool, detail: str = "") -> None:
        table.add_row(item, f"[green]✔ {detail}[/]" if ok else f"[red]✘ {detail}[/]")

    import sys

    row("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])

    deps = ["pydantic", "httpx", "langgraph", "typer", "PIL", "sqlalchemy", "yaml"]
    missing = []
    for d in deps:
        try:
            __import__(d.lower() if d != "PIL" else "PIL")
        except ImportError:
            missing.append(d)
    row("核心依赖", not missing, "全部就绪" if not missing else f"缺失: {missing}")

    from fastvideo3.utils.ffmpeg import find_ffmpeg

    try:
        row("FFmpeg", True, find_ffmpeg())
    except Exception as e:  # noqa: BLE001
        row("FFmpeg", False, str(e))

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

    from fastvideo3.core.config import MODELS_DIR
    from fastvideo3.utils.yolo import WEIGHT_NAME

    weight_ok = (MODELS_DIR / WEIGHT_NAME).exists()
    row(
        "yolov8n-seg.pt(双人同屏)",
        weight_ok,
        "就位" if weight_ok else "缺失(运行 fetch_models 下载, 缺失时自动降级拼接)",
    )

    try:
        from fastvideo3.core.jobs import JobStore

        JobStore(cfg.db_url).dispose()
        row("数据库", True, cfg.db_url.split("://")[0])
    except Exception as e:  # noqa: BLE001
        row("数据库", False, str(e)[:80])

    console.print(table)


@app.command()
def cost(run_id: str = typer.Argument(...)):
    """查看某次运行的成本台账。"""
    from fastvideo3.core.storage import RunStorage

    data = RunStorage(run_id).load_json("cost.json")
    total = sum(e["amount_cny"] for e in data)
    table = Table(title=f"成本台账 {run_id}（合计 ¥{total:.2f}）")
    for col in ("能力", "供应商", "模型", "金额(¥)", "备注"):
        table.add_column(col)
    for e in data:
        table.add_row(
            e["capability"], e["provider"], e["model"], f"{e['amount_cny']:.3f}", e.get("note", "")
        )
    console.print(table)


@app.command("ls-runs")
def ls_runs():
    """列出历史运行。"""
    from fastvideo3.core.storage import list_runs

    for r in list_runs():
        console.print(r)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8000, help="监听端口"),
    reload: bool = typer.Option(False, help="开发热重载（uvicorn --reload）"),
):
    """启动 Web 面板：FastAPI 接口 + Vue 前端（需 pip install "fastvideo3[web]"）。"""
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError('Web 面板需要: pip install "fastvideo3[web]"') from e

    if not (PROJECT_ROOT / "web" / "dist" / "index.html").exists():
        console.print(
            "[yellow]⚠ 未检测到 web/dist 前端构建产物：先在 web/ 目录执行 "
            "npm install && npm run build；开发模式可用 npm run dev (http://127.0.0.1:5173)[/]"
        )
    console.print(f"[bold]▶ FastVideo3 Web 面板[/]  http://{host}:{port}")
    uvicorn.run("fastvideo3.app.api:app", host=host, port=port, reload=reload)


def app_entry() -> None:
    app()


if __name__ == "__main__":
    app_entry()
