"""图节点：导演（Supervisor）视角的阶段调度。

教学说明：并行性有两处 ——
① 图级：storyboard 之后 shots/audio 双分支并行（LangGraph 原生扇出）；
② 节点内：m4/m5/m6 用信号量并发提交多个异步任务。
把节点内并发进一步拆成 Send() 逐镜头 map-reduce 是进阶练习（见 TDD §5.4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import Asset, BaseInfo, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import Capability, Provider, ProviderError
from fastvideo3.providers.factory import make_fallback, make_provider


@dataclass
class GraphContext:
    """运行时上下文（不可序列化部分：供应商连接、存储、账本）。"""

    cfg: AppConfig
    storage: RunStorage
    ledger: CostLedger
    jobs: JobStore
    providers: dict[str, Provider] = field(default_factory=dict)

    def provider(self, cap: Capability) -> Any:
        # 返回 Any：调用方按具体协议（TextProvider/T2IProvider/…）直接传参，
        # 静态层面不再逐点 ignore（运行时由 factory 保证能力匹配）。
        if cap not in self.providers:
            self.providers[cap] = make_provider(cap, self.cfg)
        return self.providers[cap]


def build_context(cfg: AppConfig, storage: RunStorage) -> GraphContext:
    return GraphContext(
        cfg=cfg,
        storage=storage,
        ledger=CostLedger(storage, cfg.budget_limit_cny),
        jobs=JobStore(cfg.db_url),
    )


def build_nodes(ctx: GraphContext) -> dict[str, Any]:
    """工厂：闭包注入上下文，返回节点函数表。"""

    async def intake(state) -> dict:
        ctx.storage.log_event("intake", "start", topic=state["base_info"].get("topic"))
        if ctx.ledger.exceeded():
            raise ProviderError(f"预算已超限: ¥{ctx.ledger.total()}")
        return {"status": "running"}

    async def screenwriter(state) -> dict:
        from fastvideo3.modules.m1_scripting import generate_script

        base = BaseInfo(**state["base_info"])
        script = await generate_script(
            ctx.cfg,
            ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
            base,
            ctx.storage,
            ctx.ledger,
            critic=None,  # 评审回路在图级 script_gate 体现；此处保持单轮快速
            external_feedback=state.get("script_feedback") or None,
        )
        return {"script": script.model_dump(mode="json"), "script_feedback": ""}

    async def script_gate(state) -> dict:
        if ctx.cfg.review_mode != "gated" or state.get("script_rejected"):
            return {}
        decision = interrupt(
            {
                "phase": "script",
                "message": "剧本已生成，请审阅 output/<run_id>/script.json。回复 approve 或 reject:<意见>",
            }
        )
        text = str(decision or "approve")
        if text.lower().startswith("reject"):
            feedback = text.split(":", 1)[1].strip() if ":" in text else "请打磨剧情冲突"
            return {"script_feedback": feedback, "script_rejected": True}
        return {}

    async def art(state) -> dict:
        from fastvideo3.modules.m2_assets import build_assets

        script = Script.model_validate(state["script"])
        assets = await build_assets(
            ctx.cfg,
            ctx.provider(Capability.T2I),
            script,
            ctx.storage,
            ctx.ledger,  # type: ignore[arg-type]
        )
        return {"assets": {k: a.model_dump() for k, a in assets.items()}}

    async def storyboard(state) -> dict:
        from fastvideo3.modules.m3_storyboard import build_storyboard

        script = Script.model_validate(state["script"])
        assets = {k: Asset(**v) for k, v in state["assets"].items()}
        frames = await build_storyboard(
            ctx.cfg,
            ctx.provider(Capability.T2I),
            script,
            assets,  # type: ignore[arg-type]
            ctx.storage,
            ctx.ledger,
        )
        return {"frames": frames}

    async def shots(state) -> dict:
        from fastvideo3.modules.m4_shots import generate_shots
        from fastvideo3.providers.factory import make_r2v_provider

        script = Script.model_validate(state["script"])
        assets = {k: Asset(**v) for k, v in state.get("assets", {}).items()}
        vlm = None
        if ctx.cfg.candidates_per_shot > 1:
            try:
                vlm = ctx.provider(Capability.VLM)  # type: ignore[assignment]
            except ProviderError:
                vlm = None
        clips = await generate_shots(
            ctx.cfg,
            ctx.provider(Capability.I2V),
            script,
            state["frames"],  # type: ignore[arg-type]
            ctx.storage,
            ctx.ledger,
            ctx.jobs,
            vlm=vlm,  # type: ignore[arg-type]
            r2v=make_r2v_provider(ctx.cfg),  # type: ignore[arg-type]
            assets=assets,
        )
        return {"clips": clips}

    async def audio(state) -> dict:
        from fastvideo3.modules.m5_audio import synthesize_dialogues

        script = Script.model_validate(state["script"])
        lines = await synthesize_dialogues(
            ctx.cfg,
            ctx.provider(Capability.TTS),
            script,
            ctx.storage,
            ctx.ledger,  # type: ignore[arg-type]
            fallback_tts=make_fallback(Capability.TTS, ctx.cfg),  # type: ignore[arg-type]
        )
        return {"audio": [ln.model_dump() for ln in lines]}

    async def talk(state) -> dict:
        from fastvideo3.core.schema import AudioLine
        from fastvideo3.modules.m6_talk import make_talk_clips

        script = Script.model_validate(state["script"])
        assets = {k: Asset(**v) for k, v in state["assets"].items()}
        lines = [AudioLine(**ln) for ln in state.get("audio", [])]
        talk_provider = None
        if ctx.cfg.talk_mode != "audio_overlay":
            try:
                talk_provider = ctx.provider(Capability.TALK)  # type: ignore[assignment]
            except ProviderError:
                talk_provider = make_fallback(Capability.TALK, ctx.cfg)
        talks, fallbacks = await make_talk_clips(
            ctx.cfg,
            talk_provider,
            ctx.provider(Capability.I2V),
            script,  # type: ignore[arg-type]
            state["frames"],
            assets,
            lines,
            ctx.storage,
            ctx.ledger,
            ctx.jobs,
        )
        # 降级片段并入 clips 通道 → m7 会自动叠加对白音轨
        return {"talks": talks, "clips": fallbacks}

    async def edit(state) -> dict:
        from fastvideo3.core.schema import AudioLine
        from fastvideo3.modules.m7_edit import compose_final

        script = Script.model_validate(state["script"])
        lines = [AudioLine(**ln) for ln in state.get("audio", [])]
        final = await compose_final(
            ctx.cfg,
            script,
            state.get("clips", {}),
            state.get("talks", {}),
            lines,
            ctx.storage,
        )
        return {"final_video": str(final)}

    async def final_review(state) -> dict:
        from pathlib import Path

        from fastvideo3.modules.m8_review import review_final

        try:
            vlm = ctx.provider(Capability.VLM)
        except ProviderError:
            return {"final_review": {"passed": True, "feedback": "无 VLM，跳过终审"}}
        script = Script.model_validate(state["script"])
        result = await review_final(
            vlm, Path(state["final_video"]), script, ctx.storage, ctx.ledger
        )  # type: ignore[arg-type]
        return {"final_review": result.model_dump(mode="json")}

    async def final_gate(state) -> dict:
        if ctx.cfg.review_mode != "gated" or state.get("final_rejected"):
            return {}
        decision = interrupt(
            {
                "phase": "final",
                "message": f"成片已生成: {state.get('final_video')}。回复 approve 或 reject:<意见>",
            }
        )
        text = str(decision or "approve")
        if text.lower().startswith("reject"):
            return {"final_rejected": True}
        return {}

    async def finalize(state) -> dict:
        summary = ctx.ledger.summary()
        ctx.storage.save_json(
            "run_summary.json",
            {"cost": summary, "final_video": state.get("final_video", "")},
        )
        meta_path = ctx.storage.root / "run_meta.json"
        if meta_path.exists():
            meta = ctx.storage.load_json("run_meta.json")
            meta["status"] = "done"
            ctx.storage.save_json("run_meta.json", meta)
        ctx.storage.log_event("finalize", "done", cost=summary.get("total"))
        return {"status": "done"}

    return {
        "intake": intake,
        "screenwriter": screenwriter,
        "script_gate": script_gate,
        "art": art,
        "storyboard": storyboard,
        "shots": shots,
        "audio": audio,
        "talk": talk,
        "edit": edit,
        "final_review": final_review,
        "final_gate": final_gate,
        "finalize": finalize,
    }


# ── 条件边 ──
def route_after_script_gate(state) -> str:
    if state.get("script_feedback") and state.get("script_rejected"):
        return "screenwriter"  # 驳回 → 带意见重写（仅一次，防死循环）
    return "art"


def route_after_final_gate(state) -> str:
    if state.get("final_rejected") and not state.get("_final_rerun"):
        return "edit"
    return "finalize"
