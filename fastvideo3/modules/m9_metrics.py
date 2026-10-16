"""m9 一致性度量：把"前后人物一致性"变成可量化、可跨配置对比的证据。

背景：改造前盘上**没有任何**一致性数值——终审是兜底 0.7、抽卡无评分、一致性抽查
只在 Fake run 出现过 3 次。本模块产出 consistency_report.json，两层互相印证：

- Tier A（VLM 结构化属性位）：能给出"头盔颜色错了"这种可行动原因；
- Tier B（纯 PIL 色相/直方图）：零成本客观数值，用来**证伪** Tier A（metric_conflict）。

三重反"模型自觉"设计：自校准锚（[ref,ref] 必须全过）、左右顺序互换（抓顺序偏置）、
双人强制二选一（抓"两个都说对"的幻觉）。锚不过 → 整批标 judge_trusted=false，
而不是让管线失败或采信可疑分数。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger, llm_meta
from fastvideo3.core.prompts import render_strict
from fastvideo3.core.schema import Asset, ConsistencyVerdict, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import VLMProvider
from fastvideo3.utils import faceid
from fastvideo3.utils import quality as q

ANCHOR_MIN_SCORE = 0.85

_FINGERPRINT_KEYS = (
    "use_reference_keyframes",
    "require_reference_keyframes",
    "ref_max_chars",
    "keyframe_seed_scope",
    "use_r2v_for_characters",
    "asset_candidates",
    "asset_style_gate",
    "candidates_per_shot",
    "storyboard_sample_rate",
    "consistency_metric",
    "look_extraction",
)


def config_fingerprint(cfg: AppConfig) -> str:
    """一致性相关开关的有序哈希：跨 run 对比表按它分组，防止"配置不同却比分数"。"""
    blob = "|".join(f"{k}={getattr(cfg, k, None)}" for k in _FINGERPRINT_KEYS)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


async def measure_consistency(
    cfg: AppConfig,
    script: Script,
    assets: dict[str, Asset],
    frames: dict[str, str],
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider | None = None,
) -> dict:
    """对含人物的镜头做一致性度量并落盘 consistency_report.json。"""
    if cfg.consistency_metric == "off":
        return {}
    full = cfg.consistency_metric == "full"
    char_map = script.character_map()
    events = storage.read_events()

    coverage = _coverage(cfg, script, events, ledger)
    # 口型同步度（m6 度量后落事件）：并入逐镜行与汇总，让"嘴不动但话在响"可见可拦
    lipsync_by_shot: dict[str, dict] = {}
    for e in events:
        if e.get("status") == "lipsync_check" and e.get("shot"):
            lipsync_by_shot[e["shot"]] = {
                "corr": e.get("corr"), "method": e.get("method"),
                # verdict/located 是"这把尺子到底量没量到嘴"的元信息：经验框量不到嘴时
                # located=False、verdict=inconclusive，闸门必须据此**拒绝**据此重做
                "verdict": e.get("verdict"), "located": e.get("located"),
                "eyes_corr": e.get("eyes_corr"), "bg_corr": e.get("bg_corr"),
                "reason": e.get("reason"),
            }

    # 自校准锚：同一张参考图自比，属性位必须全过 —— 否则这个 judge 的分不可信
    judge_trusted = True
    if vlm is not None:
        judge_trusted = await _anchor_check(cfg, script, assets, storage, ledger, vlm)

    char_shots = [s for s in script.all_shots() if s.characters]
    random.seed(42)
    sampled = char_shots if full else (
        random.sample(char_shots, k=max(1, len(char_shots) // 2)) if char_shots else []
    )

    per_shot: list[dict] = []
    per_char: dict[str, dict] = {}
    for shot in sampled:
        frame = frames.get(shot.shot_id)
        ref_asset = assets.get(f"char_{shot.characters[0]}")
        if not frame or not Path(frame).exists() or ref_asset is None:
            continue
        ref = Path(ref_asset.uri)
        if not ref.exists():
            continue
        row = _tier_b(ref, Path(frame))
        # Tier C（客观身份层）：SFace 人脸 embedding 的余弦相似度 —— 离线、零边际成本、可复现。
        # 它回答的是 Tier B 回答不了的问题："这还是不是**同一张脸**"（Tier B 量的是服装主色，
        # 只能证伪"衣服换了"）。权重缺失/检不到正脸 → verdict=inconclusive，**不计入达标率**
        # （把"没量到"算成不合格就会制造假阴性 → 触发付费重拍，那是口型度量踩过的坑）。
        #
        # 必须拿**本镜全部声明角色**去比，不能只比 characters[0]：双人同框时把第一个人
        # 拍成侧脸/背光，余弦就会掉到阈值下，于是"另一个角色明明还在、脸也对得上"的合格
        # 镜头被判成身份漂移 → 白花一次重拍钱（2026-10-12 权重就位后首次真跑实测撞上：
        # 同帧唐僧 0.306 / 国王 0.4286）。只有"画面里有脸、但没有任何一个角色对得上"
        # 才是真的身份问题。
        char_refs: dict[str, Path] = {}
        for name in shot.characters:
            asset = assets.get(f"char_{name}")
            if asset is not None and Path(asset.uri).exists():
                char_refs[name] = Path(asset.uri)
        fid = faceid.identity_similarity_multi(char_refs or {shot.characters[0]: ref}, Path(frame))
        row.update(
            {
                "face_cosine": fid["cosine"],
                "face_verdict": fid["verdict"],
                "face_best_char": fid["best_char"],
                "face_per_char": fid["per_char"],
                "face_frame_faces": fid["frame_faces"],
                "face_reason": fid["reason"],
                "faceid_available": fid["available"],
            }
        )
        if fid["verdict"] == "different":
            # 客观层与 VLM 冲突时以客观层为准记事件：VLM 说"脸一致"而余弦远低于阈值，
            # 这正是 metric_conflict 要抓的形状（此前只有 hue 一路证伪，脸漂了抓不到）
            storage.log_event(
                "m9_metrics",
                "faceid_conflict",
                shot=shot.shot_id,
                cosine=fid["cosine"],
                best_char=fid["best_char"],
                per_char=fid["per_char"],
                threshold=faceid.COSINE_THRESHOLD,
                reason=fid["reason"],
                vlm_face=row.get("face"),
                hint="本帧检到的脸对不上任何一个声明角色 → 客观判身份漂移（VLM 的判定只作解释）",
            )
        row.update({"shot": shot.shot_id, "scene": shot.scene_id,
                    "chars": list(shot.characters), "verified": False, "judge_agree": None})
        if vlm is not None:
            verdict, agree = await _tier_a(
                cfg, ref, Path(frame), shot.characters, char_map, storage, ledger, vlm, full
            )
            if verdict is not None:
                row.update(
                    {
                        "face": verdict.face,
                        "hair": verdict.hair,
                        "outfit": verdict.outfit,
                        "headwear": verdict.headwear,
                        "photoreal": verdict.photoreal,
                        "score": verdict.score,
                        "reason": verdict.reason,
                        "verified": True,
                        "judge_agree": agree,
                    }
                )
                if verdict.outfit and row["hue_dist"] is not None:
                    if row["hue_dist"] > q.HUE_CONFLICT_DEG:
                        row["metric_conflict"] = True
                        storage.log_event(
                            "m9_metrics",
                            "metric_conflict",
                            shot=shot.shot_id,
                            hue_dist=round(row["hue_dist"], 1),
                            hint="VLM 判 outfit 一致但服装主色色相差超阈值",
                        )
        row["lipsync"] = lipsync_by_shot.get(shot.shot_id)
        per_shot.append(row)
        for cid in shot.characters:
            acc = per_char.setdefault(
                cid, {"face": 0, "outfit": 0, "photoreal": 0, "n": 0, "max_hue_dist": 0.0}
            )
            acc["n"] += 1
            acc["face"] += 1 if row.get("face") else 0
            acc["outfit"] += 1 if row.get("outfit") else 0
            acc["photoreal"] += 1 if row.get("photoreal") else 0
            if row["hue_dist"] is not None:
                acc["max_hue_dist"] = max(acc["max_hue_dist"], row["hue_dist"])

    summary = _summary(per_shot, per_char)
    summary.update(_lipsync_summary(per_shot, cfg))  # 口型同步率 + 无口型镜头清单
    # Tier C 汇总：只统计**真量到**的镜头（inconclusive 不进分母，理由见 faceid.summarize）
    summary.update(
        faceid.summarize(
            [
                {
                    "cosine": r.get("face_cosine"),
                    "verdict": r.get("face_verdict"),
                    "available": r.get("faceid_available"),
                }
                for r in per_shot
            ]
        )
    )
    methods = ["pil_hue"]
    if vlm is not None:
        methods.insert(0, "vlm_attributes")
    if summary.get("faceid_available"):
        methods.append("sface_embedding")
    report = {
        "run_id": storage.run_id,
        "resolution": cfg.i2v_resolution,
        "ablation": cfg.ablation,
        "config_fingerprint": config_fingerprint(cfg),
        "method": "+".join(methods),
        "judge_trusted": judge_trusted,
        "coverage": coverage,
        "per_shot": per_shot,
        "per_char": {
            cid: {
                "face_rate": round(v["face"] / v["n"], 3) if v["n"] else None,
                "outfit_rate": round(v["outfit"] / v["n"], 3) if v["n"] else None,
                "photoreal_rate": round(v["photoreal"] / v["n"], 3) if v["n"] else None,
                "max_hue_dist": round(v["max_hue_dist"], 1),
                "n": v["n"],
            }
            for cid, v in per_char.items()
        },
        "summary": summary,
    }
    storage.save_json("consistency_report.json", report)
    storage.log_event(
        "m9_metrics",
        "checked",
        shots=len(per_shot),
        verified=sum(1 for r in per_shot if r["verified"]),
        anchor_ok=judge_trusted,
    )
    return report


def _coverage(cfg: AppConfig, script: Script, events: list[dict], ledger: CostLedger) -> dict:
    """从事件/台账重建"一致性机制实际通电比例"——这是因果证据的地基。"""
    char_shot_ids = {s.shot_id for s in script.all_shots() if s.characters}
    kf = [e for e in events if e.get("status") == "keyframe_gen" and e.get("shot") in char_shot_ids]
    with_ref = [e for e in kf if (e.get("ref_count") or 0) > 0]
    gens = [e for e in events if e.get("status") == "shot_gen"]
    r2v = [e for e in gens if e.get("route") == "r2v"]
    vlm_entries = [e for e in ledger.entries if e.capability == "vlm"]
    verified = [e for e in vlm_entries if (e.meta or {}).get("verified")]
    return {
        "ref_attached_ratio": round(len(with_ref) / len(kf), 3) if kf else None,
        "r2v_route_ratio": round(len(r2v) / len(gens), 3) if gens else None,
        "vlm_verified_ratio": round(len(verified) / len(vlm_entries), 3) if vlm_entries else None,
        "keyframes_with_chars": len(kf),
    }


def _tier_b(ref: Path, frame: Path) -> dict:
    return {
        "hue_dist": q.hue_distance(
            q.dominant_hue(ref, q.OUTFIT_BOX), q.dominant_hue(frame, q.OUTFIT_BOX)
        ),
        "face_hist_dist": round(q.hist_distance(ref, frame, q.FACE_BOX), 3),
    }


async def _tier_a(
    cfg: AppConfig,
    ref: Path,
    frame: Path,
    characters: list[str],
    char_map,
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider,
    full: bool,
) -> tuple[ConsistencyVerdict | None, bool | None]:
    ch = char_map.get(characters[0])
    look = ch.look.model_dump(exclude_none=True) if ch is not None and ch.look else {
        "appearance": ch.appearance if ch else ""
    }
    rubric = render_strict("consistency_metric.v1", look_json=json.dumps(look, ensure_ascii=False))
    try:
        verdict = await vlm.review_json([ref, frame], schema=ConsistencyVerdict, rubric=rubric)
    except Exception as e:  # noqa: BLE001  # 度量失败不阻塞交付，但要可见
        storage.log_event("m9_metrics", "tier_a_failed", error=str(e)[:200])
        return None, None
    ledger.record(
        "vlm",
        vlm.name,
        vlm.model,
        note=f"metric:{frame.stem}",
        meta=llm_meta(vlm, purpose="consistency_metric", n_media=2, verified=True),
    )
    agree: bool | None = None
    if full:
        # 左右顺序互换：同一对图换序再问，属性位不一致 = judge 有顺序偏置
        try:
            swapped = await vlm.review_json(
                [frame, ref], schema=ConsistencyVerdict, rubric=rubric
            )
            agree = (
                swapped.face == verdict.face
                and swapped.outfit == verdict.outfit
                and swapped.headwear == verdict.headwear
            )
            if not agree:
                storage.log_event("m9_metrics", "judge_order_bias", frame=frame.name)
            ledger.record(
                "vlm", vlm.name, vlm.model, note=f"metric_swap:{frame.stem}",
                # 这也是一次**真实调用**（换序复核），同样要带 token 用量
                meta=llm_meta(vlm, purpose="consistency_metric", n_media=2, verified=True),
            )
        except Exception:  # noqa: BLE001
            agree = None
    return verdict, agree


async def _anchor_check(
    cfg: AppConfig,
    script: Script,
    assets: dict[str, Asset],
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider,
) -> bool:
    """自校准锚：[ref, ref] 自比必须全属性 True 且 score≥阈值，否则整批不可信。"""
    ch = script.characters[0] if script.characters else None
    ref_asset = assets.get(f"char_{ch.character_id}") if ch else None
    if ch is None or ref_asset is None or not Path(ref_asset.uri).exists():
        storage.log_event("m9_metrics", "anchor_skipped", reason="no_ref_asset")
        return False
    look = ch.look.model_dump(exclude_none=True) if ch.look else {"appearance": ch.appearance}
    rubric = render_strict("consistency_metric.v1", look_json=json.dumps(look, ensure_ascii=False))
    ref = Path(ref_asset.uri)
    try:
        v = await vlm.review_json([ref, ref], schema=ConsistencyVerdict, rubric=rubric)
    except Exception as e:  # noqa: BLE001
        storage.log_event("m9_metrics", "judge_unreliable", reason=str(e)[:200])
        return False
    ledger.record(
        "vlm", vlm.name, vlm.model, note="metric_anchor",
        meta=llm_meta(vlm, purpose="consistency_metric", n_media=2, verified=True),
    )
    ok = all((v.face, v.hair, v.outfit, v.headwear, v.photoreal)) and v.score >= ANCHOR_MIN_SCORE
    if not ok:
        storage.log_event(
            "m9_metrics", "judge_unreliable", score=v.score,
            hint="同一张图自比都过不了，本批 VLM 分数不可采信",
        )
    return bool(ok)


def _summary(per_shot: list[dict], per_char: dict[str, dict]) -> dict:
    verified = [r for r in per_shot if r.get("verified")]
    n = len(verified) or 1
    # 最差场景：按 scene 聚合 face 通过率（教学上直接指出"哪一场戏换脸最严重"）
    by_scene: dict[str, list[int]] = {}
    for r in verified:
        sc = str(r.get("scene") or r.get("shot", "").split("_")[0])
        by_scene.setdefault(sc, []).append(1 if r.get("face") else 0)
    worst_scene = None
    if by_scene:
        worst_scene = min(by_scene, key=lambda k: sum(by_scene[k]) / len(by_scene[k]))
    return {
        "face_consistency": round(sum(1 for r in verified if r.get("face")) / n, 3),
        "outfit_consistency": round(sum(1 for r in verified if r.get("outfit")) / n, 3),
        "style_photoreal_rate": round(sum(1 for r in verified if r.get("photoreal")) / n, 3),
        "mean_hue_dist": round(
            sum(r["hue_dist"] for r in per_shot if r["hue_dist"] is not None)
            / max(1, sum(1 for r in per_shot if r["hue_dist"] is not None)),
            1,
        )
        if any(r["hue_dist"] is not None for r in per_shot)
        else None,
        "shots_measured": len(per_shot),
        "worst_scene": worst_scene,
    }


def _lipsync_summary(per_shot: list[dict], cfg: AppConfig) -> dict:
    """口型同步汇总：真口型通道的达标率 + 被叠音降级（无口型）的镜头清单。"""
    synced = [
        r["lipsync"] for r in per_shot
        if r.get("lipsync") and r["lipsync"].get("method") == "lipsync"
        and r["lipsync"].get("corr") is not None
    ]
    return {
        "lipsync_rate": (
            round(
                sum(1 for x in synced if x["corr"] >= cfg.lipsync_min_corr) / len(synced), 3
            )
            if synced
            else None
        ),
        "lipsync_no_mouth_shots": [
            r["shot"] for r in per_shot
            if r.get("lipsync") and r["lipsync"].get("method") == "overlay"
        ],
    }


def gate_failures(report: dict, cfg: AppConfig) -> list[dict]:
    """一致性闸门：挑出需要自动重做的镜头。**两条独立判据，客观层优先**。

    - **客观层（Tier C，SFace 人脸 embedding 余弦）**：离线、可复现、不花钱、不依赖任何模型
      自觉 → 只要 `face_verdict == "different"` 就进闸门，**即使 VLM 自校准锚不可信也照拦**。
      这是这次改造的关键：以前锚不过就整批放弃自动重做（`judge_trusted=False → return []`），
      等于"裁判可疑时连客观证据也不用了"。
      `different` 的含义是"**这一帧检到的脸，对不上本镜任何一个声明角色**"（见
      `faceid.identity_similarity_multi`）：只比 `characters[0]` 会把双人镜头里"第一个人
      这一帧正好侧脸"误判成身份漂移，那是直接烧钱的假阳性。检不到脸 = `inconclusive`，不进闸门。
    - **主观层（Tier A，VLM 属性位）**：能给出"头盔颜色错了"这种可行动原因，但已多次解析失败
      （落兜底分/整批跳过）→ 仅在 `judge_trusted` 不为 False 时采信。
    - Tier B（色相/直方图）只做**证伪**与解释，不单独触发重做：它量的是服装主色，
      夜景/换装剧情都会让它超阈值，拿它当闸门会误伤。

    返回 [] 的情况都是"不该自动烧钱"：度量关闭、报告为空、两层都没有可信判据。
    """
    if cfg.consistency_gate != "reshoot" or not report:
        return []
    trusted = report.get("judge_trusted") is not False
    bad: list[dict] = []
    seen: set[str] = set()
    for row in report.get("per_shot", []):
        shot = row.get("shot")
        objective = row.get("face_verdict") == "different"
        face, outfit = row.get("face"), row.get("outfit")
        subjective = trusted and bool(row.get("verified")) and (face is False or outfit is False)
        if not (objective or subjective):
            continue
        if shot in seen:
            continue
        seen.add(shot or "")
        bad.append(
            {
                "shot": shot,
                "face": face,
                "outfit": outfit,
                "headwear": row.get("headwear"),
                "hue_dist": row.get("hue_dist"),
                # 客观层证据单独给出来：重做之后要靠它复核，而不是再问一次 VLM
                "face_cosine": row.get("face_cosine"),
                "evidence": "sface" if objective else "vlm",
                "reason": (
                    # 措辞要说清"比的是本镜全部角色"：只比一个人时的"不同人"是假阳性来源
                    f"本帧 {row.get('face_frame_faces')} 张脸对不上任何声明角色"
                    f"（最高余弦 {row.get('face_cosine')} < {faceid.COSINE_THRESHOLD}，"
                    f"逐角色 {row.get('face_per_char')}）"
                    if objective
                    else row.get("reason", "")
                ),
            }
        )
        # 口型**故意不进自动重做**（2026-10-04 实测）：度量在一条人眼确认同步的
        # videoretalk 产物上只给 原始 corr 0.163 / 偏相关 0.113（阈值 0.25）—— 假阴性。
        # 拿它去重做 = 反复重抽本来合格的镜头，比不拦更贵。看板照常显示口型状态，
        # 判定走人工：`scripts/probe_lipsync.py sheet <run> --shot <id>` 出一张
        # 带逐帧响度条的嘴部接片，10 秒就能定案。
    return bad


def report_table(report: dict) -> str:
    """consistency_report.json → 终端/Markdown 表格。"""
    lines = [
        f"# 一致性报告 {report.get('run_id', '')}",
        "",
        f"- resolution={report.get('resolution')} ablation={report.get('ablation') or '-'} "
        f"fingerprint={report.get('config_fingerprint')} method={report.get('method')}",
        f"- judge_trusted={report.get('judge_trusted')} coverage={report.get('coverage')}",
        "",
        "| shot | face | outfit | headwear | photoreal | hue_dist | verified |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.get("per_shot", []):
        lines.append(
            f"| {r.get('shot')} | {r.get('face')} | {r.get('outfit')} | {r.get('headwear')} "
            f"| {r.get('photoreal')} | {r.get('hue_dist')} | {r.get('verified')} |"
        )
    lines += ["", "| char | face_rate | outfit_rate | photoreal_rate | max_hue_dist | n |",
              "|---|---|---|---|---|---|"]
    for cid, v in (report.get("per_char") or {}).items():
        lines.append(
            f"| {cid} | {v.get('face_rate')} | {v.get('outfit_rate')} "
            f"| {v.get('photoreal_rate')} | {v.get('max_hue_dist')} | {v.get('n')} |"
        )
    lines += ["", f"summary: {report.get('summary')}"]
    return "\n".join(lines)


def compare_table(reports: list[tuple[str, dict]]) -> str:
    """多 run 对比矩阵：消融的产出物。"""
    head = "| arm | ref_ratio | r2v_ratio | face | outfit | hue | photoreal | shots |"
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for name, rep in reports:
        cov = rep.get("coverage") or {}
        s = rep.get("summary") or {}
        lines.append(
            f"| {name} | {cov.get('ref_attached_ratio')} | {cov.get('r2v_route_ratio')} "
            f"| {s.get('face_consistency')} | {s.get('outfit_consistency')} "
            f"| {s.get('mean_hue_dist')} | {s.get('style_photoreal_rate')} "
            f"| {s.get('shots_measured')} |"
        )
    return "\n".join(lines)
