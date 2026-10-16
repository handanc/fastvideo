"""m2 资产生成（教学第 2 讲：角色一致性与资产锚定）。

角色立绘 + 场景概念图 → 资产库（全片唯一来源）。资产是后续一切生成的锚点，
失败阻断（不做静默降级）。cinema 档角色立绘走 best-of-N + VLM 选优
（asset_candidates>1）：立绘崩会被 m3 参考图/r2v/双人首帧全链路继承放大，源头把关。

2026-10 一致性改造：
- 立绘带确定性种子（同 run 重跑/续跑不跳变；此前完全无种子，锚点本身随机）；
- 风格闸门（asset_style_gate）：实测出现过"定妆照是动漫插画、下游全片真人"的
  源头污染，现在最优候选过不了写实判别会 warn/block 并落降级台账；
- best-of-N 的分数不可伪造：评审异常记 verified=False 且排序恒判负。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastvideo3.core import pause
from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger, llm_meta
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import Asset, Script
from fastvideo3.core.storage import RunStorage, preserve_original, safe_promote
from fastvideo3.providers.base import (
    ModerationError,
    ProviderError,
    T2IProvider,
    VLMProvider,
    filter_kwargs_by_capability,
)

_MAX_RETRY = 2

# 画幅 → 万相文生图 size 参数。单边上限 1440（2026-09-25 真实探单：
# 1080*1920 被拒 "width or height should be between 512 and 1440"）。
# hi_res 供 high/cinema 的 1080P 时间线取最大档，减少上采样细节损失。
ASPECT_SIZES = {"9:16": "720*1280", "16:9": "1280*720", "1:1": "1024*1024"}
ASPECT_SIZES_HI = {"9:16": "810*1440", "16:9": "1440*810", "1:1": "1440*1440"}


#: 场景概念图的负向清单：人物相关的词一个不留。全局 negative_prompt 管画风劣化，
#: 这一份管"画面里不该有谁"——两者职责不同，不能合并（合并后改画风会误删禁人词）。
_SCENE_NEGATIVE = (
    "人物，人像，人影，人群，群像，角色，行人，路人，脸，面孔，手，手臂，肢体，剪影人形，"
    "person, people, crowd, face, hands, human figure, silhouette of person"
)

#: summary 里出现这些字就整句丢掉：场景描述是 LLM 写的，十有八九带着角色名与动作，
#: 直接拼进提示词等于请模型画人。按子句过滤比整段丢弃保留更多可用的氛围信息。
#: 黑名单只能挡住"点名某人"，挡不住**动作与心理状态**：2026-10-14 实测有剧情残片
#: 漏进场景图的"氛围与陈设"槽（"拒绝前行"、"瞬间也被迷住，忘记了自己的身份和誓言"、
#: "打破僵局"）—— 那是人的状态，不是空间的状态，模型据此就会往画面里补人。
_PERSON_HINTS = (
    "人", "僧", "侠", "王", "后", "娘", "娃", "孩", "众", "群", "他", "她",
    "角色", "身影", "面孔", "手", "脸", "走", "说", "喊", "笑", "看",
    # 动作/位移/表情
    "行", "赴", "立", "站", "坐", "躺", "跪", "抱", "拉", "推", "拽", "打", "踢",
    "跳", "转身", "回", "望", "盯", "瞥", "皱眉", "怒", "喜", "哭", "泪", "呆",
    "愣", "点头", "摇头", "挥手", "指", "吸引", "迷", "恋", "吻", "握", "拖",
    # 心理/状态/言语
    "忘记", "想", "念", "觉", "意识", "害羞", "心动", "僵局", "劝", "诫", "誓",
    "身份", "拒绝", "不肯", "不愿", "舍不得", "试图", "决定", "惊呼", "质问",
)

#: 白名单：能进"氛围与陈设"槽的小句必须**说的是空间本身**。summary 按定义是剧情
#: （谁做了什么），靠黑名单一条条追是追不完的 —— 反过来只放行含空间/陈设/光线词的
#: 小句，漏进来的概率从"每条都要想到"变成"必须命中这几十个名词之一"。
_SPACE_WORDS = (
    "街", "道", "路", "巷", "殿", "宫", "堂", "楼", "阁", "亭", "台", "墙", "院",
    "门", "窗", "廊", "阶", "梯", "桥", "塔", "庙", "寺", "村", "城", "市", "集",
    "摊", "帐", "帘", "幕", "桌", "椅", "床", "案", "镜", "画", "灯", "烛", "香",
    "钟", "鼓", "旗", "伞", "车", "船", "马", "树", "林", "草", "花", "山", "石",
    "岩", "洞", "水", "河", "湖", "海", "泉", "井", "田", "园", "沙", "泥", "雪",
    "冰", "雨", "雾", "云", "天", "空", "阳", "光", "影", "色", "烟", "尘", "风",
    "瀑", "池", "阶前", "庭院", "屋", "檐", "柱", "栏", "铺", "坛", "墓", "寨",
)


def _space_clauses(text: str, names: list[str]) -> list[str]:
    """从自由文本里挑出**只描述空间**的小句：先滤掉任何人相关词，再要求命中空间名词。"""
    out: list[str] = []
    for raw in re.split(r"[，。；,;、]", str(text or "")):
        c = raw.strip()
        if not c:
            continue
        if any(n and n in c for n in names):
            continue
        if any(w in c for w in _PERSON_HINTS):
            continue
        if not any(w in c for w in _SPACE_WORDS):
            continue
        out.append(c)
    return out


#: 风格锚里属于"人像/皮肤"的词：对定妆照是对的，对**空景背景**就是在正向提示里
#: 提人 —— 与"画面中没有任何人物"自相矛盾，而模型听正向的那句。
#: 「真人写实」也是正向词：它是"不要卡通"的写法，但 t2i 读到「真人」就可能在空景里
#: 摆一个人 —— 与"没有任何人物"对着干。换成不提人的等价说法。
_PORTRAIT_STYLE_REWRITES = (
    ("真人写实电影剧照", "实拍质感电影剧照"),
    ("真人写实", "实拍质感"),
    ("自然皮肤纹理", ""),
    ("皮肤纹理", ""),
    ("realistic human skin", ""),
    ("human skin", ""),
    ("肤质", ""),
)


def _scene_style(style_suffix: str) -> str:
    """把人像专属的风格词换成不提人的等价说法，并清理剥完留下的空逗号。"""
    s = str(style_suffix or "")
    for w, repl in _PORTRAIT_STYLE_REWRITES:
        s = s.replace(w, repl)
    s = re.sub(r"，\s*[，,]+", "，", s)          # 中文里连出现的逗号
    s = re.sub(r",\s*(?:,\s*)+", ", ", s)        # 英文侧 ", ," 这类悬空分隔
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ，,")


#: 禁人尾句（人工改写全文时若缺了就补上）。背景图里长出人来会污染**每一段**场景片段的
#: 参考素材（模型把参考图里的路人当成场景的一部分画进镜头），这个后果比"提示词不够尊重人"大。
_NO_PERSON_TAIL = "画面中没有任何人物、演员、路人、剪影或人影，也不得出现文字与水印。"


def scene_background_prompt(
    cfg: Any, script: Any, scene: Any, *, override: str = ""
) -> str:
    """纯背景的场景概念图提示词：**空景在前、禁人在后**，正文里绝不出现人或演员。

    为什么过滤是必须的而不是"加一句无人物"就够：t2i 模型对"画面里有什么"的服从度
    远高于对"没有什么"的服从度 —— 提示词正文里写着"猪八戒拉着唐僧"，尾缀一句
    "无人物"压不住它。把带人物的子句从正文里删掉，才是把人物从**正向描述**里移除。

    同一原理适用于两处容易忽略的泄漏：
    - 剧情残片（"拒绝前行"、"被迷住"）说的是人的状态，也要挡在正文外 → 小句必须
      命中空间名词才放行（`_space_clauses`）；
    - 风格锚里的"自然皮肤纹理 / realistic human skin"是给人像用的，背景图带着它
      等于正向着陆强调皮肤 → 剥掉（`_scene_style`）。
    - 收尾构图句也不能写"演员""人物"：改成说地面与景深。
    """
    names = [str(c.name or "") for c in script.characters] + [
        str(c.character_id) for c in script.characters
    ]
    # 场景自己的标题也是剧情（"八戒滞留女儿国街头"），同样不进正文。
    # 但**单字不做子串匹配**（与 `schema._ALIAS_MIN` 同一条规矩）：标题写成"街"时，
    # 子串匹配会把"街道繁华"这种正经空间描述一起误杀 —— 误杀比漏杀更糟，因为它安静地
    # 把提示词变成空壳，没人会去查。单字的线索本来已由 _PERSON_HINTS 覆盖。
    guards = [n for n in names + [str(scene.title or "")] if len(str(n).strip()) >= 2]
    # 人工改写 = **整条提示词全文**（与场景片段的提示词同一套语义：机器模板是默认值，
    # 不是上限；所见即所发）。这里只做一件保命的事：全文里没有禁人声明时补上尾句。
    # 为什么不套用 `_space_clauses` 过滤：那是给"机器从剧情概要里抽空间描述"用的，
    # 拿来筛人写的整条提示词会把风格锚、构图要求一起切碎 —— 人写的东西优先。
    # 残余风险写在 UI 上（改成人物出现在背景图里会污染整段的参考素材），并靠禁人尾句压一层。
    human = (override or "").strip()
    if human:
        if "没有任何人物" in human or "没有人物" in human:
            return human
        return human + _NO_PERSON_TAIL
    bg = _space_clauses(scene.summary, guards)
    parts = [
        "空景场景概念图，画面中没有任何人物、人影或肢体，只画空间本身。",
        f"地点：{scene.location}。时间：{scene.time_of_day}。",
    ]
    if bg:
        parts.append(f"氛围与陈设：{'，'.join(bg[:3])}。")
    else:
        # 滤完什么都没有 —— 这是常态（summary 按定义是剧情）。此时用地点本身做空间描述，
        # 宁可平实也不要塞进"忘记誓言"这种心理残句，那只会把画面带向有人。
        parts.append(
            f"{scene.location}的全景与近景细节完整：地面、墙面、陈设与"
            f"{scene.time_of_day}间的光线方向都要成立。"
        )
    parts.append(f"风格：{script.base_info.style}。{_scene_style(cfg.style_suffix)}".strip())
    parts.append(
        "构图：中景到远景机位，地面与空间留白充足、景深层次清晰，"
        "陈设与透视关系完整，画面以空间本身为主体。"
    )
    return " ".join(p for p in parts if p).strip()


def size_for_aspect(aspect_ratio: str, hi_res: bool = False) -> str:
    table = ASPECT_SIZES_HI if hi_res else ASPECT_SIZES
    return table.get(aspect_ratio, "1440*1440" if hi_res else "1024*1024")


def _cand_name(character_id: str, take: int, cand: int) -> str:
    """候选文件名带 take：删除重生成（take+1）不会复用用户删掉不要的那批脸。"""
    return f"_cand_char_{character_id}_t{take}_{cand}.png"


def _char_seed(character_id: str, appearance: str, cand: int, take: int = 0) -> int:
    """角色立绘确定性种子：同角色同描述同候选号同 take → 同种子。

    续跑/重跑时立绘不跳变（锚点稳定）；候选号参与 key 以保证 best-of-N
    的 N 张确实不同。`take` 是"删除后重生成"的计数（takes.json）：确定性种子下
    删图重画会拿到同一张脸，take+1 才能让重生成真的换一张。
    """
    import hashlib

    key = f"char|{character_id}|{appearance}|{cand}|{take}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


async def build_assets(
    cfg: AppConfig,
    t2i: T2IProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
    vlm: VLMProvider | None = None,
    fallback: T2IProvider | None = None,
    kinds: set[str] | None = None,
    only: set[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, Asset]:
    """生成角色立绘与场景概念图。

    分步生成依赖三个参数：
    - `kinds`：只跑 `{"character"}` 或 `{"scene"}`（第 2/3 步要能拆开）；
    - `only`：只补缺口（`{"c1"}` 这类 cid/sid 集合），其余复用盘上产物；
    - `skip_existing`：在盘且未被血缘标 stale 的资产直接复用 —— 反复点击
      不该把已花钱的图重新生成覆盖掉。
    """
    from fastvideo3.core.schema import script_fingerprint

    kinds = kinds or {"character", "scene"}
    fp = f"script#{script_fingerprint(script)}"
    # 旧 registry 先读进来 merge：部分生成时不能把别的键丢掉（此前整体覆写）
    old_registry: dict = (
        storage.load_json("asset_registry.json") if storage.exists("asset_registry.json") else {}
    )
    registry: dict[str, Asset] = {k: Asset(**v) for k, v in old_registry.items()}

    def _publish(a: Asset) -> None:
        """登记一张资产：本地表 + **合并写**落盘。

        整体覆写会让"角色图与场景图并行生成"互相抹掉对方的条目（各自开头读全表、
        结尾写全表）。合并写只动自己那一个键，两条路可以安全并发。
        """
        registry[a.asset_id] = a
        storage.merge_json("asset_registry.json", {a.asset_id: a.model_dump()})
    size = size_for_aspect(cfg.aspect_ratio, hi_res=cfg.resolution[0] >= 1080)
    n_cand = max(1, cfg.asset_candidates) if vlm is not None else 1
    # 人工上传的资产是"用户意图"，重跑本阶段绝不能覆盖（否则换定妆照毫无意义）
    uploaded: dict = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}
    # 删除计数：删过几次就换几次种子，保证"删除→重新生成"不是同一张脸
    takes: dict = storage.load_json("takes.json") if storage.exists("takes.json") else {}

    def _reuse(name: str, owner: str, kind: str) -> Asset | None:
        """返回可复用的 Asset；None 表示必须重新生成。"""
        rel = f"assets/{name}"
        dest = storage.path("assets", name)
        if not dest.exists() or dest.stat().st_size == 0:
            return None
        # 台账键是 char_{cid} / scene_{sid}（与文件名同），不是 kind_{owner}
        # （kind 是 "character"/"scene"）—— 拼错会让 registry.get 恒为 None，
        # "只补 c1" 就会把 c2 也重画一遍（白花钱 + 换脸）。
        aid = name.rsplit(".", 1)[0]
        if only is not None and owner not in only:
            return registry.get(aid)  # 不在本轮补齐名单里：沿用旧登记，不重画
        if any(k == name or k.endswith("/" + name) for k in uploaded):
            storage.log_event("m2_assets", "reuse_uploaded", asset=name)
            meta = {"uploaded": True}
        elif skip_existing and not storage.is_stale(rel):
            storage.log_event("m2_assets", "reuse_existing", asset=name)
            meta = {}
        else:
            return None  # 被血缘标 stale → 必须重做
        base_meta = dict(registry[aid].meta) if aid in registry else {}
        base_meta.update(meta)
        return Asset(asset_id=aid, kind=kind, owner_id=owner, uri=str(dest), meta=base_meta)  # type: ignore[arg-type]

    # 角色立绘（style_suffix 全片风格锚，防止每格随机画风）
    for ch in script.characters:
        if "character" not in kinds:
            continue
        reused = _reuse(f"char_{ch.character_id}.png", ch.character_id, "character")
        if reused is not None:
            _publish(reused)
            continue
        if pause.stop_for_budget(storage, ledger, node="m2_assets", unit=ch.character_id):
            break  # 成本到线：停下来等人工确认（core/pause.py），剩下的仍是缺口
        prompt = render(
            "char_portrait.v1",
            char_name=ch.name,
            appearance=ch.appearance,
            style=script.base_info.style,
            style_suffix=cfg.style_suffix,
        )
        dest = storage.path("assets", f"char_{ch.character_id}.png")
        meta: dict = {"appearance": ch.appearance}
        take = int(takes.get(ch.character_id, 0))
        if n_cand > 1:
            meta, dest = await _gen_best_char(
                cfg, t2i, vlm, prompt, ch, dest, storage, ledger, size, n_cand,  # type: ignore[arg-type]
                fallback=fallback,
                take=take,
            )
        else:
            name, model, gen_meta = await _gen_with_retry(
                t2i,
                prompt,
                dest,
                ledger,
                f"char:{ch.character_id}",
                size=size,
                negative_prompt=cfg.negative_prompt,
                seed=_char_seed(
                    ch.character_id, ch.appearance, 0, int(takes.get(ch.character_id, 0))
                ),
                fallback=fallback,
                storage=storage,
            )
            meta.update({"provider": name, "model": model, "seed": gen_meta.get("seed"),
                         **gen_meta.get("extra", {})})
            gate = await _style_gate(cfg, vlm, dest, ch.character_id, storage, ledger)
            if gate:
                meta.update(gate)
        asset = Asset(
            asset_id=f"char_{ch.character_id}",
            kind="character",
            owner_id=ch.character_id,
            uri=str(dest),
            meta=meta,
        )
        _publish(asset)
        # 血缘：立绘 ← 剧本指纹；重画立绘 → 所有消费过它的帧/片段/成片自动作废
        storage.record_regeneration(
            f"assets/char_{ch.character_id}.png", [fp], reason="定妆照重生成"
        )

    # 场景概念图必须是**纯背景**：它现在是场景级生成的 `reference_image` 之一，
    # 提示词里写明"空景背景（图中无人）"。若图里带了人，模型会把参考图里的人
    # 当成场景的一部分画进每个镜头 —— 多出来的路人比缺参考更难查。
    for scene in script.scenes:
        if "scene" not in kinds:
            continue
        reused = _reuse(f"scene_{scene.scene_id}.png", scene.scene_id, "scene")
        if reused is not None:
            _publish(reused)
            continue
        if pause.stop_for_budget(storage, ledger, node="m2_assets", unit=scene.scene_id):
            break
        prompt = scene_background_prompt(
            cfg, script, scene,
            override=_scene_prompt_override(storage, scene.scene_id),
        )
        dest = storage.path("assets", f"scene_{scene.scene_id}.png")
        await _gen_with_retry(
            t2i,
            prompt,
            dest,
            ledger,
            f"scene:{scene.scene_id}",
            size=size,
            negative_prompt=_SCENE_NEGATIVE + (f"，{cfg.negative_prompt}" if cfg.negative_prompt else ""),
            fallback=fallback,
            storage=storage,
        )
        asset = Asset(
            asset_id=f"scene_{scene.scene_id}",
            kind="scene",
            owner_id=scene.scene_id,
            uri=str(dest),
            meta={"location": scene.location},
        )
        _publish(asset)
        storage.record_regeneration(
            f"assets/scene_{scene.scene_id}.png", [fp], reason="场景图重生成"
        )

    storage.log_event("m2_assets", "done", count=len(registry))
    return registry


async def _style_gate(
    cfg: AppConfig,
    vlm: VLMProvider | None,
    image: Path,
    char_id: str,
    storage: RunStorage,
    ledger: CostLedger,
) -> dict:
    """立绘写实风格闸门。返回写入 Asset.meta 的字段。

    warn：记事件+降级台账，交付继续（meta.style_verified=False 供下游/报告辨识）；
    block：抛错 —— 资产崩比带病交付便宜（m2 现有"失败阻断"哲学）。
    """
    if vlm is None or cfg.asset_style_gate == "off":
        return {}
    from fastvideo3.core.schema import StyleVerdict

    try:
        verdict = await vlm.review_json([image], schema=StyleVerdict, rubric=render("style_gate.v1"))
        ledger.record(
            "vlm",
            vlm.name,
            vlm.model,
            note=f"style_gate:{char_id}",
            meta=llm_meta(vlm, purpose="style_gate", n_media=1, verified=True),
        )
    except Exception as e:  # noqa: BLE001  # 闸门自身异常不阻塞交付，但要可见
        storage.log_degradation(
            "m2_assets", "style_gate_error", str(e), impact="立绘风格未校验即放行"
        )
        return {"style_verified": None, "medium": ""}
    if verdict.photoreal:
        return {"style_verified": True, "medium": verdict.medium, "gate_score": verdict.score}
    storage.log_event(
        "m2_assets",
        "char_style_gate_failed",
        char=char_id,
        medium=verdict.medium,
        score=verdict.score,
    )
    storage.log_degradation(
        "m2_assets",
        "char_style_gate_failed",
        f"{char_id} 立绘被判为 {verdict.medium}",
        impact="动漫/3D 立绘会经 m3 参考图继承到全片首帧",
    )
    if cfg.asset_style_gate == "block":
        raise ProviderError(f"角色立绘未过写实闸门（medium={verdict.medium}）: {char_id}")
    return {"style_verified": False, "medium": verdict.medium, "gate_score": verdict.score}


def _scene_prompt_override(storage: RunStorage | None, scene_id: str) -> str:
    """界面上改写过的项目背景提示词正文（`run_meta.scene_image_prompts`）。

    读在 m2、写在 API，两边共用 run_meta 这一个来源 —— 另存一份就会出现
    「界面显示的是改过的、实际画的是机器模板」。
    """
    if storage is None:
        return ""
    try:
        meta = storage.load_json("run_meta.json") or {}
    except Exception:  # noqa: BLE001  # 读不到就当没改过，退回机器模板
        return ""
    return str((meta.get("scene_image_prompts") or {}).get(scene_id) or "")


async def _gen_with_retry(
    t2i: T2IProvider,
    prompt: str,
    dest,
    ledger: CostLedger,
    tag: str,
    size: str = "1024*1024",
    negative_prompt: str = "",
    seed: int | None = None,
    fallback: T2IProvider | None = None,
    storage: RunStorage | None = None,
) -> tuple[str, str, dict]:
    """生成一张资产图。返回 (provider, model, meta)。

    fallback 分支按能力位过滤 seed/ref（此前一律丢弃且无痕迹）。
    """
    # 重画会直接盖掉盘上那张脸 —— 先留一份「最原始」，界面才有「恢复」可点
    preserve_original(Path(dest))
    kwargs: dict = {"size": size, "negative_prompt": negative_prompt, "seed": seed}
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await t2i.generate(prompt, dest=dest, **kwargs)
            ledger.record(
                "t2i", t2i.name, t2i.model, note=tag,
                meta={"route": "asset", "seed": seed},
            )
            return t2i.name, t2i.model, {"seed": seed, "extra": {}}
        except ModerationError:
            prompt = prompt + "。画面积极健康，无任何敏感元素。"
            last_err = ProviderError(f"{tag}: moderation rejected, retry {attempt}")
        except Exception as e:  # noqa: BLE001
            last_err = e
    if fallback is not None:
        fb_kwargs, dropped = filter_kwargs_by_capability(fallback, kwargs)
        try:
            await fallback.generate(prompt, dest=dest, **fb_kwargs)
            ledger.record(
                "t2i", fallback.name, fallback.model, note=f"{tag}#fallback",
                meta={"route": "asset", "dropped": dropped, "seed": seed},
            )
            if storage is not None:
                storage.log_event(
                    "m2_assets", "fallback_used", tag=tag, provider=fallback.name,
                    dropped=dropped, seed_value=seed,
                )
            return fallback.name, fallback.model, {
                "seed": fb_kwargs.get("seed"),
                "extra": {"dropped": dropped, "fallback": True},
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ProviderError(f"资产生成失败（已重试{_MAX_RETRY}次）: {tag}: {last_err}")


async def _gen_best_char(
    cfg: AppConfig,
    t2i: T2IProvider,
    vlm: VLMProvider,
    prompt: str,
    ch,
    dest: Path,
    storage: RunStorage,
    ledger: CostLedger,
    size: str,
    n_cand: int,
    fallback: T2IProvider | None = None,
    take: int = 0,
) -> tuple[dict, Path]:
    """角色立绘 best-of-N + VLM 选优（<¥1/片，防立绘崩被全链路继承放大）。

    返回 (写入 Asset.meta 的字段, 实际生效的定妆照路径) —— 提升失败时后者指向候选
    文件，保证锚点可用、管线不中断。
    落选候选保留在 assets/_cand_*.png 供人工排查与 ✔ 改选。评审异常的候选记
    verified=False 且排序恒判负 —— 此前 except→0.5 会让"没评上"压过"评了但分低"。
    """
    cands: list[Path] = []
    seeds: dict[str, int] = {}
    # 复用**同 take** 已在盘的候选：上次崩在"提升"那一步时，图其实已经花钱生成好了，
    # 续跑不该再抽一遍。旧 take 的候选不复用（那是用户删掉不要的脸），但留在盘上
    # 供看板回看/✔ 改选 —— 文件名里的 t{take} 就是这道区分。
    for i in range(n_cand):
        existing = storage.path("assets", _cand_name(ch.character_id, take, i))
        if existing.exists() and existing.stat().st_size > 0:
            cands.append(existing)
            seeds[str(existing)] = _char_seed(ch.character_id, ch.appearance, i, take)
    for i in range(len(cands), n_cand):
        cdest = storage.path("assets", _cand_name(ch.character_id, take, i))
        seed = _char_seed(ch.character_id, ch.appearance, i, take)
        try:
            _, _, _ = await _gen_with_retry(
                t2i,
                prompt,
                cdest,
                ledger,
                f"char:{ch.character_id}#c{i}",
                size=size,
                negative_prompt=cfg.negative_prompt,
                seed=seed,
                fallback=fallback,
                storage=storage,
            )
            cands.append(cdest)
            seeds[str(cdest)] = seed
        except ProviderError as e:
            storage.log_event(
                "m2_assets", "candidate_failed", char=ch.character_id, cand=i, error=str(e)[:200]
            )
    if not cands:
        raise ProviderError(f"角色立绘候选全部失败: char:{ch.character_id}")

    rubric = render("char_qc.v1", appearance=ch.appearance)
    best, best_score = cands[0], None
    scores: list[float | None] = []
    verified_any = False
    for c in cands:
        try:
            r = await vlm.review([c], rubric=rubric)
            score: float | None = float(r.score)
            verified = bool(r.verified)
            ledger.record(
                "vlm", vlm.name, vlm.model, note=f"char_qc:{ch.character_id}",
                meta=llm_meta(vlm, purpose="char_qc", n_media=1, verified=verified),
            )
        except Exception:  # noqa: BLE001  # 评审失败不阻塞：记 None，排序恒判负
            score, verified = None, False
        verified_any = verified_any or verified
        scores.append(score)
        if score is not None and (best_score is None or score > best_score):
            best, best_score = c, score
    pick_name = best.name
    # 最优候选 → 正式路径（registry/下游 m3 引用点不变）。
    # 复制而非 rename：Windows 上候选图常被文件请求/杀软短暂握住，rename 抛
    # WinError 32 曾把整条 run 打死（run_20261002_132909 实测）。
    effective = dest
    preserve_original(Path(dest))  # 抽卡选优同样会覆盖已有图
    if not safe_promote(best, dest):
        effective = best  # 提升失败也要继续跑：直接把候选路径当锚点
        storage.log_degradation(
            "m2_assets",
            "portrait_promote_failed_use_candidate_path",
            f"{pick_name} → {dest.name} 复制失败（文件被占用）",
            impact="定妆照路径指向候选文件；下游一致性不受影响，但文件名不合约定",
        )
    storage.log_event(
        "m2_assets",
        "char_best_of_n" if verified_any else "best_of_n_unscored",
        char=ch.character_id,
        n=len(cands),
        pick=pick_name,
        scores=scores,
        verified=verified_any,
        margin=round(best_score, 3) if best_score is not None else None,
    )
    gate = await _style_gate(cfg, vlm, effective, ch.character_id, storage, ledger)
    return (
        {
            "provider": t2i.name,
            "model": t2i.model,
            "seed": seeds.get(str(best)),
            "qc_score": best_score,
            **gate,
        },
        effective,
    )
