""""这张脸是谁给的"必须可对账 —— 在场漏标与参考图错配。

来源是 run_20261007_130730 的逐帧核对（2026-10-08）：声明了在场的镜头确实带对了定妆照
（猪八戒/孙悟空的首帧与立绘同人），但**开场镜头 `characters=[]` 而 action 写着"女王在
台阶上回眸一笑"、台词说话人就是女王** → 没带任何人物参考图，画出来的女王与定妆照是两
个人（米色短裙 vs 橙色曳地礼服）；四人同框的收尾全景同理。m3 只按 `shot.characters`
取参考图，于是"人物来自角色图"这件事只在剧本标全时才成立。

本文件锁住三件事：
1. `effective_cast`：台词说话人（结构化，精确）+ action/camera 点名（别名，只在唯一时认）
   补进在场，且**声明的永远排在前面**（不漏标的角色不能挤掉已锁脸的槽位）；
2. 参考图与"图1是谁"严格成对：某角色资产缺失被跳过时，锁定语料与血缘必须跟着实际的图走
   （旧实现按 `characters[:len(refs)]` 反推，会把张三的脸标注成李四）；
3. VLM 抽查驳回重画时用的是**该镜头自己的**参考图与种子（旧实现复用生成循环留下的循环
   变量，等于把最后一个镜头的脸塞进被驳回的镜头）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.schema import (
    Asset,
    BaseInfo,
    Script,
    Shot,
    ShotType,
    effective_cast,
)
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.providers.fake import FakeT2IProvider, FakeVLMProvider, fake_script


class RefSpy(FakeT2IProvider):
    """记录 (目标文件名, 参考图文件名, 提示词)：证明"哪张图给了哪一镜、说了谁"。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, list[str], str]] = []

    async def generate(self, prompt, **kw):
        raw = kw.get("ref_image") or []
        refs = [raw] if isinstance(raw, Path) else list(raw)
        self.calls.append(
            (Path(kw["dest"]).name, [Path(p).name for p in refs], prompt)
        )
        return await super().generate(prompt, **kw)


def _script(payload: dict) -> Script:
    return Script.model_validate(payload)


QUEEN_SCRIPT = {
    "base_info": {"topic": "女儿国", "style": "真人实拍电影感"},
    "title": "测试",
    "characters": [
        {"character_id": "queen", "name": "女儿国国王", "appearance": "米色短裙，玉坠耳环"},
        {"character_id": "tang_seng", "name": "唐僧", "appearance": "白色僧袍，五佛冠"},
        {"character_id": "ba_jie", "name": "猪八戒", "appearance": "蓝色敞怀长袍，猪耳"},
    ],
    "scenes": [
        {
            "scene_id": "s1",
            "title": "开场",
            "location": "皇宫广场",
            "shots": [
                {
                    "shot_id": "s1_shot1",
                    "scene_id": "s1",
                    "type": "scenery",
                    "duration_sec": 4.0,
                    "characters": [],  # ← 漏标：画面里就是女王
                    "action": "宫殿全景，女王在台阶上回眸一笑",
                    "camera": "Pan Right",
                    "dialogues": [
                        {"character_id": "queen", "text": "御弟哥哥，可愿随臣妾回宫一叙？"}
                    ],
                },
                {
                    "shot_id": "s1_shot2",
                    "scene_id": "s1",
                    "type": "dialogue",
                    "duration_sec": 5.0,
                    "characters": ["tang_seng"],
                    "action": "唐僧抬头瞥见女王，手中佛珠掉落",
                    "camera": "Medium Shot",
                    "dialogues": [{"character_id": "tang_seng", "text": "色即是空……"}],
                },
                {
                    "shot_id": "s1_shot3",
                    "scene_id": "s1",
                    "type": "scenery",
                    "duration_sec": 4.0,
                    "characters": [],
                    "action": "镜头拉远：唐僧、八戒同框发呆",
                    "camera": "Wide Shot",
                },
            ],
        }
    ],
}


def test_speaker_and_text_mentions_join_the_cast():
    script = _script(QUEEN_SCRIPT)
    s1, s2, s3 = (script.all_shots()[i] for i in range(3))

    # 台词说话人是结构化信号：漏标的在场也能精确补上
    assert effective_cast(script, s1) == (["queen"], ["queen"])
    # 声明的永远在前，且没有新线索时名单不变
    assert effective_cast(script, s2) == (["tang_seng"], [])
    # 别名：全名"唐僧"、后缀"八戒"都认；"女王"既非前缀也非后缀 → 不猜
    cast, implied = effective_cast(script, s3)
    assert set(cast) == {"tang_seng", "ba_jie"} and "queen" not in cast
    assert implied == cast


def test_ambiguous_alias_is_not_guessed():
    """两个角色共享同一别名时整个放弃：认错人比不补更糟。"""
    script = _script(
        {
            "base_info": {"topic": "t"},
            "characters": [
                {"character_id": "zhang", "name": "张师傅", "appearance": "蓝工装"},
                {"character_id": "li", "name": "李师傅", "appearance": "灰工装"},
            ],
            "scenes": [
                {
                    "scene_id": "s1",
                    "title": "",
                    "location": "车间",
                    "shots": [
                        {
                            "shot_id": "x1",
                            "scene_id": "s1",
                            "type": "action",
                            "characters": [],
                            "action": "师傅推门进来",
                        }
                    ],
                }
            ],
        }
    )
    assert effective_cast(script, script.all_shots()[0]) == ([], [])


def _assets(script: Script, tmp_path: Path, skip: set[str] = frozenset()) -> dict[str, Asset]:
    out = {}
    for c in script.characters:
        if c.character_id in skip:
            continue
        p = tmp_path / f"char_{c.character_id}.png"
        Image.new("RGB", (8, 8), (200, 30, 30)).save(p)
        out[f"char_{c.character_id}"] = Asset(
            asset_id=f"char_{c.character_id}",
            kind="character",
            owner_id=c.character_id,
            uri=str(p),
        )
    return out


async def test_implied_character_portrait_becomes_the_reference(
    cfg, storage, ledger, tmp_path
):
    """漏标的女王必须真的拿到 char_queen.png：这是"分镜图的人来自角色图"的全部内容。"""
    cfg.use_reference_keyframes = True
    cfg.ref_max_chars = 1
    script = _script(QUEEN_SCRIPT)
    spy = RefSpy()
    await build_storyboard(cfg, spy, script, _assets(script, tmp_path), storage, ledger)

    by_dest = {d: (refs, prompt) for d, refs, prompt in spy.calls}
    assert by_dest["shot_s1_shot1.png"][0] == ["char_queen.png"], "开场镜头没拿到女王的定妆照"
    assert "女儿国国王" in by_dest["shot_s1_shot1.png"][1]
    assert by_dest["shot_s1_shot2.png"][0] == ["char_tang_seng.png"]

    # 血缘记的是"实际送出去的那张图"：换女王定妆照必须作废这一帧
    edges = storage.load_json("lineage.json")["edges"]
    assert "assets/char_queen.png" in edges["storyboard/shot_s1_shot1.png"]["inputs"]

    implied_events = [e for e in storage.read_events() if e.get("status") == "cast_implied"]
    assert {e["shot"] for e in implied_events} == {"s1_shot1", "s1_shot3"}
    queen_ev = next(e for e in implied_events if e["shot"] == "s1_shot1")
    assert queen_ev["implied"] == ["女儿国国王"] and queen_ev["anchored"] == ["女儿国国王"]


async def test_lockface_names_follow_the_images_not_the_declaration(
    cfg, storage, ledger, tmp_path
):
    """资产缺失被跳过时，"图1是谁"必须跟着图走（旧实现会张冠李戴）。"""
    cfg.use_reference_keyframes = True
    cfg.ref_max_chars = 2
    script = _script(QUEEN_SCRIPT)
    assets = _assets(script, tmp_path, skip={"tang_seng"})  # s1_shot3 在场 = [唐僧, 八戒]
    spy = RefSpy()
    await build_storyboard(cfg, spy, script, assets, storage, ledger)

    refs, prompt = next((r, p) for d, r, p in spy.calls if d == "shot_s1_shot3.png")
    assert refs == ["char_ba_jie.png"], "唐僧资产缺失时必须跳过，而不是送一个不存在的文件"
    head = prompt.split("你是 AI 绘画提示词专家")[0]  # 锁定语料在元提示词之前
    assert "画面中猪八戒的脸" in head and "唐僧" not in head, "图1 被标注成了没送出去的那个人"
    edges = storage.load_json("lineage.json")["edges"]
    assert edges["storyboard/shot_s1_shot3.png"]["inputs"] == [
        "assets/char_ba_jie.png",
        f"script#{_fp(script)}",
    ]


async def test_multi_ref_order_matches_the_lockface_labels(cfg, storage, ledger, tmp_path):
    """两张都在时：图序 = 在场顺序 = 锁定语料里的名字顺序（多图协议的命门）。"""
    cfg.use_reference_keyframes = True
    cfg.ref_max_chars = 2
    script = _script(QUEEN_SCRIPT)
    spy = RefSpy()
    await build_storyboard(
        cfg, spy, script, _assets(script, tmp_path), storage, ledger
    )
    refs, prompt = next((r, p) for d, r, p in spy.calls if d == "shot_s1_shot3.png")
    assert refs == ["char_tang_seng.png", "char_ba_jie.png"]
    head = prompt.split("你是 AI 绘画提示词专家")[0]
    assert "图1是唐僧" in head and "图2是猪八戒" in head


def _fp(script: Script) -> str:
    from fastvideo3.core.schema import script_fingerprint

    return script_fingerprint(script)


async def test_spot_check_retry_uses_that_shot_own_reference(cfg, storage, ledger, tmp_path):
    """驳回重画不能拿"上一个镜头"的参考图与种子（旧实现复用循环变量的病灶）。"""
    cfg.use_reference_keyframes = True
    cfg.storyboard_sample_rate = 1.0
    script = fake_script(BaseInfo(topic="t"))
    # 造两个单人镜头，且**最后生成**的那个与被抽查的那个不是同一人
    script.scenes = script.scenes[:1]
    script.scenes[0].shots = [
        Shot(
            shot_id="qa", scene_id="s1", type=ShotType.ACTION, action="小明走进咖啡馆",
            characters=["c1"],
        ),
        Shot(
            shot_id="qb", scene_id="s1", type=ShotType.ACTION, action="小红起身打招呼",
            characters=["c2"],
        ),
    ]
    spy = RefSpy()
    await build_storyboard(
        cfg, spy, script, _assets(script, tmp_path), storage, ledger,
        vlm=FakeVLMProvider(always_pass=False),
    )
    per_shot: dict[str, set[tuple[str, ...]]] = {}
    for dest, refs, _prompt in spy.calls:
        per_shot.setdefault(dest, set()).add(tuple(refs))
    assert spy.calls, "抽查驳回没有触发重画"
    assert per_shot["shot_qa.png"] == {("char_c1.png",)}, per_shot
    assert per_shot["shot_qb.png"] == {("char_c2.png",)}, per_shot


@pytest.mark.parametrize("mode", ["off", "scenery_only"])
async def test_scene_image_never_shares_a_request_with_a_person_ref(
    cfg, storage, ledger, tmp_path, mode
):
    """补进来的人也算"有人"：场景图不再与他同框挤一次请求（scenery_only 的原意）。"""
    cfg.use_reference_keyframes = True
    cfg.scene_ref_mode = mode
    script = _script(QUEEN_SCRIPT)
    assets = _assets(script, tmp_path)
    p = tmp_path / "scene_s1.png"
    Image.new("RGB", (8, 8), (10, 200, 10)).save(p)
    assets["scene_s1"] = Asset(
        asset_id="scene_s1", kind="scene", owner_id="s1", uri=str(p)
    )
    spy = RefSpy()
    await build_storyboard(cfg, spy, script, assets, storage, ledger)
    by_dest = {d: refs for d, refs, _ in spy.calls}
    assert by_dest["shot_s1_shot1.png"] == ["char_queen.png"]
    assert by_dest["shot_s1_shot3.png"] == ["char_tang_seng.png"]
