"""资产装载：把"登记表 + 盘上实际有的图"合并成下游能发的资产字典。

为什么需要这一层（真实事故）：用户可以在界面上**跳过 m2** 直接上传角色图/场景图，
`PUT /artifacts` 只写文件与 `uploads.json` 台账，不写 `asset_registry.json`。
下游一律读登记表 → 登记表不存在就等于"本片没有任何参考图"，于是：

- 场景级生成（wan3.0 参考形态）组不到一张图，`references=[]`；
- 本地规格守卫把请求拦下（官方要求首帧或参考素材二选一），**账号一分没扣**，
  但界面上只看到"片段还没出来"，谁也说不清为什么。

所以装载规则是：**盘上存在的图就是资产**，登记表只是补充元信息（外观描述、候选池）。
登记表优先，盘上有而登记表没有的补一条，uri 一律解析成绝对路径。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core.schema import Asset
from fastvideo3.core.storage import RunStorage

_PREFIX_KIND = (("char_", "character"), ("scene_", "scene"))


def resolve_path(storage: RunStorage, rel_or_abs: str, *, fallback_rel: str = "") -> Path:
    """把一个 uri 字符串解析成**绝对路径**：绝对 → 原样；相对 → 相对 run 根。

    `fallback_rel` 是同一资产的规范落盘位置（如 `assets/char_c1.png`）：登记表里的 uri
    可能指向候选目录或被手工清理过，规范位置存在就用它，别把"图其实就在盘上"误判成缺图。
    """
    raw = str(rel_or_abs or "").strip().replace("\\", "/")
    cands: list[Path] = []
    if raw:
        p = Path(raw)
        cands.append(p if p.is_absolute() else storage.root / p)
        name = Path(raw).name
        if name:
            cands.append(storage.root / name)
    if fallback_rel:
        cands.append(storage.root / str(fallback_rel).replace("\\", "/"))
    for c in cands:
        if c.is_file():
            return c
    return cands[0] if cands else Path(raw)


def _kind_of(stem: str) -> tuple[str, str] | None:
    for prefix, kind in _PREFIX_KIND:
        if stem.startswith(prefix):
            return kind, stem[len(prefix):]
    return None


def load_assets(storage: RunStorage) -> dict[str, Asset]:
    """run 的可用资产：登记表 ∪ `assets/` 目录里实际存在的角色图/场景图。

    返回键沿用登记表命名（`char_{character_id}` / `scene_{scene_id}`），
    `uri` 一律是绝对路径 —— 下游 `Path(uri).exists()` 的判据从此与进程 CWD 无关。
    """
    raw = storage.load_json("asset_registry.json") if storage.exists("asset_registry.json") else {}
    out: dict[str, Asset] = {}
    for key, val in (raw or {}).items():
        try:
            a = Asset(**val)
        except Exception:  # noqa: BLE001 - 登记表被手改坏时不能让整步崩掉
            continue
        kind_owner = _kind_of(str(key))
        if kind_owner is None:
            out[key] = a
            continue
        kind, owner = kind_owner
        a.uri = str(resolve_path(storage, a.uri, fallback_rel=f"assets/{key}.png"))
        if not a.asset_id:
            a.asset_id = str(key)
        if a.kind != kind:
            a.kind = kind  # type: ignore[assignment]
        if not a.owner_id:
            a.owner_id = owner
        out[key] = a

    adir = storage.root / "assets"
    if adir.is_dir():
        for png in sorted(adir.glob("*.png")):
            kind_owner = _kind_of(png.stem)
            if kind_owner is None:
                continue
            kind, owner = kind_owner
            key = png.stem
            if existing := out.get(key):
                if not Path(str(existing.uri)).is_file():
                    existing.uri = str(png)
                continue
            out[key] = Asset(
                asset_id=key,
                kind=kind,  # type: ignore[arg-type]
                owner_id=owner,
                uri=str(png),
                meta={"source": "disk"},
            )
    return out


def refs_on_disk(
    storage: RunStorage,
    *,
    characters,
    scene_ids,
    overrides: dict | None = None,
) -> list[Path]:
    """本组**盘上真有**的参考素材路径（立绘 + 空景背景），用于选路前的物理前提判断。

    与 m4_scene 的解析同一套规则（登记表为空时回落到规范落盘位置），但这里不读登记表 ——
    `compute_progress` 每 3 秒跑一次，为一个"有没有图"的判断去解析整张登记表不值当。
    """
    ov = overrides or {}
    picked = ov.get("char_images") or {}
    out: list[Path] = []
    for cid in characters or ():
        rel = str(picked.get(cid) or f"assets/char_{cid}.png")
        p = resolve_path(storage, rel)
        if p.is_file():
            out.append(p)
    for sid in scene_ids or ():
        p = storage.root / f"assets/scene_{sid}.png"
        if p.is_file():
            out.append(p)
    return out


def asset_file(
    storage: RunStorage, assets: dict[str, Asset], key: str, *, owner_id: str = ""
) -> Path | None:
    """某资产在盘上的文件；一个都没有时返回 None（调用方据此降级，别发空素材）。"""
    a = assets.get(key)
    fallback = f"assets/{key}.png"
    if a is not None:
        p = resolve_path(storage, str(a.uri), fallback_rel=fallback)
        if p.is_file():
            return p
    direct = storage.root / fallback
    return direct if direct.is_file() else None
