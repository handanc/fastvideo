"""运行产物存储：output/<run_id>/ 目录契约（TDD §4.3）。"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from fastvideo3.core.config import OUTPUT_DIR

STAGES = ("assets", "storyboard", "audio", "clips", "talk", "review", "final")


def _atomic_write_json(path: Path, data: Any) -> None:
    """临时文件 + os.replace 原子落盘：子进程与 Web API 会并发写同一台账。

    直接 write_text 的读-改-写会丢更新（丢 mark = 漏重做留下旧脸；丢 clear = 白花钱）。
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


_MERGE_LOCKS: dict[str, "threading.Lock"] = {}
_MERGE_LOCKS_GUARD = threading.Lock()


def _merge_lock(key: str) -> "threading.Lock":
    with _MERGE_LOCKS_GUARD:
        return _MERGE_LOCKS.setdefault(key, threading.Lock())


def safe_unlink(path: Path, attempts: int = 4, delay: float = 0.4) -> bool:
    """删除文件，带重试且**永不抛异常**（Windows 占用时返回 False，由调用方回 409）。

    与 safe_promote 同理：产物文件可能正被 /api/files 请求流式发送、被浏览器 <video>
    持有、或被杀软扫描；直接 unlink 抛 PermissionError 会把整个请求打成 500。
    """
    for i in range(attempts):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if i < attempts - 1:
                time.sleep(delay)
    return False


class RunStorage:
    """一个 run 的全部落盘产物。"""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = (root or OUTPUT_DIR) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (self.root / stage).mkdir(exist_ok=True)

    # ── 路径 ──
    def dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: str, name: str) -> Path:
        return self.dir(stage) / name

    # ── JSON 读写 ──
    def save_json(self, name: str, data: Any) -> Path:
        p = self.root / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def merge_json(self, name: str, updates: dict) -> None:
        """把 `updates` 合并进一个 JSON 映射表：读-改-写互斥 + 原子落盘。

        为什么需要：并行生成角色图与场景图时，两边各自"开头读全表、结尾整体覆写"，
        后写完的那路会把先写完的整批抹掉 —— 表现是"图明明在盘上、登记表里没有它"，
        而下游只信登记表。合并写只增改自己那几个键，互斥锁保证两个协程不会
        同时读到旧表再先后覆写。
        """
        path = self.root / name
        with _merge_lock(str(path)):
            data: dict = {}
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        data = raw
                except (OSError, ValueError):
                    data = {}
            data.update(updates)
            _atomic_write_json(path, data)

    # ── 事件流（可观测性）──
    def log_event(self, node: str, status: str, **data: Any) -> None:
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "node": node, "status": status}
        entry.update(data)
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def read_events(self) -> list[dict]:
        p = self.root / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    # ── 口型豁免台账 ──
    def record_lipsync_waiver(self, shot_id: str, reason: str) -> None:
        """记录"该对白镜头按策略显式放弃口型"（如双人镜头 overlay 策略）。

        与"口型失败"必须严格区分：失败是缺口（可修、「开始生成」会继续补），
        豁免是**决定**（不该让「开始生成」无限补一个永远补不出的东西）。
        没有这本台账时，双人镜头会让 videos 步骤永远 gap —— 点多少次都没变化，
        那比静默无口型更糟：它把用户困在死循环里。
        """
        name = "lipsync_waived.json"
        data = self.load_json(name) if self.exists(name) else {}
        data[shot_id] = reason
        self.save_json(name, data)

    def lipsync_waivers(self) -> dict[str, str]:
        if not self.exists("lipsync_waived.json"):
            return {}
        return self.load_json("lipsync_waived.json") or {}

    # ── 降级台账 ──
    def log_degradation(self, node: str, what: str, why: str, impact: str = "") -> None:
        """统一降级台账：事件流 + degradations.json 双写（事件可 grep、json 可汇总）。

        背景：art/storyboard/shots 三处 `except ProviderError → vlm=None` 曾完全静默，
        "vlm 计费 0 条 + best_of_n 无评分"这种全局性失效在盘上不留任何痕迹。
        现在任何"保交付但降级"的路径都必须在此登记。
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_event(node, "degradation", what=what, why=why[:300], impact=impact)
        items: list[dict] = (
            self.load_json("degradations.json") if self.exists("degradations.json") else []
        )
        items.append(
            {"ts": ts, "node": node, "what": what, "why": why[:300], "impact": impact}
        )
        self.save_json("degradations.json", items)

    def read_degradations(self) -> list[dict]:
        if not self.exists("degradations.json"):
            return []
        return self.load_json("degradations.json")

    # ── 血缘与作废台账（lineage.json）──────────────────────────────
    # 每个产物记下它**实际消费了哪些输入**；改/删任一输入 → 沿血缘反查、只标真正
    # 依赖它的下游为 stale。这是"换了定妆照但旧脸混在成片里"与"改了 scene_s1 却
    # 把整个片子重做一遍（烧钱）"两个相反故障的唯一可靠解法。
    LINEAGE = "lineage.json"

    def _lineage(self) -> dict:
        if self.exists(self.LINEAGE):
            try:
                data = self.load_json(self.LINEAGE)
                if isinstance(data, dict):
                    return dict(data.get("edges") or {})
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_lineage(self, edges: dict) -> None:
        _atomic_write_json(self.root / self.LINEAGE, {"edges": edges})

    def _lineage_lock(self) -> "threading.Lock":
        """血缘的读-改-写互斥锁。

        并行生成角色图与场景图时，两路都会各自 `_lineage()` 读全表、改一条、再整表覆写 ——
        后写的把先写那条边的更新抹掉，表现是"下游没被作废、旧片段仍显示完成"。
        只锁最里层的三个改写口（`record_lineage`/`mark_stale`/`clear_stale`），外层组合
        调用不重复取锁，避免自锁。
        """
        return _merge_lock(str(self.root / self.LINEAGE))

    def record_lineage(self, produced: str, inputs: list[str]) -> None:
        """登记"produced 这个产物由 inputs 生成"。重做同一产物即覆盖其上游。"""
        with self._lineage_lock():
            edges = self._lineage()
            edges[produced] = {"inputs": list(inputs), "stale": False, "reason": ""}
            self._save_lineage(edges)

    def is_stale(self, rel: str) -> bool:
        return bool(self._lineage().get(rel, {}).get("stale"))

    def stale_set(self) -> set[str]:
        return {k for k, v in self._lineage().items() if v.get("stale")}

    def dependents_of(self, rel: str) -> list[str]:
        """沿血缘反查所有（含传递）下游产物 —— 作废范围的唯一权威来源。"""
        edges = self._lineage()
        out: list[str] = []
        frontier = [rel]
        seen = {rel}
        while frontier:
            cur = frontier.pop()
            for produced, meta in edges.items():
                if produced in seen:
                    continue
                if cur in (meta.get("inputs") or []):
                    seen.add(produced)
                    out.append(produced)
                    frontier.append(produced)
        return out

    def mark_stale(self, rels: Iterable[str], reason: str) -> int:
        """标记作废，并**自动传递到全部下游**（改场景图 → 帧 → 片段 → 成片）。"""
        with self._lineage_lock():
            return self._mark_stale_locked(rels, reason)

    def _mark_stale_locked(self, rels: Iterable[str], reason: str) -> int:
        edges = self._lineage()
        targets: set[str] = set()
        for rel in rels:
            targets.add(rel)
            targets.update(self.dependents_of(rel))
        changed = 0
        for rel in targets:
            meta = edges.get(rel)
            if meta is None:
                # **凭空写一条"已作废"是垃圾状态**：这条产物既不在盘上、也从没有过血缘
                # （典型来源是把组名打错成 `scene_scene_1.mp4` 这类不存在的相对路径）。
                # 它会永远挂在 lineage 里 —— 本项目反复踩过"遗留状态被当成缺口 → 步骤
                # 永久 gap"，所以在这里一次性拦住：没有实体也没有来路的产物不作废。
                if not (self.root / rel).exists():
                    continue
                meta = {"inputs": [], "stale": True, "reason": reason}
                edges[rel] = meta
                changed += 1
            elif not meta.get("stale"):
                meta["stale"] = True
                meta["reason"] = reason
                changed += 1
        if changed:
            self._save_lineage(edges)
        return changed

    def record_regeneration(self, produced: str, inputs: list[str], reason: str = "") -> int:
        """登记"produced 刚被（重新）生成"：更新血缘、清除自身 stale、**作废全部下游**。

        下游作废是必须的：帧是按旧定妆照画的，定妆照一旦重画，这些帧就是旧脸 ——
        `_find_done_job` 那类"文件在盘即跳过"的逻辑感知不到上游变化，只有血缘能。
        返回被标 stale 的下游数量。
        """
        self.record_lineage(produced, inputs)
        self.clear_stale(produced)
        return self.mark_stale(self.dependents_of(produced), reason or f"{produced} 已重生成")

    def clear_stale(self, rel: str) -> None:
        with self._lineage_lock():
            edges = self._lineage()
            meta = edges.get(rel)
            if meta and meta.get("stale"):
                meta["stale"] = False
                meta["reason"] = ""
                self._save_lineage(edges)

    def upstream_of(self, rel: str) -> list[str]:
        return list(self._lineage().get(rel, {}).get("inputs") or [])


def list_runs(root: Path | None = None) -> list[str]:
    base = root or OUTPUT_DIR
    if not base.exists():
        return []
    runs = [d.name for d in base.iterdir() if d.is_dir() and (d / "run_meta.json").exists()]
    return sorted(runs, reverse=True)


def original_of(path: Path) -> Path:
    """`char_c1.png` → `char_c1.orig.png`（同目录、同后缀，中间插 `.orig`）。"""
    return path.with_name(path.stem + ".orig" + path.suffix)


def preserve_original(path: Path) -> bool:
    """**第一次**要覆盖某个已有产物前，把它快照成 `.orig`；已有就什么都不做。

    为什么不是复用 `.bak.png`：`.bak` 是"上一版"，一路改下去它会变成"上一次生成的那张脸"，
    人想要的『恢复』是回到**最原始**那一版（第一次跑出来、他还没动过的）。只写一次正是
    这个语义：之后再覆盖多少次，`.orig` 始终是原点。
    与 safe_promote 同样绝不抛异常 —— 留底失败最多是"没有恢复按钮"，不能打死生成。
    """
    import shutil

    try:
        src = Path(path)
        if not src.is_file():
            return False
        orig = original_of(src)
        if orig.is_file():
            return False
        shutil.copy2(src, orig)
        return True
    except Exception:  # noqa: BLE001
        return False


def safe_promote(src: Path, dest: Path, attempts: int = 4, delay: float = 0.4) -> bool:
    """把 src 提升为 dest：带重试的复制，**任何情况下都不抛异常**。

    为什么不用 Path.replace()：Windows 上 rename 只要源文件被别的进程握一下
    （浏览器拉 /api/files 缩略图、FileResponse 还在流式发送、杀软实时扫描、
    Explorer 预览面板）就抛 WinError 32。实测这条把整条 cinema run 的 art 节点
    打死了 —— 提升产物是"锦上添花"，绝不能成为管线的单点故障。
    复制而非移动：源（候选图）留着给前端当"落选候选"排查用。
    """
    import shutil
    import time as _time

    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        try:
            shutil.copy2(src, dest)
            # copy2 会保留**源文件**的 mtime → 看板用 mtime 做缓存版本号时会以为
            # 文件没变（浏览器继续显示旧图）。提升即"内容换了"，mtime 必须是现在。
            os.utime(dest, None)
            return True
        except OSError:
            if i < attempts - 1:
                _time.sleep(delay)
    return False
