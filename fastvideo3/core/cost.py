"""成本台账：按 pricing.yaml 记账、汇总、熔断（TDD §7.3 / §11）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fastvideo3.core.config import PROJECT_ROOT
from fastvideo3.core.schema import CostEntry
from fastvideo3.core.storage import RunStorage

_pricing_cache: dict[str, Any] | None = None


def load_pricing(path: Path | None = None) -> dict[str, Any]:
    global _pricing_cache
    if _pricing_cache is None or path is not None:
        p = path or PROJECT_ROOT / "pricing.yaml"
        with open(p, encoding="utf-8") as f:
            _pricing_cache = yaml.safe_load(f) or {}
    return _pricing_cache


def price_of(model: str, capability: str, quantity: float = 1.0) -> float:
    """按模型单价计费；未知模型回退到该能力的默认单价。"""
    pricing = load_pricing()
    spec = (pricing.get("models") or {}).get(model) or (pricing.get("defaults") or {}).get(
        capability, {}
    )
    return float(spec.get("price_cny", 0.0)) * quantity


class CostLedger:
    """单个 run 的成本流水，落盘于 cost.json。"""

    def __init__(self, storage: RunStorage, budget_limit_cny: float = 150.0) -> None:
        self.storage = storage
        self.budget_limit_cny = budget_limit_cny
        self.entries: list[CostEntry] = []
        if storage.exists("cost.json"):
            self.entries = [CostEntry(**e) for e in storage.load_json("cost.json")]

    def record(
        self,
        capability: str,
        provider: str,
        model: str,
        quantity: float = 1.0,
        note: str = "",
    ) -> CostEntry:
        amount = price_of(model, capability, quantity)
        entry = CostEntry(
            capability=capability,
            provider=provider,
            model=model,
            amount_cny=round(amount, 4),
            note=note,
        )
        self.entries.append(entry)
        self._flush()
        return entry

    def _flush(self) -> None:
        self.storage.save_json("cost.json", [e.model_dump(mode="json") for e in self.entries])

    def total(self) -> float:
        return round(sum(e.amount_cny for e in self.entries), 4)

    def exceeded(self) -> bool:
        return self.total() >= self.budget_limit_cny

    def summary(self) -> dict[str, float]:
        by_cap: dict[str, float] = {}
        for e in self.entries:
            by_cap[e.capability] = round(by_cap.get(e.capability, 0.0) + e.amount_cny, 4)
        by_cap["total"] = self.total()
        return by_cap
