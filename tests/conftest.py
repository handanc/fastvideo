"""测试基座：Fake 优先、完全离线（TDD §12.1）。"""

from __future__ import annotations

import pytest

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.storage import RunStorage


@pytest.fixture
def outdir(tmp_path, monkeypatch):
    """把全局输出目录指向 tmp，保证测试不污染项目目录。"""
    from fastvideo3.core import config as config_mod

    d = tmp_path / "output"
    monkeypatch.setattr(config_mod, "OUTPUT_DIR", d)
    return d


@pytest.fixture
def cfg(outdir) -> AppConfig:
    return AppConfig(
        fake=True,
        resolution=(160, 288),  # 小规格加速 ffmpeg 冒烟
        max_parallel_jobs=2,
        budget_limit_cny=100.0,
    )


@pytest.fixture
def storage(tmp_path) -> RunStorage:
    return RunStorage("t_run", root=tmp_path)


@pytest.fixture
def ledger(storage) -> CostLedger:
    return CostLedger(storage, budget_limit_cny=100.0)


@pytest.fixture
def jobs(tmp_path) -> JobStore:
    return JobStore(f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}")
