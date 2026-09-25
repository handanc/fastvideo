"""存储 / 成本台账 / 任务台账测试。"""

from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import JobStatus, RunJob
from fastvideo3.core.storage import RunStorage, list_runs


def test_run_storage_layout_and_json(storage):
    assert (storage.root / "assets").is_dir()
    assert (storage.root / "final").is_dir()
    storage.save_json("x.json", {"a": 1})
    assert storage.load_json("x.json") == {"a": 1}
    assert storage.exists("x.json")


def test_events_roundtrip(storage):
    storage.log_event("node_a", "start", foo=1)
    storage.log_event("node_a", "done")
    events = storage.read_events()
    assert len(events) == 2
    assert events[0]["node"] == "node_a"
    assert events[0]["foo"] == 1


def test_list_runs(tmp_path):
    RunStorage("run_b", root=tmp_path).save_json("run_meta.json", {})
    RunStorage("run_a", root=tmp_path).save_json("run_meta.json", {})
    (tmp_path / "not_a_run").mkdir()
    assert list_runs(tmp_path) == ["run_b", "run_a"]


def test_cost_ledger_record_and_persist(storage):
    ledger = CostLedger(storage, budget_limit_cny=1.0)
    ledger.record("i2v", "fake", "wan2.2-i2v-flash", quantity=5)
    assert ledger.total() > 0
    assert not ledger.exceeded() or ledger.total() >= 1.0
    # 重新加载可恢复
    again = CostLedger(storage, budget_limit_cny=1.0)
    assert again.total() == ledger.total()
    assert "total" in ledger.summary()


def test_cost_budget_exceeded(storage):
    ledger = CostLedger(storage, budget_limit_cny=0.0001)
    ledger.record("talk", "fake", "wan2.2-s2v", quantity=10)
    assert ledger.exceeded()


def test_job_store_crud(tmp_path):
    store = JobStore(f"sqlite:///{(tmp_path / 'j.db').as_posix()}")
    job = RunJob(job_id="j1", run_id="r1", capability="i2v", provider="p", params={"a": 1})
    store.submit(job)
    store.update("j1", status=JobStatus.RUNNING, output_uri="ext-123")
    got = store.get("j1")
    assert got is not None and got.status is JobStatus.RUNNING
    assert store.list_open("r1")[0].job_id == "j1"
    store.update("j1", status=JobStatus.SUCCEEDED, output_uri="out.mp4")
    assert store.get("j1").status is JobStatus.SUCCEEDED
    assert store.list_open("r1") == []
    assert store.get("missing") is None
    store.dispose()


def test_pricing_covers_all_configured_models():
    """定价覆盖一致性：所有档位 yaml 里出现的模型名必须在 pricing.yaml 有条目。

    缺失条目会静默回退到能力默认价（cost.price_of），如 wan2.2-i2v-plus 曾按
    ¥0.11/s 记账（实际 ≈¥0.7/s），预算熔断严重失真。此测试守住护栏本身。
    """
    import yaml

    from fastvideo3.core.config import CONFIGS_DIR, PROJECT_ROOT

    with open(PROJECT_ROOT / "pricing.yaml", encoding="utf-8") as f:
        priced = set((yaml.safe_load(f).get("models") or {}).keys())

    configured: set[str] = set()
    for name in ("default.yaml", "profile_high.yaml", "profile_cinema.yaml"):
        with open(CONFIGS_DIR / name, encoding="utf-8") as f:
            yml = yaml.safe_load(f) or {}
        configured.update((yml.get("models") or {}).values())

    missing = configured - priced
    assert not missing, f"pricing.yaml 缺少模型定价条目（熔断会低估成本）: {sorted(missing)}"
