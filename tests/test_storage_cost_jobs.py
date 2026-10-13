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


def test_price_table_round_trips_without_losing_entries():
    """单价表（pyproject.toml）渲染→再解析必须一条不丢。

    旧版这条查的是 `pricing.yaml` 的**重复键**（PyYAML 静默取最后一个：实测
    worst_case 里 `dashscope_wan22_s2v` 出现两次，480P 的预估被按 720P 记）。
    表搬到 TOML 后重复键由格式本身挡住（tomllib 直接解析失败），风险移到了
    `refresh-prices` 的**写回**环节：渲染时把 `tiers` 摊平成同名键、或者漏掉一个
    字段，都会静默覆盖条目 —— 那和重复键是同一类"只看账单才发现"的失真，
    所以这里做往返检查。
    """
    import tomllib

    from fastvideo3.core import pricing
    from fastvideo3.core.pricing import render_table

    doc = pricing.raw_table()
    assert doc.get("models"), "单价表必须在 pyproject.toml 里（记账的单价来源）"
    again = tomllib.loads(render_table(doc))["tool"]["fastvideo3"]["pricing"]
    assert set(again["models"]) == set(doc["models"]), "渲染/解析往返丢了模型条目"
    assert set(again["worst_case"]) == set(doc["worst_case"]), "渲染/解析往返丢了 worst_case 键"
    for name, spec in doc["models"].items():
        got = again["models"][name]
        assert set(got) >= set(spec) - {"tiers"}, f"{name} 字段被渲染吞掉：{set(spec) - set(got)}"
        if spec.get("tiers"):
            assert set(got["tiers"]) == {k.lower() for k in spec["tiers"]}, f"{name} 分档丢了"


def test_pricing_covers_all_configured_models():
    """定价覆盖一致性：所有档位 yaml 里出现的模型名必须在**单价表**有条目。

    缺失条目会静默回退到能力默认价（`cost.price_of`），如 wan2.2-i2v-plus 曾按
    ¥0.11/s 记账（实际 ¥0.70/s），预算熔断严重失真。此测试守住护栏本身。
    表已搬到 `pyproject.toml` 的 `[tool.fastvideo3.pricing]`，所以读合并视图
    （`pricing.sheet()`：远端 → pyproject → 旧 yaml）而不是只读那两份文件之一。
    """
    import yaml

    from fastvideo3.core.config import CONFIGS_DIR
    from fastvideo3.core.pricing import sheet

    priced = set(sheet()["models"])

    configured: set[str] = set()
    for name in ("default.yaml", "profile_high.yaml", "profile_cinema.yaml"):
        with open(CONFIGS_DIR / name, encoding="utf-8") as f:
            yml = yaml.safe_load(f) or {}
        configured.update((yml.get("models") or {}).values())

    # 分档计价的模型（fal InfiniteTalk：`端点@480p`/`端点@720p`）用前缀匹配认账：
    # 只要它有任何一个档位键就算"有定价"，但**必须覆盖全部档位**（下面的第二道断言），
    # 否则实际发送的那一档会静默回落到能力默认价 —— 熔断失真的老毛病。
    def has_price(model: str) -> bool:
        return model in priced or any(k.startswith(f"{model}@") for k in priced)

    missing = {m for m in configured if not has_price(m)}
    assert not missing, f"单价表缺少模型定价条目（熔断会低估成本）: {sorted(missing)}"

    # 分档模型：**裸键必须存在且有价**。这条是 ¥0 事故的直接防线 —— 裸键缺失时
    # `resolve` 会回落到能力默认价，而实际发的那一档（如 1080P）压根没被计过价。
    # 不再硬编码"必须有 480p+720p"：官网各型号列出的档位本来就不同
    # （wan2.7-i2v 只列 720P/1080P，wan2.2-i2v-plus 只列 480P/1080P），
    # 把不存在的档逼出来只会让人往表里编数字，那比缺档更糟。
    tiered = {m for m in configured if any(k.startswith(f"{m}@") for k in priced)}
    models_table = sheet()["models"]
    for model in tiered:
        tiers = sorted(k.split("@", 1)[1] for k in priced if k.startswith(f"{model}@"))
        assert tiers, f"{model} 声称分档却没有任何档价"
        bare = models_table.get(model) or {}
        assert float(bare.get("price_cny") or 0.0) > 0, (
            f"{model} 缺裸键兜底价（档位对不上时会静默回落默认价）: {tiers}"
        )
        # 裸键取的是**最高档**：红线/预估按最坏情况，不能比任何一档便宜
        top = max(float((models_table.get(f"{model}@{t}") or {}).get("price_cny") or 0.0) for t in tiers)
        assert float(bare["price_cny"]) == top, f"{model} 裸价 {bare['price_cny']} 应等于最高档 {top}"
