"""真实成本对账：阿里云账单 vs 项目台账（`cost.json`）。

为什么要有这个脚本：台账金额 = 单价 × 用量。用量已经优先取 API 回传的 `usage`，
但**单价**永远可能失真 —— 免费额度转正式计费、官方调价、模型没登记进价目表
（此时按能力默认价估算）。唯一不会骗人的数字是账单，所以定期拿它对一次。

用法：
    # 看某个账期（默认上个自然月）的真实账单与台账差额
    python scripts/real_cost.py output/run_20261011_231024
    python scripts/real_cost.py output/run_20261011_231024 --cycle 2026-10
    python scripts/real_cost.py --all --cycle 2026-10     # 全部 run 的台账合计 vs 账单

前置：`.env` 里配 `FV3_ALIYUN_ACCESS_KEY_ID` / `FV3_ALIYUN_ACCESS_KEY_SECRET`
（阿里云 RAM 的 AK/SK，需要账单只读权限；**与百炼的 DASHSCOPE_API_KEY 不是同一个东西**）。
没配就如实说"取不到账单"，不会假装这个月没花钱。

⚠ 账单按账期/产品聚合，粒度到"产品+实例"，通常拿不到"哪一次生成花了多少"。
  所以它用来对**总额与量级**，逐条对账仍以 cost.json 的 usage 为准。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastvideo3.core import billing  # noqa: E402


def _last_cycle() -> str:
    today = date.today()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def _ledger_total(run_dir: Path) -> float:
    p = run_dir / "cost.json"
    if not p.is_file():
        return 0.0
    try:
        rows = json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return 0.0
    return round(sum(float(r.get("amount_cny") or 0) for r in rows if isinstance(r, dict)), 4)


def _uncertain(run_dir: Path) -> float:
    """台账里单价不可信（只能按能力默认价估算）的那部分金额。"""
    p = run_dir / "cost.json"
    if not p.is_file():
        return 0.0
    try:
        rows = json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return 0.0
    return round(
        sum(
            float(r.get("amount_cny") or 0)
            for r in rows
            if isinstance(r, dict) and (r.get("meta") or {}).get("price_certain") is False
        ),
        4,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", help="run 目录（不给则需 --all）")
    ap.add_argument("--cycle", default="", help="账期 yyyy-MM（默认上个自然月）")
    ap.add_argument("--all", action="store_true", help="把 output/ 下所有 run 的台账合计起来对账")
    ap.add_argument("--product", action="append", default=[], help="产品码（默认 dashscope、bailian）")
    args = ap.parse_args()

    cycle = args.cycle or _last_cycle()
    if not billing.available():
        _ok, why = billing.query_bill(cycle)
        print(f"✘ 取不到真实账单：{why}")
        print("  → 台账金额仍是「单价 × 用量」的估算；配好 AK/SK 后重跑本脚本即可对账。")
        return 2

    codes = tuple(args.product) if args.product else billing.DEFAULT_PRODUCT_CODES
    bill, why = billing.query_bill(cycle, product_codes=codes)
    if bill is None:
        print(f"✘ 账单查询失败（{cycle}）：{why}")
        return 1
    print(f"账期 {cycle} 真实账单：{bill['currency']} {bill['pretax_amount']}")
    for prod, amt in sorted(bill["by_product"].items()):
        print(f"  {prod}: {amt}")
    if bill.get("errors"):
        print("  ⚠ 部分查询失败：" + "；".join(bill["errors"]))

    if args.all:
        out = Path(args.run_dir) if args.run_dir else Path("output")
        runs = sorted(p for p in out.glob("run_*") if (p / "cost.json").is_file())
        ledger = round(sum(_ledger_total(r) for r in runs), 4)
        unc = round(sum(_uncertain(r) for r in runs), 4)
        print(f"\n台账合计（{len(runs)} 条 run）：¥{ledger}，其中单价为估算的 ¥{unc}")
    elif args.run_dir:
        rd = Path(args.run_dir)
        ledger = _ledger_total(rd)
        unc = _uncertain(rd)
        print(f"\n该 run 台账：¥{ledger}，其中单价为估算的 ¥{unc}")
        print(
            "  ⚠ 账单是**整个账期所有 run 与所有调用**的合计，单条 run 的台账只会小于它；"
            "要逐条对账请看 cost.json 每行的 meta.quantity_source（api_usage=服务端回传，"
            "requested=按请求值估算）与 meta.price_source。"
        )
    else:
        print("\n（给了 run 目录或 --all 才会打印台账对比）")
        return 0

    if args.all:
        rep = billing.reconcile(bill, ledger)
        print(
            f"\n对账：账单 {rep['bill_cny']} vs 台账 {rep['ledger_cny']} "
            f"→ 差 {rep['delta_cny']}（{rep['verdict']}）"
        )
        if rep["delta_cny"] > 0.01:
            print(
                "  处理办法：① 给 pricing.yaml 补真实单价；② 或配 FV3_PRICING_URL 指向"
                "实时价目表（与 pricing.yaml 同构的 JSON，带本地缓存与 TTL）；"
                "③ 免费额度转计费的模型必须从 0.0 改成真实价，否则成本上限永远形同虚设。"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
