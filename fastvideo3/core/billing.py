"""真实账单（阿里云 BSS OpenAPI）：把"我们自己算的成本"与"云账单实际扣了多少"对上。

**为什么需要它**：台账金额 = 单价 × 用量。用量这一侧已经改成优先取 API 回传的 `usage`
（官方"只对成功的结果计数"），但单价这一侧永远可能失真 —— 模型从免费额度转正式计费、
官方调价、我们表里漏登记（此时按能力默认价估算），台账都不会知道。唯一不会骗人的数字
是**账单**。所以：账单为准，台账为解释。

**边界（都如实说，不假装）**：
- 需要阿里云 **AK/SK**（`FV3_ALIYUN_ACCESS_KEY_ID` / `FV3_ALIYUN_ACCESS_KEY_SECRET`），
  与百炼的 `DASHSCOPE_API_KEY` **不是同一个东西**；没配就返回 unavailable + 原因，
  不猜、不编、不静默返回 0。
- 本机没装阿里云 SDK，所以按官方 ACS3-HMAC-SHA256 规范自己签名（`hmac`/`hashlib` 标准库）。
  规范请求串的构造有单元测试钉住；**首次真实调用前无法验证签名被服务端接受**，
  失败会返回原始报文而不是伪装成"这个月没花钱"。
- 账单按**账期/产品**聚合，粒度到"产品 + 实例"，通常拿不到"哪一次生成花了多少"。
  所以它的用途是对总额与量级，不是逐条对账。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

ENDPOINT = "business.aliyuncs.com"
API_VERSION = "2017-12-14"
ALGORITHM = "ACS3-HMAC-SHA256"
#: 百炼/灵积在账单里的产品码（不同账号可能记在其中一个下，两个都查再合并）
DEFAULT_PRODUCT_CODES = ("dashscope", "bailian")


def credentials() -> tuple[str, str]:
    return (
        (os.getenv("FV3_ALIYUN_ACCESS_KEY_ID") or "").strip(),
        (os.getenv("FV3_ALIYUN_ACCESS_KEY_SECRET") or "").strip(),
    )


def available() -> bool:
    ak, sk = credentials()
    return bool(ak and sk)


def _rfc3986(value: str) -> str:
    """阿里云要求的百分号编码：空格编成 %20（不是 +），`*` 编成 %2A，`~` 不编。"""
    return quote(str(value), safe="~").replace("+", "%20").replace("*", "%2A")


def canonical_query(params: dict[str, Any]) -> str:
    """规范化查询串：**按 key 字典序**排序后 k=v 用 & 连接（顺序错签名就错）。"""
    return "&".join(
        f"{_rfc3986(k)}={_rfc3986(v)}" for k, v in sorted(params.items()) if v is not None
    )


def canonical_request(
    method: str, uri: str, query: dict[str, Any], headers: dict[str, str], payload: bytes = b""
) -> tuple[str, str]:
    """构造规范请求串，返回 (CanonicalRequest, SignedHeaders)。

    格式（官方 V3 签名）：
        HTTPMethod \\n CanonicalURI \\n CanonicalQueryString \\n
        CanonicalHeaders \\n SignedHeaders \\n HashedRequestPayload
    其中 CanonicalHeaders 只收 `host` 与 `x-acs-*`，按名字字典序、`名:值\\n` 逐行拼。
    """
    lower = {k.lower(): str(v).strip() for k, v in headers.items()}
    signed = sorted(k for k in lower if k == "host" or k.startswith("x-acs-"))
    canon_headers = "".join(f"{k}:{lower[k]}\n" for k in signed)
    hashed_payload = hashlib.sha256(payload or b"").hexdigest()
    req = "\n".join(
        [
            method.upper(),
            uri,
            canonical_query(query),
            canon_headers,
            ";".join(signed),
            hashed_payload,
        ]
    )
    return req, ";".join(signed)


def sign(secret: str, canonical: str) -> str:
    """StringToSign = 算法名 + \\n + Hex(SHA256(CanonicalRequest))，再用 SK 做 HMAC。"""
    string_to_sign = ALGORITHM + "\n" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def query_bill(
    billing_cycle: str,
    *,
    product_codes: tuple[str, ...] = DEFAULT_PRODUCT_CODES,
    timeout: float = 20.0,
) -> tuple[dict[str, Any] | None, str]:
    """查一个账期（`yyyy-MM`）的实例账单。返回 `(汇总或 None, 说明)`；**任何失败都不抛**。

    汇总形如：`{"cycle": "2026-10", "currency": "CNY", "pretax_amount": 12.34,
                "by_product": {...}, "items": [...], "source": "aliyun_bss"}`。
    """
    if not available():
        return None, (
            "未配置阿里云 AK/SK（FV3_ALIYUN_ACCESS_KEY_ID / FV3_ALIYUN_ACCESS_KEY_SECRET）；"
            "注意它与百炼的 DASHSCOPE_API_KEY 不是同一个凭据。"
            "没配就取不到真实账单 —— 台账金额仍是「单价×用量」的估算。"
        )
    ak, sk = credentials()
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return None, "缺 httpx，无法请求账单接口"

    total = 0.0
    by_product: dict[str, float] = {}
    items: list[dict[str, Any]] = []
    currency = "CNY"
    errors: list[str] = []
    for code in product_codes:
        params = {
            "BillingCycle": billing_cycle,
            "ProductCode": code,
            "Granularity": "MONTHLY",
            "MaxResults": 300,
        }
        headers = {
            "host": ENDPOINT,
            "x-acs-action": "QueryInstanceBill",
            "x-acs-version": API_VERSION,
            "x-acs-date": _utc_now(),
            "x-acs-signature-nonce": uuid.uuid4().hex,
        }
        canon, signed_headers = canonical_request("GET", "/", params, headers)
        signature = sign(sk, canon)
        headers["Authorization"] = (
            f"{ALGORITHM} Credential={ak},SignedHeaders={signed_headers},Signature={signature}"
        )
        try:
            url = f"https://{ENDPOINT}/?{canonical_query(params)}"
            r = httpx.get(url, headers=headers, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{code}: {type(e).__name__}: {e}")
            continue
        if r.status_code != 200:
            # 原文回传：签名/权限/账期错都长得不一样，吞掉就等于让用户猜
            errors.append(f"{code}: HTTP {r.status_code} {r.text[:300]}")
            continue
        try:
            data = r.json()
        except ValueError:
            errors.append(f"{code}: 响应不是 JSON")
            continue
        body = (data.get("Data") or {})
        rows = body.get("Items") or []
        if isinstance(rows, dict):  # 有的版本把列表包在 Item 里
            rows = rows.get("Item") or []
        for it in rows:
            amt = float(it.get("PretaxAmount") or 0.0)
            total += amt
            prod = str(it.get("ProductCode") or code)
            by_product[prod] = round(by_product.get(prod, 0.0) + amt, 6)
            items.append(
                {
                    "product": prod,
                    "instance_id": it.get("InstanceID") or "",
                    "item": it.get("Item") or "",
                    "usage": it.get("Usage"),
                    "usage_unit": it.get("UsageUnit") or "",
                    "pretax_amount": amt,
                    "billing_date": it.get("BillingDate") or "",
                }
            )
        currency = str(body.get("Currency") or currency)
    if errors and not items:
        return None, "；".join(errors)
    return {
        "cycle": billing_cycle,
        "currency": currency,
        "pretax_amount": round(total, 6),
        "by_product": by_product,
        "items": items,
        "source": "aliyun_bss",
        "errors": errors,
    }, ("ok" if not errors else "部分产品码查询失败：" + "；".join(errors))


def cycles_of(entries: list[dict[str, Any]]) -> list[str]:
    """台账条目涉及的自然月（`yyyy-MM`），账单就按这些账期取。"""
    out: set[str] = set()
    for e in entries:
        ts = str(e.get("ts") or "")[:7]
        if len(ts) == 7 and ts[4] == "-":
            out.add(ts)
    return sorted(out)


def ledger_by_cycle(output_dir: Path) -> dict[str, dict[str, float]]:
    """`账期 → {run_id: 台账金额}`：扫全部 run 的 cost.json。

    要扫全部 run 是因为账单是**账户×月**粒度：想知道"这个月账单里属于本 run 的那份"，
    必须知道同月还有哪些 run 在花钱，否则会把别人的消费算到这条 run 头上。
    """
    out: dict[str, dict[str, float]] = {}
    if not output_dir.is_dir():
        return out
    for run_dir in sorted(output_dir.iterdir()):
        p = run_dir / "cost.json"
        if not p.is_file():
            continue
        try:
            entries = json.loads(p.read_text(encoding="utf-8")) or []
        except (OSError, ValueError):
            continue  # 半截 JSON 不该让整次对账失败
        if not isinstance(entries, list):
            continue
        total = sum(float(e.get("amount_cny") or 0.0) for e in entries if isinstance(e, dict))
        for cyc in cycles_of([e for e in entries if isinstance(e, dict)]):
            out.setdefault(cyc, {})[run_dir.name] = round(total, 4)
    return out


def reconcile_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    product_codes: tuple[str, ...] = DEFAULT_PRODUCT_CODES,
) -> dict[str, Any]:
    """**真实账单优先**的 run 级对账：拿 BSS 账单校正台账估算。

    为什么台账不能自证：金额 = 单价 × 用量。用量已经尽量取 API 回传的 usage，
    但单价那一侧永远可能失真（免费额度转正式计费、官方调价、模型没登记）。
    2026-10-12 那条 run 就是极端例子：i2v/talk/text 全按 ¥0 记，合计 ¥6.30，
    而账单在正常跑 —— 只有账单能揭穿这种失真。

    诚实的边界：账单粒度是**账户×月×产品**，拿不到"某一次生成花了多少"。
    所以本 run 的真实金额按"本 run 台账 ÷ 同月全部 run 台账"分摊，并在返回里
    写明这是分摊值。同月还有别的东西用这个账号花钱时，那份也会被摊进来 ——
    这是账单口径的固有粒度，不是实现缺陷，界面上必须一起说出来。
    """
    root = Path(run_dir)
    cost_file = root / "cost.json"
    if not cost_file.is_file():
        return {"available": False, "reason": f"没有台账文件：{cost_file}"}
    try:
        entries = json.loads(cost_file.read_text(encoding="utf-8")) or []
    except (OSError, ValueError) as e:
        return {"available": False, "reason": f"台账读不出：{e}"}
    entries = [e for e in entries if isinstance(e, dict)]
    run_total = sum(float(e.get("amount_cny") or 0.0) for e in entries)
    cycles = cycles_of(entries) or [date.today().strftime("%Y-%m")]
    if not available():
        return {
            "available": False,
            "ledger_cny": round(run_total, 4),
            "reason": (
                "未配置阿里云 RAM AK/SK（FV3_ALIYUN_ACCESS_KEY_ID / _SECRET，"
                "与百炼 DASHSCOPE_API_KEY 不是同一个凭据）→ 取不到真实账单，"
                "台账金额仍是「单价×用量」的估算。"
            ),
        }
    out_dir = output_dir or root.parent
    by_cycle = ledger_by_cycle(out_dir)
    per_cycle: list[dict[str, Any]] = []
    real_total = 0.0
    for cyc in cycles:
        bill, why = query_bill(cyc, product_codes=product_codes)
        if not bill:
            per_cycle.append({"cycle": cyc, "ok": False, "reason": why})
            continue
        shares = by_cycle.get(cyc) or {}
        cycle_ledger = sum(shares.values())
        run_in_cycle = float(shares.get(root.name) or run_total)
        # 分摊比：本 run 占同月台账的比例。同月台账为 0 时无从分摊，如实说。
        ratio = (run_in_cycle / cycle_ledger) if cycle_ledger > 0 else None
        share = round(float(bill.get("pretax_amount") or 0.0) * ratio, 4) if ratio is not None else None
        if share is not None:
            real_total += share
        per_cycle.append(
            {
                "cycle": cyc,
                "ok": True,
                "bill_cny": round(float(bill.get("pretax_amount") or 0.0), 4),
                "by_product": bill.get("by_product") or {},
                "ledger_all_runs_cny": round(cycle_ledger, 4),
                "ledger_this_run_cny": round(run_in_cycle, 4),
                "share_ratio": round(ratio, 4) if ratio is not None else None,
                "allocated_cny": share,
                "errors": bill.get("errors") or [],
            }
        )
    result = {
        "available": any(c.get("ok") for c in per_cycle),
        "run_id": root.name,
        "ledger_cny": round(run_total, 4),
        "real_cny": round(real_total, 4),
        "cycles": per_cycle,
        "granularity": "账单按 账户×月×产品 聚合，本 run 的金额是按台账占比**分摊**出来的",
        "verdict": (
            "台账偏低：单价表可能失真（免费额度转正式计费 / 漏登记 / 官方调价）"
            if real_total - run_total > 0.01
            else "台账与账单量级一致"
            if abs(real_total - run_total) <= 0.01
            else "台账偏高（单价表高于实际结算价）"
        ),
        "reconciled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not result["available"]:
        result["reason"] = "；".join(
            str(c.get("reason") or "") for c in per_cycle if not c.get("ok")
        )[:300]
    return result


def save_reconciliation(run_dir: Path, data: dict[str, Any]) -> Path:
    p = Path(run_dir) / "real_cost.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_reconciliation(run_dir: Path) -> dict[str, Any] | None:
    """上次对账的结果（看板要用真实账单口径显示，而不只是台账估算）。"""
    p = Path(run_dir) / "real_cost.json"
    if not p.is_file():
        return None
    try:
        got = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def reconcile(bill: dict[str, Any], ledger_total_cny: float) -> dict[str, Any]:
    """账单 vs 台账：差额与比值。台账偏低说明单价表失真（免费额度转计费/漏登记/官方调价）。"""
    real = float(bill.get("pretax_amount") or 0.0)
    delta = round(real - float(ledger_total_cny or 0.0), 4)
    return {
        "bill_cny": round(real, 4),
        "ledger_cny": round(float(ledger_total_cny or 0.0), 4),
        "delta_cny": delta,
        "ratio": round(real / ledger_total_cny, 3) if ledger_total_cny else None,
        "verdict": (
            "台账偏低：单价表可能失真（免费额度转正式计费 / 漏登记 / 官方调价）"
            if delta > 0.01
            else "台账偏高或一致"
            if delta < -0.01
            else "一致"
        ),
    }
