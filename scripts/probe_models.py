"""模型协议探测：用最小花费试出 .env 里指定的模型**真正接受哪种请求协议**。

为什么需要它：`fastvideo3/providers/protocols.py` 的协议表是按百炼命名习惯**推断**的；
新模型（wan3.0-video / qwen-image-3.0 等）的官方请求体无法离线确认。推断错了不是
"效果差一点"，而是每个请求都 400 —— 而 400 会被 fallback 掩盖成"付费模型也能跑"，
让人误判改动无效。本脚本把"哪个协议被接受"变成一次可观测的实验。

花费控制（重要）：
- 默认只探 text + t2i（`--caps` 可选）；**i2v/talk 每次"提交成功"就消耗 1 个视频额度**
  （你列的 wan3.0-video 只有 30 次），所以默认不探，需显式 `--caps text,t2i,i2v,talk`。
- 提交成功即停止（不等出片、不轮询），避免重复计费。

用法：
    python scripts/probe_models.py                      # 探 text+t2i
    python scripts/probe_models.py --caps i2v --write   # 探视频并把结论写入 configs/model_protocols.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from fastvideo3.core.config import CONFIGS_DIR, load_config
from fastvideo3.providers import protocols
from fastvideo3.providers.dashscope import API_BASE, COMPAT_BASE

for _stream in (sys.stdout, sys.stderr):  # Windows GBK 控制台 → UTF-8
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, pragma: no cover
        pass

TINY_PROMPT = "一个戴黄色头盔的年轻人站在面馆门口，真人写实电影剧照"
_ASYNC_HEADER = {"X-DashScope-Async": "enable"}


async def _post(key: str, url: str, body: dict, extra_headers: dict | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(url, headers=headers, json=body)
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        data = {"_raw": r.text[:200]}
    return r.status_code, data


def _variants(cap: str, model: str, image_url: str | None) -> list[tuple[str, str, dict, dict | None]]:
    """(协议名, 端点, 请求体, 额外头) 候选序列 —— 按推断协议优先、其余兜底。"""
    if cap == "text":
        return [
            (
                "openai_compat",
                f"{COMPAT_BASE}/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "回复 OK 两个字母"}],
                 "max_tokens": 8},
                None,
            )
        ]
    if cap == "t2i":
        return [
            (
                protocols.T2I_MULTIMODAL,
                f"{API_BASE}/services/aigc/multimodal-generation/generation",
                {"model": model,
                 "input": {"messages": [{"role": "user", "content": [{"text": TINY_PROMPT}]}]},
                 "parameters": {"size": "512*512", "n": 1, "watermark": False}},
                None,
            ),
            (
                protocols.T2I_ASYNC,
                f"{API_BASE}/services/aigc/text2image/image-synthesis",
                {"model": model, "input": {"prompt": TINY_PROMPT},
                 "parameters": {"size": "512*512", "n": 1, "watermark": False}},
                _ASYNC_HEADER,
            ),
        ]
    if cap in ("i2v", "r2v"):
        media_type = "reference_image" if cap == "r2v" else "first_frame"
        out: list[tuple[str, str, dict[str, Any], dict[str, str] | None]] = [
            (
                protocols.I2V_MEDIA,
                f"{API_BASE}/services/aigc/video-generation/video-synthesis",
                {"model": model,
                 "input": {"prompt": TINY_PROMPT, "media": [{"type": media_type,
                                                             "url": image_url}]},
                 "parameters": {"resolution": "480P", "duration": 5, "prompt_extend": False}},
                _ASYNC_HEADER,
            ),
            (
                protocols.I2V_LEGACY,
                f"{API_BASE}/services/aigc/video-generation/video-synthesis",
                {"model": model, "input": {"prompt": TINY_PROMPT, "img_url": image_url},
                 "parameters": {"resolution": "480P", "duration": 5, "watermark": False}},
                _ASYNC_HEADER,
            ),
        ]
        return out
    if cap == "talk":
        return [
            (
                "i2v_driving_audio",
                f"{API_BASE}/services/aigc/video-generation/video-synthesis",
                {"model": model,
                 "input": {"prompt": TINY_PROMPT,
                           "media": [{"type": "first_frame", "url": image_url}]},
                 "parameters": {"resolution": "480P", "duration": 5}},
                _ASYNC_HEADER,
            ),
            (
                protocols.TALK_S2V,
                f"{API_BASE}/services/aigc/video-generation/video-synthesis",
                {"model": model, "input": {"image_url": image_url, "audio_url": image_url},
                 "parameters": {"resolution": "480P", "watermark": False}},
                _ASYNC_HEADER,
            ),
        ]
    return []


def _classify(status: int, data: dict) -> str:
    """ok / gate / reject / transient —— gate 表示模型未开通，换协议也没用。"""
    text = json.dumps(data, ensure_ascii=False).lower()
    if status < 400:
        return "ok"
    if any(k in text for k in ("access denied", "overdue", "good standing", "not activated",
                               "未开通", "欠费", "arrearage")):
        return "gate"
    if status in (429, 500, 502, 503, 504):
        return "transient"
    return "reject"


async def probe_cap(cfg, cap: str, image_url: str | None, results: dict) -> None:
    routed = cfg.routing.get(cap)
    provider_name = routed.default if routed else ""
    model = cfg.model_overrides.get(cap) or cfg.models.get(
        {"text": "dashscope_text", "t2i": "dashscope_t2i", "i2v": "dashscope_i2v",
         "r2v": "dashscope_r2v", "talk": "dashscope_s2v"}.get(cap, ""), ""
    )
    if not model:
        print(f"[{cap}] 无模型名，跳过")
        return
    inferred = protocols.infer_protocol(model, cap)
    print(f"\n[{cap}] model={model}  推断协议={inferred or '未知'}  yaml路由={provider_name}")
    for proto, url, body, headers in _variants(cap, model, image_url):
        status, data = await _post(cfg.dashscope_api_key, url, body, headers)
        verdict = _classify(status, data)
        msg = (data.get("message") or data.get("code") or "")[:120]
        print(f"    {proto:22s} → HTTP {status} {verdict}{'  ' + msg if msg else ''}")
        if verdict == "gate":
            results[cap] = {"model": model, "status": "not_activated", "protocol": None}
            print(f"    ⇒ 模型未开通/账户欠费：去百炼控制台开通 {model} 后重探（换协议无用）")
            return
        if verdict == "ok":
            results[cap] = {"model": model, "status": "ok", "protocol": proto}
            print(f"    ⇒ 可用协议 = {proto}；{protocols.describe(model, cap)}")
            if proto != inferred:
                print("    ⚠ 与推断不一致 → 建议 --write 回写 configs/model_protocols.yaml")
            return
        time.sleep(1)
    results[cap] = {"model": model, "status": "all_rejected", "protocol": None}
    print("    ⇒ 所有候选协议都被拒：请把上面报错原文贴回来，或改用控制台示例请求")


def _write_table(results: dict) -> None:
    p = CONFIGS_DIR / "model_protocols.yaml"
    existing = _yaml_load(p) if p.exists() else {}
    table = dict(existing.get("protocols") or {})
    for cap, r in results.items():
        if r.get("protocol"):
            table[f"{cap}/{r['model']}"] = r["protocol"]
    header = (
        "# scripts/probe_models.py 实测结论（优先级高于内置命名推断）。\n"
        "# 键 = <capability>/<model> 或裸模型名；值 = 协议标识（见 providers/protocols.py）。\n"
        "protocols:\n"
    )
    body = "".join(f"  {k}: {v}\n" for k, v in sorted(table.items()))
    p.write_text(header + (body or "  {}\n"), encoding="utf-8")
    print(f"✔ 结论已写入 {p}")


def _yaml_load(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


async def main() -> int:
    ap = argparse.ArgumentParser(description="探测 .env 指定模型的真实调用协议")
    ap.add_argument("--caps", default="text,t2i",
                    help="逗号分隔：text,t2i,i2v,talk,r2v（i2v/talk 每次成功消耗 1 个视频额度）")
    ap.add_argument("--write", action="store_true", help="把结论写入 configs/model_protocols.yaml")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.dashscope_api_key:
        print("缺少 DASHSCOPE_API_KEY")
        return 1
    image_url = None
    if any(c in args.caps for c in ("i2v", "talk", "r2v")):
        from fastvideo3.utils.dashscope_upload import upload_local_file
        from fastvideo3.utils.image import make_placeholder_image

        tmp = Path(".probe_frame.png")
        make_placeholder_image(tmp, label="probe", color="#3c3c5a")
        image_url = await upload_local_file(cfg.dashscope_api_key, "probe", tmp)
        print(f"探测用首帧已上传: {image_url[:60]}…")

    results: dict = {}
    for cap in [c.strip() for c in args.caps.split(",") if c.strip()]:
        await probe_cap(cfg, cap, image_url, results)
    if args.write:
        _write_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
