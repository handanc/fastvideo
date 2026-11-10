"""DashScope 临时文件上传：本地文件 → oss:// 临时 URL（48h 有效，仅百炼可读）。

wan2.2-s2v / videoretalk 等视频类模型**不接受 base64 data URI**（报 400 url error），
官方流程（help.aliyun.com/zh/model-studio/get-temporary-file-url）：

1. GET  /api/v1/uploads?action=getPolicy&model=<model>  → 上传策略
2. POST {upload_host}（OSS 表单直传）                    → 得到 oss://<upload_dir>/<name>
3. 模型调用请求头加 ``X-DashScope-OssResourceResolve: enable`` 解析 oss:// URL
"""

from __future__ import annotations

from pathlib import Path

import httpx

from fastvideo3.core import endpoints
from fastvideo3.providers.base import ProviderError

# 默认值只在 `core/endpoints.py` 定义一份。上传策略是**原生任务 API** 的一部分，
# 所以它必须跟提交用同一个端点：换业务空间/套餐端点后仍打公共端点拿到的 oss:// 临时
# URL，那边根本不认（表现为 400 url error，像"图片坏了"而不是"端点没换全"）。
API_BASE = endpoints.DEFAULT_TASK_BASE_URL
OSS_RESOLVE_HEADER = {"X-DashScope-OssResourceResolve": "enable"}


async def get_upload_policy(api_key: str, model: str, base_url: str | None = None) -> dict:
    base = (base_url or API_BASE).rstrip("/")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{base}/uploads",
            params={"action": "getPolicy", "model": model},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code >= 400:
        raise ProviderError(f"getPolicy 失败 HTTP {resp.status_code}: {resp.text[:200]}")
    data = (resp.json() or {}).get("data") or {}
    if not data.get("upload_host"):
        raise ProviderError(f"getPolicy 无 upload_host: {str(data)[:200]}")
    return data


async def upload_local_file(
    api_key: str, model: str, path: Path, base_url: str | None = None
) -> str:
    """本地文件直传百炼临时 OSS，返回 oss:// URL（供 s2v/retalk 等模型输入）。

    `base_url` 传调用方那个 provider 的端点（`self.api_base`）—— 见模块头的说明：
    上传与提交不同端点，拿到的 oss:// URL 会在提交时被拒。
    """
    path = Path(path)
    policy = await get_upload_policy(api_key, model, base_url=base_url)
    key = f"{policy['upload_dir']}/{path.name}"
    fields = {
        "OSSAccessKeyId": policy["oss_access_key_id"],
        "Signature": policy["signature"],
        "policy": policy["policy"],
        "x-oss-object-acl": policy["x_oss_object_acl"],
        "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
        "success_action_status": "200",
        "key": key,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            policy["upload_host"],
            data=fields,
            files={"file": (path.name, path.read_bytes())},
        )
    if resp.status_code not in (200, 201, 204):
        raise ProviderError(f"OSS 直传失败 HTTP {resp.status_code}: {resp.text[:200]}")
    return f"oss://{key}"
