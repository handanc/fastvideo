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

from fastvideo3.providers.base import ProviderError

API_BASE = "https://dashscope.aliyuncs.com/api/v1"
OSS_RESOLVE_HEADER = {"X-DashScope-OssResourceResolve": "enable"}


async def get_upload_policy(api_key: str, model: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{API_BASE}/uploads",
            params={"action": "getPolicy", "model": model},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code >= 400:
        raise ProviderError(f"getPolicy 失败 HTTP {resp.status_code}: {resp.text[:200]}")
    data = (resp.json() or {}).get("data") or {}
    if not data.get("upload_host"):
        raise ProviderError(f"getPolicy 无 upload_host: {str(data)[:200]}")
    return data


async def upload_local_file(api_key: str, model: str, path: Path) -> str:
    """本地文件直传百炼临时 OSS，返回 oss:// URL（供 s2v/retalk 等模型输入）。"""
    path = Path(path)
    policy = await get_upload_policy(api_key, model)
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
