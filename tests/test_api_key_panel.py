"""面板里的「百炼 API KEY」输入框：掩码输入、失焦即写进 `.env` 的 DASHSCOPE_API_KEY。

⚠ 全部用例都指向 tmp 里的假 `.env`（`monkeypatch.setattr(envfile, "env_path", …)`）：
真实 `.env` 装着活密钥，测试既不能改它，也不能把它的内容读进断言或输出。

钉住的行为，一条都不能少：
1. 写回只动那一行：别的变量、注释、行尾注释、空行原样保留（`core/envfile.py` 唯一实现）；
2. **值绝不出现在响应里**（也不出现在错误说明里）—— GET 只回"有没有 / 从哪读到"；
3. 留空＝不改动，**永远不清空**已有 KEY（误触失焦不该让用户下次跑不出片）；
4. 明显不完整的输入（太短、含空格/换行/#）当场拒绝，而不是存一个静默失效的 KEY；
5. 与当前值相同时不写盘（少一次无意义的文件改写，也就少一次把 .env 写坏的机会）；
6. 别的站点打不到这个写接口（`allow_origins=["*"]` 配一个"改本地密钥"的端点，
   等于任何网页都能换掉用户本机这把 KEY）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import fastvideo3.app.api as api_mod
from fastvideo3.core import envfile
from fastvideo3.core.config import load_config

SECRET = "sk-live-key-abcdef0123456789"
NEW_KEY = "sk-brand-new-key-99887766"

SAMPLE = """# 密钥区
DASHSCOPE_API_KEY=sk-live-key-abcdef0123456789
FAL_KEY=fal-secret-keep-me

FV3_TEXT_MODEL=qwen3.7-flash
FV3_T2I_MODEL=wan2.7-image   # 行尾注释要保留
# DASHSCOPE_API_KEY=  (注释掉的例子，不许被"替他打开")
"""


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(envfile, "env_path", lambda: p)
    # 进程环境变量优先于 .env：不清掉的话测的是这台机器的 shell，不是被测代码
    for k in ("DASHSCOPE_API_KEY", "FAL_KEY", "DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return p


@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


def _put(client, value, headers=None):
    return client.put("/api/api-keys", json={"dashscope_api_key": value}, headers=headers or {})


# ───────────────────────── 状态：只回名字与布尔 ─────────────────────────
def test_get_reports_presence_without_the_value(client, dotenv):
    body = client.get("/api/api-keys").json()
    row = next(k for k in body["keys"] if k["name"] == "DASHSCOPE_API_KEY")
    assert row["set"] is True and row["source"] == ".env"
    assert row["field"] == "dashscope_api_key"
    # 整份响应里不许出现密钥本体（连长度/前缀这类可拼回的线索也不给）
    assert SECRET not in json.dumps(body, ensure_ascii=False)
    assert not any(len(str(v)) > 12 and str(v).startswith("sk-") for v in row.values()), row


def test_get_says_unset_when_absent(client, tmp_path, monkeypatch):
    empty = tmp_path / "other.env"
    empty.write_text("FV3_TEXT_MODEL=qwen3.7-flash\n", encoding="utf-8")
    monkeypatch.setattr(envfile, "env_path", lambda: empty)
    body = client.get("/api/api-keys").json()
    row = next(k for k in body["keys"] if k["name"] == "DASHSCOPE_API_KEY")
    assert row["set"] is False and row["source"] == ""


def test_process_environment_wins_the_label(client, dotenv, monkeypatch):
    """进程环境变量里有 KEY 时，状态要说是 `env` 来的 —— 否则用户改 `.env` 会"改了没反应"
    还以为是面板坏了（`load_config` 的优先级就是 env > .env）。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", SECRET)
    row = next(k for k in client.get("/api/api-keys").json()["keys"] if k["name"] == "DASHSCOPE_API_KEY")
    assert row["set"] is True and row["source"] == "env"


# ───────────────────────── 写回：只动那一行 ─────────────────────────
def test_put_writes_only_the_key_line_and_keeps_everything_else(client, dotenv):
    r = _put(client, NEW_KEY)
    assert r.status_code == 200, r.text
    assert r.json()["changed"] is True and r.json()["name"] == "DASHSCOPE_API_KEY"
    out = dotenv.read_text(encoding="utf-8")
    assert f"DASHSCOPE_API_KEY={NEW_KEY}" in out
    assert "FAL_KEY=fal-secret-keep-me" in out                     # 别的密钥没动
    assert "FV3_TEXT_MODEL=qwen3.7-flash" in out                   # 模型行没动
    assert "FV3_T2I_MODEL=wan2.7-image   # 行尾注释要保留" in out   # 行尾注释原样
    assert "# DASHSCOPE_API_KEY=  (注释掉的例子，不许被\"替他打开\")" in out  # 注释行不打开
    assert SECRET not in out, "旧值应被替换掉，不留残值"


def test_written_key_is_the_one_config_reads(client, dotenv):
    """写进去的名字必须就是 `load_config` 读的那个键 —— 名字错一位就是"存了但跑不通"。"""
    _put(client, NEW_KEY)
    cfg = load_config(dotenv_path=dotenv, resolution="480P")
    assert cfg.dashscope_api_key == NEW_KEY


def test_blank_input_never_wipes_the_stored_key(client, dotenv):
    """留空＝不改动。把"没填"当成"想删掉"是最贵的一种误触：下一个任务直接跑不动。"""
    r = _put(client, "")
    assert r.status_code == 200 and r.json()["saved"] is False
    assert r.json()["changed"] is False
    assert f"DASHSCOPE_API_KEY={SECRET}" in dotenv.read_text(encoding="utf-8")
    _put(client, "   ")   # 只有空白也算留空
    assert f"DASHSCOPE_API_KEY={SECRET}" in dotenv.read_text(encoding="utf-8")


def test_same_value_does_not_rewrite_the_file(client, dotenv):
    before = dotenv.read_bytes()
    r = _put(client, SECRET)
    assert r.status_code == 200 and r.json()["changed"] is False
    assert dotenv.read_bytes() == before, "同值还要改写一次文件 = 多一次把 .env 写坏的机会"


@pytest.mark.parametrize(
    "bad,why",
    [
        ("sk-1", "太短（多半是没贴全）"),
        ("sk has space", "含空格：dotenv 不加引号会把值截断"),
        ("sk-a\nsk-b", "含换行"),
        ("sk-secret#tail", "含 #：会被当成行尾注释而截断"),
    ],
)
def test_suspicious_input_is_refused_not_stored(client, dotenv, bad, why):
    r = _put(client, bad)
    assert r.status_code == 400, (why, r.text)
    assert f"DASHSCOPE_API_KEY={SECRET}" in dotenv.read_text(encoding="utf-8"), "拒绝时不许动文件"
    # 错误说明里不得出现用户贴进来的内容（错误信息经常会被顺手贴进截图/日志）
    assert bad not in r.text


def test_padded_input_is_trimmed_before_writing(client, dotenv):
    r = _put(client, f"  {NEW_KEY}\n")
    assert r.status_code == 200
    assert f"DASHSCOPE_API_KEY={NEW_KEY}\n" in dotenv.read_text(encoding="utf-8")


def test_response_and_error_texts_never_carry_the_secret(client, dotenv, monkeypatch):
    """连失败路径也不能回显：异常文本可能带路径，但不许带值。"""
    monkeypatch.setattr(envfile, "env_path", lambda: dotenv.parent / "missing" / ".env")
    r = _put(client, NEW_KEY)
    assert r.status_code == 500, r.text
    assert NEW_KEY not in r.text and SECRET not in r.text


# ───────────────────────── 跨站请求改不到本机的 KEY ─────────────────────────
def test_cross_site_request_cannot_rewrite_the_key(client, dotenv):
    r = _put(client, NEW_KEY, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403, r.text
    assert f"DASHSCOPE_API_KEY={SECRET}" in dotenv.read_text(encoding="utf-8")


def test_same_site_and_bare_calls_are_allowed(client, dotenv):
    assert _put(client, NEW_KEY, headers={"Origin": "http://testserver"}).status_code == 200
    dotenv.write_text(SAMPLE, encoding="utf-8")
    assert _put(client, "sk-another-key-123456").status_code == 200  # 无 Origin（curl/本机）放行


def test_cannot_swap_key_mid_generation(client, dotenv, outdir):
    """生成中不许换 KEY：换 KEY 就是换账号，前半段扣 A、后半段扣 B，
    "已花 ¥X / 成本上限"这套账当场作废（用户还会误以为停的是自己设的上限）。"""
    from fastvideo3.core import tasklock

    ok, _st = tasklock.acquire(outdir, "run_locked", "scene")
    assert ok, _st
    try:
        r = _put(client, NEW_KEY)
        assert r.status_code == 409, r.text
        assert "正在生成中" in r.text
        assert f"DASHSCOPE_API_KEY={SECRET}" in dotenv.read_text(encoding="utf-8")
    finally:
        tasklock.release(outdir, "run_locked")
    # 空闲后照常能存
    assert _put(client, NEW_KEY).status_code == 200


def test_dotenv_is_git_ignored():
    """这把 KEY 落进 `.env`，所以"`.env` 不入库"是这个功能的**前提**，不是一个习惯。"""
    from fastvideo3.core.config import PROJECT_ROOT

    lines = [
        ln.strip()
        for ln in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert ".env" in lines, "面板能写 KEY 了，.env 必须仍然被忽略"
