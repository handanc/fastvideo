"""项目内置运行时：把"跑起来需要的东西"全部收进项目目录。

为什么要有这一层：一条 run 要调 ffmpeg 几十次，而 ffmpeg 原本来自 site-packages
（`imageio_ffmpeg`）或系统 PATH —— 两者都在项目外面。换一台机器、换一个 venv、
或者宿主机根本没装 ffmpeg，项目就跑不起来；更糟的是"本机跑得通"永远解释不了
线上为什么跑不通。所以这里的规则是：

1. **解析顺序**（`ffmpeg_source()`）：显式 `FV3_FFMPEG_PATH` → **项目内 `bin/`**
   → venv 里的 imageio-ffmpeg → 系统 PATH。项目内置排在系统 PATH **之前**：
   部署包带着自己的二进制，行为由项目决定，不由宿主机决定。
2. **补齐**（`provision_ffmpeg()`）：venv 里有静态二进制就直接复制进 `bin/`
   （离线、零下载），否则从镜像列表下载。复制后**必须跑一次 `-version`** 才认，
   半截文件比没有文件更坏（后面每一步都只看到 "ffmpeg failed"）。
3. **自检**（`checks()`）：doctor 的 CLI 与 API **共用这一份**清单。以前两处各写
   一遍，加一项就要同步两次，迟早漂。

`bin/` 与 `web/dist/` 都在 `.gitignore` 里：二进制不入库，入库也解决不了问题
（版本对不上照样炸）。真正入库的是 `requirements.lock.txt` —— 见 `lock_status()`。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastvideo3.core.config import PROJECT_ROOT

BIN_DIR = PROJECT_ROOT / "bin"
LOCK_FILE = PROJECT_ROOT / "requirements.lock.txt"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_TEMPLATE = PROJECT_ROOT / ".env.example"
WEB_DIST = PROJECT_ROOT / "web" / "dist"

#: 项目内的二进制名（Windows 带 .exe；解析时两种都认，避免同一份代码分平台判断）
FFMPEG_NAMES = ("ffmpeg.exe", "ffmpeg")

#: 兜底下载镜像（venv 里没有 imageio-ffmpeg 时才用）。gyan 是 ffmpeg 官方推荐的
#: Windows 构建源，且镜像站本身把包下下来再解压，所以需要 zip 解包。
_FFMPEG_MIRRORS = (
    "https://gh-proxy.gitee.ac/https://github.com/BtbN/FFmpeg-Builds/releases/latest/"
    "download/ffmpeg-master-latest-win64-gpl.zip",
    "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-bin_build.zip",
)


def _is_executable(path: Path) -> bool:
    return path.is_file() and (sys.platform != "win32" or path.suffix == ".exe")


def _display(path: Path) -> str:
    """自检文本里的路径：项目内就写相对路径，不在项目内就写绝对路径。

    直接 `relative_to(PROJECT_ROOT)` 会在路径不属于项目时抛 ValueError（软链接安装、
    测试里换掉 PROJECT_ROOT 都会碰到），自检崩在格式化的那一刻很难看。
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def ffmpeg_target_path() -> Path:
    """项目内应该放 ffmpeg 的位置（已存在就用它，否则用 ffmpeg.exe 这个名字）。"""
    for name in FFMPEG_NAMES:
        p = BIN_DIR / name
        if p.is_file():
            return p
    return BIN_DIR / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")


def _candidates() -> list[tuple[str, Path]]:
    """按优先级排好的 (来源, 路径) 列表；调用方挑第一个真能跑的。"""
    out: list[tuple[str, Path]] = []
    env_path = os.getenv("FV3_FFMPEG_PATH", "").strip()
    if env_path:
        out.append(("显式配置 FV3_FFMPEG_PATH", Path(env_path)))
    for name in FFMPEG_NAMES:
        p = BIN_DIR / name
        if p.is_file():
            out.append(("项目内置 bin/", p))
            break
    try:  # imageio-ffmpeg 的静态二进制装在 site-packages 里，随 venv 走（在项目外）
        import imageio_ffmpeg

        out.append(("虚拟环境 imageio-ffmpeg", Path(imageio_ffmpeg.get_ffmpeg_exe())))
    except Exception:  # noqa: BLE001  # 没装/装了但取不到 → 继续往系统 PATH 找
        pass
    which = shutil.which("ffmpeg")
    if which:
        out.append(("系统 PATH", Path(which)))
    return [(src, p) for src, p in out if p.is_file()]


def _version_word(first_line: str) -> str:
    """`ffmpeg -version` 首行 → 版本号。

    别直接 `split()[0]`：首行是 "ffmpeg version 7.1 Copyright ..."，那取到的是
    "ffmpeg" 这个词，自检里写"（87.6 MB，ffmpeg）"等于什么也没说。
    """
    parts = first_line.replace("ffmpeg version ", "").split()
    if not parts:
        return ""
    # 版本号是第一个"长得像版本"的词，别按位置取：不同构建的首行字段数不一样
    # （gyan 版是 `7.1-essentials_build-www.gyan.dev`，BtbN 版是 `n7.1-...`），
    # 按 parts[1] 取会拿到 "Copyright"。
    for tok in parts:
        if re.match(r"^v?\d+(\.\d+)+", tok):
            return tok
    return parts[0]


def _runnable(path: Path) -> tuple[bool, str]:
    """真跑一次 `-version`。只看文件存在会认下截断的下载（87MB 下到一半也是"存在"）。"""
    try:
        proc = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:160]
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[:160]
    first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
    return True, first


@dataclass(frozen=True)
class FFmpegStatus:
    path: str
    source: str
    version: str
    #: 是否在项目目录内（= 部署包自带，换机器行为不变）
    bundled: bool = False

    def as_detail(self) -> str:
        where = "项目内" if self.bundled else self.source
        ver = f"｜{self.version}" if self.version else ""
        # 不在项目内 = 换机器就得重装 → 补齐指令直接放在这里（自检面板逐条显示 detail）
        tail = "" if self.bundled else " → 想要项目自带：provision（复制进 bin/）"
        return f"{self.path}（{where}）{ver}{tail}"


def ffmpeg_source() -> FFmpegStatus:
    """当前实际会用到的 ffmpeg 及其来源。找不到抛 RuntimeError（带补齐指令）。"""
    tried: list[str] = []
    for src, path in _candidates():
        ok, version = _runnable(path)
        if ok:
            return FFmpegStatus(
                path=str(path),
                source=src,
                version=_version_word(version),
                bundled=_inside_project(path),
            )
        tried.append(f"{src} {path} → {version}")
    raise RuntimeError(
        "未找到可用 ffmpeg。"
        + (f" 试过：{'；'.join(tried)}。" if tried else "")
        + "补齐（优先从当前 venv 复制，离线）：python -m fastvideo3.app.cli provision"
    )


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def provision_ffmpeg(force: bool = False, timeout: float = 180.0) -> tuple[str, str | None]:
    """把 ffmpeg 收进 `bin/`。返回 `(做了什么, 错误原因)`，失败不抛（自检不能因为修不好而崩）。"""
    target = ffmpeg_target_path()
    if target.is_file() and not force:
        ok, version = _runnable(target)
        if ok:
            return f"项目内置 ffmpeg 已就位：{target}（ffmpeg {_version_word(version)}）", None
        # 存在但跑不起来：当作没就位，重装修复（截断/权限问题的典型表现）

    # 1) venv 里就有静态二进制 → 离线复制，零下载，版本与开发机完全一致
    try:
        import imageio_ffmpeg

        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if src.is_file():
            BIN_DIR.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            shutil.copy2(src, tmp)
            os.replace(tmp, target)  # 原子替换：中途断电也不会留下半个 exe
            if sys.platform != "win32":
                target.chmod(0o755)
            ok, detail = _runnable(target)
            if ok:
                mb = target.stat().st_size / 1e6
                return (
                    f"已复制进项目：{src} → {target}（{mb:.1f} MB，ffmpeg {_version_word(detail)}）",
                    None,
                )
            target.unlink(missing_ok=True)
            return "", f"从 imageio-ffmpeg 复制后仍无法运行：{detail}"
    except Exception as e:  # noqa: BLE001
        reason = f"venv 里的 imageio-ffmpeg 取不到二进制：{str(e)[:120]}"
    else:
        reason = "venv 里的 imageio-ffmpeg 没带静态二进制（pip install --force-reinstall imageio-ffmpeg）"

    # 2) 下载兜底（需要外网）。只在 1) 不可用时走
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in _FFMPEG_MIRRORS:
        try:
            got = _download_ffmpeg(url, target, timeout)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url.split('/')[2]}: {str(e)[:160]}")
            continue
        if got:
            return got, None
        errors.append(f"{url.split('/')[2]}: 下载完成但校验未通过")
    if errors:
        return "", f"{reason}；下载也失败：" + "；".join(errors)
    return "", reason


def _download_ffmpeg(url: str, target: Path, timeout: float) -> str:
    """下载 zip 并解出里面的 ffmpeg 可执行文件；校验通过才落地，返回说明文本。"""
    import io
    import zipfile

    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    body = resp.content
    if len(body) < 1_000_000:  # 镜像挂了通常回一个几百字节的错误页
        raise RuntimeError(f"响应只有 {len(body)} 字节，不像二进制")
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        names = [n for n in zf.namelist() if re.search(r"(^|/)ffmpeg(\.exe)?$", n)]
        if not names:
            raise RuntimeError(f"压缩包里没找到 ffmpeg 条目：{zf.namelist()[:5]}")
        data = zf.read(names[0])
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    if sys.platform != "win32":
        tmp.chmod(0o755)
    ok, detail = _runnable(tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"解出的二进制跑不起来：{detail}")
    os.replace(tmp, target)
    return f"已下载并放进项目：{target}（{len(data) / 1e6:.1f} MB，ffmpeg {_version_word(detail)}）"


# ── 其余"该待在项目里"的东西 ────────────────────────────────────────────────


def env_file_status() -> tuple[bool, str]:
    """`.env` 只看**在不在**：自检输出会原样贴到界面上，绝不能把内容读出来。"""
    if ENV_FILE.is_file():
        keys = _env_key_count(ENV_FILE)
        return True, f"{ENV_FILE.name} 已就位（{keys} 项配置，内容不外显）"
    if ENV_TEMPLATE.is_file():
        return False, f"缺 {ENV_FILE.name}：复制 {ENV_TEMPLATE.name} 为 .env 并填入 KEY（cp .env.example .env）"
    return False, "缺 .env，而且 .env.example 也不在仓库里"


def _env_key_count(path: Path) -> int:
    """只数**变量名**，不返回任何值。"""
    n = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                n += 1
    except Exception:  # noqa: BLE001
        return -1
    return n


def _normalize_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _installed_version(pkg: str) -> str | None:
    """优先用 importlib.metadata（含连字符包名的正确写法），再退到模块 __version__。"""
    from importlib import metadata

    for guess in (pkg, pkg.replace("-", "_")):
        try:
            return metadata.version(guess)
        except Exception:  # noqa: BLE001
            continue
    try:  # 少数包（numpy/PIL/cv2）装的名字和导入名不同，按模块兜一次
        mod = __import__(pkg)
        v = getattr(mod, "__version__", None)
        return str(v) if v else None
    except Exception:  # noqa: BLE001
        return None


def lock_status() -> tuple[bool, str]:
    """跑着的解释器与 `requirements.lock.txt` 是否一致 —— "本机好使、线上不行"的探测器。

    只比**版本漂移**，不把"锁文件里多写了本项目自己"当成错误。
    """
    if not LOCK_FILE.is_file():
        return False, f"缺 {LOCK_FILE.name}：python -m fastvideo3.app.cli provision --lock 生成"
    rows = _read_lock()
    if not rows:
        return False, f"{LOCK_FILE.name} 是空的"
    drift, missing = [], []
    for pkg, want in rows.items():
        have = _installed_version(pkg)
        if have is None:
            missing.append(pkg)
        elif want and have != want:
            drift.append(f"{pkg} 装的是 {have}，锁文件写的是 {want}")
    total = len(rows)
    if missing or drift:
        bits = []
        if missing:
            bits.append(f"缺 {len(missing)} 个：{'、'.join(missing[:4])}")
        if drift:
            bits.append(f"版本漂移 {len(drift)} 个：{'；'.join(drift[:3])}")
        return False, f"{total} 项里 {'；'.join(bits)}" + ("…" if len(drift) > 3 else "")
    return True, f"{total} 项与锁文件一致（{LOCK_FILE.name}）"


def _read_lock() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in LOCK_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(\S+)", s)
        if m:
            rows[_normalize_pkg(m.group(1))] = m.group(2)
    return rows


def write_lock(path: Path | None = None) -> tuple[str, str | None]:
    """用当前环境生成锁文件（当前环境就是"验证过能跑"的那一份）。"""
    target = path or LOCK_FILE
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0:
        return "", f"pip freeze 失败：{proc.stderr[-200:]}"
    lines = [
        "# FastVideo3 运行环境锁文件 —— 由 `provision --lock` 从验证过的环境生成，别手改。",
        "# 复现：python -m venv .venv && .venv 的 pip  install -r requirements.lock.txt",
        f"# 生成于：Python {sys.version.split()[0]}（{sys.platform}）",
        "# 本项目自己用 `pip install -e .` 装，所以不在清单里（-e 路径换机器就失效）。",
        "# 清单含可选重组件（torch/torchvision/ultralytics…）：那是「验证过的环境」真装过的东西。",
        "#   只要最小可跑环境就按 requirements.txt 装；跨平台时 torch 的 CUDA/CPU 构建由 index 决定，",
        "#   需要 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu",
    ]
    kept, skipped = [], 0
    for raw in proc.stdout.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("-e ") or s.startswith("file://"):
            skipped += 1  # 可编辑安装的本项目：路径是本机的，写进锁文件等于写死
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+) @ .*$", s)
        if m:  # pip 有时输出 `pkg @ file:///...`，退化成等号版本以便跨机可用
            ver = _installed_version(_normalize_pkg(m.group(1)))
            if not ver:  # 查不到版本就**跳过**：写成 `pkg==` 会让 `pip install -r` 直接报错
                skipped += 1
                continue
            kept.append(f"{m.group(1)}=={ver}")
            continue
        kept.append(s)
    if not kept:
        return "", "pip freeze 没解析出任何包"
    body = "\n".join(lines + kept) + "\n"
    # 原子替换（与本模块复制二进制同一做法）：写一半被打断不该留下半份锁文件，
    # 那会让下一次 `pip install -r requirements.lock.txt` 装出个残缺环境
    tmp = target.with_suffix(".part")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, target)
    return f"已写入 {target.name}：{len(kept)} 个包（跳过 {skipped} 项本机路径依赖/无版本）", None


def web_dist_status() -> tuple[bool, str]:
    """前端构建产物：API 进程在导入时挂载它，缺了就只有接口没有页面。"""
    index = WEB_DIST / "index.html"
    if not index.is_file():
        return False, f"缺 {_display(WEB_DIST)}：上线前必须 cd web && npm ci && npm run build"
    assets = list((WEB_DIST / "assets").glob("*.js")) if (WEB_DIST / "assets").is_dir() else []
    return True, f"{_display(WEB_DIST)} 已构建（index.html + {len(assets)} 个 JS 包）"


def weights_status() -> tuple[bool, str]:
    """身份一致性客观层（Tier C）的 SFace 权重 + cv2 能力：缺了就退回 A/B 层，不是不能跑。

    权重在 `models/`（项目内），但**光有权重不够** —— cv2 还得带 `FaceDetectorYN` /
    `FaceRecognizerSF` 这两个类，所以这里报的是"这一层到底可用吗"，不是"文件在不在"。
    """
    from fastvideo3.core.config import MODELS_DIR
    from fastvideo3.utils import faceid

    here = _display(MODELS_DIR)
    missing = faceid.missing_weights()
    if missing:
        return False, (
            f"缺权重 {missing}（应放在 {here}/）→ 身份一致性退回 A/B 层；"
            "provision 会试下载，离线环境手工放进该目录同样算就位"
        )
    if not faceid.available():
        return False, f"权重已在 {here}/，但当前 opencv 不带 FaceDetectorYN/FaceRecognizerSF → 退回 A/B 层"
    return True, f"{here} 内权重齐备，opencv 支持（身份一致性客观层可用）"


def checks(cfg: Any) -> list[dict[str, Any]]:
    """doctor 的一整套检查项；CLI 与 API 共用，返回 [{item, ok, detail}]。

    口径说明（与上次排查一致，别再改回"一律报错"）：
    - 可选项（DEEPSEEK_KEY / FAL_KEY / yolov8n / 锁文件缺失）**不算失败**时要在
      detail 里写清影响面；算不算 ok 由"这条 run 会不会因此跑不了"决定。
    """
    out: list[dict[str, Any]] = []

    def row(item: str, ok: bool, detail: Any = "") -> None:
        out.append({"item": item, "ok": bool(ok), "detail": str(detail)})

    row("Python >= 3.11", sys.version_info >= (3, 11), f"{sys.version.split()[0]}（{sys.executable}）")

    missing = []
    for dep in ("pydantic", "httpx", "typer", "PIL", "sqlalchemy", "yaml"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    row("核心依赖", not missing, "全部就绪" if not missing else f"缺失: {missing}")

    try:
        row("FFmpeg", True, ffmpeg_source().as_detail())
    except Exception as e:  # noqa: BLE001
        row("FFmpeg", False, e)

    ok, detail = env_file_status()
    row("配置文件 .env", ok, detail)

    ok, detail = lock_status()
    row("依赖锁 requirements.lock.txt", ok, detail)  # 漂移 = 与验证过的环境不一致，算异常

    ok, detail = web_dist_status()
    row("前端构建 web/dist", ok, detail)

    row("数据库", *db_status(cfg))

    ok, detail = weights_status()
    row("身份一致性权重（可选）", ok, detail)

    # KEY 三项放进来是因为 CLI 与 API 两个 doctor 各写了一遍、文案已经不一致
    # （"主通道"这种说法在两处相反）。口径统一为：缺 KEY 只说明那条通道不可用，
    # 不影响"项目能不能独立跑起来"，所以可选 KEY 一律 ok=True，把影响面写在 detail 里。
    # DEEPSEEK / FAL 是**备选通道**：缺了不影响一条 run 跑完，所以不算失败（缺什么、
    # 影响哪条通道写在 detail 里）。DASHSCOPE 不同 —— 没有它真实调用一步都走不了。
    row(
        "DEEPSEEK_API_KEY",
        True,
        "已配置"
        if cfg.deepseek_api_key
        else "未配置(仅备选文本供应商，全百炼栈可忽略)"
        + ("" if cfg.dashscope_api_key else "；注意 DASHSCOPE 也未配置"),
    )
    row(
        "DASHSCOPE_API_KEY",
        bool(cfg.dashscope_api_key) or cfg.fake,
        "已配置" if cfg.dashscope_api_key else "未配置(真实调用不可用；--fake 仍可跑通全流程)",
    )
    row(
        "FAL_KEY",
        True,
        "已配置"
        if cfg.fal_api_key
        else "未配置(可选：口型兜底通道 fal_infinitalk 不可用；主通道走百炼)",
    )

    from fastvideo3.core.config import MODELS_DIR
    from fastvideo3.utils.yolo import WEIGHT_NAME

    row(
        "yolov8n-seg.pt(备用)",
        True,  # 信息项：双人镜头已改首帧同框单条 S2V，权重本期未接入管线
        "就位(分割合成进阶备用)"
        if (MODELS_DIR / WEIGHT_NAME).exists()
        else "未下载(管线不依赖；双人镜头走首帧同框方案)",
    )

    return out


def db_status(cfg: Any) -> tuple[bool, str]:
    try:
        from fastvideo3.core.jobs import JobStore

        JobStore(cfg.db_url).dispose()
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]
    url = str(cfg.db_url)
    inside = "（项目内 output/）" if "sqlite" in url and str(PROJECT_ROOT) in url else ""
    return True, url.split("://")[0] + inside


def provision_all(force: bool = False, with_lock: bool = False) -> list[dict[str, str]]:
    """把该进项目的东西收进项目，返回逐项结果（给 CLI 打印、给 API 回显）。"""
    done: list[dict[str, str]] = []

    def add(what: str, did: str, err: str | None) -> None:
        done.append({"what": what, "done": did, "error": err or ""})

    add("ffmpeg", *provision_ffmpeg(force=force))

    if with_lock:
        msg, err = write_lock()
        add("requirements.lock.txt", msg, err)

    # 权重：能下就下，下不到只记录原因（身份一致性会退回 A/B 层，不该卡住上线）
    try:
        from fastvideo3.utils import faceid

        # fetch_weights 把"本来就在"的也算进 got，所以按"就位数/总数"报，不谎称下载过
        got, err = faceid.fetch_weights()
        add("身份一致性权重", f"models/ 内 {len(got)}/{len(faceid.weight_paths())} 就位", err)
    except Exception as e:  # noqa: BLE001
        add("身份一致性权重", "", str(e)[:200])

    if not ENV_FILE.is_file() and ENV_TEMPLATE.is_file():
        shutil.copyfile(ENV_TEMPLATE, ENV_FILE)
        add(
            ".env",
            f"已从 {ENV_TEMPLATE.name} 复制生成 → 请填入 DASHSCOPE_API_KEY（内容不外显）",
            None,
        )
    else:
        add(".env", "已存在，未改动", None)

    dist_ok, dist_detail = web_dist_status()
    add("web/dist", dist_detail if dist_ok else "未构建（需 node：cd web && npm ci && npm run build）", None)

    return done
