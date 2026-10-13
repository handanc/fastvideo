# bin/ —— 项目自带的可执行文件

一条 run 要调 ffmpeg 几十次。它原本来自 `site-packages`（`imageio-ffmpeg`）或系统 PATH，
两者都在项目外面 —— 换一台机器、换一个 venv、或者宿主机没装 ffmpeg，项目就跑不起来。
所以把这个目录当成交付物的一部分：

```powershell
fastvideo3 provision          # 优先从当前 venv 复制（离线、零下载），否则走镜像下载
```

解析顺序（`fastvideo3/core/runtime.py`）：

1. 环境变量 `FV3_FFMPEG_PATH`（显式指定，最高优先级）
2. **本目录**的 `ffmpeg.exe` / `ffmpeg` —— 排在系统 PATH **之前**，部署包的行为由项目决定
3. venv 里的 `imageio-ffmpeg` 静态二进制
4. 系统 PATH

放进来之后会被真跑一次 `ffmpeg -version` 校验，校验不过不算就位（半截文件比没有文件更坏，
后面每一步只会看到 `ffmpeg failed`）。

本目录内容不入库（`.gitignore` 里 `bin/*`），入库靠 `requirements.lock.txt` 复现环境；
真正需要跨机器一致的是那份锁文件里的 Python 依赖版本。
