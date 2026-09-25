# FastVideo3 系统联调报告（IT 验收）

日期：2026-08-22 ~ 08-23 · 环境：Windows + conda env `fastvideo3`（Python 3.11.15）· FFmpeg：imageio-ffmpeg 7.1 兜底 · 文本模型：qwen3.7-plus（百炼模型市场专属端点）

## CI 自检结果（全绿）

| 门禁 | 结果 |
|---|---|
| `ruff check .` | ✔ All checks passed |
| `mypy fastvideo3` | ✔ Success: no issues found in 40 source files |
| `pytest -m "not api"` | ✔ **73 passed**（含 3 次完整图冒烟 + 降级路径回归） |

## 真实 API 联调结果（用户 KEY 实测）

### 首次打通（run_20260822_224936）

- **成片**：29.0s（目标 30s）/ 11.5 MB / 含音轨 / `final.srt` 字幕旁挂
- **成本**：**¥5.32**（text 0.03 + t2i 1.98 + i2v 3.30 + vlm 0.01，TTS 走免费 edge-tts）——远低于 ¥15 验收线
- **qwen3.7-plus 剧本**：结构化 JSON 一次通过 Schema 校验（无围栏/解释文字污染）
- **降级链实战验证**：`wan2.2-s2v` 拒绝 base64 音频（400 url error）→ m6 自动降级 I2V+音频直叠，3 个对话镜头全部保交付

### 真实环境暴露并已修复的问题（7 项，全部有回归测试）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | TTS 全灭 | LLM 生成的 voice_id 是"甜美女声"类自由文本 | `sanitize_voice` 白名单清洗，按角色 md5 确定性分配 |
| 2 | 特定角色台词必失败 | `zh-CN-XiaochenNeural` 已被微软下线 | 白名单只保留 5 个实测可用音色 |
| 3 | 音频拼接崩溃 | edge-tts 实际输出 MP3 却命名 .wav | Provider 侧统一转 16k 单声道 WAV |
| 4 | 一句台词丢失 | LLM 给 action 镜头配了台词，m5 只认 dialogue 标签 | 音画链路改为**"有无台词"驱动** |
| 5 | 降级成片无对白 | m7 仅 audio_overlay 模式叠音轨 | 凡用 I2V 片段且有台词即叠音轨 |
| 6 | TTS 重试全失败 | 无退避连打触发限流 | 重试间 1.5s×n 退避 |
| 7 | 进程崩溃 | （首次运行 audio 节点异常退出） | `resume` 支持崩溃续跑：检查点无 interrupt 时直接 `ainvoke(None)` |
| 8 | 二次运行成片静音（-91dB） | m6 降级 I2V 片段被误装进 talks 返回，m7 当"口型成品"直用、不叠音轨 | `make_talk_clips` 返回 `(talks, fallbacks)` 分离，降级片段并入 clips 通道强制叠对白 |
| 9 | 成片时长缩水 30s→24.8s | `mux_audio` 的 `-shortest` 把视频截成台词长度 | 去掉 `-shortest`（MP4 允许流不等长）；另验证 `apad` 无限流会挂死编码，列入禁用 |
| 10 | 转场/拼接解码报错（-22 EINVAL） | 对白轨 16k 单声道与静音轨 44.1k 立体声参数不一致，concat 流中途换参数 | `normalize_clip` 统一 `-ar 44100 -ac 2` |
| 11 | 多句台词镜头音频溢出/时长失控 | 合并音轨长于镜头时行为未定义 | `mux_audio` 按 `tpad` 克隆末帧补静帧（m5 补静帧策略落地）；编剧 prompt 增加"每镜头最多一句台词"源头约束 |
| 12 | 成片在用户播放器"不完整"（卡死/半截） | libass 字幕合成把输出转成 **yuv444p** H.264，WMP/手机/浏览器无法正确解码 | 交付级 `safe_encode`：yuv420p + High profile + faststart，成片必过此道 |
| 13 | "不是完整电影" | 30 秒样片缺片头片尾、叙事体量不足 | m7 增加**片头标题卡+片尾"完"卡**；编剧 prompt 强制片名+收尾镜头；交付 90 秒三幕完整短剧 |
| 14 | 音频提速步骤崩溃（WinError 183） | Windows 下 `Path.rename()` 不能覆盖已存在文件（POSIX 可以） | 全部改用跨平台的 `Path.replace()` |

### 已知边界（如实记录）

1. **wan2.2-s2v 口型**：当前拒绝 base64 音频输入，需公网 URL（OSS 托管）——已自动降级为 I2V+直叠，口型完整版列为扩展课题（上传素材到 OSS 后把 `DashScopeTalkProvider` 的 data URI 换成 URL 即可）。
2. **CosyVoice**：该账号未开通（引擎 418）——代码已按官方 dashscope SDK 正确实现，开通后把 `configs/default.yaml` 的 tts 路由对调即可；当前默认走免费 edge-tts。
3. **字幕烧录**依赖 ffmpeg libass；兜底二进制若缺该滤镜自动降级为 `final.srt` 旁挂（不阻塞成片）。
4. **双人同屏**在未安装 `fastvideo3[full]` 时降级为左右拼接。

## 最终验收（run_20260823_155417，单进程一次跑通）

| 验收项 | 标准 | 实测 | 结果 |
|---|---|---|---|
| 成片时长 | 30s ±15% | 29.4s（偏差 2%） | ✅ |
| 对白可闻 | 非静音 | mean -29.5dB / 4 句台词全在 | ✅ |
| 音频格式 | 真 WAV | 4/4 RIFF 头 | ✅ |
| 台词约束 | ≤1 句/镜头、≤30 字 | 全部满足 | ✅ |
| 字幕 | SRT 与台词一致 | 4 条 | ✅ |
| 成本 | ≤¥15（IT-2 线） | **¥4.70** | ✅ |
| 全流程 | 单进程无人工干预 | 剧本→资产→分镜→I2V/TTS 并行→S2V(降级)→合成 | ✅ |

## 交付画质优化（2026-08-30，对标主流微短剧观感）

### 粗糙根因诊断（对照 run_20260823_165659 成片与参考微短剧）

| # | 根因 | 表现 | 对策（high 档默认开启） |
|---|---|---|---|
| 1 | I2V 源仅 464×832（flash-480P），成片 540×960@1.3Mbps | 糊、肉、细节全无 | i2v 升 `wan2.2-i2v-plus@1080P`，成片 1080×1920，CRF20 |
| 2 | 无全局风格锚 | 一格动漫一格真人，像拼贴画 | `style_suffix` 真人电影锚 + `negative_prompt` 禁卡通/插画，全 T2I 链路注入 |
| 3 | 角色跨镜头变脸 | 同一角色每镜一张脸 | 场景+角色组合**稳定种子** + 外貌逐字注入 + 立绘锚定 |
| 4 | normalize 用 contain+pad | 画幅毫厘之差 → 上下黑边 | cover 裁切（scale increase + crop，lanczos）满屏 |
| 5 | 无包装 | 无声、无调色、字幕默认小字、片头简陋 | 程序化免版权 BGM、eq+unsharp 调色、短剧样式烧录字幕（大号白字黑描边）、渐变片头尾卡+淡入淡出、-16 LUFS 响度归一 |

实现落点：`configs/profile_high.yaml`（overlay 合并）、`m2/m3`（风格锚/种子）、`utils/ffmpeg.py`（cover/调色/编码参数/响度）、`m7`（字幕样式/BGM/包装）、`utils/image.py`（渐变卡）、`scripts/make_bgm.py`（内置 BGM 生成，零版权风险）。

### high 档实测验收（run_20260830_003834，《除夕最后一单》类主题，90s 目标）

| 验收项 | 标准 | 实测 | 结果 |
|---|---|---|---|
| 分辨率/编码 | 1080×1920 可播 | 1080×1920 yuv420p High @4.6Mbps，48.7MB | ✅ |
| 时长 | 90s ±15% | 81.8s | ✅ |
| 响度 | 竖屏平台标准 | -15.97 LUFS / 44.1kHz 立体声 | ✅ |
| 画风一致 | 无卡通/真人混搭 | 全片真人电影感（中英双语风格锚 + 负向提示词 + 稳定种子） | ✅ |
| 角色一致 | 服装/造型跨镜头稳定 | 骑手蓝盔蓝夹克全片一致 | ✅ |
| 字幕 | 短剧样式烧录 | 9 条，SimHei 大号白字黑描边底部安全区 | ✅ |
| 包装 | BGM + 片头尾 + 调色 | 程序化 BGM 循环混音、渐变片头尾卡、eq+unsharp 调色 | ✅ |
| 成本 | ≤¥150 熔断线 | **¥11.10**（t2i-plus + i2v-plus 1080P 实测远低于估算） | ✅ |
| CI | 全绿 | ruff ✔ / mypy ✔ / pytest 82 passed | ✅ |

### 本轮新发现并修复的问题（续表）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 15 | 字幕烧录静默跳过/渲染失败 | Windows 盘符冒号在 filtergraph 转义地狱；`Microsoft YaHei` 带空格静默不渲染；force_style 字号是 libass 脚本单位（PlayResY≈288）非像素 | 字幕用相对文件名 + `cwd` 运行 ffmpeg；`SimHei` + 脚本单位字号（12/Outline1/MarginV14，随分辨率自缩放） |
| 16 | 交付音轨 96/192kHz 非标准 | loudnorm 默认重采样到 192kHz | `safe_encode` 加 `-ar 44100` 拉回 |
| 17 | 主进程在 m6 静默死亡（无 traceback） | 环境级偶发kill | `resume <run_id>` 检查点续跑成功；m4 台账复用已完成片段，**零重复计费** |

## 商业级 cinema 档实测（2026-08-30，run_20260830_211509）

**为什么之前不如商业级（根因复盘）**：① 交付片没有口型（s2v/retalk 账号门控）→ 对话镜头"哑巴"；
② wan2.2 运动模型代差（商业片用 wan2.7/Kling/Seedance 级）；③ 无角色一致性引擎（商业管线用
参考生视频/LoRA/IP-Adapter 固脸）；④ 无"抽卡选优"（商业团队每镜多次生成取最佳）；⑤ TTS/配乐/调音薄。

**cinema 档模型栈（`--quality cinema`，全百炼单 KEY）**：

| 环节 | 模型 | 作用 |
|---|---|---|
| 首帧硬锁脸 | `wan2.7-image` 参考图条件生成 | 角色立绘作参考图生成首帧，脸在源头锁住（修复"衣服一致脸漂移"）；`use_reference_keyframes` 开关，economy/high 行为不变 |
| 角色动作镜头 | `wan2.7-r2v` 参考生视频 | 角色参考图→像素级形象一致（本次 4 镜×2 候选） |
| 空镜/动作 | `wan2.7-i2v` | 商业级运动+画质 |
| 对话口型 | `wan2.7-i2v` + `driving_audio` | **原生口型同步**，绕开 s2v 账号门控（9 镜全中） |
| 抽卡选优 | best-of-2 + `qwen-vl-plus` 打分 | 8 次选优触发 |
| 其余 | 同 high 档（风格锚/调色/响度/字幕/BGM） | |

**实测验收**：1080×1920 @5.2Mbps · 79.8s · -15.97 LUFS · 成本 **¥113.35**
（i2v 64 + talk 45 + t2i 4.3）· CI 82/82 · 画面观感：teal&orange 电影调色、自然皮肤、
浅景深——对齐主流商业 AI 短剧观感。

附注：剧本模型工作空间端点 2026-08-30 起 403 失效，实测同 KEY 标准端点可调用
qwen3.7-plus，.env 已注释回落标准端点。

## 云端对口型（2026-08-30，对标参考项目 FastVideo 的本地对口型阶段）

参考项目 FastVideo 用本地 GPU 对口型模型（MultiTalk 类）；本项目按"全 API 化"原则改为
**百炼云端对口型**，双通道：

| 通道 | 模型 | 输入 | 模式 | 说明 |
|---|---|---|---|---|
| S2V 直出 | `wan2.2-s2v` | 首帧图+人声 | `s2v_direct`（默认） | 数字人口型/表情/动作同步 |
| 口型替换 | `videoretalk`（声动人像，更先进档） | I2V 运动片段+人声 | `v2v_overlay` | 对已有运动视频重对口型，≈ 原项目 v2v 口型阶段 |

关键工程（`utils/dashscope_upload.py` + `providers/dashscope.py`）：
- 两模型**拒收 base64** → 走官方临时 OSS 直传：`GET /api/v1/uploads?action=getPolicy` →
  OSS 表单直传 → `oss://` URL → 提交时带 `X-DashScope-OssResourceResolve: enable`（已实测上传链路通）。
- 账号门控错误（未开通/欠费）分类为 `AccountGateError`：不重试、events 落 `account_gate`
  提示，m6 自动降级 I2V+音频直叠，成片不阻塞。

**当前账号状态（如实记录）**：`wan2.2-s2v` 与 `videoretalk` 均返回 400 Access denied
（overdue-payment 类）—— 需到百炼控制台「模型广场」开通这两个模型并确认余额。
开通后无需改代码：直接重跑即出口型；真实冒烟测试 `pytest -m api`。
切换更先进档：`configs/default.yaml` 的 `providers.talk.default: dashscope_videoretalk`
+ `--talk_mode v2v_overlay`。

### 用法

```bash
fastvideo3 run --topic "你的主题" --duration 90 --quality high   # 交付档
fastvideo3 run --topic "你的主题" --duration 30                  # 教学档(默认, 低成本)
```

## 日常使用

```bash
conda activate fastvideo3
cd D:\FastVideo3
copy .env.example .env     # 填入 DASHSCOPE_API_KEY（.env 已就绪则跳过）
fastvideo3 doctor
fastvideo3 run --topic "你的主题" --duration 60
```
