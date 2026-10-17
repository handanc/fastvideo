# FastVideo3 系统联调报告（IT 验收）

> **2026-10-06 架构变更说明（本文历史记录的前置阅读）**
> 端到端（LangGraph 图）生成路径已**整体删除**：`graph/builder.py`/`nodes.py`/`state.py`、
> `interrupt()` 人工审片门、`SqliteSaver` 检查点、`POST /continue`、`POST /resume`、
> CLI `run` / `resume` 命令、`review_mode` 配置位都不再存在。现在只有一条路径：
> `fastvideo3 new`（建任务）→ 六次 `fastvideo3 step`（Web 面板的「开始生成」）。
> 因此本文中所有关于 **resume / 检查点续跑 / 审片门 / 端到端整图跑** 的条目都是**历史记录**，
> 保留是为了可追溯当时的问题与结论，不代表现在的行为；其能力对应物是：
> 续跑 = 再点一次「开始生成」（按盘上产物 + 血缘只补缺口）、人工把关 = 每一步之间的介入。
> 下面的「用法 / 日常使用」两节已按新命令改写，其余带日期的段落不改写。

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
| 首帧硬锁脸 | `wan2.7-image` 参考图条件生成 | 角色立绘作参考图生成首帧，脸在源头锁住（修复"衣服一致脸漂移"）；`use_reference_keyframes` 开关。**2026-10-17 起三档出厂全开**（本表写的是当年只有这一档开的状态）：一致性不是画质项，480P 也要用同一张角色图出同一个角色；只有"不达标自动重拍"仍留在 720P/1080P |
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

### 用法（2026-10-06 起：建任务 + 一步一步生成）

```bash
fastvideo3 new --topic "你的主题" --duration 90 --quality high   # 交付档；只建任务，不花钱
fastvideo3 step <run_id>                                          # 点一次补一步，共六步
# 教学档(默认, 低成本)：fastvideo3 new --topic "你的主题" --duration 30
```

## 日常使用

```bash
conda activate fastvideo3
cd D:\FastVideo3
copy .env.example .env     # 填入 DASHSCOPE_API_KEY（.env 已就绪则跳过）
fastvideo3 doctor
fastvideo3 serve           # 推荐：Web 面板里点「开始生成」，每步之间能改剧本/换图/删图
# 或纯命令行：fastvideo3 new --topic "你的主题" --duration 60  →  fastvideo3 step <run_id> ×6
```

## 一致性链路审计与修复（2026-10-01）

**结论：本文档 2026-08-30 那节"cinema 档实测"里的两条核心质量结论不成立，已修正实现并重写文档。**

审计方法：逐条比对 `output/*/cost.json`（60 个 run 的计费台账）、`events.jsonl`、
`final_review.json` 与实际产物图像。三条硬证据：

| 声称 | 台账实际 | 根因 |
|---|---|---|
| "首帧硬锁脸（wan2.7-image 参考图条件生成），修复衣服一致脸漂移" | 该 run 24 次 t2i **全是 `wanx2.1-t2i-plus`，`wan2.7-image` 0 次**（全 60 run 合计亦 0 次），且无 `fallback_used` 事件 | `DashScopeT2IProvider.generate` 收 `ref_image` 形参但 body 从不引用（死参数）；当时根本没路由到支持参考图的 provider |
| "抽卡选优 best-of-2 + qwen-vl 打分，8 次选优触发" | 该 run **`vlm` 计费 0 条**；全项目 15 条 `best_of_n` 事件 **pick 恒为第一个候选**、无 scores 字段 | 评审因账号欠费（`Arrearage`）失败，`except → score=0.5` 兜底把"没评上"伪装成"评过了" |
| "对齐商业级观感" | 真跑的 6 个 `final_review.json` 全是 `score:0.7 / feedback:""` —— 是 `dashscope.py` 解析失败的**默认值**，不是模型给的分；cinema run 连 `final_review.json` 都没有（欠费），分数只以 0.6 兜底值残留在 checkpoint | 评审异常被写成"通过" |

目视证据：主角 C1 定妆照本身是**动漫插画**（违反 cinema 的 photoreal 风格锚），跨镜头至少 4 张
不同的脸、头盔 4 种颜色（固化文本写"黄色头盔"）；即"衣服一致、脸漂移"并未被修复。

**已修（P0+P1，全部离线可验证，145 测试 / ruff / mypy 全绿）**：

1. 死参数变硬约束：新增 `CapabilityMismatchError` + `supports_seed`/`supports_reference_image` 能力位，
   不支持参考图的 provider 收到 `ref_image` 直接抛错；cinema 加 `require_reference_keyframes`
   在花钱前校验路由。
2. 降级变响亮：`fallback_used{dropped:[seed,ref_image]}`、`ref_condition_dropped{reason}`、
   `degradations.json`、`run_summary.review_unavailable`；台账新增 `CostEntry.meta`
   （`seed/ref_count/ref_ids/route/verified`），只看盘即可重建请求内容。
3. 分数不可伪造：删除 0.7/0.5 兜底，评审异常记 `verified=False` 且排序恒判负；
   `best_of_n`/`char_best_of_n` 事件带 `scores`，无真实分时改记 `best_of_n_unscored`。
4. 覆盖面扩大：参考图条件从"仅 cinema、仅 `characters[0]`"扩展到 **high 档 + 双人镜头多图**
   （`ref_max_chars`，默认 1，待 `scripts/probe_multi_ref.py` 探单通过后放开 2）；对白镜头经首帧
   继承锁脸并新增 `talk_identity_drift` 校验。
5. 源头与种子：立绘加确定性种子 + 写实风格闸门（`asset_style_gate`）；种子作用域
   `scene_cast → cast`（旧实现把场景编进种子，角色换场景即换脸）。
6. 可度量：新增 m9 `consistency_report.json`（VLM 结构化属性位 + 自校准锚 + 顺序互换 +
   纯 PIL 色相证伪）、`fastvideo3 consistency` / `compare-consistency`、
   `configs/ablations/` 四臂与 `scripts/run_ablation.py`（兑现 TDD 第 2 讲的消融承诺）。
7. 顺带修掉两个真实缺陷：`FakeI2VProvider.submit` 强制要 `image` 导致 fake cinema 的 r2v 分支
   必崩（`run_20260925_222322/_223246` 事件原文可查）；参考图锁定语料曾被写进
   `shot.visual_prompt` 并被 m6 原样传给口型模型（无效指令）。

**待办（需充值后执行）**：`scripts/probe_multi_ref.py` 探单多图协议 → 通过后设
`ref_max_chars: 2`；跑 `scripts/run_ablation.py --with-shots` 出真实四臂对比表；
重跑一次真实 cinema 并按新判据核对 `wan2.7-image` 计费 > 0。

## image2video / 对口型链路审计与优化（2026-10-03）

对照参考项目（本地 GPU 单体，口型用 [MultiTalk](https://github.com/MeiGen-AI/MultiTalk)：
Wan2.1-I2V-14B + wav2vec2，**每人一条音轨 + 人脸 bbox 绑定**，最多 3 人同屏各自对口型），
本项目当前链路与结论：

| 镜头类型 | 本项目实现 | 合理性结论 |
|---|---|---|
| 动作/空镜 | 首帧 → i2v（wan2.2/2.7/3.0，media 数组协议），motion-only 提示词 | 合理，与参考项目同构（参考用 Wan2.2-TI2V-5B 本地） |
| 单人对白 | 首帧 + 该角色台词音轨 → 口型模型（cinema/high: `driving_audio` 原生口型；economy: wan2.2-s2v 数字人） | 合理；音轨时长驱动视频时长，不硬截 |
| 多人对白 | **默认运动片段 + 音轨叠加，不模拟口型**（`two_person_talk: overlay`） | 有意取舍：API 口型模型只有"单音轨驱动整幅画面"，没有 MultiTalk 的音轨-人脸绑定；单音轨喂双人画面会让**没说话的人也跟着动嘴**，比无口型更糟。要真多人口型只能上 MultiTalk 级本地模型，与"全 API、无 GPU"定位冲突 |

本轮修掉的三个真实缺陷：

1. **`.env` 模型覆盖污染兜底通道**：`FV3_TALK_MODEL=wan3.0-video` 同时作用于主通道与
   fallback，而 wan3.0 的 media 枚举不含 `driving_audio`（实测 400：`input.media.1.type`）
   → 主备同一个坏模型，"换道重试"等于原地重试，对白镜头全部降级成无口型片段。
   现 `make_fallback` 一律 `overrides={}`（兜底只用 yaml 里它自己平台的模型），且 m6
   识别协议错误后**立即止损**换备用通道（`s2v_protocol_mismatch` / `talk_fallback_used` 事件）。
2. **TTS 单句失败 abort 整个视频步**：edge-tts 限流（`No audio was received`）曾让与配音
   无关的动作镜头也白等一轮。videos 步内顺序改为 片段(m4) → 配音(m5) → 口型(m6)。
3. **语音平直/抢话/响度漂移**（对口型输入质量的三处损伤）：edge-tts 此前**完全丢弃
   emotion** → 现映射为 rate/pitch 韵律偏移；多句拼接零间隔 → 现句间 0.25s 换气；
   合并音轨响度不一 → 现归一到 -16 LUFS（`shot_{id}_mix.wav`，母带不动）再喂口型模型。

仍待办（需真实额度探单）：wan3.0-video 的口型参数形态（是否改收 `reference_audio`）未知，
`scripts/probe_multi_ref.py` 探通前 cinema 的口型主通道建议保持 `wan2.7-i2v`（留空
`FV3_TALK_MODEL` 即可）。

## 口型度量证伪与重建 · s2v 通道定性与探单工具（2026-10-03 续）

起因是一个必须澄清的问题：**"目前最优的对口型方案是不是 wan2.7-i2v + driving_audio"**。
回答它需要两样东西：各通道的真实能力证据，以及一把**可信的尺子**。本轮的结论是
——两样此前都没有，而且旧尺子一直在产出看起来像结论的噪声。

### 1. 旧尺子被自己的对照实验证伪

对 8 月 run（唯一真跑过 `wan2.7-i2v + driving_audio` 的一轮，9 条口型片段、按 ¥1/s 计费）
测得的"嘴部运动 × 音频包络"相关系数为 -0.474 / -0.387 / 0.007 / 0.243 / -0.061 / 0.198 /
0.120 / -0.398 / 0.072（均值 ≈ -0.05），当时读作"这个通道不做口型"。

加对照区复测同素材：

| 片段 | 嘴区 corr | 整脸 corr | 背景 corr | 三区运动幅度（嘴/脸/背景） |
|---|---|---|---|---|
| shot_S2_08 | 0.243 | -0.181 | -0.576 | **0.8** / 4.4 / 4.2 |
| shot_S1_03 | -0.474 | 0.186 | -0.203 | 23.1 / 15.1 / 18.5 |

写死的比例框（画面下缘中心）在特写/过肩/全身混排的真实分镜下**根本不落在嘴上**：
S2_08 的"嘴区"运动幅度 0.8（一块几乎静止的区域），而脸和背景都在 4 以上；背景对照
相关 -0.576（本应≈0）说明数字里混的是镜头运动。→ "wan2.7 没有口型"这个结论作废。

### 2. 新尺子：先定位、再对照、量不准就明说无结论

- `RegionBoxes`（core/schema.py）+ `prompts/lipsync_boxes.v1.md`：视觉模型在首帧上定位
  **嘴唇框与双眼框**（归一化，容忍 0~1 / 0~100 / 0~1000 三种坐标制与反序），`sane()`
  拒绝退化框、占满全画面、以及"眼框=嘴框"这类对照失效。
- `lipsync_diag()`：嘴 / 眼（同脸对照）/ 背景（画面外对照）三区一起测；
  **嘴区运动幅度须 ≥ 1.5× 眼区**才算"量到了嘴"，否则 `verdict=inconclusive`。
- `scan_lipsync()`（免模型兜底）：上百个候选框滑窗取最大相关，并用 **8 个"打乱顺序"的
  音频包络**跑同一套扫描当零分布 —— 只报 max(corr) 而不报零分布，就是把噪声当口型。
  两个非显然的实现约束（都踩过）：
  ① 对照**不能用循环移位**：对白节律近似周期（4 帧说/4 帧停），整周期移位＝原序列本身，
     实测把对照抬到 0.769 vs 真实 0.729，检验当场失效；
  ② 候选必须先按**运动幅度**筛（≥ 全场最大的一半）再比相关：h.264 压缩噪声会随说话帧
     整幅起伏，只看相关会把"嘴"定到离真嘴半个画面之外。
- **接线纪律**：`inconclusive` **绝不**进自动重做回路（m9 `gate_failures` 要求
  `located is True and verdict == "unsynced"`），看板徽标也新增第四态"未测准"（warn，
  不是 err）—— 拿量不准的尺子标红，用户会去重做一个其实没问题的镜头，那是真花钱。

### 3. 用新尺子重测那 9 条 wan2.7-i2v 片段（免费，本地）

1 条 `synced`（corr 0.823 vs 乱序对照 0.344，确有随语音运动的区域），
8 条 `inconclusive`（画面运动本身就能造出 0.25~0.63 的最大相关，超不过对照就不算证据）。
→ 现有盘上证据**既不能证明也不能否证** "wan2.7-i2v 做口型"。

### 4. `wan2.2-s2v` 的定性：它没被证伪过，因为从没跑到它

- 官方规格（百炼 wan2.2-s2v 模型页/API 参考）：**它就是音频驱动对口型专用模型**，
  480P ¥0.5/秒、720P ¥0.9/秒，音频 <15MB 且 <20s，输出上限 720P。
- run_20261002_235904 那三次 `s2v_attempt_failed` 的原文是
  `Input should be 'first_frame'|'reference_image'|'reference_audio'|...: input.media.1.type`
  —— 服务端在校验 **`input.media[]` 数组**，即 wan2.7/3.0 新协议；而 s2v 发的是
  `input.image_url + input.audio_url`，**没有 media 数组**。那次 run 的 cost.json 里
  **talk 计费 0 条**：请求压根没到 s2v。真凶是当时生效的 `FV3_TALK_MODEL=wan3.0-video`
  把主通道与兜底同时改道（已修，见上一节第 1 条）。
- 但本轮默认测试套件意外真实提交了一次 s2v，返回 **400 `url error, please check url`**
  → s2v 今天**仍然**没接通，且卡在**我们的输入 URL 形态**上（临时 `oss://` +
  `X-DashScope-OssResourceResolve` 这条流程对 s2v 端点是否成立，尚未证实）。
  所以准确表述是：**"s2v 不能对口型"从未被证明；被证明的是"我们还没提交成功"。**

### 5. 本轮顺带修掉的两处"会伪装成模型能力问题"的配置越界

- m6 曾把 `cfg.i2v_resolution`（cinema = 1080P）原样透传给所有口型通道，而 s2v 上限 720P
  → 提交即 400，极易误诊成"s2v 不行"。现按 provider 官方规格钳位（`Provider.max_resolution`
  + `clamp_resolution()`），钳位时落 `resolution_clamped` 事件；主通道与兜底通道各自钳。
- s2v 的音频 <20s 硬上限此前无守卫 → 现抛 `CapabilityMismatchError` 立即止损（不再空转
  3 次重试），并把"规格外"与"瞬时失败"分开上报，事件里看得出是哪种。

### 6. 事故与加固：默认 `pytest` 会真花钱

`tests/test_talk_lipsync_api.py` 带 `pytest.mark.api`，但 `pyproject.toml` 只**声明**了
marker、没有 deselect 规则 → 一次普通 `pytest` 真向百炼提交了 s2v 任务（400 未建任务，
未产生计费）。现加 `addopts = "-m 'not api'"`，真冒烟必须显式 `pytest -m api`。
**教训**：任何"读真实 .env、打真实端点"的测试都必须默认排除，否则回归套件就是隐性付费入口。

### 7. 定档与探单工具

`scripts/probe_lipsync.py`：
```
python scripts/probe_lipsync.py -v scan output/run_20260830_211509      # 免费，测已有片段
python scripts/probe_lipsync.py scan <run> --with-vlm                   # 加视觉模型精确定位
python scripts/probe_lipsync.py probe --frame 首帧.png --wav 混音.wav \
    --channels wan2.2-s2v,wan2.7-i2v --cap 5 --spend                    # 花钱，逐通道各 1 条
```
`probe` 会先按 `pricing.yaml` 把单价 > `--cap`（默认 ¥5/秒，用户红线）的通道直接跳过。
**建议顺序**：先解决 s2v 的 `url error`（400 不花钱，可用 `pytest -m api` 迭代）→
再用同一首帧+同一段混音各出 1 条 5 秒短片（s2v ≈¥2.5 + i2v ≈¥5）→ 用 `scan`/`--with-vlm`
测 → 三条以上素材一致才定档。在此之前，cinema 的 talk 主通道保持 `wan2.7-i2v`
（留空 `FV3_TALK_MODEL`），**不要**把"当前最优"当成已证事实。

### 定档（2026-10-04）：口型主通道换 **videoretalk**，s2v 降为兜底

人工审核 `run_20261002_235904/final/final.mp4` 判定"没有对口型"。核对盘上事实：

- `talk/` 目录 0 个 mp4，`cost.json` 里 **talk 计费 0 条**（只有 i2v ¥32）→ 那三个对白镜头
  是"i2v 运动片段 + 音轨硬叠"，成片里**一个口型镜头都没有**。所以这条片子否证的不是
  某个口型模型，而是**我们的降级路径**：主通道被 `.env` 的 `FV3_TALK_MODEL=wan3.0-video`
  改道到拒绝 `driving_audio` 的模型，兜底又被同一个覆盖污染 → 全部失败 → 静默叠音。
  两处污染上一轮已修，本轮补的是"修好之后该用谁"。

选 videoretalk 作主通道的三条理由（按分量排序，**便宜不是理由**）：

1. **它不重画画面**：输入是 m4 已经生成的运动片段，只重绘嘴部区域。s2v 是"首帧 + 音频
   重新发明一个镜头"——运镜、构图、身体动作全部重来，等于把 m3+m4 辛苦锁住的身份
   再洗一遍牌；对"前后人物一致是致命项"这个项目定位来说，这是决定性差异。
2. **输出分辨率/帧率与输入一致**：cinema 的 1080×1920 原样保住。wan2.2-s2v 官方上限
   720P，混进 1080P 成片必然上采样发糊（正是此前"对白镜头按 480P 出片"那类缺陷的根源）。
3. ¥0.08/秒 + 新账号 1800 秒免费额度：口型这一步事实上可以按 ¥0 试、按 ¥0 跑。

顺带修掉的三处：

- **`pricing.yaml` 把 videoretalk 记成 ¥0.50/秒（官方 ¥0.08，高估 6 倍）** —— 这个错误
  直接扭曲过通道选型：让"i2v 之后再 retalk 一遍"看起来和 s2v 直出一样贵，于是被跳过。
  同时 `wan2.2-s2v` 按我们**实际发送的 720P 档**记 ¥0.9/秒（熔断宁可早触发不能晚触发）。
- **wan2.2-s2v 的端点是错的**：官方创建任务地址是 `services/aigc/image2video/video-synthesis`，
  我们打在 `services/aigc/video-generation/video-synthesis` 上，服务端回 `400 url error,
  please check url`。报文说"url"，于是被读成"s2v 不接受我们的输入"，差点把整条便宜的
  专用口型通道判死。**原先的 respx 用例 mock 的就是那个错端点，所以它一直绿**——
  自证式测试抓不到路由错误；现在用例同时 mock 两个端点并断言只走对的那个。
- retalk 缺运动片段时抛 `CapabilityMismatchError`（一次止损、换 s2v），不再空转 3 次
  然后静默降级成叠音——那正是这次人工审核看到的缺陷形态。

配置落地：`configs/default.yaml`、`profile_high/cinema.yaml` 的
`talk: {default: dashscope_videoretalk, fallback: dashscope_wan22_s2v}`，
`talk_mode: v2v_overlay`（`.env` 的 `FV3_TALK_MODE` 同步改掉，否则环境变量压过 yaml，
配置改动会静默不生效）。

**零成本验证路径（待授权执行）**：对白镜头的运动片段用 `wan3.0-video` 的**免费额度**出
（该账号 i2v 已验证可用），再喂 videoretalk 的 1800 秒免费额度 → 1 个镜头约 ¥0.55-1.0
甚至 ¥0，产出的片段用 `scripts/probe_lipsync.py scan` 度量 + 人工看片。
两条通道的实测表一旦出来，本节的"押注"部分才算落地。

### 实测结论（2026-10-04，一条镜头的真实 A/B）：videoretalk **确实**在做口型，而我的度量不能信

`scripts/verify_lipsync_channel.py` 在 run_20261002_235904 的 `shot_1_2` 上真跑了一条：
首帧 → `wan3.0-video` i2v 运动片段（试用额度，¥0）→ `videoretalk` 重对口型（¥0.08/秒，
1800 秒免费额度）→ 两段产物都落盘在 `output/run_20261002_235904/_lipsync_verify/`。

- **videoretalk 通道可用**，输出 1080×1920 / 30fps（与输入同规格，兑现"不降级"）；
  时长 3.40s 与驱动音频 3.41s 对齐。
- **人眼判定为"确有口型"**：`probe_lipsync.py sheet` 出的嘴部接片上，每格顶部橙条是该帧
  响度 —— 响度 0.96 / 0.99 / 0.93 的帧嘴张开露齿，响度 0.00 的帧嘴闭合；
  而**改动前**的同源片段嘴与响度无关（0.00 时张着、0.61 时闭着）。这是一真一负对照组。
- **但自动度量判错了它**：同一条人眼确认同步的片段，原始 corr 0.163、扣眼区对照的
  偏相关 0.113，都低于 0.25 阈值 → 会被判"不同步"。原因不是框错位（背景运动幅度 3.4
  远低于嘴 17.5，说明框在脸上），而是"相邻帧灰度差"这个特征本身：说话时嘴**保持张开**
  的时段帧间差反而接近 0。
- 处理：**把口型从自动重做回路里摘出来**（m6 不再标 stale、m9 闸门不再据此 reshoot、
  看板由红色"不同步"改为琥珀"待人工"），并留下 `probe_lipsync.py sheet` 作为人眼定案工具。
  理由写进代码注释：度量假阴性比假阳性贵得多 —— 它会让我们反复重抽本来合格的镜头。

顺带修掉的度量自身缺陷（都有回归用例）：
- 取样先缩帧再裁框 → 中远景的嘴只剩 3×3 像素，量的是锯齿；改为**原始分辨率上先裁后缩**。
- 幅度门槛 1.5× 眼区是按"眼区完全静止"的合成素材校的，真人整脸一起动（实测 嘴18.1/眼18.2）
  → 把好片段误判成"量不准"；降到 0.5× 只用来挡严重错位，判同步与否交给偏相关。
- 零分布对照不能用循环移位（周期性节律下移整周期＝原序列），必须打乱顺序。

## 底层逻辑修正（2026-10-04）：让"无口型成片"在结构上不可能静默出现

用户裁定："修正某个任务的视频毫无意义，要修正整个项目的底层生成对口型视频的逻辑。"
复盘发现"全片没有对口型"不是某次运行的事故，而是**五层逻辑共同允许**的结果，逐层修：

1. **通道层**：口型主通道换成 videoretalk（见上节定档），s2v 端点修正、单价修正。
   旧主通道（s2v 直出 / wan2.7+driving_audio）一个端点错、一个口型不可证。
2. **模式层**：`talk_mode` 默认 `v2v_overlay`，且**四处默认值一起改**（yaml、.env、
   `api.RunCreate`、CLI `--talk-mode`、前端表单）。只改 yaml 的话，从网页新建的 run
   仍会拿到 `s2v_direct` —— 默认值散落多处是本项目反复出现的静默失效模式。
3. **完成度层（根因）**：旧判据"clips 在盘即视为该镜头有视频"，于是 m6 降级出的叠音
   片段让对白镜头**算完成** → 逐步生成不再提供修复、成片照常合成、判定照常通过。
   现改为：`talk_mode != audio_overlay` 时，对白镜头完成 = `talk/shot_x.mp4` 存在。
   降级产物从此是**缺口**，逐步生成会继续提供"补口型"。
4. **交付层**：新增 `lipsync_coverage()`（progress.py，进度/判定/看板共用一个函数），
   合成时写入 `run_summary.lipsync_coverage`；high/cinema 有缺口即
   `quality_verdict.passed=False` + 看板红幅"本片部分没有对口型"。economy 只报告不判死。
   判定并入逻辑抽成 `apply_lipsync_verdict()` 单份实现，图路径与逐步路径共用
   （"同一逻辑两份实现"是本项目所有静默退化的共同根因，不再新增第二份）。
5. **豁免层**：双人镜头按 `two_person_talk=overlay` 策略**显式放弃**口型（API 单音轨
   驱动整幅画面，会让没说话的人动嘴）。没有台账时它会永远 gap、把逐步生成困在死循环；
   现 m6 落 `lipsync_waived.json`，coverage 单列 `waived`（看板琥珀提示），
   既不冒充有口型、也不冒充缺口。

度量层的自我否定（同轮）：自动口型度量在人眼确认同步的 videoretalk 产物上只给
0.163/0.113（假阴性），因此**从自动重做回路中摘除**（m6 不标 stale、m9 闸门不据此
reshoot、看板改琥珀"待人工"），人眼定案工具为 `probe_lipsync.py sheet`。

## 口型通道换档（2026-10-04 二次定档）：fal.ai 托管 MeiGen **InfiniteTalk**

### 为什么换

上一档定的是百炼 videoretalk，依据是**架构与规格**（不重画画面、输出分辨率=输入、
¥0.08/秒）。用户人工看片后的裁定是：对口型效果不合格 —— 嘴部动作幅度小、同步感平。
"便宜且不降级"不能替代"像真的在说话"，所以主通道换成 InfiniteTalk。

InfiniteTalk（MeiGen-AI，arXiv 2508.12946）是**稀疏帧视频配音**模型：给"视频 + 音频"
重画口型，并**同步对齐头部运动、身姿与表情**；也支持"图像 + 音频"直出。底座
Wan2.1-I2V-14B-480P + `chinese-wav2vec2-base`（中文语音原生）。本地推理需 24GB 级显存
（官方低显存目标是单张 4090），本机 RTX 5070 Laptop 8GB 且其 torch 2.4.1+cu121 栈与
50 系不兼容 → **按 fal.ai serverless API KEY 调用**，KEY 手动写在 `.env` 的 `FAL_KEY`。

### 官方规格（2026-10-04 从 `fal.ai/api/openapi/queue/openapi.json?endpoint_id=…` 实取）

| 端点 | 必填 | 可选（枚举/默认/上下限） | 单价 |
|---|---|---|---|
| `fal-ai/infinitalk` | `image_url` `audio_url` `prompt` | `resolution` 480p\|720p（默认480p）、`num_frames` 41–721（默认145）、`acceleration` none\|regular\|high（默认regular）、`seed` 0–4294967295（默认42） | **$0.20/秒**，720p 翻倍 |
| `fal-ai/infinitalk/video-to-video` | `video_url` `audio_url` `prompt` | 同上，但 `num_frames` 41–**241** | **$0.30/秒**，720p 翻倍 |
| `fal-ai/infinitalk/single-text` | `image_url` `text_input` `voice` `prompt` | 自带 TTS（20 个英文音色） | $0.20/秒 |
| `fal-ai/infinitalk/multi-text` | `image_url` `first_text_input` `second_text_input` `prompt` | 双说话人 | endpointBilling 仍 price=0 / `enterprise_status: pending` → **未 GA，不接** |

协议要点：提交 `POST https://queue.fal.run/{endpoint}`（头 `Authorization: Key <FAL_KEY>`，
请求体就是模型输入本身，没有 DashScope 的 input/parameters 包装）→ `request_id`；
`GET .../requests/{id}/status`（`IN_QUEUE|IN_PROGRESS|COMPLETED`）→ `GET .../requests/{id}`
→ `{"video":{"url":…},"seed":…}`。输入必须是公网 URL：先
`POST https://rest.fal.ai/storage/upload/initiate?storage_type=gcs` 拿 `upload_url`/`file_url`
再 `PUT` 字节（与百炼临时 OSS 同构，实现在 `providers/fal.py`，失败**直接抛错不偷偷改走
base64** —— 静默兜底是本项目反复踩的病）。

帧数换算：底座 `shared_config.sample_fps = 16`、VAE 时间步长 `(4,8,8)` → 帧数必须 `4n+1`。
所以 `num_frames = ceil((时长×16 − 1)/4)×4 + 1`，向上取整（帧数少了音频会被截，口型停在
半句上）。据此 v2v 实际上限 ≈15.06s、i2v ≈45s → 类属性 `max_audio_sec = 15.0`（取严的那档，
因为默认模式是 v2v_overlay）；`max_resolution = "720p"`。

### 与既有链路的接缝（谁改了什么）

| 层 | 改动 | 为什么必须改在这里 |
|---|---|---|
| 通道 | 新增 `providers/fal.py`（上传 + 队列 + 分档计价 + `probe_key` 零成本探活） | 外部平台不该污染 dashscope.py |
| 端点选择 | 同一 provider 内按输入形态自动选：有运动片段→`/video-to-video`，只有首帧→i2v 端点 | `talk_mode` 的两种语义都覆盖，m6 不用加分支 |
| 规格 | `clamp_resolution` 把档位 1080P 钳到 720p 并记 `resolution_clamped`；>15s 音频抛 `CapabilityMismatchError` 立刻换通道 | 枚举外值 422 的重试价值为 0（s2v 的 `400 url error` 教训） |
| 计价 | `Provider.price_key()/worst_case_price_key()` + `pricing.yaml` 四个 `端点@档位` 键（汇率 ¥7.15/USD 向上取整） | 同模型 480p/720p 差 2 倍、i2v/v2v 差 1.5 倍；笼统模型价会让熔断失真 |
| 红线 | 装配期 `_video_cost_cap(cfg, provider)` 改用 `worst_case_price_key()` | 用户 ¥5/秒红线必须按**最坏档**校验（¥4.34/秒 刚好在内） |
| 预估 | `estimate_run_cost_cny()` + `GET /api/cost-estimate`；cli 预飞、新建表单、看板都用它 | 口型涨 25–50 倍后，"熔断够用"不能再靠旧量级直觉；前端不再手抄单价表 |
| 完成度/交付 | 沿用上一轮的 `lipsync_coverage` + 红幅 + `quality_verdict` 并入（未改） | 通道换了，"没口型就是缺口"的结构约束不变 |
| 自检 | `doctor`（CLI 与 API）加 `FAL_KEY` 行与 `FAL 账户探活(¥0)` | KEY 缺失要在花钱前说出来，不是等 run 失败 |

### 成本量级（30s 成片，口型按对白约 18s 估）

| 档 | 口型单价 | 全片预估 | 说明 |
|---|---|---|---|
| economy | ¥2.17/秒（480p v2v） | ≈¥44–50 | 熔断 ¥120 够用 |
| high | ¥4.34/秒（钳到 720p） | ≈¥90–102 | 熔断 ¥120 勉强，建议 ¥150 |
| cinema | ¥4.34/秒 | ≈¥114–144 | **¥120 熔断会被口型步骤吃穿** → 提到 ≥¥190 |

⚠ 熔断低于预估时的表现是**静默降级成无口型叠音**（正是用户投诉的形态），所以现在
`RunDetail` 有红幅、`fastvideo3 new` 预飞会打印"熔断低于预估下限"、`fal_infinitalk` 的
`budget_skip` 仍进 `events.jsonl`。**看片前先确认 talk/ 目录有产物**，或
`python scripts/probe_fal_infinitalk.py check`。

### 未证实项（下一条该跑的实测）

1. **竖屏承接**：i2v 端点的 `image_url` 说明写"不符合所选画幅会 resize+center crop"，
   但 schema 里没有 `aspect_ratio` 参数 → 9:16 首帧到底是被裁还是保持竖幅，必须真出一条才知道；
   v2v 端点理论上承接输入片段的画幅。用 `python scripts/probe_fal_infinitalk.py shot --clip … --wav …`
   （它会打印产物 WxH/fps/时长）。
2. **720p 上限对 cinema 的代价**：1080P 交付档的对白镜头现在按 720p 出片再由 m7 cover 上去。
   这是"口型准"换"分辨率高"的显式取舍（用户裁定取口型），但画质影响未量化。
3. **多镜头稳定性**：InfiniteTalk README 自陈 v2v "会模仿原片运镜但不完全一致"，长视频有
   色偏倾向（FusionX LoRA 更明显）；单镜头效果 ≠ 全片稳定。身份漂移由
   `_identity_drift_check`（服装主色 hue 距离）与 m9 一致性度量继续盯着。
4. **双人镜头**：本地 InfiniteTalk 有多人模式（需 `multi/` 权重 + 多路音轨），fal 的
   `multi-text` 端点未 GA 且只带英文音色 → 本期仍按策略显式豁免（`lipsync_waived.json`），
   看板显示"⚠ N 个对白镜头按策略无口型"。

## 口型通道三次定档（2026-10-05）：百炼 `wan2.2-s2v` 数字人为主，fal InfiniteTalk 降为兜底

用户裁定：按官方 API 参考（`help.aliyun.com/zh/model-studio/wan-s2v-api`，文档更新
2026-09-22）把对口型视频改由 **阿里云百炼 wan2.2-s2v** 生成，并回答 KEY 问题。

### 官方规格（照抄进代码注释与类属性，不靠记忆）

| 项 | 值 |
|---|---|
| 能力 | **单张图片 + 音频** → 说话/唱歌/表演视频（口型、表情、动作随音频）；真人与卡通皆可 |
| 创建任务 | `POST {base}/services/aigc/image2video/video-synthesis`，头 `X-DashScope-Async: enable`、`Authorization: Bearer <KEY>` |
| 查询 | `GET {base}/tasks/{task_id}`（默认 QPS 20；结果 URL 保留 24h） |
| 图像 | jpg/jpeg/png/bmp/webp，**宽高均在 [400, 7000] px**；只接受公网 HTTP(S) URL |
| 音频 | wav/mp3，**<15MB 且时长 <20s**；要清晰响亮人声、去噪去 BGM |
| `parameters.resolution` | **480P / 720P**（默认 480P）；官方明写"尽量保持与输入图像相同的宽高比" → **竖屏 9:16 不被裁** |
| 时长 | 输出视频时长 = 输入音频时长；**音频 >20s 任务报错**，官方解决办法是"切成 <20s 多段分别生成再拼接" |
| 单价 | 480P **¥0.5/秒**、720P **¥0.9/秒**，按**输出秒**计费；**调用失败不计费也不消耗免费额度** |
| 额度/限流 | 新账号 **100 秒**免费额度；提交 RPS 5、**同时处理中任务 1**（m6 逐镜头串行提交天然满足） |
| 地域 | 文档要求**华北2（北京）**与该地域 KEY；默认公共端点 `dashscope.aliyuncs.com/api/v1`，业务空间专属端点用 `FV3_S2V_BASE_URL` 覆盖 |

### 本轮改的东西

| 层 | 改动 | 实测/官方依据 |
|---|---|---|
| 路由 | 三档 `talk: {default: dashscope_wan22_s2v, fallback: fal_infinitalk}` | 兜底必须与主通道**同输入形态**（图片+音频）；videoretalk 只吃视频，在 s2v_direct 下等于没有兜底 |
| 模式配对 | `talk_mode` 五处默认统一到 **s2v_direct**（schema / AppConfig / load_config 回落 / api.RunCreate / cli）+ 三份 yaml + 前端 | s2v 不接受视频；`DashScopeTalkProvider.submit` 收到 `video=` 现在抛 `CapabilityMismatchError`，**不再静默丢掉 m4 花钱生成的运动片段** |
| 出参 bug | `_first_media_url` 同时支持 `output.results` 为**对象**（s2v/retalk）与**数组**（t2i） | 官方成功响应示例是 `"results": {"video_url": …}`；旧实现写死数组 → **任务成功那一刻 KeyError(0)**：钱已花、产物就在响应里却被判失败 |
| 分辨率 | provider 内再钳一次：超上限档名**靠上取 720P**，完全认不出才用官方默认 480P | 旧行为是"1080P 原样透传"（每次必 400）；静默降到 480P 等于悄悄丢画质不记账 |
| 输入硬规格 | 提交**前**检查：图片宽高 [400,7000]、音频 <15MB（超限 `CapabilityMismatchError`） | 先传几 MB 到临时 OSS 再吃 400 = 白跑一趟且看不出是谁的错 |
| 超长对白 | 新增 `plan_talk_groups`（**句子边界**分组）+ `talk_for_shot`（多段 → 各段一次提交 → `concat_demux(-c copy)` 拼回一个镜头片段）+ `_stitch_groups` | 官方唯一推荐做法；上限取**主/兜底里更严的那个**（fal 15s < s2v 19.5s）；段号进 `job_id` 与台账 `note`，否则续跑会把后段判成"已完成"；事件 `s2v_split{shot,groups,cap_sec}` |
| 计价 | `wan2.2-s2v@480p`=0.50、`@720p`=0.90 两档键 + `worst_case` 表；`price_key()` 先按模型上限钳位再取档 | 同模型两档差 80%；不存在的档名会让 `price_of` 静默回落默认价 → 熔断失真 |
| 预估/红线 | `estimate_run_cost_cny` 走 `price_of_route(路由名, 分辨率档)`；装配期红线走 `worst_case_price_key()` | 预估要贴账单、红线要宁紧勿松，两个方向不能反 |
| 账号探测 | 新增 `probe_model_activation()` 并接进 `doctor --probe`（CLI 与 API 两处） | **实测**：本账号 `wan2.2-s2v` 探针返回 400 `InvalidParameter.DataInspection`（媒体不可达，未建任务）→ 模型可路由；而 `Model not exist` 会被分类成 `arrears`。**同时纠正探针写法**：只发 `{"model":…}` 会被接受并真的建任务（实测该任务 0.2s 后自己 FAILED，按官方规则不计费，但白占任务号与并发位） |
| 前端 | 新建表单默认 `s2v_direct`、通道说明、预估改问后端 `GET /api/cost-estimate`；看板红幅文案与"熔断低于预估"提示跟随实际路由 | 单价表只存在 pricing.yaml，JS 里那份只作离线兜底 |

### 成本量级（30 秒成片，口型按对白约 18s 计）

| 档 | 口型单价 | 全片预估 | 结论 |
|---|---|---|---|
| economy | ¥0.50/秒（480P） | ≈¥14-20 | 熔断 ¥120 绰绰有余 |
| high | ¥0.90/秒（钳到 720P） | ≈¥28-40 | 同上 |
| cinema | ¥0.90/秒 | ≈¥52-82 | 同上；**上一档 fal 主通道时的 ¥114-144 不再适用** |

`python scripts/probe_fal_infinitalk.py check` 与 `GET /api/cost-estimate` 都按当前路由实时取价。

### KEY 问题的回答（用户问的原话）

**不需要新申请 KEY**：现有 `DASHSCOPE_API_KEY` 就够 —— 实测它对 `wan2.2-s2v` 的提交请求
返回"媒体不可达"而不是"模型不存在/未开通/欠费"，说明这把 KEY（华北2/北京公共端点）
可以路由到该模型。两个前提：① 该 KEY 必须是**北京地域**创建的（`dashscope.cn-beijing`
系；跨地域 KEY 会 401/403）；② 若哪天报 `Model not exist`/`未开通`，去百炼控制台
「模型广场 → 万相-数字人」开通即可（`doctor --probe` 会把这条直接说出来）。
`.env` 新增可选项 `FV3_S2V_BASE_URL`（默认公共端点，业务空间隔离时才需要填）。

顺带记录：`.env` 里那把 `FAL_KEY` 当前探测为 **403 invalid_key**（无效或过期）。它现在
只影响**兜底**通道，不影响主链路；想让兜底生效需到 fal.ai dashboard 重新生成。

### 已证实 / 未证实

- ✔ 已证实：官方端点与请求体形态、出参 `results` 为对象、图片/音频硬规格、两档单价与
  100 秒免费额度、账号可路由（¥0 探测）、286 条回归全绿 + mypy/ruff 干净 + 前端构建通过。
- ✘ **尚未证实**：**真实出一条口型片段**（需要一次 `--spend` 探单：5s@480P ≈ ¥2.5，
  其中口型 ¥2.5；或直接用 `fastvideo3 step <run>` 跑 videos 步骤）。s2v 的中文嘴型
  自然度、以及"从首帧重画整段画面"会不会漂身份（`_identity_drift_check` 会记事件），
  都要看产物。
- ✘ 未证实：`usage.fps` 官方示例给 16（另有文档写 30）；m7 统一 normalize 到
  `output_fps` 所以不影响交付，但 s2v 片段与 i2v 片段的帧率是否一致值得一并看。
- 双人镜头仍按策略**显式豁免**（`two_person_talk=overlay`）：s2v 是单音轨驱动整幅画面，
  多人各自口型需要每人一条音轨 + 人脸绑定，API 侧没有该能力。

## 人物音频修复（2026-10-05）：为什么"配音听起来很奇怪"

用户反馈成片人物音频奇怪。查下来**不是模型选错，而是四件事叠在一起**，全部有盘上证据
（`output/run_20261005_130208`，cinema 档、talk_mode=s2v_direct）：

| # | 病灶 | 证据 | 修法 |
|---|---|---|---|
| 1 | 配置写 cosyvoice，**实际每句都出自免费 edge-tts** | `cost.json` 三条 `('tts','edge_tts','edge-tts','line:xxx(fallback)')`；而 `degradations.json` 里**一条都没有** | ①`m5` 音色池改成按模型版本给名（`COSY_V2_VOICES` 全带 `_v2`）；②永久错误（418/未开通）**不空转重试**直接止损；③降级必须写 `log_degradation("m5_audio","tts_provider_failed")` + 事件 `tts_fallback_used` |
| 2 | 音色是 **md5 抽签**，不看性别 | `script.json` 里 5 个角色 `voice_id` 全为空 → `md5(cid)%5` 在 5 个 edge 音色里任选 | `plan_voices()`：按角色性别分桶、桶内轮流取（同性别不撞嗓子）；`Character` 新增 `gender`/`age_band`，编剧提示词把这两项写成硬要求；老剧本缺字段时从 `appearance`/`name` 关键词兜底推断（两边同时命中判"未知"，不押 50% 的性别） |
| 3 | **情绪 100% 丢失** | 剧本写「焦急」「震惊转羞涩」「戏谑」，而 `edge_tts` 的韵律表键是英文 `angry/happy` → 查不到 → rate/pitch 恒 `+0%` | 新增 `utils/emotion.py`：中文自由文本→归一档→(语速,音调)乘子；edge 换算成 `+12%/+2Hz`，cosyvoice 换算成 SDK 真实存在的 `speech_rate`/`pitch_rate`（`inspect` 核实过签名里**没有** `emotion` 参数，此前那条透传分支永远不会命中） |
| 4 | 成片对白听的是**模型回传的音轨**，不是我们的母带 | `ffprobe talk/shot_s3_shot_1.mp4` → `aac 44100Hz stereo 101kbps`，而驱动它的是 `shot_s3_shot_1_mix_16k.wav`（16k 单声道）；且**片段 4.30s < 音频 4.51s**（句尾被吃 0.21s）。`m7` 旧逻辑 `if not from_talk` 才挂我们的音轨 | m6 用 `_record_driver_audio()` 落 `audio/shot_{id}_driver.wav`（**真实驱动音频**：单段=那份混音；分段=各段混音零间隔拼接，与拼接出的视频时间轴严格对齐）；m7 对 `from_talk` 也换轨，音频更长时 `mux_audio` 克隆末帧补足而不是截句 |

### 实测过的音色（不是抄文档）

同一把 KEY、同一句台词逐个合成：`cosyvoice-v2 + longwan` → **418 InvalidParameter**；
`cosyvoice-v2 + longxiaochun_v2 / longshu_v2 / longcheng_v2 / longwan_v2 / longxiaoxia_v2 /
loongbella_v2 / longshuo_v2 / longlaotie_v2 / longmiao_v2 / longjielidou_v2 / longxian_v2 /
longling_v2` → 全部 ✔；`cosyvoice-v1 + longwan` → ✔；`cosyvoice-v3` → AccessDenied（未开通）。
所以自动池只收**性别确定**的 8 个 v2 音色（4 女 4 男），童声/老年音色不进自动池，
只能在剧本里显式写 `voice_id`。

### 口径变化

- 三档配音统一 `dashscope_cosyvoice` 主、`edge_tts` 兜底（economy 不再是"零成本 TTS"，
  但 ¥2/万字符 → 30s 短片全部对白约 **¥0.02-0.05**，相对口型/运镜开销可忽略）。
- 台账按**字符数**计（`quantity=len(text)`）：此前每句恒记 `quantity=1` → 配音成本被记成
  ¥0.0002/句，比真实值低约两个数量级（台账失真的老毛病）。
- `AudioLine`/`Character` 新增字段都是带默认值的可选项，老 `script.json` 照常反序列化续跑。

### 复现与验证

- `fastvideo3 doctor --probe`：一条命令分别探「百炼口型通道(¥0)」「DASHSCOPE 文本」「FAL」；
  配音是否被降级改看 `degradations.json` 的 `tts_provider_failed`。
- 回归：新增 4 条音频用例（性别分桶/版本配套/中文情绪/降级留痕+按字计费）与
  2 条音轨用例（m6 落 driver、m7 换轨后片段必须装得下整条对白）。

## 字幕修复（2026-10-05）：两层字幕 → 单层、单一字体、零 AI 参与

用户反馈成片"出现了两层字幕"。取证（`output/run_20261005_130208`，抽帧 + 读 `final.srt`）：

```
1  00:00:08,320 --> 00:00:10,240  色即是空...哎？
2  00:00:09,519 --> 00:00:14,032  师傅！观音菩萨还在天上看着呢！   ← 与上一条重叠 0.72s
```

**不是模型画了字幕**（逐帧检查过口型片段：画面里没有文字），而是我们自己的 SRT 时间轴
重叠：口型模型回传的片段**比它的驱动音频短**（s2v 片段 1.70s vs 音频 1.92s、4.30s vs
4.51s），旧 `build_srt` 又把"字幕时长 = 音频时长、下一条起点 = 下一镜头起点 + 偏移"直接
拼接，上一条于是溢出进下一个镜头的时间窗 —— 两条同时挂在画面上。

### 改了什么

| 项 | 改动 |
|---|---|
| 时间轴 | `build_srt` 重写：每条字幕只在自己的镜头窗口 `[t+0.3, t+dur-0.15]` 内排布；窗口不够时**后一条被推迟**到前一条结束之后（重叠是观感事故，轻微滞后不是）；同镜头内相邻句再留 0.12s 间隔。实测同一批数据：修复前重叠 0.72s/0.74s，修复后间隔 0.45s/0.12s |
| 音轨（根因之一） | m7 对口型片段换回我们的 TTS 母带（见上一节），片段被补足到音频长度 → 字幕窗口不再被"短片段"挤破 |
| 字体 | 全片唯一中文字体：`utils/image.cjk_font()` 返回 (libass 家族名, 字体文件)，**字幕与标题卡共用同一行配置**。旧配置是两份各写各的（标题卡 msyh.ttc 微软雅黑、字幕写死 SimHei 黑体）。家族名必须不带空格（带空格的在 libass 里静默渲染失败） |
| 样式唯一来源 | `m7.subtitle_style()`：终审母版与母版失败的回退链共用同一份 force_style，杜绝"两条路径各写一份 → 一深一浅像两层" |
| 禁止 AI 画字 | `core/prompts.NO_ONSCREEN_TEXT` 钉进运动提示词与口型提示词；`default.yaml` 的 negative_prompt 从空串改为含「字幕/文字/水印/text/caption」（此前 economy 档对文字**零防线**） |
| 零成本重看效果 | 新脚本 `scripts/recompose_final.py`：只重跑 m7（纯本地 ffmpeg），改字幕/音轨后**不必重花口型与运镜的钱**；覆盖前自动备份旧成片 |

### 验证

- 新增 4 条用例：音频溢出镜头时字幕不重叠、同镜头两句不重叠、字幕字体与标题卡同源、
  三条生成路径都禁止画面文字。
- 对旧 run 实跑 `recompose_final.py --write`：新 `final.srt` 三条字幕互不重叠
  （8.320→9.949 / 10.069→14.119 / 14.239→20.250），抽帧确认 9.0s 与 10.2s 处画面
  各只有一层字幕；旧成片备份为 `final_prev_195552.mp4`。
- 顺带修掉一个潜伏 bug：`concat_demux` 把**相对路径**写进 concat 列表，而列表按自身
  目录解析 → 从仓库根用相对 run 目录重合成时拼出双重路径报 "Impossible to open"；
  现在列表一律写绝对路径。

---

## 双层字幕第二次（2026-10-07）：旁挂 srt 与成片同基名，播放器自动再挂一层

**结论：片内字幕只有一层；多出来的一层是播放器自动挂载的旁挂 `final.srt`。**

用户报告 `output/run_20261007_130730/final` 的成片"双层字幕"。取证：

- 抽帧 2.5s / 5.0s / 8.0s / 13.0s / 4.60s / 9.35s / 16.5s：任意时刻画面里**至多一条**字幕，
  且与 `final.srt` 的时间轴逐条对齐（1.76-4.56 / 4.68-6.849 / …），SRT 自身零重叠；
- `final/_norm/merged_all.mp4`（烧录前的合并母带）同位置抽帧**没有任何字幕** → 烧录只发生一次；
- `final_master` 只 `-map 0:v` + 音频，**没有内嵌字幕流** → 容器里也没有第二层；
- 事件流只有 `final_master_ok`，没有 `final_master_fallback` → 不存在"烧两次"。

于是只剩一个来源：`final/final.srt` 与 `final/final.mp4` **同基名**，而 VLC / PotPlayer /
MPC-HC 的默认行为就是自动挂载同名 srt。烧录层 + 旁挂层叠在一起 = 用户看到的两层。
这与 2026-10-05 那次（SRT 时间轴重叠）是**完全不同的机制**，所以那次的所有回归用例都绿着。

### 改了什么

| 项 | 改动 |
|---|---|
| 旁挂文件名 | `final/final.srt` → `final/subtitles.srt`（`m7.SUBTITLE_SIDECAR_NAME`），与成片不同基名，播放器不再自动挂载；想单独要字幕轨仍可从看板「下载字幕」拿到 |
| 读取兼容 | `m7.subtitle_sidecar_path()`：新名优先、旧 run 回退 `final.srt`；看板 `board.subtitle` 与 `scripts/recompose_final.py` 都走它 |
| 回归锁 | `test_m7_edit` 断言 `subtitles.srt` 存在**且** `final.srt` 不存在（同名即复发） |

### 验证

- `tests/test_m7_edit.py` / `test_board_api.py` 更新后全绿；全量 `pytest -q` 332 passed。
- 对 `run_20261007_130730` 抽帧确认片内单层（见上），问题定位不依赖猜测。

## 分镜人物溯源核查（2026-10-08）：声明在场的镜头锁脸成立，漏标在场的镜头人物是自由发挥

**结论：机制成立，失效面是剧本的"在场"（`characters`）漏标。**

用户问"分镜图中的人物，是否来源于角色图中对应人物"。对 `run_20261007_130730`
（cinema 档，`use_reference_keyframes=true`）四重取证：

- **事件**：4 个带角色镜头的 `keyframe_gen` 均 `ref_count=1` 且 `ref_ids` 与 `characters` 对应；
- **血缘**：`storyboard/shot_*.png` 的 `inputs` 里确有 `assets/char_*.png`；
- **度量**：`consistency_report.json` `ref_attached_ratio=1.0`、4 镜 `face=true`
  （0.75 / 0.98 / 0.60 / 0.95，judge 自校准通过）；
- **目视逐帧**：猪八戒（猪耳/发髻/蓝色敞怀袍）、孙悟空（金箍/红金甲/金箍棒）、
  沙僧（络腮胡/灰袍/大佛珠）的首帧与各自定妆照**同人**；qwen-image-3.0（免费额度）与
  wan2.7-image（兜底）两种模型都真消费了参考图。

反例同样确凿：开场镜头 `characters=[]`，可 action 写着"女王在台阶上回眸一笑"、台词说话人
就是女王；收尾全景 `characters=[]` 而 action 点名四人。m3 只按 `characters` 取参考图，
于是这些镜头**没有任何人物参考图** —— 画出的女王（橙色曳地礼服+凤冠）与定妆照
（米色短裙+玉坠）根本不是同一形象，四人全景里每个人也都漂了。

### 改了什么

| 项 | 改动 |
|---|---|
| 有效在场 | `schema.effective_cast()`：`characters` + 台词说话人（结构化，精确）+ action/camera 点名（别名只在**全片唯一**时认，如"八戒"←"猪八戒"；两角色共享同一别名则整个放弃）。声明的永远排在前面，漏标的不会挤掉已锁脸的槽位 |
| m3 按有效在场装配 | 参考图、外貌文本、蒸馏校验词表、种子分组、血缘上游全部改用有效在场：漏标的人也拿到定妆照槽位，换掉他的定妆照会作废消费过它的首帧 |
| 图与名字成对 | `_char_refs` 返回 `(图片, character_id)` 两个等长列表。旧实现按 `characters[:len(refs)]` 反推"图1是谁"：某角色资产缺失被跳过后，refs[0] 是第二个人的立绘，锁定语料与血缘却标成第一个人 |
| 抽查重画用本镜头的参考图 | 旧实现复用生成循环结束后的循环变量 → 被驳回镜头的重画拿的是"最后一个镜头"的参考图与种子（把别人的脸塞进来）。现在生成时把每镜头上下文存进 `gen_ctx` |
| 抽查与锁脸开关解耦 | 抽查的对照对象 = 有效在场里第一个有立绘的人，**不看** `ref_cids`：锁脸关着的档（economy）抽查照做。中途一度把 anchor 取自 `ref_cids` 导致抽查静默消失，`test_m2_m3` 三个用例当场抓出 |
| 漏标可见 | 新事件 `cast_implied`：`declared / implied / anchored / unanchored`（槽位不够时谁没锚上也说清） |

### 验证与未做

- 回归锁：`tests/test_implied_cast.py` 8 例（说话人补在场、别名唯一性、参考图=定妆照、
  资产缺失时图名不错配、多图图序=语料名字顺序、抽查重画用本镜头参考图、场景图不与人物同框）。
  全量 `pytest -q` 343 passed。
- **未做（有意）**：`ref_max_chars=1` 未放开 → 双人/多人镜头第二个人仍只有文字外貌可依，
  放开需 `scripts/probe_multi_ref.py` 探单通过（账号欠费中）；m4 的 r2v 参考与 m9 的度量仍按
  **声明**的 `characters` 取 —— i2v 从首帧继承脸、风险有限，而让推断出的在场进 m9 闸门会
  触发付费重拍，先把"补在场"做对、把闸门口径留给人工确认。

## 口型通道四次定档（2026-10-11）：分档位两条链 —— A 链 `wan2.2-s2v` / B 链 `运动片段 + videoretalk`

### 起因：用户报"卡在生成镜头视频，进不到拼接成片"

`run_20261011_164152`（cinema，`talk_mode=v2v_overlay`）盘上事实，四条全部指向同一个根因：

- `events.jsonl`：`s2v_capability_mismatch`（主通道 `wan2.2-s2v` 只吃「图片+音频」，拒收 m4 的
  运动片段）→ 换兜底 → `fal.ai HTTP 403「User is locked. Reason: Exhausted balance」`
  → `failed_fallback` ×4 → `m6_talk done clips=0, fallbacks=4`；
- `cost.json`：**talk 计费 0 条** —— 用户以为在跑的 `videoretalk` **从未被调用过**，
  因为三档的 `providers.talk.default` 一直写着 `dashscope_wan22_s2v`；
- `compute_progress`：`videos = gap（missing = 4 个对白镜头）`、`compose = pending`。
  **这不是 bug 而是判据**：`talk_mode != audio_overlay` 时对白镜头的"完成"= `talk/shot_x.mp4`
  在盘（`progress.py:198-214`），而第一个 gap 之后的步骤一律 `pending`（`progress.py:241-244`）
  —— 这条规则正是 2026-10-04 为消灭"全片无口型却标 done"而立的（见上文"底层逻辑修正"）；
- i2v 侧确实在用 `wan3.0-video`（14 条），但 `pricing.yaml` 把它记 **¥0（免费额度期）**
  → 看板"累计成本 ¥30"里看不见这份运动费。

### 为什么定成"两条链"而不是继续押一条

| | A 链 `wan2.2-s2v`（economy/high） | B 链 `videoretalk`（cinema） |
|---|---|---|
| 输入 | 首帧图 + 音频 | m4 运动片段 + 音频 |
| 身份 | 从一张静帧**重发明整个镜头**（要 `_identity_drift_check` 盯着） | **只重绘嘴部**，画面不重画 → m3 参考图锁脸一路继承 |
| 分辨率 | 上限 720P → 1080P 成片的每个对白镜头都吃 `resolution_clamped`，再由 m7 cover 上去 | 官方规格「输出分辨率与帧率=输入」→ 1080×1920 原样保住 |
| 长对白 | 音频 <20s → 按句分段生成再拼（拼接点嘴型跳变风险） | 120s → 不需要分段 |
| 钱 | ¥0.5–0.9/秒，但对白镜头**不过 i2v**（省一份运动费） | ¥0.08/秒 + 1800 秒免费额度，但多一个 i2v 步骤 |
| 门槛 | 三档通用 | **要求运动片段边长 ≥640** → economy 的 480P（480×854）会被服务端拒，所以 B 链只给 high 以上 |

**`wan3.0-video` 不能当口型通道**（这条被反复误解， hence 记录）：官方 media 枚举是
`text/first_frame/last_frame/reference_image/reference_video/reference_audio/file/link`，
**没有 `driving_audio`**；`reference_audio` 的定位是"参考风格/节奏，不支持音频驱动生成"。
项目里存着真实报错原文（`tests/test_talk_fallback.py`）。它的"原生音画同步"是模型**自己**生成
配音/音效，与我们"用 m5 的 TTS 母带驱动嘴型"是反方向的需求。B 链因此是
`wan3.0-video 出运动 + videoretalk 对口型`**两段**，而不是一段。

### 改了什么

| 层 | 改动 |
|---|---|
| 定档 | `configs/profile_cinema.yaml`：`talk.default → dashscope_videoretalk` + `talk_mode → v2v_overlay`（**成对**，两行一起改）；economy/high 保持 A 链不动 |
| 底座说明 | `configs/default.yaml` 的口型段与 `talk_mode` 段重写为"两条链 + 配对判据 + 三条不成对代价" |
| 环境变量陷阱 | `.env` / `.env.example` 的 `FV3_TALK_MODE` **改为留空**（原先写死 `s2v_direct`，env > yaml 会静默盖掉档位差异 —— 本次故障的直接成因之一） |
| 前端 | `NewRunForm.vue`：新增 `TIER_TALK_PAIR` 配对表，换档自动换模式（表单默认曾是 `s2v_direct` + `cinema` = **一打开就不成对**）；不成对时红字当场说破"会先烧 i2v 再没口型"；`web/dist` 已重新构建 |
| 模块文档 | `m6_talk.py` 文档头此前仍写"二次定档：主 = fal InfiniteTalk"、"v2v_overlay（默认）"，与配置矛盾 → 改为四次定档口径 |
| 回归 | `tests/test_model_config.py`、`tests/test_fal_talk.py` 的 4 处"三档都是 s2v_direct"断言改为**档位感知**，并新增 cinema 侧装配断言（`DashScopeRetalkProvider` / `supports_video_input=True` / 与 `v2v_overlay` 成对）；用例改吃 `dotenv_path=none.env`，不再被开发者 .env 影响 |

### 未做（下一步）

1. **首尾帧（`last_frame`）尚未接入**：官方支持且约束已核实（提示词**必填**、首帧边长
   [240,8000]、尾帧 ≤20MB、**首尾帧宽高比必须一致**、duration 2–30 或 -1、输出 24/30fps）。
   `_submit_media` 本就按 `(type, url)` 拼装，缺的是"尾帧从哪来"：计划由 m3 每镜头额外生成
   一张「动作结束态」图（约 +¥0.25/镜）并接血缘/`is_stale`。注意首尾帧与 `reference_image`
   **互斥** → cinema 走 `wan2.7-r2v` 的角色动作镜头拿不到这项收益。
2. **B 链的嘴型质量仍需人眼定案**：`videoretalk` 在 2026-10-04 的单帧 A/B 里"确实在对口型"
   但被判"幅度小、同步感平"。本次换了素材来源（1080P、起止构图受控、脸更大），
   **有可能翻案** —— 请用 `python scripts/probe_lipsync.py sheet <run> --shot <id>` 看接片。
   自动度量仍只作记录（假阴性已被实测证伪，见"口型度量证伪与重建"）。
3. **fal 兜底等于没有**：账户 403 余额耗尽被锁。要么充值，要么把 `fallback` 摘掉，
   否则主通道每次失败都多一次无谓 403 往返。
4. **`pricing.yaml` 里 `wan3.0-video: 0.0`**：额度耗尽后真实账单会记进台账但**预估与熔断
   看不见**（`price_of` 命中 0 价键）。B 链把运动费纳入了对白镜头的成本，这个失真必须补。

## 口型五次定档（2026-10-11）：模式的名字就是通道 —— 三条链 + 一个音轨开关

用户实测结论（本轮前提）：`wan2.2-s2v` **可用**，优点是输入语音、保证语音一致，缺点是只能单人
同屏、不出动作视频；`wan3.0-video 出动作 → videoretalk 加口型`**可用**，缺点是链路太长太慢；
`wan3.0-video 直接由图片+提示词出口型+动作`**待验证**。本轮把这三条做成 UI 上的三个选项，
并实现 C 链。

### 结构改动：`core/talkmode.py` 取代"两处各写一份"

`talk_mode` 与 `providers.talk` 曾经是两份事实（一份决定语义、一份决定装配），漂一次就是
"钱花了但成片没口型"。现在映射表只有一份，且**在唯一装配口 `factory._talk_route` 里生效**：

| 模式（UI 名） | 通道 provider | 输入形态 | 对白镜头是否过 m4 | 成片音轨 |
|---|---|---|---|---|
| `s2v_direct`（wan2.2-s2v） | `dashscope_wan22_s2v` | 图片+音频 | 否 | 我们的母带 |
| `v2v_overlay`（videoretalk） | `dashscope_videoretalk` | 片段+音频 | **是** | 我们的母带 |
| `scripted_talk`（wan3.0-video） | `dashscope_wan30_talk`（新） | 图片+提示词 | 否 | **由 `voice_source` 决定** |
| `audio_overlay`（不做口型） | 无 | — | 是 | 我们的母带 |

yaml 路由降级为"形态相容时才生效的供应商覆盖"；`.env` 的 `FV3_TALK_MODEL` 仍是最高优先级的
专家级覆盖（通道自己会响亮拒绝错误输入，不是静默降级）。前端选项、单价、档位冲突全部问后端
（`GET /api/talk-modes`；`/api/cost-estimate` 新增 `talk_channel/talk_eats/tier_warning`）。

### C 链的官方规格（照抄进 `DashScopeWan30TalkProvider` 类文档，不靠记忆）

media type 枚举 `first_frame/last_frame/reference_image/reference_video/reference_audio/file/link`
——**没有 `driving_audio`**（真实 400 原文存在 `tests/test_talk_fallback.py`；这也是"wan3.0 能不能
直接对口型"曾经被误判的根源）。口型来自**提示词里的台词**，指南原文："提示词重点描述动态过程、
剧情、**台词**与运镜"，写法「角色 + 说话动词 + 冒号 + 引号内容」。
`resolution` 到 **1080P**、`ratio` 可显式 `9:16`、`duration` [2,30] 或 `-1` 智能时长、
`audio` 默认 true（"开关声音价格相同"）、`prompt_extend` 默认 true（我们显式传 false，否则服务端
改写会吞字改字）、**没有 `negative_prompt` 参数** → 负向必须写进提示词的「负向清单」段，当参数发
会被静默丢弃。

### 人物一致性与语音一致性（用户点名的两项，怎么落地）

- **人物一致性**：C 链的输入仍是 m3 的**参考图锁脸首帧**（`use_reference_keyframes` 不变），
  `_scripted_prompt` 再把 `Character.appearance` **逐字**写进提示词（与 m3/m4 同一来源）作为文本侧
  锚，并声明"全程保持与首帧相同的人物长相、服装配色与场景"。`seed` **不作为**一致性手段 ——
  官方明写"即使使用相同 seed，也不能保证每次生成结果完全一致"。产物照旧过
  `_identity_drift_check`（服装主色色相距离）与 m9 度量。
- **语音一致性**：这是 C 链唯一的真取舍，所以做成开关而不是替用户赌。
  `voice_source=master`（**当时默认，2026-10-12 已翻案为 `model`，见下一节**）：m6 落 `audio/shot_{id}_driver.wav` → m7 把成片音轨换回 m5 按
  性别分桶的 cosyvoice 母带 → 全片音色统一（"唐僧拿女声"那条红线不破），代价是嘴型对着模型自己
  的音长、贴母带后可能有零点几秒节奏偏差。
  `voice_source=model`：什么都不写（m7 的 `if driver.exists()` 自然放行）→ 嘴型与声音严格同轨，
  代价是同一角色跨镜头（每镜一次独立请求）可能换嗓子。**探单时两条各出一条对听再定默认**。
- **音字一致**：台词逐字取剧本 `DialogueLine.text`（与 m7 字幕同一来源），说话动词按
  `utils/emotion` 归一后的情绪档**确定性**映射（不搞随机修饰 —— 随机会让同一句每次表演都不同）。
- **多人同屏**：C 链**不走 `two_person_talk=overlay` 的豁免** —— 台词逐条点名后模型自己分配嘴型，
  这是三条链里唯一能做多人各自对口型的（A/B 受"单音轨驱动整幅画面"限制）。

### 改动清单

`core/talkmode.py`（新，唯一映射表）· `providers/dashscope.py`（`DashScopeWan30TalkProvider`：
`first_frame`/`last_frame`、图片硬规格检查、对 audio/video 的响亮拒绝）·
`providers/protocols.py`（新协议 `wan30_scripted_talk`，并**纠正**了原先
`wan3.0-video*→i2v_driving_audio` 这条错误映射）· `providers/factory.py`（`_talk_route`
模式解析 + 注册新通道）· `modules/m6_talk.py`（C 链分支 + `_scripted_prompt` /
`_scripted_duration` / `_apply_voice_source`；`_s2v_one` 的 `wav` 改为可选，让三条链共用同一条
重试/止损/兜底/计价实现）· `modules/m4_shots.py`（是否为对白镜头出运动片段改问映射表）·
`core/cost.py` + `pricing.yaml`（预估按**模式解析出的通道**取价，新增 `dashscope_wan30_talk` 映射）·
`core/schema.py` / `core/config.py` / `app/api.py` / `app/cli.py`（新枚举值 + `voice_source` 贯穿 +
`GET /api/talk-modes` + CLI 预飞打印通道与档位冲突）· `web`（选项由后端映射表生成、C 链音轨开关与
取舍说明、档位冲突红字；`dist` 已重建）· `tests/test_wan30_talk.py`（新增 17 例）·
`tests/test_model_config.py` / `tests/test_fal_talk.py`（把"三档都是 s2v"的策略断言改为模式感知，
并钉住"形态不相容的路由与兜底会被忽略"）。

回归：`360 passed`（原 343 + 新 17）· ruff · mypy · `npm run build`。

### 未做（下一步）

1. **C 链尚未真跑过一条**：需要一次探单（免费额度期内 ≈¥0），验三件事 —— 嘴型是否与它自己发的
   音同步、台词是否**逐字**没被吞改、音色性别对不对。自动度量只作记录（假阴性已被实测证伪），
   定案仍用 `scripts/probe_lipsync.py sheet`；`voice_source=model` 时对照音轨由 m6 从产物里抽。
2. **尾帧（`last_frame`）通道已支持、上游还没接**：m3 目前每镜头只产首帧。接上"动作结束态"尾帧
   即可锁定起止（官方约束：提示词必填、首尾帧宽高比必须一致、尾帧 ≤20MB）。
3. **首帧/首尾帧 与 `reference_audio` 互斥**是官方硬规定 → C 链拿不到音色克隆。想要"锁脸 +
   钉音色"只能改用参考生视频模式（`reference_image` + `reference_audio`），但那会失去"以这张图
   作第一帧"的起止锁定，属于另一条链路，需要单独定档。
4. **C 链没有兜底通道**（fal 要音频、videoretalk/kling 要视频，都不吃"纯提示词自发声"）→
   主通道失败即降级为无口型叠音，看板红幅会说出来。

## 模型选择进前端（2026-10-11）：面板点名 → 落 run_meta + 写回 .env

### 起因

`.env` 的 `FV3_TEXT_MODEL/FV3_T2I_MODEL/FV3_I2V_MODEL` 是**全局**开关，而且优先级高于
`configs/*.yaml`。本轮排查撞到的实况：三份档位表写着 wanx2.1-t2i-turbo / wan2.2-i2v-plus /
wan2.7-i2v，实际跑出来的是 qwen-image-3.0(-pro) 与 wan3.0-video —— 看 10-06、10-07 两条 run 的
`cost.json` 就能对上。用户提的要求很明确：**多数用户只用 Web 面板，不会动 `.env`**，所以面板
选的模型必须优先级最高，并且**写回 `.env`** 成为之后的默认（否则下一个任务又读回旧值）。

### 两道落地（缺一道都会"看着没生效"）

1. **落 `run_meta.json`**：`ProjectConfig.model_overrides`（能力→模型名，空＝跟随档位），
   恢复入口只有一个 —— `config.apply_run_overrides(cfg, meta)`，CLI 的 `_cfg_from_meta`
   与 API 的 `_cfg_for_run` 都调它（两条路径各写一遍就会出现"网页和命令行用不同模型"）。
   优先级：**本 run 点名 > `.env` > yaml**，所以续跑永不受后续 `.env` 变化影响。
2. **写回 `.env`**：`core/envfile.py`，非 Fake 建任务时执行（`fastvideo3 new --*-model` 同样写，
   两条路径不能各有各的默认）。

| 能力 | 可选项（目录见 `core/model_catalog.py`） | 对应环境变量 |
|---|---|---|
| text | qwen3.7-flash（**新出厂默认**）、qwen3.7-flash-2026-07-15、qwen3.7-plus、qwen3.7-plus-2026-05-26、qwen3.7-max、qwen3.7-max-2026-06-08、qwen-plus、qwen3.8-flash、qwen3.8-max、qwen3.8-max-0902、qwen3.8-27b | `FV3_TEXT_MODEL` |
| t2i | qwen-image-2.0(-pro)、qwen-image-3.0(-pro)、wan2.7-image(-pro) | `FV3_T2I_MODEL` |
| i2v | wan3.0-video、wan3.0-video-prime、wan2.7-i2v | `FV3_I2V_MODEL` |

**口型不在这里选**：通道由「口型模式」决定（`core/talkmode.py`，模式即通道）。让面板写
`FV3_TALK_MODEL` 会把模式与通道重新拆成两份事实 —— 正是本项目反复出问题的形状。

### 2026-10-17 补的三处缺口（"面板选了 A、实际跑 B"的另一种形状）

上面那套优先级本身是对的，但**显示侧各自另算了一遍**，于是显示与装配分家：

1. `api.model_choices` / `cli doctor` / `model_catalog.configured_models` 三处都硬写
   `cfg.model_for("t2i", "dashscope_t2i", …)`，而 t2i 路由早已是 `dashscope_wan27_image`
   （它读 `dashscope_t2i27` 那个键）。实测三档面板"默认"都显示 `wanx2.1-t2i-turbo/plus`
   —— 那两个只是**兜底路由**的模型，这次 run 根本不会调用它们。现在唯一来源是
   `providers.factory.model_in_effect`（与 `mm()` 同一条判据，且**不需要 API KEY**，
   没配 KEY 也显示得出来），并由 `tests/test_text2image_choice_is_one_knob.py` 逐档、
   逐个可选项与真实装配比对。
2. 档位文件不再各写一份文生图（`profile_720P/1080P` 删掉 `providers.t2i` 与
   `dashscope_t2i*`）：三档共用 `default.yaml` 那一份，差别只可能来自面板点名。
   这条有结构判据守着（直接读 yaml 正文），因为留一份在档位里，下次有人只改那一档，
   三档一致性就又变成看运气。
3. `stage --run-id` 以前只吃命令行参数 → 同一个 run 网页用 A 模型、命令行用 B 模型。
   现在复用已有 run 时按 `run_meta` 恢复（分辨率档与 fake 也一样，不一致会打印说明）。

顺带：文生图可选项**只列支持参考图的那一族**（wan2.7-image / qwen-image 系）。三档出厂都
要求「参考图锁脸」，把收不了参考图的 `wanx2.1-t2i-*` 留在下拉里就是"选得出来、后端必然
拒绝"的自相矛盾界面；它们仍会作为 `.env`/configs 的生效值显示出来，并标出"不支持参考图
→ 锁脸会失效"（`choices[].locks_face` 一位来自 `model_catalog.locks_face_capable`，
判据是协议而不是模型名清单，JS 里没有第二份）。

### `.env` 写入的安全边界（为什么单开一个模块）

`.env` 里躺着 API 密钥，所以：
- **只改这几行的值**，其余每一行（密钥、注释、空行、排版）原样保留；行尾注释连前面的空白一起
  保留；文件里没有这个变量时追加一行；**注释掉的行不替用户打开**；
- **按字节读写**（`read_text()` 会把 CRLF 归一化成 LF —— 那会让"只改一行"变成全文件 diff，
  而这个文件的 diff 恰恰最需要人眼审）；
- 临时文件 + `os.replace` 原子落盘，失败清理，不留半截；
- 返回值只含**变量名**，绝不含值或原文（不写日志、不进 HTTP 响应）；
- **写失败不阻断建任务**：run_meta 里已有精确选择，`model_env_error` 落进 meta、详情页红条提醒。

### 校验（在建任务时，不在提交 API 时）

`model_catalog.validate/protocol_error/ref_image_warning` 三道：能力是否允许、名字是否在目录内、
**t2i/i2v 的名字能否解析出协议**（provider 键同时决定端点与请求体，解析不出来就是打错端点、
必然 400 —— 这是 `.env` 换模型时代偿过的坑）。非法就 400/`BadParameter` 并列出可选清单。
另外 high/cinema 强制「参考图锁脸」，选了不支持参考图的 t2i 会在 m3 花钱前报错 → 这里先说。

### 改动清单与验证

新增 `core/model_catalog.py`、`core/envfile.py`、`tests/test_model_catalog.py`、
`tests/test_envfile.py`；`schema.ProjectConfig`（4 个新字段：model_overrides /
model_env_written / model_env_error + 已有 voice_source）、`config.apply_run_overrides`、
`api`（`GET /api/model-choices`、create_run 校验与写回）、`cli new`（`--text-model` /
`--t2i-model` / `--i2v-model`）、前端 `NewRunForm.vue`（高级折叠区，选项与"当前生效值"全问后端）、
`RunDetail.vue`（模型溯源与写回失败红条）、`configs/default.yaml`（text 出厂默认改 flash，
与 `.env` 一致，免得换台机器默认又变）、`.env.example`。

验证：`393 passed` · ruff · mypy 全绿 · `npm run build`；测试一律指向 tmp 的假 `.env`，
真实 `.env` 未被改动（已核对）。

### 未做

1. **面板选择的模型不在 `pricing.yaml` 表内时**，台账会回落到"能力默认价"（面板已用
   ⚠ 标出 `priced=false`，如 qwen-image-2.0、qwen3.7-max 等），需要回填真实单价。
2. **文本模型不做协议校验**（`dashscope_qwen` 走 OpenAI 兼容端点，名字只是路由参数）：
   选了账号未开通的名字会在调用时 `AccountGateError` 并回落 deepseek，面板无法预先知道。
3. 写回是"最后一个任务赢"：多标签页同时建任务会互相覆盖 `.env`（单机自用场景可接受），
   真要并发应当把默认值挪进数据库。

## C 链首条成片复盘（2026-10-12）：语音领先口型 0～2.2 秒 → 默认音轨改为模型原声

`run_20261011_231024`（cinema + `scripted_talk`）用户看片反馈：**口型与语音不一致，语音早于口型；
但口型与字幕一致**。三句话就把病灶锁死了：字幕与语音同源（都来自我们的 TTS 时间轴），
而嘴是模型对着**它自己**的发声画的 —— 说明成片音轨被换过，而口型没跟着换。

### 逐镜测量（嘴部运动能量 × 音频包络，取相关最大时的平移量）

| 镜头 | clip | 我们的母带 | 语音领先口型（换母带后） |
|---|---|---|---|
| shot_1_2 | 4.04s | 2.53s | +0.83s |
| shot_2_1 | 3.02s | 1.65s | +2.17s |
| shot_2_2 | 3.02s | 1.65s | +1.50s |
| shot_3_1 | 4.04s | 2.19s | +0.00s |
| shot_3_2 | 4.04s | 2.72s | +0.00s |

两个结论，都不容忍"再等等看"：

1. 偏差最大 2.2 秒，观众一定看得出来；
2. **偏移不是常数**（0 / 0.83 / 1.5 / 2.17）→ "给母带加一个固定 delay"这条路**修不好**。
   而想按镜头测出偏移再各自对齐也不可靠：这把尺子在人眼确认过同步的素材上本就会假阴性
   （见"口型度量证伪与重建"），拿它做自动对齐等于把猜测写进交付链路。

根因是**结构性的**：C 链的口型与声音都是模型生成的，二者同源；把音轨换成 TTS 母带就必然
分家。`_scripted_duration` 里那 1 秒"呼吸余量"是放大器（clip 4.04s vs 母带 2.53s，模型可以把
台词安排在窗口内的任意位置）。

### 改了什么

| 处 | 改动 |
|---|---|
| 默认值 | `DEFAULT_VOICE_SOURCE`：`master` → **`model`**（测量表就写在它上方的注释里）。同时 `AppConfig` 字段、`schema.ProjectConfig`、`api.RunCreate`、`configs/default.yaml`、`profile_cinema.yaml`、前端表单初始值一起翻，避免"五处默认值各说各话"这个老毛病 |
| 事件与提示 | `master` 分支的 `talk_audio_source_master` 事件写明实测代价；`tier_conflict(scripted_talk, …, voice_source=master)` 直接说"嘴不上，要音色统一请改用 videoretalk"，前端建任务时就看得见 |
| 字幕时间轴 | `m7.build_srt` 新增 `model_audio_shots`：模型原声的镜头**不再按 TTS 时长排字幕**（那是另一条音轨的节奏），改为把镜头窗口按**台词字数比例**分配 —— 否则修好声画又留下"字幕跟着旧节奏"的第二层错位。判据用 `cfg.talk_mode/cfg.voice_source`，不用"driver.wav 不存在"猜（A/B 链偶发混音失败时也会没文件，那时音轨仍是母带） |
| 音色缓解 | `_timbre_anchor`：按 `Character.gender/age_band` 给每个角色一句**固定**音色描述写进提示词（确定性，不随机修饰）。是缓解不是保证 —— 官方 `first_frame/last_frame` 与 `reference_audio` 互斥，锁脸就不能克隆音色 |
| 零成本补救 | `scripts/recompose_final.py`：C 链 + model 时**跳过 driver.wav 补建**（它原来会为任何缺 driver 的口型镜头补一份，正好把补救动作抵消）；同时按 meta 恢复 `voice_source` 与 run 级模型选择 |

### 已验证

在临时副本上按"改 `run_meta.voice_source=model` → 删 5 个 `shot_*_driver.wav` →
`recompose_final.py --write`"跑通：这一次合成**没有** `talk_audio_replaced` 事件（= 保留模型原声），
字幕从"1.0+1.2 秒串排"变成铺满 4.04s 窗口（例：`05,299 --> 08,890`）。**零 API 调用**，
旧成片自动备份为 `final_prev_<时间戳>.mp4`。

回归：`400 passed`（新增 4 例：默认必须是 model、master 口径要在冲突提示里说破、
音色锚确定性、字幕两种时间轴口径不同）· ruff · mypy · `npm run build`。

### 口径定档

- 要「嘴 = 听到的声音」→ C 链（模型原声）。这是 C 链唯一结构正确的配对。
- 要「全片音色统一」**且**「嘴型对着这条音色」→ **B 链 `videoretalk`**：它先出运动片段、
  再对着我们的母带重画嘴，声/嘴/字幕三者同源。代价是链路长、慢。
- C 链 + `master` 只留作对比试验：音色统一了，但嘴必然不上。
