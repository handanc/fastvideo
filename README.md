# FastVideo3 🎬

**全 API 化的短剧视频生成教学平台 —— 按顺序分步执行，每一步之间人都能介入。**

输入一个主题，输出一部有人物对话、有人物动作、有转场动画的完整短剧视频。
所有生成推理走 API KEY（DeepSeek + 阿里云百炼），**无需 GPU**；本地仅保留 CPU 级组件（YOLOv8-seg + rembg 双人同屏、FFmpeg 剪辑）。
生成只有**一条路径**：`剧本 → 角色图 → 场景图 → 分镜图 → 镜头视频 → 成片`，点一次补一步（Web 面板的「开始生成」/ CLI 的 `step`），
每一步都能改剧本、换定妆照、删图重生成，然后再往下走 —— 一致性靠人确认 + 本地硬约束 + 自动度量重做共同保证。

- 需求与设计：[PRD.md](PRD.md) · [TDD.md](TDD.md)
- 前代项目：FastVideo2（本地推理单体版），本项目为其架构现代化重构

## 多模态 Agent 编排（LangGraph）

分步模式之外，同一个任务可以用**执行图**跑：`fastvideo3/graph/`（LangGraph `StateGraph` + SQLite/PostgreSQL 检查点）。

```
START → script → assets（角色图 ‖ 场景图，节点内并发） → scene_videos → compose → final_gate
                                          ↑                                        │
                                          └──────── 人审点名重出（条件回边）────────┘
```

```bash
fastvideo3 graph <run_id>                        # 跑到人审门停下（成片已产出）
fastvideo3 graph <run_id> --resume --redo scene_2  # 不满意：只重出这一组，再交人审
fastvideo3 graph <run_id> --resume --finish      # 收片
```

职责边界（这是本项目的硬约束，不是风格偏好）：图只管**顺序、并行、中断、恢复**；每一步做什么仍然走
`core/stages.py` 与 `cli._run_step` 那**唯一一份**执行器。自动重做 ≤2 轮的实现只在 `compose` 里，图上不重复实现。
所以同一条 run 可以两种编排混用，产物、成本台账、血缘完全一致（`tests/test_langgraph_agent.py` 用差分测试钉住）。

为什么"带口型的场景视频"能算多模态：一次 wan3.0 调用同时产出画面、台词语音与口型，输入是
角色立绘 + 空景背景图 + 音色样本 + 多镜头提示词（官方配额：参考图 ≤10、参考音频 ≤5 段且总 ≤15s，与首尾帧互斥）。

## 快速开始

```bash
# 1) 环境（默认建**项目内 .venv**，不依赖 conda；想沿用 conda 加 --conda）
./scripts/setup.sh          # Windows: .\scripts\setup.ps1
# 它做五件事：.venv → 按 requirements.lock.txt 装依赖 → pip install -e .
#             → provision（ffmpeg 收进 bin/、身份权重进 models/、缺 .env 时从模板复制）
#             → cd web && npm ci && npm run build → doctor

# 2) 零成本演示（Fake 模式，无需任何 KEY）：建任务不花钱，然后点一次补一步
fastvideo3 new --topic "校园喜剧" --duration 30 --fake     # → 打印 run_id
fastvideo3 step <run_id>     # ×6：剧本 → 角色图 → 场景图 → 分镜图 → 镜头视频 → 成片

# 3) 真实生成：编辑 .env 填入 KEY 后（先探活防欠费白跑）
fastvideo3 doctor --probe
fastvideo3 new --topic "职场新人的第一天" --duration 30 --resolution 1080P --budget 150

# 4) 或者直接开 Web 面板（推荐：每步之间能改剧本、换定妆照、删图重生成）
fastvideo3 serve            # 打开 http://127.0.0.1:8000（--port 换端口，--host 0.0.0.0 对外）
```

### 独立上线（这份目录自带运行所需的一切）

`fastvideo3 provision` 把外部依赖收进项目，之后整份目录拷到别的机器就能跑：

| 位置 | 内容 | 从哪来 |
| ---- | ---- | ------ |
| `bin/ffmpeg.exe` | 一条 run 要调几十次的二进制 | 优先从当前 venv 的 `imageio-ffmpeg` **离线复制**，否则走镜像下载；落地后真跑一次 `-version` 才算就位 |
| `models/*.onnx` | 身份一致性客观层（Tier C）的 SFace 权重 | 镜像下载；下不到手工放进 `models/` 也算就位 |
| `.env` | KEY 与模型默认值 | 缺了才从 `.env.example` 复制，**从不覆盖**已有文件，自检也从不打印其内容 |
| `web/dist/` | Vue 面板产物 | 需要 node：`cd web && npm ci && npm run build`（provision 只检查不代跑） |
| `requirements.lock.txt` | 验证过的依赖版本组合 | `provision --lock` 从当前环境生成；`setup` 优先按它装 |

ffmpeg 的解析顺序（`fastvideo3/core/runtime.py`，唯一一份）：`FV3_FFMPEG_PATH` →
**项目内 `bin/`** → venv 的 imageio-ffmpeg → 系统 PATH。项目内置排在系统 PATH **之前**：
部署包带着自己的二进制，行为由项目决定，不由宿主机装了什么决定。
`fastvideo3 doctor` 会把 ffmpeg 的**来源**一起显示出来，没在项目内就附补齐指令。

分辨率与成本（30 秒成片估算；权威口径是 `GET /api/cost-estimate` 与 `GET /api/resolutions`）：
`480P` ≈¥14-20 · `720P` ≈¥28-40 · `1080P` ≈¥52-82。

**分辨率是唯一的规格旋钮**：成片尺寸、一致性深度、抽卡数、单价量级、是否自动重拍都由它带出来
（`configs/profile_<档>.yaml`；480P 就是 `default.yaml` 本体）。以前还有一层
`economy/high/cinema` 的档位名，与分辨率一对一 —— 两个旋钮表达同一件事，就必然出现
"cinema + 480P"这种自相矛盾的组合（`talkmode` 里还专门为它写过特例），所以删掉了；
老 run 的 `run_meta.quality` 仍能被 `config.resolution_of_meta` 读出来，续跑不受影响。

场景片段按**输出秒**计价（wan3.0-video：480P ¥0.3/秒、720P ¥0.6/秒、1080P ¥1.2/秒）。
预估不够只是红幅警告；**实时累计**费用到上限时会停下来弹「是否继续生成」的人工审核窗口
（`core/pause.py`），确认即抬高上限并从停下处接着跑 —— 已出的产物一个不动、不重复花钱。
旧行为是"到线后静默跳过剩余付费提交、步骤照样报完成"，实测后果就是钱花在运动片段上、
口型一条没出而看板毫无提示。教学/演示请用 `--fake`（零成本占位产物）。

**一把 `DASHSCOPE_API_KEY` 跑通全系统**（含对口型：百炼 wan2.2-s2v 数字人）。
`FAL_KEY` 是**可选**的——只用于口型兜底通道 fal.ai InfiniteTalk（s2v 挂掉时换它再试一次），
两个都没有时对白镜头降级为"无口型叠音"并落 `degradations.json`（看板红标可见）。

| 能力     | 模型                                                  | 参考价           |
| -------- | ----------------------------------------------------- | ---------------- |
| 编剧     | `qwen3.7-flash`（出厂默认；可在 Web 面板按任务改选 plus/max/3.8 系，见「模型选择」） | flash 在免费额度期 ¥0 |
| 文生图   | `wanx2.1-t2i-turbo`                                 | ~¥0.18/张       |
| 图生视频 | `wan2.2-i2v-flash`（480P）                          | ¥0.11/秒        |
| 口型 A 链<br>`wan2.2-s2v` | 首帧图 **+ 我们的语音** + 提示词 → 口型视频（`talk_mode=s2v_direct`；对白镜头**不过 I2V**，省一份运动费；语音一致由 m5 音色池保证）。缺点：单人同屏、只到 720P | 480P ¥0.5/秒<br>720P ¥0.9/秒 |
| 口型 B 链<br>`videoretalk` | `wan3.0-video` 先出动作片段 → 再**只重绘嘴部**加口型（`talk_mode=v2v_overlay`；不重画画面 → 身份继承锁脸，**输出分辨率/帧率=输入**，1080P 不降级；语音仍是我们的）。缺点：**链路最长最慢**，且要求片段边长 ≥640 → economy 用不了 | ¥0.08/秒<br>+1800 秒免费额度 |
| 口型 C 链<br>`wan3.0-video` | 首帧图 + **提示词（台词写在里面）** → 一条调用出**口型+动作+音轨**（`talk_mode=scripted_talk`；原生 1080P、显式 9:16、**唯一能做多人同屏各自对口型**的链路）。代价：音色由模型决定（官方锁脸与克隆音色互斥）。成片**默认用模型原声**（`voice_source=model`）—— 换回我们的配音实测会让语音领先口型 0～2.2 秒 | 免费额度期 ¥0 |
| 口型兜底 | `fal-ai/infinitalk`（需 `.env` 的 `FAL_KEY`；同一 provider 按形态自动选端点；仅服务 A/B 两条链） | $0.20/秒@480p 起 |
| 配音     | **百炼 `cosyvoice-v2`**（三档同口径；失败/未开通降级免费 edge-tts 并**留痕**） | ¥2/万字符 ≈ ¥0.02-0.05/片 |
| 视觉评审 | `qwen-vl-plus`                                      | ¥1.5/百万 token |

## 常用命令

| 命令                                                                                 | 说明                                             |
| ------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `fastvideo3 new --topic ... [--fake] [--resolution 1080P]` | **建任务**（只写 run_meta.json，不花钱、不启动生成） |
| `fastvideo3 step <run_id>`                                                          | **开始生成**：点一次只跑最靠前的未完成步骤（剧本→角色图→场景图→分镜图→镜头视频→成片） |
| `fastvideo3 script-regen <run_id> --prompt "..." [--mode revise\|rewrite]`         | 按人的意见重写剧本：`revise` 保留剧情只改被点名部分（外貌卡/必留词/蒸馏提示词按 id 继承，只作废真被改到的镜头）/ `rewrite` 推倒重写 |
| `fastvideo3 stage script/assets/storyboard/shots/audio/talk/edit/metrics`         | 单阶段运行（教学；**不含** cinema 的一致性装配，别拿去出片） |
| `fastvideo3 consistency <run_id>`                                                  | 该 run 的人物一致性报告（表格 + md）             |
| `fastvideo3 compare-consistency A B C`                                             | 多 run 一致性对比矩阵（消融产出物）              |
| `python scripts/probe_fal_infinitalk.py check`                                    | 口型通道零成本体检：打印各档实际路由与单价 + fal 官方规格/价格对照 + 熔断够不够 |
| `fastvideo3 doctor --probe`                                                       | 含 **百炼 s2v 开通探测(¥0)**：用不可达媒体 URL 触发提交期 400，不建任务、不计费 |
| `python scripts/probe_fal_infinitalk.py shot --clip … --wav … [--spend]`          | 真跑一条口型（默认只打印计划与费用，加 `--spend` 才提交计费） |
| `python scripts/probe_lipsync.py sheet <run> --shot shot_1_2`                      | 出带逐帧响度条的嘴部接片，人眼 10 秒定案（免费） |
| `python scripts/recompose_final.py <run> [--write]`                                | **只重跑 m7**（字幕/音轨/调色/BGM，纯本地 ffmpeg、零 API 调用）：改字幕样式或修音轨后不用重花口型/运镜的钱 |
| `python scripts/run_ablation.py [--fake] [--with-shots]`                           | 一致性消融四臂（`configs/ablations/`，详见 TDD §2 讲） |
| `fastvideo3 new --ablation ref_only ...`                                           | 用消融配置建任务（在质量档之后叠加，路径限 configs/ 内） |
| `fastvideo3 doctor [--probe]`                                                      | 环境自检（--probe 真实 KEY 探活：欠费/未开通在跑片烧钱前暴露） |
| `fastvideo3 cost <run_id>` / `fastvideo3 ls-runs`                                | 成本台账 / 历史运行                              |
| `fastvideo3 serve`                                                                 | **Vue Web 面板**（FastAPI + Vue3，见下节） |
| `streamlit run fastvideo3/app/streamlit_app.py`                                    | Streamlit 面板（教学观测用）                     |

## Vue Web 面板

参考 FastVideo 的 "FastAPI 服务 + Web UI" 架构：`fastvideo3/app/api.py` 提供 REST 接口，`web/`（Vite + Vue 3）提供前端，构建产物由 FastAPI 同源托管。

```bash
# 1) 安装后端 Web 依赖 + 构建前端（一次性）
pip install -e ".[web]"
cd web && npm install && npm run build && cd ..

# 2) 启动（打开 http://127.0.0.1:8000）
fastvideo3 serve
```

前端开发模式（热更新）：`fastvideo3 serve` 起后端后，在 `web/` 里 `npm run dev`，访问 http://127.0.0.1:5173（`/api` 自动代理到 8000）。

页面功能（4 个视图，无路由依赖）：

| 视图     | 功能                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------- |
| 新建任务 | 主题/风格/时长/口型模式/画质/画幅/熔断/Fake 开关 + **模型选择（高级）** → 一颗 `▶ 开始生成`（**只建任务，不花钱**），说明一句「按照顺序逐步完成任务」。economy 档有醒目警示"不做人物一致性保证"；成本预估与熔断红幅来自后端 `GET /api/cost-estimate`。**模型选择**：文本 / 文生图 / 图生视频三类可选（清单与校验唯一来源 `core/model_catalog.py`，经 `GET /api/model-choices` 给出，含"当前生效值 + 是谁定的"）；点名的模型**既落进本 run 的 `run_meta.json`**（每步含续跑原样恢复，优先级高于 `.env`）**也写回 `.env`** 成为之后的默认 —— 因为多数用户只用面板不碰 `.env`，不写回就是"面板选了 A、下个任务跑 B"。`.env` 写不回（只读/被占）**不影响本任务**，详情页会红条提醒 |
| 运行历史 | 状态徽章（完成/运行中/失败/停摆）+ 成本摘要，5s 自动刷新                                       |
| 运行详情 | 顶部 **StepBar 步骤条**（剧本→角色图→场景图→分镜图→镜头视频→成片，五态：待生成/生成中/已完成/有缺口/失败），主按钮文案由服务端"下一步"驱动（`▶ 开始生成 · 生成剧本`）+ 已花/熔断。**成本闸门**（`core/pause.py`）：**实时累计**费用到线时停止提交新的付费任务并弹人工审核窗口「当前费用已超上限，是否继续生成？」——确认即抬高上限（写回 `run_meta`）并从停下处接着跑，已出产物一个不动、不重复花钱；不确认就一直拦着「开始生成」。预估不足只出红幅警告，不参与闸门（那条判据是估算，不是真花钱）。旧行为是"熔断后静默跳过剩余付费提交、步骤照样报完成"，2026-10-04 的实测后果就是口型一条没花成而看板毫无提示。两个页签：**制作看板**（产物 ↔ 镜头/角色在服务端 join）：① **剧本** —— 逐场逐镜表格 + **人工三动作**：**✎ 编辑剧本稿**（可读文本，不让人改 JSON。**任意排版都能存**：先本地确定性整理成人话格式——别称值名、字段改名、说话人中文名→角色 id、markdown 噪声、无编号镜头、无字段名散文归位，一个字的用词都不改也不花钱；整理不出来的（表格体/纯散文）才调一次 TEXT 只改排版，并逐字校验台词没被改，校验不过宁可拒绝保存。整理做了什么作为说明回显。**语音预算**（`core/voice_budget.py`，系数全在 AppConfig 的 `voice_*`）取代旧的"单句 >30 字"：**最小预估时长 = round(字数 / 4.5) + 1 秒呼吸**，镜头需要 = max(语音, 演出) (+ 静默内容的额外留白)，且不低于官方每分镜 2 秒下限—— **只读软校验**，全片念不完/单镜装不下都只出告警；唯一的写动作是**时长下限**：「需要 ≤ 你写的 ≤ 25s」一律不动（那是节奏），写短了抬到最小可行值（时长同时决定提交给 wan3.0 的 duration 与提示词里的分镜时间戳，留短了会被吞句尾），写超 25s 才封顶。**时长格子可单击手改**（单位 s，失焦即存）：手填**允许**小于最小预估时长 —— 逐格填是知情决定，后端不偷偷抬回去，只在表格上方弹 2 秒告警（建议值直接引后端算好的 `capacity.per_shot[镜号].need_sec`，公式只有一份，前端不重算）；转场多样性自动补齐）/ **🎲 按 prompt 重新生成**（弹窗二选一：`revise` 保留剧情按意见改写、字段级作废省钱，`rewrite` 推倒重写、归档旧剧本并全量作废）/ **🗑 删除剧本**（弹确认后归档删除，主按钮回到「▶ 开始生成 · 生成剧本」）；原始 JSON 折叠为「高级：直接改结构」。**剧本评审由人做**（AI critic→自动重写回路已删除），分镜一致性抽查 / 镜头 best-of-N 选优 / 成片终审 / 一致性闸门照旧自动跑；② **角色图** —— 定妆照 + 外貌/必留词/外貌卡 + 落选候选（✔ 设为定妆照）+ **⬆ 上传 / 🗑 删除** + 该角色一致性达标率；**点外貌描述文字即改**（点别处保存，走 `PUT /script` 校验+血缘作废，改完点「开始生成」按新描述重画）；③ **场景图** —— 概念图 + 上传/删除，**地点与氛围描述点文字即改**；④ **分镜与镜头视频** —— 镜头卡（首帧 + 播放器 + 台词 + 逐镜一致性徽标 + 首帧上传/删除），**画面描述与镜头语言点文字即改**；⑤ **成片** —— 播放器 + 字幕 + 一致性汇总 + **一致性闸门结论**（不合格红横幅点名镜头）+ 降级清单。**事件流与成本**：3s 轮询时间线 + 逐条成本 + 成片终审徽标 |
| 人工修片 | 角色图/场景图/分镜首帧均可**⬆ 上传替换**（裸二进制 PUT，按本片画幅 cover 归一化，旧图自动备份 `.bak.png`，登记 `uploads.json`）；候选立绘可 **✔ 设为定妆照**。改完沿血缘精确作废下游，**点「开始生成」按步骤重做**即可（**绝不覆盖你上传的图** —— 作废清单跳过 `uploads.json`，m2 也跳过人工资产） |
| 环境自检 | JSON 版 doctor（依赖/FFmpeg**及其来源**/`.env` 是否就位/依赖锁漂移/`web/dist`/KEY/权重/数据库 + **.env 模型解析表**：每个能力的模型名 → 协议 → provider，标注是否已按协议自动改道）；`POST /api/doctor/provision` 一键把运行时资产收进项目 |

REST API（Swagger 文档：`/docs`）：`GET/POST /api/runs`（POST 只建任务，不启动生成）·
`GET /api/runs/{id}` ·
`GET /api/runs/{id}/board`（制作看板：剧本+角色图+场景图+逐镜头首帧/片段/度量+成片）·
`PUT /api/runs/{id}/artifacts/{stage}/{name}`（上传替换角色图/场景图/首帧，按画幅归一化）·
`GET /api/runs/{id}/uploads`（人工上传台账）·
`POST /api/runs/{id}/characters/{cid}/portrait`（人工指定定妆照，旧图自动备份）·
`GET /api/runs/{id}/steps`（六步进度：done/gap/active/failed + 缺口清单 + 花费）·
`POST /api/runs/{id}/step`（**开始生成**：只跑最靠前的未完成步骤）·
`GET/PUT /api/runs/{id}/script`（剧本原文读取 / 编辑保存+结构校验+血缘作废）·
`DELETE /api/runs/{id}/script`（删除剧本：归档为 `script.deleted.*.json` + 下游全量标 stale，
下一步回到「生成剧本」；人工上传过的图不牵连）·
`GET/PUT /api/runs/{id}/script/text`（**剧本稿**：可读剧本文本 ⇄ Script 的本地确定性解析，带行号报错）·
`POST /api/runs/{id}/script/regenerate`（按 prompt 重写剧本：`revise` 保留剧情只改被点名部分 /
`rewrite` 推倒重写并全量作废下游）·
`DELETE /api/runs/{id}/artifacts/{stage}/{name}`（删产物并沿血缘精确作废下游）·
`GET /api/task` · `POST /api/task/unlock`（全局单任务锁状态 / 人工解锁）·
`GET /api/files/{id}/{stage}/{name}` · `GET /api/doctor` · `GET /api/health`

全链路冒烟（`fastvideo3 serve --port 8765` 启动后，另开终端）：`python scripts/smoke_web_api.py`——Fake 模式零成本验证「建任务 → 点六次开始生成 → 成片下载 → 越权防护」。

## 架构一图

```
主题 → [编剧 m1] 结构化剧本 → ✋人工：改剧本稿 / 按 prompt 重写 / 删除重生成
     → [美术 m2] 角色/场景资产库（一致性锚点）→ ✋人工：换定妆照 / 上传 / 删图
     → [分镜师 m3] 每镜头首帧 + VLM 抽查 → ✋人工：换坏帧（付视频钱前的最后一道闸）
     → [镜头 m4 ‖ 配音 m5]（模块内信号量并发）→ [口型 m6：对白镜头出真口型]
     → [剪辑 m7] 规格统一 → xfade 转场 → 字幕 → BGM → 成片
     → [一致性度量+闸门 m9]（不达标自动重做 ≤2 轮）→ [终审 VLM m8] → ✋人工看片
     → final.mp4 + run_summary.json（成本 / 判定 / 降级台账）
```

执行：**一条路径、一份装配**。`core/progress.py` 从盘上算"下一步该做什么"，
`core/stages.py` 是所有阶段唯一的装配入口（点一次 = 一步，`cli step` / API `/step`）。
中断不需要"恢复"：产物、血缘台账（`lineage.json`）与成本都在盘上，再点一次「开始生成」
只补缺口与待重做项，已付费的产物不重复生成。
（进阶 `docker compose up -d` 把 JobStore 任务台账切到 PostgreSQL。）
口型模式 `talk_mode` —— **模式的名字就是通道**（2026-10-11 五次定档，映射表唯一来源
`fastvideo3/core/talkmode.py`，前端选项由 `GET /api/talk-modes` 从它生成）：

| 模式 | 通道 | 输入 | 一句话取舍 |
|---|---|---|---|
| `s2v_direct` | `wan2.2-s2v` | 图片 + **我们的语音** | 语音一致、便宜一段（对白镜头不过 I2V）；单人、720P 上限 |
| `v2v_overlay` | `videoretalk` | **动作片段** + 语音 | 身份不重画、1080P 保真；两段付费、最慢，片段边长需 ≥640 |
| `scripted_talk` | `wan3.0-video` | 图片 + **提示词（台词在里面）** | 一条调用出口型+动作+音轨、原生 1080P、**多人同屏有解**；音色由模型决定 |
| `audio_overlay` | —（不出口型） | 只叠音轨 | 显式放弃口型的唯一方式：教学与四臂消融用它 |

因为模式自己决定通道，**"模式与路由不成对"这一整类静默失效在结构上消失了**：以前
`talk_mode` 与 `providers.talk` 各写一份，漂一次就是"对白镜头先烧一份运动片段的钱、再 100%
降级成无口型叠音，而生成镜头视频因缺口永远不绿"（`run_20261002_235904`、`run_20261011_164152`
两次实测）。yaml 路由现在只在**形态相容**时作为换供应商的覆盖（例如 B 链换成 fal 的
/video-to-video），不相容的名字会被忽略；`.env` 的 `FV3_TALK_MODEL` 是专家级显式覆盖，优先级最高。

C 链的两个新语义：`voice_source`（**默认 `model`=保留模型原声** —— C 链的口型就是对着模型自己的
发声画的，只有这条能保证"嘴 = 听到的声音"；`master`=换回我们的配音，全片音色统一，但实测
**语音领先口型 0～2.2 秒且每镜偏移不恒定**（`core/talkmode.py` 有逐镜测量表），固定延迟修不好，
只作对比试验。要"音色统一 + 嘴型对着这条配音"请走 B 链 `videoretalk`）。
台词逐字取剧本文本进提示词，与 m7 烧的字幕同一来源，所以**内容**必然一致；模型原声的镜头里
字幕时刻按"镜头窗口 × 台词字数比例"排（`build_srt` 的 `model_audio_shots`），因为我们没有模型
每句的精确时刻，硬套 TTS 时长就是第二层错位。而**音色**没有保证：官方 `first_frame/last_frame`
与 `reference_audio` 互斥，锁脸与克隆音色只能选一个（提示词里的固定音色描述只能缓解）。
"确实不要口型"仍然只能靠 `audio_overlay` 明说 —— 其余模式下"对白镜头没有口型"永远是缺口而不是完成
（判定与看板都按此约束，见 INTEGRATION.md 的"底层逻辑修正"）。

## 人物音频链路（对白为什么这样出声）

一句台词到观众耳朵里要过四道关，每一道都对应一个实测踩过的坑：

| 关 | 机制 | 曾经的真实故障 |
| 字幕 | `m7.build_srt`：文本**逐字取剧本台词**（不做语音识别、不让模型画字幕）；时间轴按镜头窗口排布并**保证任意两条不重叠**；样式唯一来源 `subtitle_style()`（SimHei 加粗白字黑描边），与片头/片尾标题卡共用同一款字体；烧录进画面，旁挂副本叫 `final/subtitles.srt`（**与成片不同基名**，见右侧） | ① 口型片段比驱动音频短 0.2s → 上一条字幕溢出进下一镜头，画面两层字幕；② 旁挂文件叫 `final.srt` 时，VLC/PotPlayer 默认自动挂载同名 srt → 烧录层 + 旁挂层叠成两层（2026-10-07 实测） |
|---|---|---|
| 音色选择 | `m5.plan_voices`：按角色**性别**分桶、桶内轮流取音色（确定性、跨镜头稳定） | 只靠 md5 抽签 → 唐僧拿女声、两个主角同一把嗓子 |
| 音色名配套 | 池子按模型版本给名：`cosyvoice-v2` 用 `*_v2`，v1 用无后缀 | v2 收旧名 → 服务端 418 → **每句静默降级 edge-tts**，看板还显示"配音正常" |
| 情绪 | `utils/emotion` 把「焦急/震惊转羞涩/戏谑」归一成档位，再换算韵律 | 韵律表是英文键，中文自由文本 100% 落空 → 全片一个语调 |
| 最终音轨 | m6 落 `audio/shot_{id}_driver.wav`，m7 用它**替换**口型片段自带音轨 | s2v 回传音轨是 16k→44.1k 上变换产物（发闷），且片段比音频短 0.21s（句尾被吃） |

自检：`fastvideo3 doctor --probe` 里"百炼口型通道探活"之外，配音降级会出现在看板降级区
与 `degradations.json`（`tts_provider_failed`），台账 note 带 `(fallback:dashscope_cosyvoice)`。

## 人物一致性链路（第 2 讲主线）

同一角色跨镜头保持同一张脸/同一身衣服，靠的是**四层机制 + 一层度量**，且每层都可验证：

| 层 | 机制 | 生效档位 | 通电判据（盘上证据） |
|---|---|---|---|
| 源头 | 角色立绘确定性种子 + best-of-N 选优 + 写实风格闸门 | 种子全档；抽卡选优仅 1080P | `char_best_of_n{scores,verified}`、`char_style_gate_failed`、`asset_registry.json` 的 `meta.seed/medium` |
| 首帧 | 立绘作**参考图条件**（`wan2.7-image` 多图协议）+ 同角色跨场景同种子 | **全档**（480P 自 2026-10-17 起也开，见下） | `keyframe_gen{ref_count,ref_ids,seed,seed_scope}`；台账 `meta.ref_count` |
| 场景片段 | 立绘 + 场景图作 `reference_image`、音色样本作 `reference_audio`，一次调用出一段戏 | 全档（与分辨率无关，一直是通的） | `clips/scene_<组>.json` 的 `mode` / `ref_images` / `dropped` |
| 运动 | 角色动作镜头走 `wan2.7-r2v` 参考生视频 | 1080P | `shot_gen{route:"r2v",ref_ids}`；台账 note `#r2v` |
| 文本 | `appearance` 逐字注入 + 外貌卡必留词蒸馏校验 + 全局风格锚/负向词 | 全档 | `distill_attr_lost{missing}`（理想 0 条） |
| 度量 | VLM 结构化属性位（自校准锚/顺序互换/双人二选一）+ SFace 人脸余弦 + 纯 PIL 色相证伪 | 全档（480P/720P 抽样，1080P 全检） | `consistency_report.json`、`metric_conflict`、`judge_unreliable` |
| 重拍 | 度量不达标就带参考图自动重做 ≤2 轮 | **仅 720P/1080P**（唯一与分辨率有关的一致性项：重拍要花钱） | `consistency_gate` 事件、`quality_verdict.rounds` |

**「用同一张角色图出同一个角色」三档都成立**（2026-10-17 裁定）。480P 曾经关掉首帧锁脸，
而且**开着也空转**：它的出厂文生图 `wanx2.1-t2i-turbo` 走异步 text2image 端点、
结构上收不了参考图，于是每一镜只落一条 `ref_condition_dropped{reason:provider_no_capability}`，
成片照常交付、看板全绿。现在三档的 t2i 主路由都是 `wan2.7-image`，代价 ¥0.06/张 × 约 21 张
≈ **+¥1.3/片**（`consistency_gate: reshoot` 仍然只有 720P/1080P 开 —— 那是 ¥6–17/段的重拍费）。

**降级一律响亮**：`ref_condition_dropped`、`fallback_used{dropped:[seed,ref_image]}`、
`best_of_n_unscored`、`degradations.json`、`run_summary.review_unavailable` —— 只看
`cost.json` + `events.jsonl` 就能重建"这次 API 实际发了什么"，杜绝"看起来通电、实际无效"。
三档都有 `require_reference_keyframes` 硬门控：t2i 主供应商不支持参考图时**在花钱前**报错
（新建任务与「模型选择」也会用同一条判据先拦，判据取自 `configs`，不在代码里抄分辨率清单）。

## 项目结构

```
fastvideo3/
├── core/       # 配置/schema/存储/成本/任务台账 + 进度状态机与阶段执行器（第1、6讲素材）
├── providers/  # DeepSeek/百炼/SiliconFlow/可灵/edge-tts + Fake 全家桶
├── modules/    # m1剧本 … m9度量（教学章节 1:1 映射）
├── app/        # Typer CLI + FastAPI Web API + Streamlit 面板
└── utils/      # ffmpeg/音频/图像/YOLO/rembg（懒加载）
web/            # Vue 3 前端（Vite 构建，dist 由 FastAPI 托管）
```

## 开发者（CI 自检）

```bash
pip install -e ".[dev]"
ruff check .          # 风格
mypy fastvideo3       # 类型
pytest -m "not api"   # 离线单测 + Fake 全链路冒烟 + Web API 单测（无需 KEY）
pytest -m api         # 真实 API 冒烟（需 .env，nightly）
cd web && npm ci && npm run build   # 前端构建门禁（CI job: web-build）
```

## 与 TDD 的实现差异说明

1. Provider 层用 **httpx 直连**（更薄、更教学友好），未强依赖 LangChain 的 LLM 封装。
2. **编排不再是状态图**（2026-10-06 删除端到端路径）：进度由 `core/progress.py` 从盘上算，
   执行由 `core/stages.py` 一份装配；并行只保留**模块内信号量并发**（m4/m5/m6 各自并发提交）。
   逐镜头 map-reduce 编排留作进阶练习（见 TDD §5.4）。
3. 双人对话镜头用分镜首帧（本就两人同框）走单条 S2V，不再做立绘左右拼接（旧路径被 9:16 cover 裁切毁帧）；YOLO/rembg 分割合成留作进阶练习（utils 中保留、暂无调用点）。
4. **一致性从"提示词问题"改造成"数据链路问题"**（2026-10）：审计真实 run 台账发现 cinema 档宣称的
   "参考图锁脸"在盘上 **0 次计费**（`wan2.7-image` 从未被调用，`ref_image` 是死参数）、抽卡**无评分**
   （15 条 `best_of_n` 全部选中第一个候选、`vlm` 计费 0 条）、终审分数是解析失败兜底的 0.7。
   现已全部接通并加"响亮降级 + 不可伪造分数 + 可量化度量"。历史 run 的观感结论请按"锁脸未生效"重读。
