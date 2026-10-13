# FastVideo3 Web 前端（Vue 3 + Vite）

简洁的单页面板，为 `fastvideo3/app/api.py`（FastAPI）提供 Web UI。无 UI 框架、无路由依赖，仅 `vue` 一个运行时依赖。

## 开发

```bash
npm install
npm run dev        # http://127.0.0.1:5173，/api 自动代理到 :8000
```

同时在项目根目录启动后端：`fastvideo3 serve`（或 `uvicorn fastvideo3.app.api:app --port 8000`）。

## 构建（生产）

```bash
npm run build      # 产物在 dist/，由 FastAPI 同源托管
fastvideo3 serve   # 打开 http://127.0.0.1:8000
```

## 结构

```
src/
├── main.js               # 入口
├── App.vue               # 顶栏 + 视图切换（新建/历史/详情/自检）
├── api.js                # fetch 封装（统一错误解包；422 的多条校验按行拼好）
├── status.js             # 运行状态 → 中文徽章映射
├── style.css             # 设计令牌 + 组件样式（亮/暗自动适配）
└── components/
    ├── NewRunForm.vue    # 新建任务表单（字段与 CLI `new` 一一对应）→ 一颗「▶ 开始生成」只建任务
    ├── RunList.vue       # 运行历史（5s 自动刷新）
    ├── RunDetail.vue     # 详情：StepBar + 制作看板/事件流与成本（3s 轮询）
    ├── StepBar.vue       # 六步进度条 + 「▶ 开始生成 · <下一步>」（点一次补一步）
    ├── ProductionBoard.vue # 制作看板：剧本三动作（改稿/按 prompt 重写/删除）、角色图/场景图/分镜/成片、上传替换
    └── DoctorView.vue    # 环境自检
```
