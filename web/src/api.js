// FastAPI 后端封装：所有请求走 /api（开发由 vite 代理，生产同源）
async function unwrap(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const data = await res.json()
      if (Array.isArray(data.detail)) {
        // 后端把"剧本稿第 N 行不合法"这类多条校验问题回成数组：换行显示才能逐条读
        msg = data.detail.join('\n')
      } else if (data.detail) {
        msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
      }
    } catch { /* 非 JSON 错误体 */ }
    throw new Error(msg)
  }
  return res.json()
}

function post(url, body = {}) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(unwrap)
}

export const api = {
  health: () => fetch('/api/health').then(unwrap),
  doctor: (probe = false) => fetch(probe ? '/api/doctor?probe=1' : '/api/doctor').then(unwrap),
  // 补齐项目内置运行时：ffmpeg 复制进 bin/、身份权重进 models/、缺 .env 时复制模板。
  // 后端只动项目内文件，不回显 .env 内容（自检文本会原样显示在浏览器里）。
  doctorProvision: () => post('/api/doctor/provision'),
  // 场景级生成（wan3.0 一次调用出一个场景的连贯多镜头片段）
  sceneGroups: (id) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/scene-groups`).then(unwrap),
  patchSceneGroup: (id, gid, payload) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/scene-groups/${encodeURIComponent(gid)}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    }).then(unwrap),
  genSceneFrames: (id, gid) =>
    post(`/api/runs/${encodeURIComponent(id)}/scene-groups/${encodeURIComponent(gid)}/frames`),
  genSceneClip: (id, gid) =>
    post(`/api/runs/${encodeURIComponent(id)}/scene-groups/${encodeURIComponent(gid)}/generate`),
  deleteSceneClip: (id, gid) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/artifacts/clips/scene_${encodeURIComponent(gid)}.mp4`,
      { method: 'DELETE' }).then(unwrap),
  // 成本预估（单价权威在后端 pricing.yaml + 模式解析出的通道，前端不再手抄量级表）
  costEstimate: (quality, duration, talkMode) =>
    fetch(`/api/cost-estimate?quality=${encodeURIComponent(quality)}&duration=${encodeURIComponent(duration)}&talk_mode=${encodeURIComponent(talkMode || '')}`).then(unwrap),
  // 口型模式清单：选项、通道名、输入形态、档位冲突提示都来自后端 core/talkmode.py
  // 模型选择器：可选清单 + 当前实际生效的模型 + 是谁定的（run/.env/yaml），
  // 唯一来源 core/model_catalog.py（前端不抄第二份模型名表）
  modelChoices: (quality, runId) =>
    fetch(
      `/api/model-choices?quality=${encodeURIComponent(quality || '')}` +
      `&run_id=${encodeURIComponent(runId || '')}`
    ).then(unwrap),
  // 中途改模型（含口型模型）：只对**之后**的步骤生效（已落盘产物不动）；持锁期间后端会拒绝。
  // payload 形如 { model_overrides: {text,t2i,i2v}, talk_mode, voice_source }
  patchRunModels: (id, payload) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/models`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(unwrap),
  // 分辨率选项（480P/720P/1080P → 内部档位 + 30s 预估），映射只在后端一份
  resolutions: () => fetch('/api/resolutions').then(unwrap),
  // 删成片：删掉后「拼接成片」回到缺口，重点一次即可零 API 费用重合成
  deleteFinal: (id) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/final`, { method: 'DELETE' }).then(unwrap),
  listRuns: () => fetch('/api/runs').then(unwrap),
  // 建任务：只写 run_meta.json，不花钱；生成由下面的 step 一步一步来
  createRun: (payload) => post('/api/runs', payload),
  runDetail: (id) => fetch(`/api/runs/${encodeURIComponent(id)}`).then(unwrap),
  board: (id) => fetch(`/api/runs/${encodeURIComponent(id)}/board`).then(unwrap),
  pickPortrait: (id, charId, candidate) =>
    post(`/api/runs/${encodeURIComponent(id)}/characters/${encodeURIComponent(charId)}/portrait`,
      { candidate }),
  // 上传替换：裸二进制 PUT（后端不依赖 multipart）
  putArtifact: (id, stage, name, file) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(stage)}/${encodeURIComponent(name)}`,
      { method: 'PUT', body: file }).then(unwrap),
  // 开始生成（一次一步）
  steps: (id) => fetch(`/api/runs/${encodeURIComponent(id)}/steps`).then(unwrap),
  // 成本到线后的人工确认：抬高上限（留空 = 后端按已花+剩余预估给一个值）
  budgetContinue: (id, new_limit_cny) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/budget/continue`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(new_limit_cny ? { new_limit_cny: Number(new_limit_cny) } : {}),
    }).then(unwrap),
  stepRun: (id) => post(`/api/runs/${encodeURIComponent(id)}/step`, {}),
  putScript: (id, script) => fetch(`/api/runs/${encodeURIComponent(id)}/script`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(script),
  }).then(unwrap),
  // 剧本稿：可读文本，人改的是剧本本身而不是 JSON
  scriptText: (id) => fetch(`/api/runs/${encodeURIComponent(id)}/script/text`).then(unwrap),
  putScriptText: (id, text) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/script/text`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    }).then(unwrap),
  delScript: (id) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/script`, { method: 'DELETE' }).then(unwrap),
  // 按 prompt 重新生成剧本：revise=保留剧情按意见改，rewrite=推倒重写
  regenerateScript: (id, prompt, mode) =>
    post(`/api/runs/${encodeURIComponent(id)}/script/regenerate`, { prompt, mode }),
  delArtifact: (id, stage, name) =>
    fetch(`/api/runs/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(stage)}/${encodeURIComponent(name)}`,
      { method: 'DELETE' }).then(unwrap),
  task: () => fetch('/api/task').then(unwrap),
  unlockTask: () => post('/api/task/unlock', {}),
}
