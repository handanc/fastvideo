// FastAPI 后端封装：所有请求走 /api（开发由 vite 代理，生产同源）
async function unwrap(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const data = await res.json()
      if (data.detail) msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch { /* 非 JSON 错误体 */ }
    throw new Error(msg)
  }
  return res.json()
}

function post(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(unwrap)
}

export const api = {
  health: () => fetch('/api/health').then(unwrap),
  doctor: (probe = false) => fetch(probe ? '/api/doctor?probe=1' : '/api/doctor').then(unwrap),
  listRuns: () => fetch('/api/runs').then(unwrap),
  createRun: (payload) => post('/api/runs', payload),
  runDetail: (id) => fetch(`/api/runs/${encodeURIComponent(id)}`).then(unwrap),
  resume: (id, decision) => post(`/api/runs/${encodeURIComponent(id)}/resume`, { decision }),
}
