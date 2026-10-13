<script setup>
import { onMounted, ref } from 'vue'
import { api } from '../api'

const result = ref(null)
const error = ref('')
const loading = ref(false)
const fixing = ref(false)
const fixMsg = ref('')

// 「可选」类检查项不该被显示成 ❌：FAL_KEY 只是口型兜底通道、yolov8n 权重本期没接入
// 管线、身份权重缺了也只是退回 A/B 层 —— 后端已经把它们算成 ok=true（口径在
// core/runtime.py），这里只补一个灰色"可选"标签，避免用户以为环境坏了。
const OPTIONAL = ['DEEPSEEK_API_KEY', 'FAL_KEY', 'yolov8n-seg.pt(备用)', '身份一致性权重（可选）']

async function load(probe = false) {
  loading.value = true
  error.value = ''
  try {
    result.value = await api.doctor(probe)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

// 补齐＝把运行需要的东西收进项目目录（ffmpeg → bin/、权重 → models/、缺 .env 时复制模板）
// 上线前点一次，这份目录就能离开本机开发环境独立运行；后端不会覆盖已有 .env，也不回显内容。
async function provision() {
  fixing.value = true
  error.value = ''
  fixMsg.value = ''
  try {
    const r = await api.doctorProvision()
    const lines = (r.done || []).map((d) => `${d.error ? '✘' : '✔'} ${d.what}：${d.done || d.error}`)
    const rest = (r.remaining || []).length ? `仍需处理：${r.remaining.join('、')}` : '运行时齐备'
    fixMsg.value = lines.concat(rest).join('\n')
    result.value = { all_ok: r.all_ok, checks: r.checks }
  } catch (e) {
    error.value = e.message
  } finally {
    fixing.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="card">
    <div class="toolbar">
      <h2 style="margin: 0">环境自检（doctor）</h2>
      <div class="spacer"></div>
      <span v-if="result" class="badge" :class="result.all_ok ? 'ok' : 'warn'">
        {{ result.all_ok ? '全部通过' : '存在异常项' }}
      </span>
      <button class="btn sm" :disabled="loading" @click="load(false)">⟳ 重新检查</button>
      <button class="btn sm" :disabled="fixing" @click="provision"
              title="把 ffmpeg 收进项目 bin/、下载身份权重、缺 .env 时从模板复制">📦 补齐项目内置运行时</button>
      <button class="btn sm" :disabled="loading" @click="load(true)"
              title="真实调用一次最小文本请求验证账户状态（≈¥0.001）">🔑 KEY 探活</button>
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>
    <pre v-if="fixMsg" class="alert ok mono" style="white-space: pre-wrap">{{ fixMsg }}</pre>

    <table v-if="result" class="tbl">
      <thead><tr><th style="width: 40px"></th><th>检查项</th><th>结果</th></tr></thead>
      <tbody>
        <tr v-for="c in result.checks" :key="c.item">
          <td>{{ c.ok ? '✅' : '❌' }}</td>
          <td>
            {{ c.item }}
            <em v-if="OPTIONAL.includes(c.item)" class="muted small">（可选）</em>
          </td>
          <td class="muted mono">{{ c.detail }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else-if="!error" class="muted">检查中…</p>
    <p class="muted small">
      「项目内置运行时」= 这份目录自带 ffmpeg、权重与依赖锁，换机器不靠宿主机装了什么。
      前端产物 <code>web/dist</code> 需要 node，构建：<code>cd web && npm ci && npm run build</code>。
    </p>
  </div>
</template>
