<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'
import { statusOf } from '../status'

const emit = defineEmits(['select'])

const runs = ref([])
const loading = ref(false)
const error = ref('')
let timer = null

async function load() {
  loading.value = true
  error.value = ''
  try {
    runs.value = await api.listRuns()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  load()
  timer = setInterval(load, 5000) // 自动刷新，跟进后台任务状态
})
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="card">
    <div class="toolbar">
      <h2 style="margin: 0">运行历史</h2>
      <div class="spacer"></div>
      <span class="muted">每 5 秒自动刷新</span>
      <button class="btn sm" :disabled="loading" @click="load">⟳ 刷新</button>
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>
    <p v-else-if="!runs.length" class="muted">暂无任务，到「新建任务」开始。</p>

    <table v-else class="tbl">
      <thead>
        <tr>
          <th>Run ID</th>
          <th>主题</th>
          <th>状态</th>
          <th>模式</th>
          <th class="num">成本(¥)</th>
          <th>最近更新</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in runs" :key="r.run_id" class="clickable" @click="emit('select', r.run_id)">
          <td class="mono">{{ r.run_id }}</td>
          <td>{{ r.topic || '—' }}</td>
          <td><span class="badge" :class="statusOf(r.status).cls">{{ statusOf(r.status).label }}</span></td>
          <td>
            <span class="tag" style="margin: 0">{{ r.fake ? 'Fake' : '真实' }}</span>
            <span class="tag">{{ r.quality }}</span>
          </td>
          <td class="num">{{ r.cost_total_cny.toFixed(2) }}</td>
          <td class="muted">{{ r.updated }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
