<script setup>
import { onMounted, ref } from 'vue'
import { api } from '../api'

const result = ref(null)
const error = ref('')
const loading = ref(false)

async function load() {
  loading.value = true
  error.value = ''
  try {
    result.value = await api.doctor()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
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
      <button class="btn sm" :disabled="loading" @click="load">⟳ 重新检查</button>
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>

    <table v-if="result" class="tbl">
      <thead><tr><th style="width: 40px"></th><th>检查项</th><th>结果</th></tr></thead>
      <tbody>
        <tr v-for="c in result.checks" :key="c.item">
          <td>{{ c.ok ? '✅' : '❌' }}</td>
          <td>{{ c.item }}</td>
          <td class="muted mono">{{ c.detail }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else-if="!error" class="muted">检查中…</p>
  </div>
</template>
