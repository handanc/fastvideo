<script setup>
import { computed, ref } from 'vue'
import { api } from '../api'

const props = defineProps({
  runId: { type: String, required: true },
  steps: { type: Object, default: null }, // GET /steps 的响应
  taskBusy: { type: Object, default: null }, // GET /api/task 的响应
})
const emit = defineEmits(['changed'])

const busy = ref(false)
const notice = ref('')
const error = ref('')

const list = computed(() => props.steps?.steps || [])
const next = computed(() => props.steps?.next || null)
const otherBusy = computed(
  () => props.taskBusy?.busy && props.taskBusy.run_id !== props.runId,
)
const disabled = computed(() => busy.value || props.steps?.running || otherBusy.value)

const STATE_CLS = { done: 'ok', gap: 'warn', active: 'run', failed: 'err', pending: 'idle' }
const STATE_LABEL = {
  done: '已完成', gap: '有缺口', active: '生成中', failed: '失败', pending: '待生成',
}

function gapText(s) {
  const n = (s.missing?.length || 0) + (s.stale?.length || 0)
  if (!n) return ''
  const sample = [...(s.missing || []), ...(s.stale || [])].slice(0, 3).join(', ')
  return `缺/待重做 ${n}：${sample}${n > 3 ? '…' : ''}`
}

async function runStep() {
  busy.value = true
  error.value = ''
  notice.value = ''
  try {
    const r = await api.stepRun(props.runId)
    notice.value = r.started
      ? '✔ 已开始'   // 只留一个字面确认：点了完全没回执，看起来就像按钮坏了
      : '✔ 全部完成'
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="card stepbar">
    <div class="steps">
      <template v-for="(s, i) in list" :key="s.id">
        <div class="step" :class="STATE_CLS[s.state] || 'idle'" :title="s.error || gapText(s)">
          <span class="idx">{{ i + 1 }}</span>
          <span class="lbl">{{ s.label }}</span>
          <span class="st">{{ STATE_LABEL[s.state] || s.state }}</span>
          <span v-if="gapText(s)" class="gapnote">{{ gapText(s) }}</span>
        </div>
        <div v-if="i < list.length - 1" class="step-connector"></div>
      </template>
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>
    <div v-if="notice" class="alert ok">{{ notice }}</div>
    <div v-if="otherBusy" class="alert warn">
      任务 {{ taskBusy.run_id }} 正在生成中（{{ taskBusy.kind }}）；一次只允许一个任务。
    </div>

    <div class="toolbar" style="margin-top: 10px">
      <button class="btn primary" :disabled="disabled || !next" @click="runStep">
        {{ busy ? '提交中…' : next ? `▶ 开始生成 · ${next.label}` : '✔ 已全部完成' }}
      </button>
      <span class="muted">
        已花 ¥{{ Number(props.steps?.spent_cny || 0).toFixed(2) }} /
        上限 ¥{{ Number(props.steps?.budget_cny || 0).toFixed(0) }}
      </span>
    </div>
    <p v-for="s in list.filter((x) => x.state === 'failed')" :key="s.id" class="alert err">
      「{{ s.label }}」失败：{{ s.error }} —— 修复后再点「开始生成」会从这一步继续。
    </p>
  </div>
</template>
