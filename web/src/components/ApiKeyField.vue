<script setup>
// 阿里云百炼 API KEY：掩码输入，点到别处 / 回车 / 按「保存」都会写进 .env 的 DASHSCOPE_API_KEY。
//
// 为什么单独一个组件而不是在每个页面里各写一遍：这个输入框要出现在「新建任务」和「运行详情」
// 两处（跑不动的时候用户先看的是详情页），而"密钥不外泄"的自律一旦抄成两份，就会有第三份漏掉。
//
// 三条自律（前端这一侧）：
// 1. 值只活在这一个 ref 里：不写 localStorage、不进 URL/查询串、不进任何提示文案；
// 2. 保存成功后**立刻清空输入框** —— 密钥留在 DOM 里就等于留在内存与崩溃转储里；
// 3. 显示一律用后端的"有没有 / 从哪读到"（`GET /api/api-keys` 只回名字与布尔），永不回显值。
// 留空＝不改动：不发请求，更不会把已存的 KEY 清掉（误触失焦不该让用户下次跑不出片）。
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'

const props = defineProps({
  // 详情页里已经知道这条 run 缺 KEY（降级台账报了），用来把提示说得具体一点
  context: { type: String, default: '' },
})

const row = ref(null)
const draft = ref('')
const busy = ref(false)
const note = ref('')
const error = ref('')
const noBackend = ref(false)
// 探活：`set=True` 只证明"字符串存在"，证明不了账户能用。真实失败长这样：绿勾出现了、
// 第一步却报欠费/KEY 无效/连不上。所以给一颗**用户自己按**的探活按钮（它要发一次真请求，
// ≈¥0.001），前端绝不自动调用，后端也不在别的接口里顺手探。
const probing = ref(false)
const probeResult = ref(null)
const probeError = ref('')

const label = computed(() => (row.value && row.value.label) || '阿里云百炼 API KEY')
const isSet = computed(() => !!(row.value && row.value.set))
const stateText = computed(() => {
  if (!row.value) return noBackend.value ? '后端未就绪' : '状态未知'
  if (!row.value.set) return '未配置'
  return row.value.source === 'env' ? '已配置（来自环境变量）' : '已配置（来自 .env）'
})
const placeholder = computed(() => (isSet.value ? '••••••••' : 'sk-…'))
const trimmed = computed(() => (draft.value || '').trim())
// 探活结果的状态位翻成人话：芯片上显示 `invalid_key` 这种内部名字，用户只能去猜
const probeChipText = computed(() => {
  const map = {
    ok: '可用',
    arrears: '账户欠费/未开通',
    invalid_key: 'KEY 被拒',
    unreachable: '连不上',
  }
  const st = probeResult.value && probeResult.value.status
  return map[st] || st || ''
})

async function load() {
  try {
    const data = await api.apiKeys()
    row.value = (data.keys || []).find((k) => k.field === 'dashscope_api_key') || null
    noBackend.value = false
  } catch (e) {
    // 问不到状态时输入框照常可用（保存失败会说明原因），表单不能因为一处问不到就不能填
    row.value = null
    noBackend.value = /\b(404|405)\b/.test(String(e && e.message))
  }
}
onMounted(load)

async function probe() {
  if (probing.value) return
  probing.value = true
  probeError.value = ''
  try {
    probeResult.value = await api.probeApiKey()
  } catch (e) {
    const msg = String((e && e.message) || e)
    probeError.value = /\b(404|405)\b/.test(msg)
      ? '后端还是旧版本（没有探活接口）：重启 `fastvideo3 serve` 后刷新页面'
      : msg
    probeResult.value = null
  } finally {
    probing.value = false
  }
}

async function save() {
  const value = trimmed.value
  note.value = ''
  error.value = ''
  if (!value || busy.value) return
  busy.value = true
  probeResult.value = null      // 换了 KEY 就是换了账户，上一次的结论作废
  probeError.value = ''
  try {
    const r = await api.saveApiKey(value)
    note.value = r.changed ? `✔ 已写入 .env 的 ${r.name}` : `✔ ${r.name} 已是当前值`
    draft.value = ''
    await load()
  } catch (e) {
    const msg = String((e && e.message) || e)
    error.value = /\b(404|405)\b/.test(msg)
      ? '后端还是旧版本（没有 /api/api-keys 这个接口）：重启 `fastvideo3 serve` 后刷新页面'
      : `保存失败：${msg}`
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="keyField" :class="{ keyUnset: row && !row.set }">
    <div class="keyHead">
      <b>{{ label }}</b>
      <span class="keyChip" :class="isSet ? 'keyChipOk' : 'keyChipWarn'">{{ stateText }}</span>
    </div>
    <div class="keyLine">
      <input
        v-model="draft"
        type="password"
        :placeholder="placeholder"
        autocomplete="off"
        spellcheck="false"
        data-lpignore="true"
        @blur="save"
        @keyup.enter="save"
      />
      <button class="btn sm" type="button" :disabled="busy || !trimmed" @click="save">
        {{ busy ? '保存中…' : '保存' }}
      </button>
    </div>
    <div v-if="noBackend" class="keyNote keyWarn">
      ⚠ 后端没有这个接口：重启 <code>fastvideo3 serve</code>（或 <code>npm run dev</code> 的代理目标）
      后刷新页面
    </div>
    <div v-else-if="error" class="keyNote keyWarn">⚠ {{ error }}</div>
    <div v-else-if="note" class="keyNote keyOk">
      {{ note }}
    </div>

    <!-- 探活：用户自己按的按钮。"存下了"与"账户能用"是两件事：欠费/未开通/复制截断都只有
         真打一次才知道，而让人一路点到第一步报错才知道是最费时间的那一种。 -->
    <div class="keyLine">
      <!-- 成本披露放进 tooltip 而不是正文：用户刚要求把说明文字去掉，但"这一按会发一次真请求"
           属于点下去之前该知道的事 —— 藏起来省了版面，也省掉了知情。 -->
      <button
        class="btn sm"
        type="button"
        title="发一次最小请求确认账户可用（约 ¥0.001）"
        :disabled="probing || !isSet || !!noBackend"
        @click="probe"
      >
        {{ probing ? '测试中…' : '测试KEY' }}
      </button>
      <span v-if="probeResult" class="keyChip" :class="probeResult.usable ? 'keyChipOk' : 'keyChipWarn'">
        {{ probeChipText }}
      </span>
    </div>
    <div v-if="probeResult" :class="['keyNote', probeResult.usable ? 'keyOk' : 'keyWarn']">
      {{ probeResult.usable ? '✔' : '⚠' }} {{ probeResult.advice }}
      <span v-if="!probeResult.usable && probeResult.detail" class="muted">
        （{{ probeResult.detail }}）
      </span>
    </div>
    <div v-if="probeError" class="keyNote keyWarn">⚠ 测试失败：{{ probeError }}</div>

    <div v-if="context" class="keyNote muted">{{ context }}</div>
  </div>
</template>

<style scoped>
.keyField {
  margin: 14px 0;
  padding: 12px 14px;
  border: 1px solid var(--line, #333);
  border-radius: 10px;
}
.keyField.keyUnset { border-color: var(--err, #e5533d); }
.keyHead { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 14px; }
.keyLine { display: flex; align-items: center; gap: 8px; margin: 6px 0; flex-wrap: wrap; }
.keyLine input {
  flex: 1 1 260px;
  min-width: 0;
  padding: 7px 10px;
  letter-spacing: .14em;
}
.keyChip {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--line, #333);
  white-space: nowrap;
}
.keyChipOk { color: var(--ok, #2e9e5b); }
.keyChipWarn { color: var(--err, #e5533d); border-color: var(--err, #e5533d); }
.keyNote { font-size: 12px; margin-top: 4px; }
.keyOk { color: var(--ok, #2e9e5b); }
.keyWarn { color: var(--err, #e5533d); }
</style>
