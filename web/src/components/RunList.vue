<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'
import { statusOf } from '../status'

const emit = defineEmits(['select'])

const runs = ref([])
const loading = ref(false)
const error = ref('')
const notice = ref('')
const picked = ref([])      // 勾选的 run_id
const deleting = ref(false)

async function load() {
  loading.value = true
  error.value = ''
  try {
    runs.value = await api.listRuns()
    // 列表变了之后，勾里可能留着已经不存在的任务（刚删掉的、别处删掉的）
    const alive = new Set(runs.value.map((r) => r.run_id))
    picked.value = picked.value.filter((id) => alive.has(id))
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

const allPicked = computed(() => runs.value.length > 0 && picked.value.length === runs.value.length)

function toggleAll() {
  picked.value = allPicked.value ? [] : runs.value.map((r) => r.run_id)
}

function toggle(id) {
  picked.value = picked.value.includes(id)
    ? picked.value.filter((x) => x !== id)
    : [...picked.value, id]
}

// 删除是**不可撤销**的：盘上的产物、成本台账、血缘、事件流一起没。所以确认框必须写清
// 删几个、哪几个、以及"已花的钱的记录也会消失"—— 只写「确定删除？」是不够的。
async function removePicked() {
  const ids = [...picked.value]
  if (!ids.length) return
  const rows = runs.value.filter((r) => ids.includes(r.run_id))
  const cost = rows.reduce((s, r) => s + Number(r.cost_total_cny || 0), 0)
  const list = ids.slice(0, 8).join('\n') + (ids.length > 8 ? `\n…共 ${ids.length} 个` : '')
  const ok = window.confirm(
    `确定删除 ${ids.length} 个任务？\n\n${list}\n\n`
    + '会连同盘上目录一起删除：视频片段、角色图/场景图、成本台账、事件流全部消失，不可恢复。\n'
    + `这些任务已花费约 ¥${cost.toFixed(2)}（花费记录也会一并删除）。\n\n`
    + '正在生成中的任务会被拒绝删除。',
  )
  if (!ok) return
  deleting.value = true
  error.value = ''
  notice.value = ''
  const failed = []
  let done = 0
  let freed = 0
  // 一个个删：并发删在 Windows 上会撞文件占用，而且失败时说不清是哪个任务没删掉
  for (const id of ids) {
    try {
      const r = await api.deleteRun(id)
      done += 1
      freed += Number(r.freed_mb || 0)
    } catch (e) {
      failed.push(`${id}：${e.message}`)
    }
  }
  picked.value = []
  await load()
  deleting.value = false
  if (done) notice.value = `✔ 已删除 ${done} 个任务，释放 ${freed.toFixed(1)} MB 磁盘空间`
  if (failed.length) error.value = `以下任务没删掉：\n${failed.join('\n')}`
}

// 不做自动刷新：这一页只是任务清单，5 秒一次的全量拉取换不来什么，
// 反而会在人正勾着一半时把列表刷掉（勾还在、行没了）。进页面拉一次，删完再拉一次。
onMounted(load)
</script>

<template>
  <div class="card">
    <div class="toolbar">
      <h2 style="margin: 0">运行历史</h2>
      <div class="spacer"></div>
      <!-- 删除放在窗口最右侧：它是这一页唯一的破坏性动作，不该混在列表行里 -->
      <button
        class="btn sm del"
        :disabled="!picked.length || deleting"
        :title="picked.length ? `删除勾选的 ${picked.length} 个任务（连盘上数据一起）` : '先勾选要删除的任务'"
        @click="removePicked"
      >
        {{ deleting ? '删除中…' : `🗑 删除${picked.length ? ` (${picked.length})` : ''}` }}
      </button>
    </div>

    <div v-if="error" class="alert err" style="white-space: pre-line">{{ error }}</div>
    <div v-if="notice" class="alert ok">{{ notice }}</div>
    <p v-if="!runs.length && !loading" class="muted">暂无任务，到「新建任务」开始。</p>

    <table v-else class="tbl">
      <thead>
        <tr>
          <th class="pick">
            <input type="checkbox" :checked="allPicked" title="全选 / 全不选" @change="toggleAll" />
          </th>
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
          <!-- 勾选格必须 stop：否则点勾的同时把任务打开了 -->
          <td class="pick" @click.stop>
            <input
              type="checkbox"
              :checked="picked.includes(r.run_id)"
              :title="`勾选 ${r.run_id}`"
              @change="toggle(r.run_id)"
            />
          </td>
          <td class="mono">{{ r.run_id }}</td>
          <td>{{ r.topic || '—' }}</td>
          <td><span class="badge" :class="statusOf(r.status).cls">{{ statusOf(r.status).label }}</span></td>
          <td>
            <span class="tag" style="margin: 0">{{ r.fake ? 'Fake' : '真实' }}</span>
            <span class="tag">{{ r.resolution }}</span>
          </td>
          <td class="num">{{ r.cost_total_cny.toFixed(2) }}</td>
          <td class="muted">{{ r.updated }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
/* 勾选列压到最窄：它不该把 Run ID 挤到换行 */
.pick { width: 28px; text-align: center; padding: 2px 4px; }
.pick input { cursor: pointer; }
</style>
