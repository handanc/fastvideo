<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { statusOf } from '../status'

const props = defineProps({ runId: { type: String, required: true } })
const emit = defineEmits(['back'])

const detail = ref(null)
const error = ref('')
const notice = ref('')
const showReject = ref(false)
const rejectMsg = ref('')
let timer = null

async function load() {
  try {
    detail.value = await api.runDetail(props.runId)
    error.value = ''
  } catch (e) {
    error.value = e.message
  }
}

function schedulePolling() {
  clearInterval(timer)
  const st = detail.value?.status
  // 运行中/待审片时 3 秒轮询跟进；结束后停止
  if (st === 'running' || st === 'paused' || st === 'idle') {
    timer = setInterval(load, 3000)
  }
}

async function decide(decision) {
  notice.value = ''
  try {
    await api.resume(props.runId, decision)
    notice.value = `已提交「${decision === 'approve' ? '通过' : '驳回'}」，后台续跑中…`
    showReject.value = false
    rejectMsg.value = ''
    await load()
    schedulePolling()
  } catch (e) {
    error.value = e.message
  }
}

const base = computed(() => detail.value?.meta?.base_info || {})
const reviewBadge = computed(() => {
  const r = detail.value?.review
  if (!r) return null
  const fb = r.feedback ? ' · ' + String(r.feedback).slice(0, 60) : ''
  if (r.review_error) return { cls: 'warn', label: `⚠ 终审异常，已放行${fb}` }
  if (r.passed === false) return { cls: 'err', label: `✘ 终审未通过 (${Number(r.score || 0).toFixed(2)})${fb}` }
  return { cls: 'ok', label: `✔ 终审通过 (${Number(r.score || 0).toFixed(2)})` }
})
const images = computed(() => [
  ...(detail.value?.images?.assets || []).map((u) => ({ url: u, cap: 'assets' })),
  ...(detail.value?.images?.storyboard || []).map((u) => ({ url: u, cap: 'storyboard' })),
])
const st = computed(() => statusOf(detail.value?.status))

watch(detail, schedulePolling)
onMounted(load)
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div>
    <div class="toolbar">
      <button class="btn sm" @click="emit('back')">← 返回列表</button>
      <span class="mono muted">{{ props.runId }}</span>
      <span v-if="detail" class="badge" :class="st.cls">{{ st.label }}</span>
      <div class="spacer"></div>
      <button class="btn sm" @click="load">⟳ 刷新</button>
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>
    <div v-if="notice" class="alert ok">{{ notice }}</div>
    <p v-if="!detail && !error" class="muted">加载中…</p>

    <template v-if="detail">
      <!-- 概要 + 审片操作 -->
      <div class="card">
        <h2>{{ base.topic || '(无主题)' }}</h2>
        <p class="muted">
          风格：{{ base.style || '—' }} · 时长：{{ base.target_duration_sec || '—' }}s ·
          画幅：{{ base.aspect_ratio || '—' }} · 档位：{{ detail.meta.quality }} ·
          {{ detail.meta.fake ? 'Fake 模式' : '真实生成' }}
          <span v-if="detail.meta.fake" class="badge warn">占位演示产物</span>
          <span v-if="reviewBadge" class="badge" :class="reviewBadge.cls">{{ reviewBadge.label }}</span>
        </p>

        <div v-if="detail.status !== 'done'" class="toolbar" style="margin-top: 8px">
          <button class="btn primary sm" @click="decide('approve')">✔ 审片通过并续跑</button>
          <button class="btn danger sm" @click="showReject = !showReject">✘ 驳回…</button>
          <span class="muted">未暂停时点击「通过」等效于从中断处恢复运行</span>
        </div>
        <div v-if="showReject" class="toolbar">
          <input
            v-model="rejectMsg" type="text" style="flex: 1"
            placeholder="驳回意见（可选），例如：第三幕节奏太慢" @keyup.enter="decide(`reject:${rejectMsg}`)"
          />
          <button class="btn danger sm" @click="decide(`reject:${rejectMsg}`)">提交驳回</button>
        </div>
      </div>

      <div class="grid2">
        <!-- 事件流 -->
        <div class="card">
          <h2>事件流 <span class="muted">(最近 {{ detail.events.length }} 条)</span></h2>
          <div v-if="detail.events.length" class="timeline">
            <div v-for="(e, i) in detail.events" :key="i" class="ev"
                 :class="{ fail: /fail|error|gate|missing/i.test(String(e.status || '')) }">
              <span class="ts">{{ e.ts }}</span>
              <span>{{ e.node }}:{{ e.status }}<template v-if="e.name"> · {{ e.name }}</template></span>
            </div>
          </div>
          <p v-else class="muted">暂无事件（任务可能尚未启动）</p>

          <h3>成本 · 合计 ¥{{ detail.cost.total_cny.toFixed(2) }}</h3>
          <table v-if="detail.cost.items.length" class="tbl">
            <thead><tr><th>能力</th><th>模型</th><th class="num">金额(¥)</th></tr></thead>
            <tbody>
              <tr v-for="(c, i) in detail.cost.items" :key="i">
                <td>{{ c.capability }}</td>
                <td class="muted">{{ c.model }}</td>
                <td class="num">{{ Number(c.amount_cny || 0).toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted">暂无成本记录</p>
        </div>

        <!-- 成片 -->
        <div class="card">
          <h2>成片</h2>
          <video v-if="detail.final_video" class="final" :src="detail.final_video" controls></video>
          <p v-else class="muted">尚未生成成片（final/final.mp4）</p>

          <template v-if="detail.videos.clips?.length || detail.videos.talk?.length">
            <h3>镜头片段（clips {{ detail.videos.clips.length }} · talk {{ detail.videos.talk.length }}）</h3>
            <div class="vgrid">
              <video v-for="v in [...detail.videos.clips, ...detail.videos.talk].slice(0, 6)"
                     :key="v" :src="v" controls muted preload="metadata"></video>
            </div>
          </template>
        </div>
      </div>

      <!-- 产物画廊 -->
      <div class="card">
        <h2>产物画廊 <span class="muted">(角色/场景资产 + 分镜首帧，共 {{ images.length }} 张)</span></h2>
        <div v-if="images.length" class="gallery">
          <figure v-for="img in images" :key="img.url">
            <img :src="img.url" :alt="img.cap" loading="lazy" />
            <figcaption>{{ img.url.split('/').pop() }}</figcaption>
          </figure>
        </div>
        <p v-else class="muted">暂无图片产物</p>
      </div>
    </template>
  </div>
</template>
