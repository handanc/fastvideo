<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'

const props = defineProps({ runId: { type: String, required: true } })

const groups = ref([])
const error = ref('')
const msg = ref('')
const busy = ref({})   // gid -> 'frames' | 'clip' | 'delete'
let timer = null

async function load() {
  try {
    const d = await api.sceneGroups(props.runId)
    groups.value = d.groups || []
    error.value = ''
  } catch (e) {
    error.value = e.message
  }
}

// 生成是子进程异步的：点完轮询到产物出现（或 90 秒超时），别让用户盯着不动的页面猜
function pollUntil(gid, what, ms = 90000) {
  const t0 = Date.now()
  clearInterval(timer)
  timer = setInterval(async () => {
    await load()
    const g = groups.value.find((x) => x.group_id === gid)
    const done = what === 'clip' ? !!g?.clip : !!g?.first_frame && !!g?.last_frame
    if (done || Date.now() - t0 > ms) {
      clearInterval(timer)
      busy.value = { ...busy.value, [gid]: '' }
      if (!done) msg.value = `${gid}：${what === 'clip' ? '片段' : '首尾帧'}还没出来，看「事件」页签里的原因`
    }
  }, 3000)
}

async function setMode(g, mode) {
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, { mode })
    groups.value = d.groups || groups.value
    msg.value = ''
  } catch (e) { error.value = e.message }
}

async function pickCharImage(g, cid, rel) {
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, {
      char_images: { ...g.char_images, [cid]: rel },
    })
    groups.value = d.groups || groups.value
  } catch (e) { error.value = e.message }
}

async function toggleRef(g, kind, id) {
  const key = kind === 'char' ? 'char_refs' : 'scene_refs'
  const cur = g[key] || []
  const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]
  if (!next.length) { error.value = '至少保留一个参考，全去掉等于让模型自由发挥'; return }
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, { [key]: next })
    groups.value = d.groups || groups.value
  } catch (e) { error.value = e.message }
}

async function savePrompts(g) {
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, {
      prompt_first: g.prompt_first, prompt_last: g.prompt_last,
    })
    groups.value = d.groups || groups.value
    msg.value = `${g.group_id}：首尾帧提示词已保存`
  } catch (e) { error.value = e.message }
}

async function genFrames(g) {
  busy.value = { ...busy.value, [g.group_id]: 'frames' }
  error.value = ''; msg.value = ''
  try {
    await api.genSceneFrames(props.runId, g.group_id)
    await load()
    pollUntil(g.group_id, 'frames')
  } catch (e) { error.value = e.message; busy.value = { ...busy.value, [g.group_id]: '' } }
}

async function genClip(g) {
  busy.value = { ...busy.value, [g.group_id]: 'clip' }
  error.value = ''; msg.value = ''
  try {
    await api.genSceneClip(props.runId, g.group_id)
    pollUntil(g.group_id, 'clip', 600000)
  } catch (e) { error.value = e.message; busy.value = { ...busy.value, [g.group_id]: '' } }
}

async function delClip(g) {
  if (!window.confirm(`确定删除场景片段 ${g.group_id}？\n删除后该组镜头回到「待生成」，重出要再花一次钱。`)) return
  busy.value = { ...busy.value, [g.group_id]: 'delete' }
  try {
    await api.deleteSceneClip(props.runId, g.group_id)
    await load()
  } catch (e) { error.value = e.message }
  busy.value = { ...busy.value, [g.group_id]: '' }
}

onMounted(load)
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="card">
    <div class="cardHead">
      <h2>场景片段（wan3.0 场景级生成）</h2>
      <button class="btn sm" @click="load">⟳</button>
    </div>
    <div v-if="error" class="alert err">{{ error }}</div>
    <div v-if="msg" class="muted small">{{ msg }}</div>
    <p class="muted small">
      一个场景组 = 一次 wan3.0 调用（组内多镜头连贯、自带台词口型）。
      「角色+场景生成」发立绘/场景图/音色样本锁脸锁声；「首尾帧生成」锁起止构图，
      两者官方互斥、不能同发。删除片段后点「生成」只重出这一组。
    </p>

    <div v-for="g in groups" :key="g.group_id" class="sceneGroup">
      <div class="toolbar">
        <b>{{ g.group_id }}</b>
        <span class="muted small">
          {{ g.shot_ids.length }} 镜 · {{ g.duration }}s ·
          {{ g.has_dialogue ? '有对白' : '无对白' }} ·
          自动判定={{ g.auto_mode }}
        </span>
        <div class="spacer"></div>
        <select class="sm" :value="g.mode" @change="setMode(g, $event.target.value)">
          <option value="auto">自动（按规则）</option>
          <option value="reference">角色+场景生成</option>
          <option value="first_last">首尾帧生成</option>
        </select>
        <button v-if="g.clip" class="btn sm" :disabled="!!busy[g.group_id]" @click="delClip(g)">
          🗑 删除片段
        </button>
        <button class="btn sm primary" :disabled="!!busy[g.group_id]" @click="genClip(g)">
          {{ busy[g.group_id] === 'clip' ? '生成中…' : '生成' }}
        </button>
      </div>
      <div class="muted small">当前：{{ g.mode }} — {{ g.mode_reason }}</div>
      <video v-if="g.clip" :src="g.clip" controls style="max-width: 240px"></video>
      <div v-if="g.clip_stale" class="alert warn">片段已作废（上游改过），点「生成」重出</div>

      <!-- 参考形态：只允许在**已有**的图里选，不允许在这里改图 -->
      <div v-if="g.mode === 'reference'" class="modelBar">
        <div v-for="cid in g.characters" :key="cid" class="refRow">
          <label class="chk">
            <input type="checkbox" :checked="(g.char_refs || []).includes(cid)"
                   @change="toggleRef(g, 'char', cid)" />
            {{ cid }}
          </label>
          <template v-if="g.char_options && g.char_options[cid]">
            <img v-if="g.char_options[cid].portrait" :src="g.char_options[cid].portrait"
                 :class="{ picked: (g.char_images[cid] || g.char_options[cid].portrait_rel) === g.char_options[cid].portrait_rel }"
                 title="定妆照" @click="pickCharImage(g, cid, g.char_options[cid].portrait_rel)" />
            <img v-for="(u, i) in (g.char_options[cid].candidates || [])" :key="u" :src="u"
                 :class="{ picked: g.char_images[cid] === (g.char_options[cid].candidates_rel || [])[i] }"
                 title="候选立绘" @click="pickCharImage(g, cid, (g.char_options[cid].candidates_rel || [])[i])" />
          </template>
        </div>
        <div v-for="sid in g.scene_ids" :key="sid" class="refRow">
          <label class="chk">
            <input type="checkbox" :checked="(g.scene_refs || []).includes(sid)"
                   @change="toggleRef(g, 'scene', sid)" />
            {{ sid }}（空景背景）
          </label>
          <img v-if="g.scene_options && g.scene_options[sid]" :src="g.scene_options[sid]" class="picked" />
        </div>
      </div>

      <!-- 首尾帧形态：提示词可编辑，先出图再出片段 -->
      <div v-else class="modelBar">
        <label class="field"><span>首帧提示词（可改）</span>
          <textarea v-model="g.prompt_first" rows="3"></textarea>
        </label>
        <label class="field"><span>尾帧提示词（可改）</span>
          <textarea v-model="g.prompt_last" rows="3"></textarea>
        </label>
        <div class="toolbar">
          <button class="btn sm" :disabled="!!busy[g.group_id]" @click="savePrompts(g)">保存提示词</button>
          <button class="btn sm" :disabled="!!busy[g.group_id]" @click="genFrames(g)">
            {{ busy[g.group_id] === 'frames' ? '生成中…' : '生成首尾帧' }}
          </button>
        </div>
        <div class="refRow">
          <img v-if="g.first_frame" :src="g.first_frame" title="首帧" />
          <img v-if="g.last_frame" :src="g.last_frame" title="尾帧" />
          <span v-if="!g.first_frame" class="muted small">还没有首帧 → 点「生成首尾帧」</span>
        </div>
      </div>
    </div>
    <p v-if="!groups.length && !error" class="muted">还没有剧本，或本 run 不适用场景级生成。</p>
  </div>
</template>

<style scoped>
.sceneGroup { border: 1px solid var(--line, #333); border-radius: 8px; padding: 8px; margin: 8px 0; }
.refRow { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 4px 0; }
.refRow img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: pointer; opacity: .55; }
.refRow img.picked { opacity: 1; outline: 2px solid var(--accent, #4a9); }
.chk { display: flex; gap: 4px; align-items: center; }
.field { display: flex; flex-direction: column; gap: 2px; min-width: 260px; }
</style>
