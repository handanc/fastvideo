<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'

const props = defineProps({ runId: { type: String, required: true } })
// 产物增删后通知父组件刷新看板计数/步骤条（父层再拉 board/steps）
const emit = defineEmits(['changed'])

const groups = ref([])
const error = ref('')
const msg = ref('')
const busy = ref({})   // gid -> 'frames' | 'clip' | 'delete'
const notice = ref('')  // 2 秒自动消失的轻提示（"这段已经有了"这类，不该占着错误条）
let timer = null
let noticeTimer = null

function flash(text, ms = 2000) {
  notice.value = text
  clearTimeout(noticeTimer)
  noticeTimer = setTimeout(() => { notice.value = '' }, ms)
}

// 镜头文字就地改（画面描述/镜头语言）：与「剧本」页同一套 PUT /script 语义 ——
// 提交后服务端做校验 + 血缘作废，改完不删图也会重画。
const editing = ref(null)  // { gid, shot_id, field, text, original }

const typeLabel = { dialogue: '对白', action: '动作', scenery: '空镜' }

async function load() {
  try {
    const d = await api.sceneGroups(props.runId)
    groups.value = d.groups || []
    error.value = ''
  } catch (e) {
    error.value = e.message
  }
}

// 生成是子进程异步的：点完轮询到产物出现（或超时），别让用户盯着不动的页面猜
function pollUntil(gid, what, ms = 90000) {
  const t0 = Date.now()
  clearInterval(timer)
  timer = setInterval(async () => {
    await load()
    const g = groups.value.find((x) => x.group_id === gid)
    // **已作废的片段不算完成**：重出过程中文件还在盘上，只看"有没有文件"会秒退轮询，
    // 于是界面从头到尾没反应（用户报的"点了生成毫无反应"正是这条）。
    const done = what === 'clip'
      ? !!g?.clip && !g?.clip_stale
      : !!g?.first_frame && !!g?.last_frame
    if (done || Date.now() - t0 > ms) {
      clearInterval(timer)
      busy.value = { ...busy.value, [gid]: '' }
      if (done) {
        flash(`✔ ${gid} ${what === 'clip' ? '场景视频片段' : '首尾帧'}已生成`)
        emit('changed')
      } else {
        msg.value = `${gid}：${what === 'clip' ? '片段' : '首尾帧'}还没出来 —— `
          + '该组卡片下方的红字/「事件」页签里有原因（缺角色图/场景图、欠费、额度用尽是最常见的几种）'
      }
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
      prompt_first: g.prompt_first, prompt_last: g.prompt_last, prompt: g.prompt,
    })
    groups.value = d.groups || groups.value
    msg.value = `${g.group_id}：提示词已保存，点「生成」按新提示词出片`
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
  error.value = ''; msg.value = ''
  // 已经有片段且不作废 → 不重复花钱，只告诉用户"这段已经有了"（2 秒后自动消失）。
  // 想重出就点「🗑 删除片段」再点生成：删了才会重新提交一次 wan3.0（一段 9 秒 720P 是 ¥5.4 量级）。
  if (g.clip && !g.clip_stale) {
    flash(`场景 ${g.group_id} 已有视频片段，无需重复生成；要重出请先点「🗑 删除片段」`)
    return
  }
  busy.value = { ...busy.value, [g.group_id]: 'clip' }
  try {
    await api.genSceneClip(props.runId, g.group_id)
    pollUntil(g.group_id, 'clip', 900000)
  } catch (e) { error.value = e.message; busy.value = { ...busy.value, [g.group_id]: '' } }
}

async function delClip(g) {
  if (!window.confirm(`确定删除场景片段 ${g.group_id}？\n删除后该组回到「待生成」，重出要再花一次钱。`)) return
  busy.value = { ...busy.value, [g.group_id]: 'delete' }
  try {
    await api.deleteSceneClip(props.runId, g.group_id)
    await load()
    emit('changed')
  } catch (e) { error.value = e.message }
  busy.value = { ...busy.value, [g.group_id]: '' }
}

// ── 镜头文字：单击进入编辑，blur/Ctrl+Enter 提交，Esc 取消 ──────────────
function startEdit(g, sh, field, current) {
  editing.value = { gid: g.group_id, shot_id: sh.shot_id, field, text: current || '', original: current || '' }
}
function cancelEdit() { editing.value = null }
function edit(g, sh, field) {
  const e = editing.value
  return !!e && e.gid === g.group_id && e.shot_id === sh.shot_id && e.field === field
}

async function commitEdit() {
  const e = editing.value
  editing.value = null
  if (!e) return
  const text = e.text.trim()
  if (text === (e.original || '').trim()) return  // 没改：不涨版本、不作废、不打扰后端
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(props.runId)}/script`)
    if (!res.ok) throw new Error(`读取剧本失败 HTTP ${res.status}`)
    const script = await res.json()
    const shots = (script.scenes || []).flatMap((s) => s.shots)
    const target = shots.find((x) => x.shot_id === e.shot_id)
    if (!target) throw new Error(`剧本里找不到 ${e.shot_id}`)
    target[e.field] = text
    const r = await api.putScript(props.runId, script)
    msg.value = `✔ 已保存「${e.shot_id}」的${e.field === 'action' ? '画面描述' : '镜头语言'}`
      + `（剧本 v${r.version}），作废 ${r.invalidated?.length || 0} 个产物；点「生成」按新描述重出`
    await load()
    emit('changed')
  } catch (err) { error.value = err.message }
}

function win(g, sid) {
  const w = (g.windows || {})[sid] || []
  return w.length === 2 ? `${w[0]}–${w[1]}s` : ''
}

onMounted(load)
onUnmounted(() => { clearInterval(timer); clearTimeout(noticeTimer) })
</script>

<template>
  <div class="card">
    <div class="cardHead">
      <h2>场景视频（wan3.0 场景级生成）</h2>
      <button class="btn sm" @click="load">⟳</button>
    </div>
    <div v-if="error" class="alert err">{{ error }}</div>
    <div v-if="notice" class="notice">{{ notice }}</div>
    <div v-if="msg" class="muted small">{{ msg }}</div>

    <div v-for="(g, i) in groups" :key="g.group_id" class="sceneGroup">
      <div class="toolbar">
        <b>场景{{ i + 1 }} <span class="muted small">{{ g.group_id }}</span></b>
        <span v-if="g.scene_label" class="scenelabel">{{ g.scene_label }}</span>
        <span class="badge" :class="g.clip_stale ? 'warn' : (g.clip ? 'ok' : '')">
          {{ g.clip_stale ? '已作废·待重出' : (g.clip ? '已生成' : '待生成') }}
        </span>
        <span class="muted small">
          {{ g.shots.length }} 镜 · {{ g.duration }}s ·
          {{ g.has_dialogue ? '有对白' : '无对白' }} ·
          自动判定={{ g.auto_mode === 'reference' ? '角色+场景' : (g.auto_mode === 'first_last' ? '首尾帧' : g.auto_mode) }}
        </span>
        <div class="spacer"></div>
        <!-- override_mode = 用户选的；mode = 实际生效的（下一行显示）。两者不一致时说明
             物理前提不满足，原因就写在下面那行里。 -->
        <select class="sm" :value="g.override_mode || 'auto'" @change="setMode(g, $event.target.value)">
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

      <div class="muted small">当前形态：{{ g.mode === 'reference' ? '角色+场景生成' : (g.mode === 'first_last' ? '首尾帧生成' : '无法生成') }} — {{ g.mode_reason }}</div>
      <div v-if="g.scene_summary" class="muted small summary">{{ g.scene_summary }}</div>
      <div v-if="g.mode === 'none'" class="alert err">
        本组缺输入：既没有可用的角色图/场景图，也没有首帧 → 去「角色图」「场景图」页补图，
        或点「生成首尾帧」先出帧。
      </div>
      <div v-else-if="g.mode === 'reference' && !g.has_references" class="alert err">
        判定为角色+场景生成，但盘上找不到本组的角色图/场景图文件（见「角色图」「场景图」页）。
      </div>
      <video v-if="g.clip" :src="g.clip" controls style="max-width: 260px"></video>
      <div v-if="g.clip_stale" class="alert warn">片段已作废（上游改过），点「生成」重出</div>

      <!-- 组内分镜：这一段戏里按顺序发生什么（文字可点改，改完即作废本组片段） -->
      <ol class="shots">
        <li v-for="(sh, j) in g.shots" :key="sh.shot_id">
          <span class="mono">分镜{{ j + 1 }}</span>
          <span class="muted small">{{ win(g, sh.shot_id) }} · {{ typeLabel[sh.type] || sh.type }} · {{ sh.duration_sec }}s</span>
          <span v-if="sh.characters.length" class="muted small">· {{ sh.characters.join('、') }}</span>
          <template v-if="!edit(g, sh, 'action')">
            <span class="action editable" title="点击修改画面描述（本组提示词的原料）"
                  @click="startEdit(g, sh, 'action', sh.action)">{{ sh.action || '（点击写画面描述）' }}</span>
          </template>
          <textarea v-else class="inline-edit" rows="2" autofocus spellcheck="false"
                    v-model="editing.text"
                    @blur="commitEdit"
                    @keydown.esc.prevent="cancelEdit"
                    @keydown.ctrl.enter="commitEdit"></textarea>
          <div v-if="!edit(g, sh, 'camera')" class="muted small">镜头语言：
            <span class="editable" title="点击修改镜头语言"
                  @click="startEdit(g, sh, 'camera', sh.camera)">{{ sh.camera || '（点击写）' }}</span>
          </div>
          <input v-else class="inline-edit one" autofocus v-model="editing.text"
                 @blur="commitEdit"
                 @keydown.esc.prevent="cancelEdit"
                 @keydown.enter="commitEdit" />
          <div v-for="(d, k) in sh.dialogues" :key="k" class="line"><b>{{ d.who }}</b>：{{ d.text }}</div>
        </li>
      </ol>

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
            <img v-for="(u, k) in (g.char_options[cid].candidates || [])" :key="u" :src="u"
                 :class="{ picked: g.char_images[cid] === (g.char_options[cid].candidates_rel || [])[k] }"
                 title="候选立绘" @click="pickCharImage(g, cid, (g.char_options[cid].candidates_rel || [])[k])" />
            <span v-if="!g.char_options[cid].portrait" class="muted small">缺定妆照 → 去「角色图」页补</span>
          </template>
          <span v-else class="muted small">缺定妆照 → 去「角色图」页补</span>
        </div>
        <div v-for="sid in g.scene_ids" :key="sid" class="refRow">
          <label class="chk">
            <input type="checkbox" :checked="(g.scene_refs || []).includes(sid)"
                   @change="toggleRef(g, 'scene', sid)" />
            {{ sid }}（空景背景，图中无人）
          </label>
          <img v-if="g.scene_options && g.scene_options[sid]" :src="g.scene_options[sid]" class="picked" />
        </div>
      </div>

      <!-- 首尾帧形态：先出帧，再按帧出片 -->
      <div v-else-if="g.mode === 'first_last'" class="modelBar">
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

      <!-- 视频提示词：两种形态都给人看、给人改（机器拼的模板是默认值，不是上限） -->
      <label class="field wide"><span>本段视频的提示词（可改，点「生成」即按它出片）</span>
        <textarea v-model="g.prompt" rows="6" spellcheck="false"></textarea>
      </label>
      <div class="toolbar">
        <button class="btn sm" :disabled="!!busy[g.group_id]" @click="savePrompts(g)">保存提示词</button>
      </div>
    </div>
    <p v-if="!groups.length && !error" class="muted">还没有剧本（先在「剧本」页生成剧本）。</p>
  </div>
</template>

<style scoped>
.sceneGroup { border: 1px solid var(--line, #333); border-radius: 8px; padding: 8px; margin: 8px 0; }
.scenelabel { font-weight: 600; }
.notice {
  position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
  background: var(--panel, #1c1f24); color: var(--fg, #eee);
  border: 1px solid var(--line, #333); border-left: 3px solid var(--accent, #4a9);
  border-radius: 6px; padding: 8px 14px; font-size: 13px; z-index: 50;
  max-width: min(560px, calc(100vw - 32px));
}
.summary { margin: 2px 0; }
.refRow { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 4px 0; }
.refRow img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: pointer; opacity: .55; }
.refRow img.picked { opacity: 1; outline: 2px solid var(--accent, #4a9); }
.chk { display: flex; gap: 4px; align-items: center; }
.field { display: flex; flex-direction: column; gap: 2px; min-width: 260px; }
.field.wide { width: 100%; }
.field.wide textarea { width: 100%; font-size: 12px; }
.shots { margin: 6px 0; padding-left: 20px; }
.shots li { margin: 4px 0; }
.shots .action { display: block; }
</style>
