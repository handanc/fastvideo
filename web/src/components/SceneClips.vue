<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'
import { sceneLabel } from '../format'

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

const originals = ref({})   // gid -> 上次保存/加载时的提示词，用来判断"改了没保存"

async function load() {
  try {
    const d = await api.sceneGroups(props.runId)
    groups.value = d.groups || []
    for (const g of groups.value) originals.value[g.group_id] = g.prompt || ''
    error.value = ''
  } catch (e) {
    error.value = e.message
  }
}

// 提示词是这一页唯一"直接决定成片"的东西，所以它的编辑体验要单独做好：
// 改了没保存就去点生成 = 白等一次付费生成，必须在点生成前自动落盘。
function dirty(g) { return (g.prompt || '') !== (originals.value[g.group_id] || '') }
function legendCounts(g) {
  const l = g.ref_legend || []
  return {
    imgs: l.filter((x) => x.includes('立绘') || x.includes('空景')).length,
    auds: l.filter((x) => x.includes('音色')).length,
  }
}
async function autoSave(g) {
  if (!dirty(g)) return          // 没改：不涨版本、不作废、不打扰后端
  await savePrompts(g)
}

async function copyPrompt(g) {
  try { await navigator.clipboard.writeText(g.prompt || ''); flash('提示词已复制') }
  catch (e) { window.prompt('复制不了，请手动选中：', g.prompt || '') }
}
async function resetPrompt(g) {
  if (!window.confirm(`把 ${g.group_id} 的提示词恢复成机器按剧本拼的默认版？
你的人工改写会被丢弃。`)) return
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, { prompt: '' })
    groups.value = d.groups || groups.value
    for (const x of groups.value) originals.value[x.group_id] = x.prompt || ''
    flash(`${sceneLabel(g.group_id)}：已恢复默认提示词`)
  } catch (e) { error.value = e.message }
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
        flash(`✔ ${sceneLabel(gid)} ${what === 'clip' ? '场景视频片段' : '首尾帧'}已生成`)
        emit('changed')
      } else {
        msg.value = `${sceneLabel(gid)}：${what === 'clip' ? '片段' : '首尾帧'}还没出来 —— `
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
    for (const x of groups.value) originals.value[x.group_id] = x.prompt || ''
    flash(`${sceneLabel(g.group_id)}：提示词已保存`)
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
    flash(`${sceneLabel(g.group_id)} 已有视频片段，无需重复生成；要重出请先点「🗑 删除片段」`)
    return
  }
  if (dirty(g)) {
    // 改了提示词没保存就点生成：先落盘再提交，否则这次花的钱买的还是旧提示词的片子
    await savePrompts(g)
    if (dirty(g)) return   // 保存失败就别白花一次生成的钱
  }
  busy.value = { ...busy.value, [g.group_id]: 'clip' }
  try {
    await api.genSceneClip(props.runId, g.group_id)
    pollUntil(g.group_id, 'clip', 900000)
  } catch (e) { error.value = e.message; busy.value = { ...busy.value, [g.group_id]: '' } }
}

async function delClip(g) {
  if (!window.confirm(`确定删除 ${sceneLabel(g.group_id)} 的视频片段？\n删除后该组回到「待生成」，重出要再花一次钱。`)) return
  busy.value = { ...busy.value, [g.group_id]: 'delete' }
  try {
    await api.deleteSceneClip(props.runId, g.group_id)
    await load()
    emit('changed')
  } catch (e) { error.value = e.message }
  busy.value = { ...busy.value, [g.group_id]: '' }
}

onMounted(load)
onUnmounted(() => { clearInterval(timer); clearTimeout(noticeTimer) })
</script>

<template>
  <div class="card">
    <div class="cardHead">
      <button class="btn sm" @click="load">⟳ 刷新</button>
    </div>
    <div v-if="error" class="alert err">{{ error }}</div>
    <div v-if="notice" class="notice">{{ notice }}</div>
    <div v-if="msg" class="muted small">{{ msg }}</div>

    <div v-for="(g, i) in groups" :key="g.group_id" class="sceneGroup">
      <div class="toolbar">
        <b>{{ sceneLabel(g.group_id) }}</b>
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


      <!-- 参考形态：只允许在**已有**的图里选，不允许在这里改图 -->
      <!-- 编辑区跟着**用户的选择**走，不跟着"生效形态"走：选了首尾帧却只显示参考素材，
           等于把「生成首尾帧」按钮藏起来，用户无路可走（2026-10-14 实测的死角）。 -->
      <div v-if="g.mode === 'reference' && g.override_mode !== 'first_last'" class="modelBar">
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
      <div v-if="g.mode === 'first_last' || g.override_mode === 'first_last'" class="modelBar">
        <div v-if="!g.first_frame" class="alert warn">
          这一组还没有首帧：<b>先点下面的「生成首尾帧」</b>（只花 t2i 的钱）。
          在出帧之前点「生成」会按<b>参考形态</b>走 —— 官方规定首尾帧形态必须先有帧，
          没有帧就发不出 <code>first_frame</code>。
        </div>
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

      <!-- 提示词：这一页唯一"直接决定成片"的输入，所以给它最好的编辑体验 -->
      <div class="promptBox">
        <div class="promptHead">
          <b>提交给 wan3.0 的提示词</b>
          <span class="muted small">
            {{ (g.prompt || '').length }} 字 · {{ g.shots.length }} 镜 · {{ g.duration }}s ·
            参考图 {{ legendCounts(g).imgs }} · 音色 {{ legendCounts(g).auds }}
          </span>
          <span v-if="dirty(g)" class="badge warn small">已改 · 未保存</span>
          <span v-else-if="g.prompt_edited" class="badge small">人工改写版</span>
          <span v-else class="badge ok small">机器默认</span>
          <div class="spacer"></div>
          <button class="btn sm" @click="copyPrompt(g)">复制</button>
          <button class="btn sm" :disabled="!g.prompt_edited && !dirty(g)" @click="resetPrompt(g)">
            恢复默认
          </button>
        </div>
        <div v-if="(g.ref_legend || []).length" class="legend">
          <span v-for="l in g.ref_legend" :key="l" class="lg">{{ l }}</span>
        </div>
        <div v-else class="muted small legend">
          本组没有可用参考素材（角色图/场景图缺失，或形态是首尾帧）——提示词里不会出现 图N/音频N。
        </div>
        <!-- 单击即改、点到别处自动保存：这段提示词是唯一直接决定成片的东西，
             多一个"保存"按钮就多一次"改了没保存 → 白花一次生成钱"的机会 -->
        <textarea class="promptEdit" v-model="g.prompt" rows="16" spellcheck="false"
                  @blur="autoSave(g)" @keydown.ctrl.enter="autoSave(g)"></textarea>
        <div class="muted small promptFoot">
          一个板块一行（【总体】【参考素材】【人物】… 第N个镜头 各占一行），改哪段直接改哪行；
          <b>点到别处即自动保存</b>。点「生成」也会先保存再提交，不会拿旧提示词白花一次钱。
          改坏了点「恢复默认」。
        </div>
      </div>
    </div>
    <p v-if="!groups.length && !error" class="muted">还没有剧本（先在「剧本」页生成剧本）。</p>
  </div>
</template>

<style scoped>
.sceneGroup { border: 1px solid var(--line, #333); border-radius: 8px; padding: 8px; margin: 8px 0; }
.scenelabel { font-weight: 600; }
.promptBox { margin-top: 10px; border-top: 1px dashed var(--line, #333); padding-top: 8px; }
.promptHead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.lg {
  font-size: 11.5px; padding: 2px 7px; border-radius: 10px;
  background: var(--panel, #f3f4f6); border: 1px solid var(--line, #ddd); color: var(--muted, #666);
}
.promptEdit {
  width: 100%; font-family: var(--mono, ui-monospace, Consolas, monospace);
  font-size: 12.5px; line-height: 1.75; padding: 8px 10px; resize: vertical;
}
.promptFoot { margin-top: 6px; }
.notice {
  position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
  background: var(--panel, #1c1f24); color: var(--fg, #eee);
  border: 1px solid var(--line, #333); border-left: 3px solid var(--accent, #4a9);
  border-radius: 6px; padding: 8px 14px; font-size: 13px; z-index: 50;
  max-width: min(560px, calc(100vw - 32px));
}
.summary { margin: 2px 0; }
.refRow { display: flex; align-items: flex-end; gap: 6px; flex-wrap: wrap; margin: 4px 0; }
/* 缩略图按**原图比例**显示：只锁高度、宽度交给浏览器按固有比例算。
   原来是 56×56 + object-fit:cover —— 9:16 的定妆照被裁成正方形，人只看到胸像，
   而这张图正是发去锁脸的参考素材：被裁掉的构图信息（全身/头身比/服装下摆）恰恰是
   判断"这张能不能用"的依据。场景图若是横构图，同样不该被压成方块。 */
.refRow img {
  height: 88px; width: auto; max-width: 100%;
  object-fit: contain; border-radius: 4px; cursor: pointer; opacity: .55;
  background: var(--panel, #f3f4f6);
}
.refRow img.picked { opacity: 1; outline: 2px solid var(--accent, #4a9); }
.chk { display: flex; gap: 4px; align-items: center; }
.field { display: flex; flex-direction: column; gap: 2px; min-width: 260px; }
.field.wide { width: 100%; }
.field.wide textarea { width: 100%; font-size: 12px; }
</style>
