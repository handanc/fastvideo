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

// ── 参考角色的增删 ────────────────────────────────────────────
// 自动名单（`g.characters`）现在包含"只入画不说话"的人（与 m3 首帧同一份 implied cast 判据），
// 这里再让人手动加减。**候选池 = 剧本角色 ∩ 已有角色图**：没有图的人加了也发不出去
// （编号由存活名单现算，会静默少一张），不如根本不让选。
const MAX_IMG = 10  // 官方 reference_image 上限（与 core.scenegroup.MAX_REF_IMAGES 同值）

function castMap(g) {
  const m = {}
  for (const o of g.cast_options || []) m[o.id] = o
  return m
}

function shownChars(g) {
  const seen = []
  for (const cid of [...(g.characters || []), ...(g.char_refs || [])]) {
    if (cid && !seen.includes(cid)) seen.push(cid)
  }
  return seen
}

// 素材信息两个来源合并：自动名单里的人有完整候选（char_options），
// 手动加进来的"场外"角色只有 cast_options 的基本信息 —— 不给回落就会出现"缺定妆照"的假告警。
function charOpt(g, cid) {
  const base = (g.char_options && g.char_options[cid]) || {}
  const o = castMap(g)[cid] || {}
  return {
    portrait: base.portrait || o.portrait || null,
    portrait_rel: base.portrait_rel || o.portrait_rel || `assets/char_${cid}.png`,
    candidates: base.candidates || [],
    candidates_rel: base.candidates_rel || [],
  }
}

function charName(g, cid) {
  const o = castMap(g)[cid]
  return o ? o.name : cid
}

// 添加角色：弹窗里只列"这一组还没选 + 已有角色图"的人（没图的选了也发不出去）
const addFor = ref(null)
const addPicked = ref([])

function openAdd(g) {
  addPicked.value = []
  addFor.value = g
}

async function confirmAdd() {
  const g = addFor.value
  addFor.value = null
  if (!g || !addPicked.value.length) return
  const next = [...(g.char_refs || []), ...addPicked.value]
  if (next.length > MAX_IMG - 1) {
    error.value = `参考图共 ${next.length + 1} 张（含场景背景），超过官方上限 ${MAX_IMG} 张：先去掉几个人`
    return
  }
  try {
    const d = await api.patchSceneGroup(props.runId, g.group_id, { char_refs: next })
    groups.value = d.groups || groups.value
    flash(`${sceneLabel(g.group_id)}：已添加 ${addPicked.value.length} 个角色`)
  } catch (e) { error.value = e.message }
}

function addable(g) {
  const shown = shownChars(g)
  return (g.cast_options || []).filter((o) => o.portrait && !shown.includes(o.id))
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

async function restoreClip(g) {
  // 「恢复」= 回到**最原始**那段（`.orig` 快照）。没被改过就没有原点 → 弹一句提示，
  // 而不是报错：这是正常回答，红字会让人以为坏了什么。
  busy.value = { ...busy.value, [g.group_id]: 'restore' }
  try {
    await api.restoreArtifact(props.runId, 'clips', `scene_${g.group_id}.mp4`)
    await load()
    emit('changed')
    flash(`${sceneLabel(g.group_id)}：已恢复成最原始的片段（受影响的下游已作废）`)
  } catch (e) {
    if (/没有更早|还没被改动过/.test(e.message)) flash('无原始视频片段')
    else error.value = e.message
  }
  busy.value = { ...busy.value, [g.group_id]: '' }
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
      <div v-if="g.clip_stale" class="alert warn">片段已作废（上游改过），点「生成」重出</div>


      <!-- 参考形态：只允许在**已有**的图里选，不允许在这里改图 -->
      <!-- 编辑区跟着**用户的选择**走，不跟着"生效形态"走：选了首尾帧却只显示参考素材，
           等于把「生成首尾帧」按钮藏起来，用户无路可走（2026-10-14 实测的死角）。 -->
      <!-- 参考素材：这一段的输入。图块从左到右、从上到下排；场景图只能有一张，排在最后。
           角色名放在图**下面**（放上面时人和图对不上行）。取消勾选框改成每块一颗 🗑：
           勾选框+缩略图+候选图挤在一行时，人分不清"点图是换脸还是选它"。 -->
      <div v-if="g.mode === 'reference' && g.override_mode !== 'first_last'" class="refStrip">
        <figure v-for="cid in shownChars(g)" :key="cid" class="refTile">
          <img v-if="charOpt(g, cid).portrait" :src="charOpt(g, cid).portrait"
               :alt="charName(g, cid)" loading="lazy" />
          <div v-else class="missing sm">缺 char_{{ cid }}.png</div>
          <figcaption>{{ charName(g, cid) }}</figcaption>
          <button class="upl del tiny" title="从这一段的参考素材里去掉这个人（不改剧本的在场）"
                  @click="toggleRef(g, 'char', cid)">🗑 删除</button>
        </figure>

        <figure v-for="sid in g.scene_ids" :key="sid" class="refTile">
          <img v-if="g.scene_options && g.scene_options[sid]" :src="g.scene_options[sid]"
               :alt="sid" loading="lazy" />
          <div v-else class="missing sm">缺 scene_{{ sid }}.png</div>
          <figcaption>{{ sceneLabel(sid) }}（空景）</figcaption>
          <button class="upl del tiny" title="去掉背景参考图（一般不建议：空间就没有依据了）"
                  @click="toggleRef(g, 'scene', sid)">🗑 删除</button>
        </figure>

        <button class="addTile" title="添加角色（只列已有角色图的）" @click="openAdd(g)">＋</button>
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
          <b>图生视频提示词</b>
          <span class="muted small">
            {{ (g.prompt || '').length }} 字 · {{ g.shots.length }} 镜 · {{ g.duration }}s ·
            参考图 {{ legendCounts(g).imgs }} · 音色 {{ legendCounts(g).auds }}
          </span>
          <span v-if="dirty(g)" class="badge warn small">已改 · 未保存</span>
          <span v-else-if="g.prompt_edited" class="badge small">人工改写版</span>
          <span v-else class="badge ok small">机器默认</span>
          <div class="spacer"></div>
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
      </div>

      <!-- 三个动作按钮：放在提示词下面、片段上面 —— 改完提示词紧接着就是"提交/撤回"，
           再往下就是这次提交买回来的东西。 -->
      <div class="toolbar">
        <button class="btn sm primary" :disabled="!!busy[g.group_id]" @click="genClip(g)">
          {{ busy[g.group_id] === 'clip' ? '生成中…' : '生成' }}
        </button>
        <button v-if="g.clip" class="btn sm" :disabled="!!busy[g.group_id]" @click="delClip(g)">
          🗑 删除片段
        </button>
        <button class="btn sm" :disabled="!!busy[g.group_id]" title="回到最原始那一段（不是上一版）"
                @click="restoreClip(g)">
          {{ busy[g.group_id] === 'restore' ? '恢复中…' : '↩ 恢复' }}
        </button>
      </div>
      <video v-if="g.clip" :src="g.clip" controls style="max-width: 320px"></video>
    </div>
    <Teleport to="body">
      <div v-if="addFor" class="modal-mask" @click.self="addFor = null">
        <div class="modal" role="dialog" aria-modal="true">
          <h3>添加角色 · {{ sceneLabel(addFor.group_id) }}</h3>
          <p class="muted small">
            只影响这一段的参考素材，不改剧本的「在场」；只能选<b>已有角色图</b>的人
            （没有图就发不出去，等于没选）。
          </p>
          <label v-for="o in addable(addFor)" :key="o.id" class="check">
            <input v-model="addPicked" type="checkbox" :value="o.id" /> {{ o.name }}
            <span v-if="!o.in_cast" class="muted small">（本场未点名）</span>
          </label>
          <p v-if="!addable(addFor).length" class="muted small">
            没有可添加的角色：其余人要么已经在这一段里，要么还没有角色图。
          </p>
          <div v-if="error" class="alert err">{{ error }}</div>
          <div class="modal-actions">
            <button class="btn primary sm" :disabled="!addPicked.length" @click="confirmAdd">确定</button>
            <button class="btn sm" @click="addFor = null">取消</button>
          </div>
        </div>
      </div>
    </Teleport>
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
/* 小图 176px（用户裁定：88px 太小看不清是谁）。仍按**原始比例**显示、不加边框不填底色：
   描边会把竖版立绘和横版场景图"框成一样大"，看着像被裁过，也分不清谁是谁。 */
.refStrip { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end; margin: 6px 0; }
/* 图块：图在上、名字在下、删除压在块底 —— 名字放图上面时"人"和"图"对不上行 */
.refTile { margin: 0; text-align: center; min-width: 110px; }
.refTile img {
  height: 176px; width: auto; max-width: 320px; object-fit: contain; display: block; margin: 0 auto;
}
.refTile figcaption { font-size: 12px; margin: 3px 0; line-height: 1.3; }
.refTile .upl.tiny { padding: 2px 8px; font-size: 12px; width: auto; }
.refTile .missing.sm {
  height: 176px; width: 110px; display: flex; align-items: center; justify-content: center;
  font-size: 11px; text-align: center; margin: 0 auto;
  border: 1px dashed var(--border, #2a3038); border-radius: 6px;
}
/* 圆形 ➕：随图块放大到能一眼看见，排在这段素材最后（场景图之后） */
.addTile {
  width: 64px; height: 64px; border-radius: 50%; border: 1px dashed var(--border, #2a3038);
  background: none; color: var(--muted); font-size: 30px; line-height: 1; cursor: pointer;
  align-self: center;
}
.addTile:hover { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }

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
.refRow.addRow { gap: 8px; align-items: center; margin-top: 6px; }
.refRow.addRow select { max-width: 220px; }
.chk { display: flex; gap: 4px; align-items: center; }
.field { display: flex; flex-direction: column; gap: 2px; min-width: 260px; }
.field.wide { width: 100%; }
.field.wide textarea { width: 100%; font-size: 12px; }
</style>
