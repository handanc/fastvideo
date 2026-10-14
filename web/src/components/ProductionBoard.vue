<script setup>
import { computed, ref, watch } from 'vue'
import { api } from '../api'
import SceneClips from './SceneClips.vue'

const props = defineProps({
  runId: { type: String, required: true },
  board: { type: Object, default: null },
})
const emit = defineEmits(['changed'])

const tab = ref('script')
const busy = ref('')
const notice = ref('')
const error = ref('')
const uploading = ref('')

async function onUpload(stage, name, ev) {
  const file = ev.target.files?.[0]
  if (!file) return
  uploading.value = name
  error.value = ''; notice.value = ''
  try {
    const r = await api.putArtifact(props.runId, stage, name, file)
    notice.value = `✔ 已替换 ${name}（归一化到 ${r.normalized_to}${r.backed_up ? '，旧图已备份' : ''}），`
      + `作废下游 ${(r.invalidated || []).length} 个产物`
    emit('changed')
  } catch (e) {
    error.value = `上传失败：${e.message}`
  } finally {
    uploading.value = ''
    ev.target.value = ''
  }
}

const b = computed(() => props.board || {})
const scenes = computed(() => b.value.scenes || [])
const chars = computed(() => b.value.characters || [])
const allShots = computed(() => scenes.value.flatMap((s) => s.shots.map((sh) => ({ ...sh, scene: s }))))
const counts = computed(() => ({
  script: allShots.value.length,
  cast: chars.value.length,
  scenes: scenes.value.length,
  shots: allShots.value.length,
  frames: allShots.value.filter((s) => s.frame).length,
  clips: allShots.value.filter((s) => s.clip || s.talk).length,
  final: b.value.final_video ? 1 : 0,
}))

const tabs = computed(() => [
  { id: 'script', label: '剧本', n: counts.value.script },
  { id: 'cast', label: '角色图', n: counts.value.cast },
  { id: 'scenes', label: '场景图', n: counts.value.scenes },
  { id: 'shots', label: '场景视频', n: counts.value.shots },
  { id: 'final', label: '成片', n: counts.value.final },
])

const typeLabel = { dialogue: '对白', action: '动作', scenery: '空镜' }
const transLabel = { cut: '硬切', dissolve: '叠化', fade: '淡入淡出', scenery: '空镜转场' }

function metricBadges(m) {
  if (!m) return []
  const out = []
  const mark = (k, v) => out.push({ k, v, ok: v === true, bad: v === false })
  mark('脸', m.face); mark('发', m.hair); mark('衣', m.outfit); mark('头饰', m.headwear)
  if (m.hue_dist != null) out.push({ k: '色距', v: m.hue_dist, bad: m.metric_conflict || m.hue_dist > 45 })
  // 口型三态：无口型(叠音) / 不同步 / **未测准**。
  // "未测准"必须与"不合格"分开显示：尺子量不到嘴时（没定位成功或对照不显著），
  // 把镜头标红会误导你去重做一个其实没问题的片段 —— 而重做是要花钱的。
  if (m.lipsync) {
    const ls = m.lipsync
    if (ls.method === 'overlay') out.push({ k: '口型', v: '无(叠音)', bad: true })
    else if (ls.verdict === 'synced') out.push({ k: '口型', v: `同步 ${ls.corr}`, ok: true })
    // 度量在真人素材上会假阴性（人眼确认同步的片段只给 0.11-0.16），所以"不同步"
    // 一律是**待人工**而不是错误标红：标红会诱导去重抽一个其实合格的镜头，那是真花钱。
    else if (ls.verdict === 'unsynced') out.push({ k: '口型', v: `待人工(${ls.corr})`, warn: true })
    else out.push({ k: '口型', v: '未测准', warn: true })
  }
  return out
}

const editing = ref(false)
const editText = ref('')
const saving = ref(false)
const editResult = ref(null)

// 看板只回 join 后的字段；编辑需要完整原文 → 直接拉 script.json 原始内容
const rawScriptSource = ref(null)
async function loadRawScript() {
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(props.runId)}/script`)
    rawScriptSource.value = res.ok ? await res.json() : null
  } catch {
    rawScriptSource.value = null
  }
}

async function startEdit() {
  if (rawScriptSource.value === null) await loadRawScript()
  editText.value = JSON.stringify(rawScriptSource.value ?? {}, null, 2)
  editResult.value = null
  editing.value = true
}

async function saveScript() {
  saving.value = true
  editResult.value = null
  try {
    const obj = JSON.parse(editText.value)
    const r = await api.putScript(props.runId, obj)
    editResult.value = {
      ok: true,
      msg: `✔ 已保存（v${r.version}），作废 ${r.invalidated} 个下游产物：` +
        `${(r.items || []).slice(0, 6).join('、')}${(r.items || []).length > 6 ? '…' : ''}`,
    }
    editing.value = false
    emit('changed')
  } catch (e) {
    editResult.value = { ok: false, msg: `保存被拒：${e.message}` }
  } finally {
    saving.value = false
  }
}

// 删除剧本 → script.json 没了 → 后端下一步回到「生成剧本」，主按钮文案随之变化
const deletingScript = ref(false)

async function delScript() {
  const n = allShots.value.length
  if (!window.confirm(
    `删除剧本「${b.value.title || '(未命名)'}」？\n\n`
    + `全部 ${n} 个镜头消失，角色图/场景图/首帧/视频/成片都标记待重做。\n`
    + '旧剧本会归档（不是直接抹掉），你上传过的图不受牵连。',
  )) return
  deletingScript.value = true
  error.value = ''
  notice.value = ''
  try {
    const r = await api.delScript(props.runId)
    notice.value = `✔ 已删除剧本，作废 ${r.count} 个下游产物（归档为 ${r.archived}）`
    editing.value = false
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    deletingScript.value = false
  }
}

// 删成片：只删 final.mp4 + 旁挂字幕 + QC 报告，**素材一个都不动** →
// 「拼接成片」回到缺口，再点一次开始生成就能零 API 费用重合成。
const deletingFinal = ref(false)

async function delFinal() {
  if (!window.confirm('确定删除成片？\n\n素材（角色图/首帧/镜头片段/口型片段）都保留，\n删除后点「开始生成」可零 API 费用重新合成。')) return
  deletingFinal.value = true
  error.value = ''
  notice.value = ''
  try {
    const r = await api.deleteFinal(props.runId)
    notice.value = `✔ 已删除成片（${(r.removed || []).join('、') || 'final.mp4'}）`
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    deletingFinal.value = false
  }
}

// ── 剧本稿：人改的是"剧本"，不是 JSON ──────────────────────────
// 保存走后端本地确定性解析（不调模型、不改你写的字），失败按行号点名。
const draftEditing = ref(false)
const draftText = ref('')
const draftSaving = ref(false)
const draftMsg = ref(null) // { ok, text, warnings[], notes[] }
const flash = ref(null) // 提示条：出现时滚到眼前
const draftLoaded = ref('') // 载入时的原文，用来判断"用户到底改过没有"
const draftStale = ref(false) // 后台把剧本改了，编辑器里还是旧内容
const draftVersion = ref(props.board?.script_version ?? null)

async function startDraft() {
  draftMsg.value = null
  draftStale.value = false
  error.value = ''
  try {
    const r = await api.scriptText(props.runId)
    draftText.value = r.text
    draftLoaded.value = r.text
    draftVersion.value = r.version
    draftEditing.value = true
  } catch (e) {
    error.value = `读取剧本稿失败：${e.message}`
  }
}

async function saveDraft() {
  draftSaving.value = true
  draftMsg.value = null
  try {
    const r = await api.putScriptText(props.runId, draftText.value)
    const items = (r.items || []).slice(0, 6).join('、')
    draftMsg.value = {
      ok: true,
      text: `✔ 已保存剧本（v${r.version}），作废 ${r.invalidated} 个下游产物`
        + `${items ? `：${items}${(r.items || []).length > 6 ? '…' : ''}` : ''}`,
      warnings: r.warnings || [],
      notes: r.notes || [],
    }
    draftEditing.value = false
    draftStale.value = false
    draftVersion.value = r.version
    emit('changed')
  } catch (e) {
    // 422 的 detail 是问题数组（带行号），api.js 已按换行拼好
    draftMsg.value = { ok: false, text: `保存被拒：${e.message}`, warnings: [], notes: [] }
  } finally {
    draftSaving.value = false
  }
}

const regenOpen = ref(false)
const regenPrompt = ref('')
const regenMode = ref('revise')
const regenBusy = ref(false)

function openRegen() {
  error.value = ''
  notice.value = ''
  regenOpen.value = true
}

async function submitRegen() {
  const p = regenPrompt.value.trim()
  if (!p) {
    error.value = '请输入 Prompt'
    return
  }
  regenBusy.value = true
  error.value = ''
  try {
    const r = await api.regenerateScript(props.runId, p, regenMode.value)
    notice.value = r.mode === 'revise'
      ? '✔ 已开始改写这份剧本（每 3 秒自动刷新）'
      : '✔ 已开始重写新剧本（旧剧本已归档，下游全部作废）'
    regenOpen.value = false
    regenPrompt.value = ''
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    regenBusy.value = false
  }
}

async function delArtifact(stage, name, what) {
  if (!window.confirm(`删除${what}「${name}」？\n它自己与下游产物都会被标记待重做。`)) return
  uploading.value = name
  error.value = ''
  notice.value = ''
  try {
    const r = await api.delArtifact(props.runId, stage, name)
    notice.value = `✔ 已删除 ${name}，作废下游 ${r.invalidated?.length || 0} 个产物`
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    uploading.value = ''
  }
}

// ── 点文字即改：外貌 / 场景 / 镜头描述 ─────────────────────────
// 单击文本进入编辑；点别处（blur）或 Ctrl+Enter 提交，Esc 取消。
// 提交走 PUT /script：服务端校验 + 血缘作废。改完**不删图也会重画**（描述变了产物即 stale），
// 删图再点「开始生成」同样按新描述出图 —— 两条路都通。
const inline = ref(null) // { kind: 'char'|'scene'|'shot', id, field, text, original }

function editKey(kind, id, field) {
  const e = inline.value
  return !!e && e.kind === kind && e.id === id && e.field === field
}

function startInline(kind, id, field, current) {
  inline.value = { kind, id, field, text: current || '', original: current || '' }
}

function cancelInline() {
  inline.value = null
}

function commitInline() {
  const e = inline.value
  inline.value = null
  if (!e) return
  const text = e.text.trim()
  if (text === (e.original || '').trim()) return // 没改：不涨版本、不作废、不打扰后端
  saveDescription(e, text)
}

async function saveDescription(e, text) {
  busy.value = `${e.kind}:${e.id}`
  error.value = ''; notice.value = ''
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(props.runId)}/script`)
    if (!res.ok) throw new Error(`读取剧本失败 HTTP ${res.status}`)
    const script = await res.json()
    const shots = (script.scenes || []).flatMap((s) => s.shots)
    const target =
      e.kind === 'char' ? (script.characters || []).find((c) => c.character_id === e.id)
      : e.kind === 'scene' ? (script.scenes || []).find((s) => s.scene_id === e.id)
      : shots.find((sh) => sh.shot_id === e.id)
    if (!target) throw new Error(`剧本里找不到 ${e.id}`)
    target[e.field] = text
    const r = await api.putScript(props.runId, script)
    notice.value = `✔ 已保存「${e.id}」的描述（剧本 v${r.version}），作废 ${r.invalidated} 个产物；`
      + '点「开始生成」即按新描述重画（不删图也会重画）'
    emit('changed')
  } catch (err) {
    error.value = err.message
  } finally {
    busy.value = ''
  }
}

async function pickPortrait(ch, candidateUrl) {
  const name = candidateUrl.split('/').pop().split('?')[0]
  busy.value = `${ch.id}:${name}`
  error.value = ''; notice.value = ''
  try {
    const r = await api.pickPortrait(props.runId, ch.id, name)
    notice.value = `✔ ${ch.name} 的定妆照已换（旧图已备份），`
      + `作废下游 ${(r.invalidated || []).length} 个产物`
    emit('changed')
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = ''
  }
}

watch(() => props.runId, () => {
  tab.value = 'script'; notice.value = ''; error.value = ''
  // 换 run 时剧本稿/JSON 编辑器不能带着上一个 run 的内容留在打开状态
  draftEditing.value = false; editing.value = false; draftStale.value = false
  draftVersion.value = props.board?.script_version ?? null
})

// 提示条在页面最上方，而操作按钮（剧本三动作/上传/删除）大多在下方 ——
// 不滚过去用户就以为"点了没反应"（2026-10-07 实测：剧本重写成功但看不见提示）。
watch([notice, error], () => flash.value?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }))

// 剧本被后台改写（按 prompt 重写完成）→ script_version 变。正开着的剧本稿编辑器
// 若不重新载入，显示的还是旧内容；没动过就直接换掉，动过则给一个"重新载入"入口。
watch(() => props.board?.script_version, (v, old) => {
  if (v == null || old == null || v === old) return
  draftVersion.value = v
  if (!draftEditing.value) return
  draftStale.value = draftText.value !== draftLoaded.value
  if (!draftStale.value) return startDraft()
})
</script>

<template>
  <div>
    <div v-if="error" ref="flash" class="alert err">{{ error }}</div>
    <div v-if="notice" ref="flash" class="alert ok">{{ notice }}</div>

    <div v-if="!b.available" class="card">
      <h2>剧本尚未生成</h2>
      <p class="muted">点上方「▶ 开始生成 · <b>生成剧本</b>」出一份剧本。</p>
    </div>

    <template v-else>
      <div class="tabs">
        <button v-for="t in tabs" :key="t.id" :class="{ active: tab === t.id }" @click="tab = t.id">
          {{ t.label }}<span class="cnt">{{ t.n }}</span>
        </button>
      </div>

      <!-- 人工上传台账：这些图重做时永远不被覆盖，所以必须看得见 -->
      <div v-if="Object.keys(b.uploads || {}).length" class="card">
        <p class="muted small">
          已上传 {{ Object.keys(b.uploads).length }} 项（重做时会跳过，不会被覆盖）：
          <span v-for="(v, k) in b.uploads" :key="k" class="chip">{{ k }}</span>
        </p>
      </div>

      <!-- ── 剧本 ────────────────────────────────────────────── -->
      <div v-if="tab === 'script'" class="card">
        <h2>{{ b.title || '(未命名)' }}</h2>
        <p class="muted">
          风格 {{ b.meta?.base_info?.style }} · 目标 {{ b.meta?.base_info?.target_duration_sec }}s ·
          画幅 {{ b.meta?.base_info?.aspect_ratio }} · 共 {{ counts.script }} 镜 /
          {{ scenes.length }} 场 · 档位 {{ b.meta?.quality }}
        </p>
        <p class="muted small">剧本评审由<b>你</b>来做；分镜抽查、镜头选优、成片终审、一致性闸门自动跑。</p>

        <!-- 角色清单（与剧本稿的「# 角色」同一份内容）：只出现在这里，看不到就以为没改出来 -->
        <div class="roster">
          <b>角色 {{ chars.length }}</b>
          <span v-for="c in chars" :key="c.id" class="chip" :class="{ warn: c.used === false }">
            {{ c.name }} <span class="mono muted">{{ c.id }}</span>
            <template v-if="c.gender === 'female'">·女</template>
            <template v-else-if="c.gender">·男</template>
            <template v-if="c.age_band"> · {{ c.age_band }}</template>
          </span>
          <p v-if="chars.some((c) => c.used === false)" class="muted small">
            ⚠ 灰底的角色没出现在任何镜头里（不会出场，但仍会生成一张定妆照）——
            要它出场就在镜头里加「在场: 角色id」，不要就删掉这行。
          </p>
        </div>

        <section v-for="s in scenes" :key="s.id" class="scene-block">
          <h3>
            <span class="tag">{{ s.id }}</span> {{ s.title }}
            <span class="muted">· {{ s.location }} · {{ s.time_of_day }}</span>
          </h3>
          <p v-if="s.summary" class="summary">{{ s.summary }}</p>
          <table class="tbl shot-table">
            <thead>
              <tr><th>镜号</th><th>类型</th><th>角色</th><th>画面 / 台词</th><th class="num">时长</th><th>转场</th></tr>
            </thead>
            <tbody>
              <tr v-for="sh in s.shots" :key="sh.shot_id">
                <td class="mono">{{ sh.shot_id }}</td>
                <td><span class="tag">{{ typeLabel[sh.type] || sh.type }}</span></td>
                <td>{{ sh.characters.join('、') || '—' }}</td>
                <td>
                  <!-- 画面描述/镜头语言是**场景提示词的原料**，就地改在这里（场景视频页只留
                       将要发出去的那段提示词，不再复述一遍分镜，免得两份文字分不清哪份算数） -->
                  <div
                    v-if="!editKey('shot', sh.shot_id, 'action')"
                    class="action editable" title="点击修改画面描述（会重算该场景的提示词并作废其片段）"
                    @click="startInline('shot', sh.shot_id, 'action', sh.action)"
                  >{{ sh.action || '（点击写画面描述）' }}</div>
                  <textarea
                    v-else class="inline-edit" rows="2" autofocus spellcheck="false"
                    v-model="inline.text"
                    @blur="commitInline"
                    @keydown.esc.prevent="cancelInline"
                    @keydown.ctrl.enter="commitInline"
                  ></textarea>
                  <div v-if="!editKey('shot', sh.shot_id, 'camera')" class="muted small">
                    镜头语言：<span class="editable" title="点击修改镜头语言"
                      @click="startInline('shot', sh.shot_id, 'camera', sh.camera)"
                    >{{ sh.camera || '（点击写）' }}</span>
                  </div>
                  <input
                    v-else class="inline-edit one" autofocus v-model="inline.text"
                    @blur="commitInline"
                    @keydown.esc.prevent="cancelInline"
                    @keydown.enter="commitInline"
                  />
                  <div v-for="(d, i) in sh.dialogues" :key="i" class="line">
                    <b>{{ d.who }}</b>：{{ d.text }}<span class="muted">（{{ d.emotion }}）</span>
                  </div>
                </td>
                <td class="num">{{ sh.duration_sec }}s</td>
                <td class="muted">{{ transLabel[sh.transition_out] || sh.transition_out }}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <!-- 人工三动作：改稿 / 按 prompt 重写 / 删除 -->
        <div class="toolbar" style="margin-top: 12px">
          <template v-if="!draftEditing">
            <button class="btn primary sm" @click="startDraft">✎ 编辑剧本稿</button>
            <button class="btn sm" @click="openRegen">🎲 按 prompt 重新生成</button>
            <button class="btn sm del" :disabled="deletingScript" @click="delScript">
              {{ deletingScript ? '删除中…' : '🗑 删除剧本' }}
            </button>
            <span class="muted">改文字用「编辑剧本稿」，换思路用「按 prompt 重新生成」</span>
          </template>
          <template v-else>
            <button class="btn primary sm" :disabled="draftSaving" @click="saveDraft">
              {{ draftSaving ? '解析保存中…' : '💾 保存剧本稿' }}
            </button>
            <button class="btn sm" @click="draftEditing = false">取消</button>
            <button v-if="draftStale" class="btn sm" @click="startDraft">↻ 载入最新剧本</button>
            <span v-if="draftStale" class="muted">后台已重写剧本（v{{ draftVersion }}），这里是旧内容</span>
            <span v-else class="muted">台词：「  角色id（情绪）: 内容」；镜头另起「###」行</span>
          </template>
        </div>

        <textarea
          v-if="draftEditing"
          v-model="draftText"
          class="script-editor"
          rows="26"
          spellcheck="false"
        ></textarea>
        <div v-if="draftMsg" class="alert" :class="draftMsg.ok ? 'ok' : 'err'" style="white-space: pre-line">
          {{ draftMsg.text }}
        </div>
        <div v-if="draftMsg?.notes?.length" class="alert ok">
          <div v-for="(n, i) in draftMsg.notes" :key="i">{{ n }}</div>
        </div>
        <div v-if="draftMsg?.warnings?.length" class="alert warn">
          <b>⚠ 已保存，但这些软规则没达标：</b>
          <div v-for="(w, i) in draftMsg.warnings" :key="i">{{ w }}</div>
        </div>

        <Teleport to="body">
          <div v-if="regenOpen" class="modal-mask" @click.self="regenOpen = false">
            <div class="modal" role="dialog" aria-modal="true" aria-labelledby="regen-title">
              <h3 id="regen-title">按 prompt 重新生成</h3>
              <textarea
                v-model="regenPrompt"
                class="script-editor"
                rows="4"
                spellcheck="false"
                placeholder="输入 Prompt，例如：第三幕节奏太慢"
              ></textarea>
              <label class="check">
                <input v-model="regenMode" type="radio" value="revise" name="regen-mode" />
                改写已有剧本
              </label>
              <label class="check">
                <input v-model="regenMode" type="radio" value="rewrite" name="regen-mode" />
                重写新剧本
              </label>
              <!-- 弹窗盖在页面上方，错误必须落在弹窗里，否则就是"点了没反应" -->
              <div v-if="error" class="alert err">{{ error }}</div>
              <div class="modal-actions">
                <button class="btn primary sm" :disabled="regenBusy || !regenPrompt.trim()" @click="submitRegen">
                  {{ regenBusy ? '启动中…' : '开始生成' }}
                </button>
                <button class="btn sm" @click="regenOpen = false">取消</button>
              </div>
            </div>
          </div>
        </Teleport>

        <details class="raw">
          <summary>高级：直接改结构（原始 JSON）</summary>
          <template v-if="!editing">
            <pre>{{ JSON.stringify(b.raw_script || { title: b.title, scenes: scenes.map(s => ({ id: s.id, title: s.title, shots: s.shots.length })) }, null, 2) }}</pre>
            <button class="btn sm" @click="startEdit">✎ 直接编辑 JSON</button>
            <p class="muted small">只用于改结构；改剧情请用「编辑剧本稿」</p>
          </template>
          <template v-else>
            <textarea v-model="editText" class="script-editor" rows="18" spellcheck="false"></textarea>
            <div class="toolbar" style="margin-top: 8px">
              <button class="btn primary sm" :disabled="saving" @click="saveScript">
                {{ saving ? '校验保存中…' : '💾 保存并校验' }}
              </button>
              <button class="btn sm" @click="editing = false">取消</button>
              <span class="muted">保存后校验结构并作废受影响产物</span>
            </div>
          </template>
        </details>
        <div v-if="editResult" class="alert" :class="editResult.ok ? 'ok' : 'err'">{{ editResult.msg }}</div>
      </div>

      <!-- ── 角色图 ──────────────────────────────────────────── -->
      <div v-if="tab === 'cast'" class="card">
        <h2>角色定妆照 <span class="muted">（一致性锚点；点外貌描述即可改，改完点「开始生成」重画）</span></h2>
        <div v-for="ch in chars" :key="ch.id" class="char-card">
          <div class="char-main">
            <img v-if="ch.portrait" :src="ch.portrait" :alt="ch.name" loading="lazy" />
            <div v-else class="missing">缺 char_{{ ch.id }}.png</div>
            <p v-if="ch.uploaded" class="badge ok small">用户上传</p>
            <label class="upl">
              <input type="file" accept="image/png,image/jpeg,image/webp" hidden
                     @change="onUpload('assets', `char_${ch.id}.png`, $event)" />
              {{ uploading === `char_${ch.id}.png` ? '上传中…' : '⬆ 上传' }}
            </label>
            <button v-if="ch.portrait" class="upl del"
                    @click="delArtifact('assets', `char_${ch.id}.png`, '角色定妆照')">🗑 删除</button>
            <p class="muted small">删除后候选图仍在，可 ✔ 选用</p>
          </div>
          <div class="char-body">
            <h3>{{ ch.name }} <span class="tag mono">{{ ch.id }}</span>
              <span v-if="ch.metrics?.face_rate != null" class="badge" :class="ch.metrics.face_rate >= 0.8 ? 'ok' : 'warn'">
                脸一致 {{ ch.metrics.face_rate }}
              </span>
              <span v-if="ch.metrics?.outfit_rate != null" class="badge" :class="ch.metrics.outfit_rate >= 0.8 ? 'ok' : 'warn'">
                服装一致 {{ ch.metrics.outfit_rate }}
              </span>
            </h3>
            <!-- 外貌描述 = 立绘提示词的原料：点文字即改，点别处保存 -->
            <p
              v-if="!editKey('char', ch.id, 'appearance')"
              class="appearance editable" title="点击修改外貌描述"
              @click="startInline('char', ch.id, 'appearance', ch.appearance)"
            >{{ ch.appearance || '（点击写外貌描述）' }}</p>
            <textarea
              v-else class="inline-edit" rows="3" autofocus spellcheck="false"
              v-model="inline.text"
              @blur="commitInline"
              @keydown.esc.prevent="cancelInline"
              @keydown.ctrl.enter="commitInline"
            ></textarea>
            <p v-if="ch.canonical_tokens?.length" class="tokens">
              必留词：<span v-for="t in ch.canonical_tokens" :key="t" class="chip">{{ t }}</span>
            </p>
            <p v-if="ch.look" class="muted mono small">
              外貌卡：{{ Object.entries(ch.look).filter(([k, v]) => v && (Array.isArray(v) ? v.length : true)).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('/') : v}`).join(' · ') }}
            </p>
            <p class="muted small">音色 {{ ch.voice_id || '—' }}</p>
            <div v-if="ch.candidates?.length" class="cands">
              <div v-for="u in ch.candidates" :key="u" class="cand">
                <img :src="u" loading="lazy" :alt="u" />
                <button class="btn sm" :disabled="busy === `${ch.id}:${u.split('/').pop()}`"
                        @click="pickPortrait(ch, u)">
                  {{ busy === `${ch.id}:${u.split('/').pop()}` ? '切换中…' : '✔ 设为定妆照' }}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── 场景图 ──────────────────────────────────────────── -->
      <div v-if="tab === 'scenes'" class="card">
        <h2>场景概念图</h2>
        <div class="scene-grid">
          <figure v-for="s in scenes" :key="s.id">
            <img v-if="s.concept" :src="s.concept" :alt="s.title" loading="lazy" />
            <div v-else class="missing">缺 scene_{{ s.id }}.png</div>
            <figcaption>
              <b><span class="tag">{{ s.id }}</span> {{ s.title }}</b>
              <span class="muted">·
                <span
                  v-if="!editKey('scene', s.id, 'location')"
                  class="editable" title="点击修改地点"
                  @click="startInline('scene', s.id, 'location', s.location)"
                >{{ s.location }}</span>
                <input
                  v-else class="inline-edit one" autofocus
                  v-model="inline.text"
                  @blur="commitInline"
                  @keydown.esc.prevent="cancelInline"
                  @keydown.enter="commitInline"
                />
                · {{ s.time_of_day }} · {{ s.shots.length }} 镜</span>
              <p
                v-if="!editKey('scene', s.id, 'summary')"
                class="summary editable" title="点击修改剧情梗概（背景图会过滤掉其中的人物描述）"
                @click="startInline('scene', s.id, 'summary', s.summary)"
              >{{ s.summary || '（点击写场景氛围描述）' }}</p>
              <textarea
                v-else class="inline-edit" rows="2" autofocus spellcheck="false"
                v-model="inline.text"
                @blur="commitInline"
                @keydown.esc.prevent="cancelInline"
                @keydown.ctrl.enter="commitInline"
              ></textarea>
              <!-- 概念图是**空景背景**：这里直接摆实际发出去的提示词，
                   而不是让人对着剧情梗概猜"背景图里是不是要画角色"。 -->
              <details v-if="s.bg_prompt" class="promptfold">
                <summary>文生图提示词 · 空景背景（图中无人）</summary>
                <p class="bgprompt">{{ s.bg_prompt }}</p>
              </details>
              <span v-if="s.uploaded" class="badge ok small">已上传</span>
              <label class="upl">
                <input type="file" accept="image/png,image/jpeg,image/webp" hidden
                       @change="onUpload('assets', `scene_${s.id}.png`, $event)" />
                {{ uploading === `scene_${s.id}.png` ? '上传中…' : '⬆ 上传替换场景图' }}
              </label>
              <button v-if="s.concept" class="upl del"
                      @click="delArtifact('assets', `scene_${s.id}.png`, '场景概念图')">🗑 删除</button>
            </figcaption>
          </figure>
        </div>
      </div>

      <!-- ── 场景视频：一个场景 = 一段 wan3.0 视频 ───────────────────
           逐镜头片段面板已删除：它展示的是"首帧图 → 单镜头短片"那条老路，
           本项目只保留场景级生成（组内多镜头由模型自己切，口型也是 wan3.0 自带）。
           镜头的画面描述/镜头语言改在场景卡片里点改（同一套 PUT /script 语义）。 -->
      <div v-if="tab === 'shots'">
        <SceneClips :run-id="props.runId" @changed="$emit('changed')" />
      </div>

      <!-- ── 成片 ────────────────────────────────────────────── -->
      <div v-if="tab === 'final'" class="card">
        <div class="cardHead">
          <h2>最终成片</h2>
          <button
            v-if="b.final_video"
            class="btn danger"
            :disabled="deletingFinal"
            @click="delFinal"
          >{{ deletingFinal ? '删除中…' : '🗑 删除成片' }}</button>
        </div>
        <video v-if="b.final_video" class="final" :src="b.final_video" controls></video>
        <p v-else class="muted">尚未合成成片（final/final.mp4）</p>

        <!-- 技术 QC：确定性数值判据（不靠 VLM），量不到的一律单列，不混进"通过" -->
        <template v-if="b.tech_qc">
          <h3>技术 QC</h3>
          <p class="muted">
            <b :class="b.tech_qc.passed ? 'ok' : 'err'">
              {{ b.tech_qc.passed ? '✔ 合格' : '✘ 不合格' }}
            </b>
            · 检查 {{ b.tech_qc.items }} 项素材
          </p>
          <ul v-if="b.tech_qc.failed?.length" class="degs">
            <li v-for="(f, i) in b.tech_qc.failed" :key="'f' + i" class="err">
              <b>{{ f.item }}</b> — {{ f.problem }}
            </li>
          </ul>
          <ul v-if="b.tech_qc.warnings?.length" class="degs">
            <li v-for="(w, i) in b.tech_qc.warnings" :key="'w' + i">
              <b>{{ w.item }}</b> — {{ w.problem }}
            </li>
          </ul>
          <p v-if="b.tech_qc.not_checked?.length" class="muted small">
            未检测：{{ b.tech_qc.not_checked.join('；') }}
          </p>
        </template>
        <p v-if="b.subtitle">
          <a :href="b.subtitle" download>下载字幕 .srt</a>
        </p>
        <h3>一致性度量</h3>
        <p v-if="b.consistency?.summary" class="muted">
          脸 {{ b.consistency.summary.face_consistency }} ·
          服装 {{ b.consistency.summary.outfit_consistency }} ·
          写实 {{ b.consistency.summary.style_photoreal_rate }} ·
          平均色距 {{ b.consistency.summary.mean_hue_dist }} ·
          已测 {{ b.consistency.summary.shots_measured }} 镜 ·
          评审可信 {{ b.consistency.judge_trusted }}
        </p>
        <p v-else class="muted">无一致性报告（480P 或未跑到度量阶段）</p>
        <template v-if="b.degradations?.length">
          <h3>本次降级记录（{{ b.degradations.length }} 条）</h3>
          <ul class="degs">
            <li v-for="(d, i) in b.degradations" :key="i">
              <b>{{ d.node }} · {{ d.what }}</b> — {{ d.why }}
              <span class="muted">（影响：{{ d.impact }}）</span>
            </li>
          </ul>
        </template>
      </div>
    </template>
  </div>
</template>
