<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { statusOf } from '../status'
import ProductionBoard from './ProductionBoard.vue'
import StepBar from './StepBar.vue'

const props = defineProps({ runId: { type: String, required: true } })
const emit = defineEmits(['back'])

const detail = ref(null)
const board = ref(null)
const steps = ref(null)
// 成本闸门（后端 core/pause.py）：**实时累计**费用到线才亮，预估不够只出红幅警告。
// 一直亮到人工确认为止 —— 这正是它存在的目的：不让"钱花完了、片没出完"变成静默事实。
const gate = computed(() => steps.value?.budget_gate || null)
const gateOpen = ref(false)
const gateBusy = ref(false)
const newLimit = ref(0)
const suggestedLimit = computed(() => {
  const g = gate.value || {}
  // 默认给"已花 + 20"，且至少比现上限高 20%：只清标记不抬上限，下一段又立刻停
  const spent = Number(g.spent_cny || 0)
  const limit = Number(g.limit_cny || 0)
  return Math.max(Math.round(limit * 1.2), Math.round(spent + 20))
})

watch(
  () => gate.value?.pending,
  (p) => { if (p) { gateOpen.value = true; newLimit.value = 0 } },
  { immediate: true },
)

async function resumeBudget() {
  gateBusy.value = true
  try {
    const want = Number(newLimit.value || 0)
    await api.budgetContinue(props.runId, want > 0 ? want : null)
    await api.stepRun(props.runId)   // 确认即继续：从停下处接着跑
    await load()
  } catch (e) {
    error.value = `继续失败：${e.message}`
  } finally {
    gateBusy.value = false
  }
}
const task = ref(null)
const pane = ref('board') // board=制作看板（剧本/角色/场景/镜头/成片）| flow=事件与成本
const error = ref('')
let timer = null

// ── 中途改模型（标题下方那个选择器）──────────────────────────────
// 为什么要放在详情页而不是只能建任务时选：一条 run 要跑六步、几分钟到几十分钟，
// 跑到一半发现选错模型（或免费额度用尽）时，唯一出路本来是"删掉重建" —— 前面几步的钱白烧。
// 后端 PATCH /api/runs/{id}/models 只在**空闲**时接受（持锁期间改配置会让同一步的前后
// 两个镜头用不同模型、产物不同源），改完只对之后的步骤生效，已落盘的产物不动。
const modelCaps = ref([])
const modelPick = ref({ text: '', t2i: '', i2v: '' })
// 口型不再是可选项：wan3.0 场景片段自带台词口型，音轨恒为模型原声。
// 这一栏只剩 文本/文生图/图生视频（wan3.0 两档）三个选择。
const modelMsg = ref('')
const modelBusy = ref(false)
const CAP_TITLE = { text: '文本', t2i: '文生图', i2v: '图生视频' }
const capTitle = (c) => CAP_TITLE[c] || c
// 分辨率标签由后端给（`resolution_of_meta` 会把老 run 的旧档位名换算过来）：
// 前端不自己维护映射表，否则新旧 run 的显示口径会各说各话。
const resOfQuality = computed(
  () => detail.value?.resolution || detail.value?.meta?.resolution_tier || '—'
)

async function loadModels() {
  const tier = detail.value?.resolution || detail.value?.meta?.resolution_tier || ''
  try {
    const data = await api.modelChoices(tier, props.runId)
    modelCaps.value = data?.capabilities || []
    // 选择器初始值 = 这条 run 已点名的模型（run_meta.model_overrides）
    const chosen = data?.run_overrides || detail.value?.meta?.model_overrides || {}
    modelPick.value = { text: chosen.text || '', t2i: chosen.t2i || '', i2v: chosen.i2v || '' }
    // 下拉没有"跟随默认"空选项 → 用后端算出的当前生效模型作初值（不在清单里就退第一项）
    for (const cap of modelCaps.value) {
      const opts = (cap.choices || []).map((c) => c.model)
      if (!opts.includes(modelPick.value[cap.capability])) {
        modelPick.value[cap.capability] =
          opts.includes(cap.effective_model) ? cap.effective_model : (opts[0] || '')
      }
    }
  } catch {
    modelCaps.value = []
  }
}

async function saveModels() {
  modelBusy.value = true
  modelMsg.value = ''
  error.value = ''
  try {
    const r = await api.patchRunModels(props.runId, {
      model_overrides: modelPick.value,
    })
    const written = (r.env_written || []).join('、')
    const parts = [`✔ 已保存${written ? `（并写回 .env：${written}）` : ''}`]
    parts.push('只对之后的步骤生效；已生成的产物要重出，请在看板里删掉它再点「开始生成」。')
    if (r.talk_warning) parts.push(`⚠ ${r.talk_warning}`)
    modelMsg.value = r.env_error ? `已保存；但 .env 写回失败：${r.env_error}` : parts.join('。')
    await load()
    await loadModels()
  } catch (e) {
    error.value = e.message
  } finally {
    modelBusy.value = false
  }
}

async function load() {
  try {
    // 四个请求一起取完再一次性赋值。此前 detail 先落、task/steps 后落，
    // 于是"要不要继续轮询"读到的还是上一次的旧值：一个已完成的 run 刚点完
    // 「按 prompt 重新生成」时判不出"有任务在跑"，轮询不再启动 ——
    // 后台明明重写完了剧本，页面却永远停在旧内容（看起来就像按钮没反应）。
    const [d, b, s, t] = await Promise.all([
      api.runDetail(props.runId),
      api.board(props.runId).catch(() => null),  // 看板/步骤/任务各自失败不拖垮其他面板
      api.steps(props.runId).catch(() => null),
      api.task().catch(() => null),
    ])
    detail.value = d
    board.value = b
    steps.value = s
    task.value = t
    error.value = ''
  } catch (e) {
    error.value = e.message
  } finally {
    schedulePolling()
  }
}

function schedulePolling() {
  clearInterval(timer)
  const st = detail.value?.status
  // 运行中/空闲/有任务在跑 → 3 秒轮询；failed 停（留手动 ⟳）
  const running = st === 'running' || st === 'idle' || steps.value?.running || task.value?.busy
  if (running) timer = setInterval(load, 3000)
}

// 失败步骤必须顶到页面最上面：StepBar 里那条红字在步骤条内部，滚动到看板里就看不见，
// 用户会以为"点了没反应"（2026-10-07 欠费导致分镜步骤连失败三次都被当成没反应）。
const failedSteps = computed(() =>
  (steps.value?.steps || []).filter((s) => s.state === 'failed'),
)
const qualityVerdict = computed(() => board.value?.quality_verdict || null)
// 模型溯源（写在 run_meta 里）：这条 run 点名用过哪些模型 + 有没有同时写回 .env
const modelProvenance = computed(() => {
  const m = detail.value?.meta || {}
  // talk 不再单列：口型就是 i2v 那同一次 wan3.0 调用，摆出来只会让人以为还有个口型开关
  const chosen = Object.entries(m.model_overrides || {}).filter(([k, v]) => v && k !== 'talk')
  if (!chosen.length) return ''
  const picks = chosen.map(([k, v]) => `${k} = ${v}`).join('、')
  const tail = m.fake ? '（Fake，未调用真实模型）'
    : (m.model_env_written || []).length ? '' : '（未写回 .env）'
  return `模型 ${picks}${tail}`
})

const lipsyncWaived = computed(() => board.value?.lipsync_coverage?.waived || [])
// 尾帧覆盖率：缺尾帧的镜头会退回**单首帧**生视频（末尾姿态/构图容易漂），而产物照样
// 出得来 —— 步骤条不会显示缺口。所以必须在看板上说出来，否则用户的观感就是
// "首尾帧改造没生效"（2026-10-12 那条 run：8 个首帧、1 个尾帧，界面上毫无提示）。
const lastFrameGap = computed(() => {
  const c = board.value?.last_frame_coverage
  return c && c.need > 0 && (c.missing || []).length ? c : null
})
// 熔断低于预估：口型通道按输出秒计价，跳闸的后果直接是"成片没对口型"，
// 所以这条必须在看板显眼处提醒，而不是等用户对完账单才发现。
// 最近一次"改写没落地"（后端记在降级台账里）：必须顶在剧本页，不能只躺在「成片」页签
const reviseFailure = computed(() => {
  const degs = board.value?.degradations || []
  const hit = degs.filter((d) => d?.what === 'revise_not_applied')
  return hit.length ? hit[hit.length - 1] : null
})
// 单价不确定的金额（后端 _read_cost 算好，前端不自己判断"哪个模型没价"）
const uncertainCost = computed(() => Number(detail.value?.cost?.uncertain_cny || 0))
const costRisk = computed(() => {
  const e = board.value?.cost_estimate
  return e && e.too_low ? e : null
})

const base = computed(() => detail.value?.meta?.base_info || {})
const reviewBadge = computed(() => {
  const r = detail.value?.review
  if (!r) return null
  const fb = r.feedback ? ' · ' + String(r.feedback).slice(0, 60) : ''
  // 欠费/未开通导致终审没跑：分数是兜底值，必须与"真评审但异常"区分开
  if (r.gate_error) return { cls: 'err', label: `⚠ 终审未真实执行（账号门），分数为兜底值${fb}` }
  if (r.review_error) return { cls: 'warn', label: `⚠ 终审异常，已放行${fb}` }
  if (r.verified === false) return { cls: 'warn', label: `⚠ 终审分数非模型给出${fb}` }
  if (r.passed === false) return { cls: 'err', label: `✘ 终审未通过 (${Number(r.score || 0).toFixed(2)})${fb}` }
  return { cls: 'ok', label: `✔ 终审通过 (${Number(r.score || 0).toFixed(2)})` }
})
const consistencyBadge = computed(() => {
  const s = detail.value?.consistency?.summary
  if (!s) return null
  const trusted = detail.value?.consistency?.judge_trusted
  return {
    cls: trusted === false ? 'warn' : 'ok',
    label: `一致性 face=${s.face_consistency ?? '-'} outfit=${s.outfit_consistency ?? '-'}` +
      (trusted === false ? '（评审不可信）' : ''),
  }
})
const st = computed(() => statusOf(detail.value?.status))

// 轮询的起停统一由 load() 结束时决定（见上），不再用 watch(detail) ——
// 那样会在 board/steps/task 还没落地时就读到旧值，做完动作后反而不再刷新。
onMounted(async () => {
  await load()
  await loadModels()   // 需要分辨率才能问对那一档的模型清单，所以排在 load 之后
})
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
    <p v-if="!detail && !error" class="muted">加载中…</p>

    <template v-if="detail">
      <!-- 概要 -->
      <div class="card">
        <h2>{{ base.topic || '(无主题)' }}</h2>
        <p class="muted">
          {{ base.style || '—' }} · {{ base.target_duration_sec || '—' }}s ·
          {{ base.aspect_ratio || '—' }} ·
          <!-- 面板只讲分辨率；档位是内部量（老 run 没有 resolution_tier 时由档位反推） -->
          {{ detail.meta.resolution_tier || resOfQuality }}
          <span v-if="detail.meta.fake" class="badge warn">Fake 模式（占位产物）</span>
          <span v-if="reviewBadge" class="badge" :class="reviewBadge.cls">{{ reviewBadge.label }}</span>
          <span v-if="consistencyBadge" class="badge" :class="consistencyBadge.cls">{{ consistencyBadge.label }}</span>
        </p>
        <!-- 模型溯源：这条 run 点名用过哪些模型、有没有同时写回 .env。
             写回改变的是**机器上的全局默认**，不留痕就没法回答"这台机器怎么突然换模型了"。 -->
        <p v-if="modelProvenance" class="muted">{{ modelProvenance }}</p>
        <div v-if="detail.meta.model_env_error" class="alert err">
          ⚠ 本次点名的模型未能写回 <code>.env</code>（{{ detail.meta.model_env_error }}）：
          这条任务仍按所选模型执行，但**之后新建的任务会读回旧值** —— 请手动改
          <code>.env</code> 的 FV3_TEXT_MODEL / FV3_T2I_MODEL / FV3_I2V_MODEL。
        </div>
      </div>

      <!-- ── 模型选择（title 下方）：允许中途改，改完继续跑后续步骤 ──────────────
           只对**之后**的步骤生效；已落盘的产物不动（要重出就在看板里删掉它再点开始生成）。
           正在生成时后端会拒绝（持锁期间改配置会让同一步的前后两个镜头用不同模型）。 -->
      <div v-if="modelCaps.length" class="card">
        <div class="cardHead">
          <h2>模型</h2>
          <button class="btn sm primary" :disabled="modelBusy" @click="saveModels">
            {{ modelBusy ? '保存中…' : '保存' }}
          </button>
        </div>
        <div class="modelBar">
          <label v-for="cap in modelCaps" :key="cap.capability" class="field">
            <span>{{ capTitle(cap.capability) }} <em class="muted">默认 {{ cap.effective_model || '—' }}</em></span>
            <select v-model="modelPick[cap.capability]">
              <option v-for="c in cap.choices" :key="c.model" :value="c.model">
                {{ c.label }}
              </option>
            </select>
          </label>
        </div>
        <p v-if="modelMsg" class="muted small">{{ modelMsg }}</p>
      </div>

      <!-- 上一步为什么没出东西：失败原因顶格显示，别让用户猜 -->
      <!-- 改写没落地必须**在剧本页就能看见**：它只记在降级台账里，而台账原本只显示在
           「成片」页签 —— 用户在剧本页点完「按 prompt 重新生成」，看不到任何异常，
           会以为改好了并接着点「开始生成」，然后花钱出一条不含那句台词的成片。 -->
      <div v-if="reviseFailure" class="alert err">
        <b>⚠ 上次「按 prompt 重新生成」没有落地，剧本未改动</b>（没花任何重画的钱）：<br />
        {{ reviseFailure.why }}<br />
        <span class="muted">
          怎么办：把要加的台词用引号括起来、并点名到<b>角色 id 与镜头号</b>（剧本稿里都能看到），
          例如「给 char_queen 加一句台词：“圣僧何不留下”，放在 shot_2_2」；
          或者直接用「✎ 编辑剧本稿」手写那一行 —— 走本地解析，必然生效。
        </span>
      </div>
      <!-- 失败只在这里说一次：步骤条下面原来还挂了一条同样的文案，两处重复
           只会让人以为发生了两件不同的事（用户反馈）。 -->
      <div v-for="s in failedSteps" :key="s.id" class="alert err">
        ⚠ 「{{ s.label }}」失败：{{ s.error }}
      </div>

      <!-- 一致性闸门结论：不达标必须顶在最显眼的地方，不能让用户以为出片即合格 -->
      <!-- 口型红条已删除：口型在场景片段里（wan3.0 一次调用自带），片段没生成时
           "缺口"本来就已经写在步骤条上，再挂一条"没有口型"只是同一件事说两遍，
           而且说法是错的（听起来像口型链路坏了）。真正的安全网在后端：
           lipsync_coverage 仍参与成片判定，缺口型照样 passed=False。 -->
      <!-- 成本闸门：到线就停下来问人。已完成的产物一个不动，确认后从停下处接着跑。 -->
      <div v-if="gate?.pending" class="alert err">
        ⏸ <b>实时费用 ¥{{ Number(gate.spent_cny || 0).toFixed(2) }} 已达上限
        ¥{{ Number(gate.limit_cny || 0).toFixed(2) }}</b>
        —— 已停在 {{ gate.asked_by || '当前步骤' }}，新的付费任务不再提交。
        <button class="btn sm primary" @click="gateOpen = true">继续生成…</button>
        <span class="muted">（点「开始生成」也会被拦住，直到你确认）</span>
      </div>

      <Teleport to="body">
        <div v-if="gate?.pending && gateOpen" class="modal-mask" @click.self="gateOpen = false">
          <div class="modal" role="dialog" aria-modal="true">
            <h3>⏸ 当前费用已超上限，是否继续生成？</h3>
            <p class="muted">
              已花 <b>¥{{ Number(gate.spent_cny || 0).toFixed(2) }}</b> ／
              上限 <b>¥{{ Number(gate.limit_cny || 0).toFixed(2) }}</b>。
              停下来不会丢任何东西：已经出来的图与片段都在盘上，确认之后从停下那一处接着跑，
              已完成的部分不重复花钱。
            </p>
            <label>
              继续用到新上限 ¥
              <input v-model.number="newLimit" type="number" min="0" step="1"
                     :placeholder="String(suggestedLimit)" />
            </label>
            <p class="muted">留空 = 用建议值 ¥{{ suggestedLimit }}（不抬高上限就没法继续，会立刻再停一次）。</p>
            <div class="toolbar">
              <button class="btn primary" :disabled="gateBusy" @click="resumeBudget">
                {{ gateBusy ? '提交中…' : '✔ 确认继续生成' }}
              </button>
              <button class="btn" @click="gateOpen = false">暂不继续</button>
            </div>
            <p class="muted">「暂不继续」只是关掉这个窗口：上限没抬，「开始生成」仍然会被拦住。</p>
          </div>
        </div>
      </Teleport>

      <div v-if="costRisk" class="alert err">
        ⚠ 成本上限 <b>¥{{ Math.round(board.meta?.budget_limit_cny || 0) }}</b>
        低于预估 <b>¥{{ Math.round(costRisk.lo) }}</b>：到成本上限时会暂停任务。
      </div>
      <!-- 单价不确定的金额：成本上限按总额判，所以这部分必须说出来，不能假装精确 -->
      <div v-if="uncertainCost > 0" class="alert warn">
        ⚠ 合计里有 <b>¥{{ uncertainCost.toFixed(2) }}</b> 的单价是估算（模型不在价目表内，
        按能力默认价计）→ 真实账单可能更高。配置 <code>FV3_PRICING_URL</code>
        指向实时价目表，或补 <code>pricing.yaml</code>。
      </div>
      <div v-if="lastFrameGap" class="alert warn">
        ⚠ <b>{{ lastFrameGap.missing.length }}/{{ lastFrameGap.need }} 个镜头没有尾帧</b>
        （{{ lastFrameGap.missing.slice(0, 6).join('、')
        }}{{ lastFrameGap.missing.length > 6 ? '…' : '' }}）
        —— 这些镜头会退回<b>单首帧</b>生视频，末尾姿态/构图容易漂、切下一镜接不上。<br />
        <span class="muted">
          补法：点「开始生成」重跑<b>场景视频</b>这一步 —— 首尾帧按自己的缺口补，
          已有的首帧不会重出（不重复花钱）。
        </span>
      </div>
      <div v-if="lipsyncWaived && lipsyncWaived.length" class="alert warn">
        ⚠ {{ lipsyncWaived.length }} 个双人镜头按策略叠音不做口型（{{ lipsyncWaived.join('、') }}）：
        单音轨驱动会让没说话的人也动嘴。
      </div>
      <div v-if="qualityVerdict && qualityVerdict.passed === false" class="alert err">
         <b>人物一致性不合格</b>（自动重做 {{ qualityVerdict.rounds || 0 }} 轮后仍未达标）：
        {{ (qualityVerdict.worst_shots || []).map((w) => `${w.shot}(${w.face === false ? '脸漂移' : ''}${w.outfit === false ? '服装变色' : ''})`).join('、') || '见度量报告' }}
        —— 可换定妆照/删图后重做。
      </div>
      <div v-else-if="qualityVerdict && qualityVerdict.passed === null" class="alert warn">
        ⚠ {{ qualityVerdict.note || '一致性未度量' }}
      </div>

      <StepBar :run-id="props.runId" :steps="steps" :task-busy="task" @changed="load" />

      <div class="tabs subtabs">
        <button :class="{ active: pane === 'board' }" @click="pane = 'board'">🎬 制作看板</button>
        <button :class="{ active: pane === 'flow' }" @click="pane = 'flow'">
          ⏱ 事件流与成本<span class="cnt">{{ detail.events.length }}</span>
        </button>
      </div>

      <!-- v-show 而非 v-if：切页签不销毁看板，编辑/上传/重生成的本地状态不丢 -->
      <div v-show="pane === 'board'">
        <ProductionBoard :run-id="props.runId" :board="board" @changed="load" />
      </div>

      <div v-show="pane === 'flow'" class="grid2">
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
        </div>

        <!-- 成本 -->
        <div class="card">
          <h2>成本 · 合计 ¥{{ detail.cost.total_cny.toFixed(2) }}</h2>
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
      </div>
    </template>
  </div>
</template>
