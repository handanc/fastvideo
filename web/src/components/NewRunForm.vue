<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'

const emit = defineEmits(['created'])

// fake 默认 false：此前默认 true，Web 面板发起的运行在未察觉时全走占位色块演示模式
const form = reactive({
  topic: '职场新人的第一天',
  style: '都市轻喜剧',
  duration: 30,
  fake: false,
  // 口型不再是选项：wan3.0 生成场景片段时原生带台词口型（音轨=模型原声）。
  // 这里不再写"该档必须配哪个模式"—— 历史上模式与路由被抄成两份，漂一次的后果不是报错而是
  // 静默降级：对白镜头先烧一份运动片段的钱、再 100% 出无口型叠音（run_20261011_164152 实测）。
  // **分辨率是唯一的规格旋钮**：成片尺寸、一致性深度、抽卡数、单价量级、是否自动重拍
  // 都由它带出来（后端 configs/profile_<档>.yaml）。成本预估与模型清单也按它问，
  // 所以不会再有"面板显示 720P、预估按另一档算"这种对不上。
  resolution: '1080P',
  aspect: '9:16',
  budget: 0,
  // 空串 = 跟随分辨率档（与改造前行为一致）。提交时原样发给后端，由 model_catalog 校验，
  // 名字打错/协议对不上会 400 并在这里摊开说明，不会先花钱再失败。
  model_overrides: { text: '', t2i: '', i2v: '' },
})

// ── 分辨率选项（后端给：成片尺寸 / 30s 预估 / 是否自动重拍 / 抽卡数）──────
const FALLBACK_RES = [
  { resolution: '480P', output_size: [540, 960], cost_30s: [5, 11], auto_reshoot: false, note: '' },
  { resolution: '720P', output_size: [720, 1280], cost_30s: [12, 24], auto_reshoot: true, note: '' },
  { resolution: '1080P', output_size: [1080, 1920], cost_30s: [36, 66], auto_reshoot: true, note: '' },
]
const resolutions = ref(FALLBACK_RES)
const currentRes = computed(
  () => resolutions.value.find((r) => r.resolution === form.resolution) || FALLBACK_RES[2]
)

async function loadResolutions() {
  try {
    const data = await api.resolutions()
    if (Array.isArray(data?.resolutions) && data.resolutions.length) resolutions.value = data.resolutions
  } catch {
    resolutions.value = FALLBACK_RES // 问不到就用本地兜底，表单不能因此开不起来
  }
}
onMounted(loadResolutions)
// 换分辨率就换内部档位（映射的权威在后端，这里只是把预估算对）
watch(() => form.resolution, () => {
  // 分辨率变了要重问预估与模型清单（这两个都按分辨率算）
  loadEst()
  loadModels()
})

// 选项清单来自后端；这份本地表只在问不到时兜底显示（价格与冲突一律不抄，永远问后端）
// ── 模型选择（text / 文生图 / 图生视频）：清单与"当前生效值"都来自后端 ─────────────
// 为什么值得单独做这一节：`.env` 里的 FV3_TEXT_MODEL / FV3_T2I_MODEL / FV3_I2V_MODEL 是
// **全局**开关 —— 改一次就把三个分辨率档的所有新 run 一起换掉，而 configs 的档位表看起来
// 还是原样（本轮排查就撞见"档位写着 wan2.2-i2v-plus、实际跑的是 wan3.0-video"）。
// 这里改成 run 级：选择落进 run_meta.json，每一步（含续跑）都按原样恢复，
// 不会出现"定妆照是新模型出的、首帧是旧模型出的"。
const modelCaps = ref([])
const CAP_TITLE = { text: '文本（编剧/评审）', t2i: '文生图（立绘/场景/首帧）', i2v: '图生视频（动作/空镜）' }

// 口型那一行的"默认 X"与初始选中值都来自后端：X = .env 的 FV3_TALK_MODEL 映射出的通道，
// 没设 .env 时 = 分辨率档位 yaml 的 talk_mode 对应通道。改 .env，这里显示跟着变。

async function loadModels() {
  try {
    const data = await api.modelChoices(form.resolution)
    modelCaps.value = data?.capabilities || []
    // 下拉里**没有"跟随默认"这个空选项**了，所以每个 select 必须有一个具体值：
    // 用后端算出的"当前生效模型"作初值（它来自 .env，其次分辨率档位）。
    // 若生效模型不在清单里（例如手改了 .env 写了个目录外的名字），退回清单第一项，
    // 免得 select 显示空白而 payload 发的是旧值。
    for (const cap of modelCaps.value) {
      const opts = (cap.choices || []).map((c) => c.model)
      const cur = form.model_overrides[cap.capability]
      if (!cur || !opts.includes(cur)) {
        form.model_overrides[cap.capability] =
          opts.includes(cap.effective_model) ? cap.effective_model : (opts[0] || '')
      }
    }
  } catch {
    modelCaps.value = []   // 问不到就不显示这一节；表单其余部分必须照常能用
  }
}
onMounted(loadModels)
watch(() => form.resolution, loadModels)

const capTitle = (cap) => CAP_TITLE[cap] || cap
const selectedNote = (cap) => {
  const model = form.model_overrides[cap.capability]
  return (cap.choices || []).find((c) => c.model === model)?.note || ''
}
// 不在 pricing.yaml 表内的模型：台账与熔断会按"能力默认价"估算，可能与真实账单不符
const selectedUnpriced = (cap) => {
  const model = form.model_overrides[cap.capability]
  if (!model) return false
  const row = (cap.choices || []).find((c) => c.model === model)
  return !!row && row.priced === false
}
const chosenModels = computed(() =>
  Object.entries(form.model_overrides).filter(([, v]) => v).map(([k, v]) => `${CAP_TITLE[k] || k} = ${v}`)
)

// 成本预估：**单价的权威来源是后端**（GET /api/cost-estimate → pricing.yaml + 供应商路由）。
// 下面这张表只在后端问不到时兜底，所以必须与 core/cost.py 的
// QUALITY_BASE_RATE_CNY_PER_SEC（不含口型）保持一致 —— 手抄表一旦漂了就等于让用户按
// 假数字填熔断：熔断中途跳闸的表现恰恰是"成片没有对口型"。
// 口型另按**路由实时单价**计入（由后端给，不在这里抄）：
//   A 链 wan2.2-s2v 480P ¥0.5/秒、720P ¥0.9/秒；B 链 videoretalk ¥0.08/秒（不分档）。
const FALLBACK_RATE = { '480P': [0.15, 0.35], '720P': [0.4, 0.8], '1080P': [1.2, 2.2] }
const est = ref(null)

async function loadEst() {
  if (form.fake) { est.value = null; return }
  try {
    est.value = await api.costEstimate(form.resolution, form.duration, 'scripted_talk')
  } catch {
    est.value = null   // 问不到就退回本地兜底量级（宁可提示不精确，也不能不提示）
  }
}
watch(() => [form.resolution, form.duration, form.fake], loadEst, { immediate: true })

const costHint = computed(() => {
  if (est.value) {
    return {
      lo: Math.round(est.value.lo), hi: Math.round(est.value.hi),
      fuse: Math.round(est.value.hi + 40),
      talk: est.value.talk_rate_per_sec, route: est.value.talk_route,
      talkSec: est.value.dialogue_sec, budgetDefault: est.value.budget_default_cny,
      // 后端按"模式解析出来的那条通道"给的价与警示（同一份来源，不会与装配漂移）
      channel: est.value.talk_channel || '',
      eats: est.value.talk_eats || '',
      warning: est.value.tier_warning || '',
    }
  }
  const [lo, hi] = FALLBACK_RATE[form.resolution] || FALLBACK_RATE['720P']
  return {
    lo: Math.round(form.duration * lo),
    hi: Math.round(form.duration * hi),
    fuse: Math.round(form.duration * hi + 40),
    talk: 0, route: '', talkSec: 0, budgetDefault: 120,
    channel: '', eats: '', warning: '',
  }
})

// 时长 1~3600：HTML 的 min/max 拦不住手输越界值，所以自己判一次并在提交前挡住
// （后端 RunCreate 同样校验，两边口径一致 —— 前端只是让用户当场看见，不用等 422）
const DURATION_MIN = 1
const DURATION_MAX = 3600
const durationInvalid = computed(
  () => !Number.isFinite(form.duration)
    || form.duration < DURATION_MIN
    || form.duration > DURATION_MAX
)

// 本次实际生效的熔断值（0 = 跟随 .env 的默认值）
const fuseEff = computed(() => (form.budget > 0 ? form.budget : costHint.value.budgetDefault))

const busy = ref(false)
const error = ref('')
const task = ref(null)

// 全局单任务锁：已有任务在跑时禁止再开（AI 短剧很贵，防重复付费）
async function pollTask() {
  try {
    task.value = await api.task()
  } catch {
    task.value = null
  }
}
onMounted(pollTask)

async function submit() {
  if (!form.topic.trim()) {
    error.value = '请填写短剧主题'
    return
  }
  if (durationInvalid.value) {
    error.value = `目标时长要在 ${DURATION_MIN}–${DURATION_MAX} 秒之间`
    return
  }
  busy.value = true
  error.value = ''
  try {
    // 建任务不花钱，也不占任务锁：它只写 run_meta.json，真正的生成在详情页一步一步来
    const res = await api.createRun({ ...form, topic: form.topic.trim() })
    emit('created', res.run_id)
  } catch (e) {
    error.value = `启动失败：${e.message}`
    await pollTask()
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="card">
    <h2>新建生成任务</h2>

    <!-- 3×2 布局：一行两个字段。字段名用外行看得懂的词（屏幕比例/成本上限），
         内部术语（aspect_ratio/budget/resolution_tier）不出现在界面上 -->
    <div class="row">
      <label class="field">
        <span>主题 *</span>
        <input v-model="form.topic" type="text" maxlength="200" placeholder="例如：职场新人的第一天" />
      </label>
      <label class="field">
        <span>风格</span>
        <input v-model="form.style" type="text" maxlength="100" placeholder="例如：都市轻喜剧" />
      </label>
    </div>

    <div class="row">
      <label class="field">
        <span>目标时长（秒）</span>
        <input v-model.number="form.duration" type="number" min="1" max="3600" step="1" />
        <span v-if="durationInvalid" class="fieldErr">时长要在 1–3600 秒之间</span>
      </label>
      <label class="field">
        <span>分辨率</span>
        <!-- 选项来自后端 GET /api/resolutions：分辨率→内部档位的映射只在后端一份 -->
        <select v-model="form.resolution">
          <option v-for="r in resolutions" :key="r.resolution" :value="r.resolution">
            {{ r.resolution }}
          </option>
        </select>
      </label>
    </div>

    <div class="row">
      <label class="field">
        <span>屏幕比例</span>
        <select v-model="form.aspect">
          <option value="9:16">9:16 竖屏</option>
          <option value="16:9">16:9 横屏</option>
          <option value="1:1">1:1 方形</option>
        </select>
      </label>
      <label class="field">
        <span>成本上限（元）</span>
        <!-- 0 = 跟随 .env 的默认值；界面不写"熔断"这种内部词 -->
        <input v-model.number="form.budget" type="number" min="0" max="100000" step="10" />
      </label>
    </div>

    <!-- 模式 × 分辨率的真实硬冲突（判据在后端，例如 videoretalk 要求片段边长 ≥640）-->
    <div v-if="!form.fake && costHint.warning" class="alert err">
      ⚠ {{ costHint.warning }}
    </div>

    <!-- ── 模型选择：清单与"当前生效值"来自 GET /api/model-choices（唯一来源 core/model_catalog.py）
         留空 = 跟随分辨率档；选择落进 run_meta.json，每一步（含续跑）都按原样恢复 ── -->
    <details v-if="modelCaps.length" class="models" :open="chosenModels.length > 0">
      <summary>模型选择 — 文本 / 文生图 / 图生视频（wan3.0 自带台词口型）</summary>
      <!-- 每行标题只说"默认 X"：X 由后端算出（.env 的 FV3_*_MODEL 优先，其次分辨率档位），
           改 .env 前端显示跟着变，不在 JS 里抄第二份默认值 -->
      <div v-for="cap in modelCaps" :key="cap.capability" class="modelRow">
        <label class="field">
          <span>{{ capTitle(cap.capability) }} <em class="muted">默认 {{ cap.effective_model || '—' }}</em></span>
          <select v-model="form.model_overrides[cap.capability]">
            <option v-for="c in cap.choices" :key="c.model" :value="c.model">
              {{ c.label }}
            </option>
          </select>
        </label>
        <div v-if="selectedNote(cap)" class="modelNote">{{ selectedNote(cap) }}</div>
        <div v-if="selectedUnpriced(cap)" class="modelNote warnText">
          ⚠ 该模型暂无单价数据 → 成本按默认价估算，可能与真实账单不符
        </div>
      </div>
    </details>

    <label class="check">
      <input v-model="form.fake" type="checkbox" />
      Fake 模式（零成本离线演示，无需任何 API KEY）
    </label>

    <div v-if="form.fake" class="alert warn">
      ⚠ Fake 模式产出<b>占位色块 + 静帧 + 静音</b>，仅演示流程，不是真实 AI 生成
    </div>
    <!-- 文案按"外行一眼看懂"收敛：只说分辨率、预估区间、建议上限，
         不出现 resolution_tier/熔断/口型单价×秒数这些内部术语 -->
    <div v-else class="alert warn">
      {{ form.resolution }} 档预估 <b>¥{{ costHint.lo }}–{{ costHint.hi }}</b>，
      建议成本上限 ≥¥{{ costHint.fuse }}
    </div>
    <div v-if="!form.fake && fuseEff < costHint.lo" class="alert err">
      ⚠ 成本上限 <b>¥{{ fuseEff }}</b> 低于预估 <b>¥{{ costHint.lo }}</b>：
      钱用完时对白镜头会<b>没有口型</b>。请提到 ≥¥{{ costHint.fuse }}，或缩短时长。
    </div>
    <div v-if="!form.fake && form.resolution === '480P'" class="alert err">
      ⚠ 480P <b>不做人物一致性保证</b>（不锁脸、不自动重做），仅试跑用；交付请选 720P 或 1080P。
    </div>
    <div v-else-if="!form.fake" class="alert ok">
      ✔ {{ form.resolution }}：参考图锁脸；成片前自动核对人物一致性，不达标自动重做 ≤2 轮。
    </div>
    <div v-if="task?.busy" class="alert warn">
      任务 <b>{{ task.run_id }}</b> 正在生成中（{{ task.kind }}）；新任务要等它完成才能开始生成。
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>

    <div class="toolbar">
      <button class="btn primary" :disabled="busy" @click="submit">
        {{ busy ? '建任务中…' : '▶ 开始生成' }}
      </button>
      <span class="muted">按照顺序逐步完成任务</span>
    </div>
  </div>
</template>
