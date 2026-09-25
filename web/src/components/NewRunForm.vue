<script setup>
import { computed, reactive, ref } from 'vue'
import { api } from '../api'

const emit = defineEmits(['created'])

// fake 默认 false：此前默认 true，Web 面板发起的运行在未察觉时全走占位色块演示模式
const form = reactive({
  topic: '职场新人的第一天',
  style: '都市轻喜剧',
  duration: 30,
  fake: false,
  review: 'auto',
  talk_mode: 's2v_direct',
  quality: 'cinema',
  aspect: '9:16',
  budget: 0,
})

// 各档位每秒成本量级（元），用于预估提示
const QUALITY_RATE = { economy: [0.15, 0.35], high: [0.4, 0.8], cinema: [1.2, 2.2] }
const costHint = computed(() => {
  const [lo, hi] = QUALITY_RATE[form.quality] || QUALITY_RATE.economy
  return {
    lo: Math.round(form.duration * lo),
    hi: Math.round(form.duration * hi),
    fuse: Math.round(form.duration * hi + 40),
  }
})

const busy = ref(false)
const error = ref('')

async function submit() {
  if (!form.topic.trim()) {
    error.value = '请填写短剧主题'
    return
  }
  busy.value = true
  error.value = ''
  try {
    const res = await api.createRun({ ...form, topic: form.topic.trim() })
    emit('created', res.run_id)
  } catch (e) {
    error.value = `启动失败：${e.message}`
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="card">
    <h2>新建生成任务</h2>

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

    <label class="field">
      <span>目标时长：{{ form.duration }} 秒</span>
      <input v-model.number="form.duration" type="range" min="15" max="300" step="15" />
    </label>

    <div class="row">
      <label class="field">
        <span>审片模式</span>
        <select v-model="form.review">
          <option value="auto">auto · 全自动</option>
          <option value="gated">gated · 人工审片</option>
        </select>
      </label>
      <label class="field">
        <span>口型模式</span>
        <select v-model="form.talk_mode">
          <option value="s2v_direct">s2v_direct · 对话直出(默认)</option>
          <option value="v2v_overlay">v2v_overlay · 视频换口型</option>
          <option value="audio_overlay">audio_overlay · 音频叠加(低成本)</option>
        </select>
      </label>
      <label class="field">
        <span>画质档位</span>
        <select v-model="form.quality">
          <option value="economy">economy · 教学</option>
          <option value="high">high · 交付</option>
          <option value="cinema">cinema · 商业级(约¥1/秒)</option>
        </select>
      </label>
    </div>

    <div class="row">
      <label class="field">
        <span>画幅</span>
        <select v-model="form.aspect">
          <option value="9:16">9:16 竖屏</option>
          <option value="16:9">16:9 横屏</option>
          <option value="1:1">1:1 方形</option>
        </select>
      </label>
      <label class="field">
        <span>成本熔断(元) · 0=跟随 .env</span>
        <input v-model.number="form.budget" type="number" min="0" max="100000" step="10" />
      </label>
    </div>

    <label class="check">
      <input v-model="form.fake" type="checkbox" />
      Fake 模式（零成本离线演示，无需任何 API KEY）
    </label>

    <div v-if="form.fake" class="alert warn">
      ⚠ Fake 模式产出<b>占位色块图 + 静帧片段 + 静音音轨</b>，仅用于流程演示，不是真实 AI 生成
    </div>
    <div v-else class="alert warn">
      {{ form.quality }} 档预估成本 <b>¥{{ costHint.lo }}-{{ costHint.hi }}</b>（{{ form.duration }}s 成片）；
      熔断填 0 跟随 .env（默认 ¥120），建议 ≥¥{{ costHint.fuse }} 防中途停跑
    </div>

    <div v-if="error" class="alert err">{{ error }}</div>

    <div class="toolbar">
      <button class="btn primary" :disabled="busy" @click="submit">
        {{ busy ? '启动中…' : '🚀 开始生成' }}
      </button>
      <span class="muted">任务将在后台运行，自动跳转到详情页跟踪进度</span>
    </div>
  </div>
</template>
