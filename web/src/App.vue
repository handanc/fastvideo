<script setup>
import { onMounted, ref } from 'vue'
import NewRunForm from './components/NewRunForm.vue'
import RunList from './components/RunList.vue'
import RunDetail from './components/RunDetail.vue'
import DoctorView from './components/DoctorView.vue'

// 轻量视图切换（不引入 vue-router，保持依赖最小）
const view = ref('new') // new | runs | detail | doctor
const selectedRun = ref('')

function openRun(id) {
  selectedRun.value = id
  view.value = 'detail'
}

// ── 配色主题：默认深黑，右上角那颗按钮切亮/暗，图标自己显示当前能切到哪一边 ──
// 存 localStorage（只存 'dark'/'light' 这个选择，不存任何内容数据）；读不到就按默认深黑。
// 首屏那一次赋值在 index.html 的内联脚本里做（避免"先黑一帧再变白"的闪烁），
// 这里 onMounted 再同步一次，保证 JS 侧的 theme 与实际 DOM 一致。
const THEME_KEY = 'fv3.theme'
const readTheme = () => {
  try {
    return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark' // 隐私模式 / 禁用存储：默认深黑照常可用
  }
}
const theme = ref(readTheme())

function applyTheme(t) {
  if (t === 'light') document.documentElement.dataset.theme = 'light'
  else delete document.documentElement.dataset.theme
}

function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  applyTheme(theme.value)
  try {
    localStorage.setItem(THEME_KEY, theme.value)
  } catch {
    /* 存不下就算了：本次会话仍然切换成功，只是下次回来是默认深黑 */
  }
}
onMounted(() => applyTheme(theme.value))
</script>

<template>
  <header class="topbar">
    <div class="brand" @click="view = 'new'">🎬 FastVideo3-AI短剧生成系统</div>
    <div class="topbarRight">
      <nav>
        <button :class="{ active: view === 'new' }" @click="view = 'new'">新建任务</button>
        <button :class="{ active: view === 'runs' || view === 'detail' }" @click="view = 'runs'">
          运行历史
        </button>
        <button :class="{ active: view === 'doctor' }" @click="view = 'doctor'">环境自检</button>
      </nav>
      <!-- 深黑时给太阳（点了变亮），亮色时给月亮（点了变暗）；title/aria 让意图不用猜 -->
      <button
        class="themeBtn"
        :title="theme === 'dark' ? '切换为白色背景' : '切换为黑色背景'"
        :aria-label="theme === 'dark' ? '切换为白色背景' : '切换为黑色背景'"
        @click="toggleTheme"
      >
        {{ theme === 'dark' ? '☀️' : '🌙' }}
      </button>
    </div>
  </header>

  <main>
    <NewRunForm v-if="view === 'new'" @created="openRun" />
    <RunList v-else-if="view === 'runs'" @select="openRun" />
    <RunDetail v-else-if="view === 'detail'" :run-id="selectedRun" @back="view = 'runs'" />
    <DoctorView v-else-if="view === 'doctor'" />
  </main>
</template>
