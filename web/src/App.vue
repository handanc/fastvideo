<script setup>
import { ref } from 'vue'
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
</script>

<template>
  <header class="topbar">
    <div class="brand" @click="view = 'new'">🎬 FastVideo3<span>多智能体短剧生成面板</span></div>
    <nav>
      <button :class="{ active: view === 'new' }" @click="view = 'new'">新建任务</button>
      <button :class="{ active: view === 'runs' || view === 'detail' }" @click="view = 'runs'">
        运行历史
      </button>
      <button :class="{ active: view === 'doctor' }" @click="view = 'doctor'">环境自检</button>
    </nav>
  </header>

  <main>
    <NewRunForm v-if="view === 'new'" @created="openRun" />
    <RunList v-else-if="view === 'runs'" @select="openRun" />
    <RunDetail v-else-if="view === 'detail'" :run-id="selectedRun" @back="view = 'runs'" />
    <DoctorView v-else-if="view === 'doctor'" />
  </main>
</template>
