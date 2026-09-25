// 运行状态 → 中文标签 + 徽章样式（与后端 _run_status 启发式一一对应）
export const STATUS = {
  done: { label: '已完成', cls: 'ok' },
  running: { label: '运行中', cls: 'run' },
  paused: { label: '待审片', cls: 'warn' },
  failed: { label: '失败', cls: 'err' },
  idle: { label: '已停止', cls: 'idle' },
}

export function statusOf(s) {
  return STATUS[s] || { label: s || '未知', cls: 'idle' }
}
