// 展示层的名字改写：**后端 id 一律不动**（scene_1 仍是 scene_1，血缘/文件名/接口都靠它），
// 只在界面上说人话。改这里不会影响任何落盘路径或 API 契约。

/** scene_1 → 场景1；s2 → 场景2；认不出的一律原样返回（宁可不改，不可改错）。 */
export function sceneLabel(id) {
  const s = String(id || '')
  const m = s.match(/^(?:scene_|s)(\d+)$/i)
  return m ? `场景${m[1]}` : s
}

/** 一批 id 的展示串（步骤条的"缺/待重做"清单用）。 */
export function sceneLabels(ids) {
  return (ids || []).map((x) => sceneLabel(x)).join(', ')
}
