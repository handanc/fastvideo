"""Streamlit 可视化面板（教学第 6 讲观测素材）。

启动：streamlit run fastvideo3/app/streamlit_app.py
功能：新建任务（只建，不花钱）、点一次补一步、事件流时间线、产物画廊、成本看板。

与 Vue 面板（`fastvideo3 serve`）同一套后端命令：`new` 建任务、`step` 推进一步。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from fastvideo3.core.config import OUTPUT_DIR
from fastvideo3.core.storage import list_runs

st.set_page_config(page_title="FastVideo3", layout="wide")
st.title("🎬 FastVideo3 —— 短剧生成面板（按顺序逐步完成）")

tab1, tab2 = st.tabs(["新建任务", "运行详情"])


def _spawn(*argv: str) -> None:
    subprocess.Popen([sys.executable, "-m", "fastvideo3.app.cli", *argv])  # noqa: S603


with tab1:
    with st.form("new_run"):
        topic = st.text_input("主题", "职场新人的第一天")
        style = st.text_input("风格", "都市轻喜剧")
        duration = st.slider("目标时长(秒)", 15, 300, 30, 15)
        fake = st.checkbox("Fake 模式（零成本演示）", value=True)
        if st.form_submit_button("开始生成"):
            cmd = ["new", "--topic", topic, "--style", style, "--duration", str(duration)]
            if fake:
                cmd.append("--fake")
            _spawn(*cmd)
            st.success("任务已建（不花钱）。请到「运行详情」选它，再点「开始生成」一步一步补。")
    st.caption("按照顺序逐步完成任务：剧本 → 角色图 → 场景图 → 分镜图 → 镜头视频 → 成片。")

with tab2:
    runs = list_runs()
    if not runs:
        st.info("暂无运行记录")
    else:
        run_id = st.selectbox("选择运行", runs)
        root: Path = OUTPUT_DIR / run_id

        if st.button("▶ 开始生成（补最靠前的未完成步骤）", type="primary"):
            _spawn("step", run_id)
            st.info("已在后台启动这一步，稍后刷新查看进度。")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("事件流")
            events_file = root / "events.jsonl"
            if events_file.exists():
                for line in events_file.read_text(encoding="utf-8").splitlines()[-30:]:
                    e = json.loads(line)
                    st.text(f"[{e.get('ts')}] {e.get('node')}:{e.get('status')}")
            if (root / "cost.json").exists():
                st.subheader("成本")
                costs = json.loads((root / "cost.json").read_text(encoding="utf-8"))
                st.metric("合计(¥)", round(sum(c["amount_cny"] for c in costs), 2))

        with col2:
            st.subheader("产物画廊")
            for label, folder in [("资产", "assets"), ("分镜", "storyboard")]:
                imgs = sorted((root / folder).glob("*.png")) if (root / folder).exists() else []
                if imgs:
                    st.caption(label)
                    st.image([str(p) for p in imgs[:6]], width=150)
            final = root / "final" / "final.mp4"
            if final.exists():
                st.caption("成片")
                st.video(str(final))
            for folder in ("clips", "talk"):
                vids = sorted((root / folder).glob("*.mp4")) if (root / folder).exists() else []
                if vids:
                    st.caption(folder)
                    st.video(str(vids[0]))
