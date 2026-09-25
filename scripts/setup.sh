#!/usr/bin/env bash
# FastVideo3 一键环境搭建（Linux / macOS）—— 对应 TDD §3.2
set -e
cd "$(dirname "$0")/.."

echo "==> [1/6] 创建 conda 环境 fastvideo3 (Python 3.11)"
conda env create -f environment.yaml || {
    echo "conda env create 失败，回退为最小环境 + pip 安装"
    conda create -n fastvideo3 python=3.11 pip -y
}
eval "$(conda shell.bash hook)"
conda activate fastvideo3

echo "==> [2/6] 安装 Python 依赖（含 ffmpeg 兜底 imageio-ffmpeg）"
pip install -r requirements.txt

echo "==> [3/6] 安装可选本地视觉组件（CPU 版 torch，双人同屏用；可跳过）"
read -r -p "安装 torch(CPU) + ultralytics + rembg + streamlit? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "fastvideo3[full]" || pip install ultralytics "rembg[cpu]" streamlit librosa soundfile
fi

echo "==> [4/6] 以可编辑模式安装本项目（获得 fastvideo3 命令）"
pip install -e .

echo "==> [5/6] 配置 API KEY"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "已生成 .env，请填入 DEEPSEEK_API_KEY 与 DASHSCOPE_API_KEY"
fi

echo "==> [6/6] 下载本地小模型 + 环境自检"
python -m fastvideo3.utils.fetch_models || true
fastvideo3 doctor
echo "✔ 完成。跑通演示: fastvideo3 run --topic '校园喜剧' --duration 30 --fake"
