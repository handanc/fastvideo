# FastVideo3 一键环境搭建（Windows PowerShell）—— 对应 TDD §3.2
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

Write-Host "==> [1/6] 创建 conda 环境 fastvideo3 (Python 3.11)"
conda env create -f environment.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "conda env create 失败，回退为最小环境 + pip 安装"
    conda create -n fastvideo3 python=3.11 pip -y
}
conda activate fastvideo3

Write-Host "==> [2/6] 安装 Python 依赖（含 ffmpeg 兜底 imageio-ffmpeg）"
pip install -r requirements.txt

Write-Host "==> [3/6] 安装可选本地视觉组件（CPU 版 torch，双人同屏用）"
$ans = Read-Host "安装 torch(CPU) + ultralytics + rembg + streamlit? [y/N]"
if ($ans -match '^[Yy]') {
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics "rembg[cpu]" streamlit librosa soundfile
}

Write-Host "==> [4/6] 以可编辑模式安装本项目（获得 fastvideo3 命令）"
pip install -e .

Write-Host "==> [5/6] 配置 API KEY"
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "已生成 .env，请填入 DEEPSEEK_API_KEY 与 DASHSCOPE_API_KEY"
}

Write-Host "==> [6/6] 下载本地小模型 + 环境自检"
python -m fastvideo3.utils.fetch_models
fastvideo3 doctor
Write-Host "完成。跑通演示: fastvideo3 run --topic '校园喜剧' --duration 30 --fake"
