# ============================================================
#  SAM 3D Body - Environment Setup Script (Windows / PowerShell)
#  Run this ONCE to install all dependencies
# ============================================================

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " SAM 3D Body - Karate Video Pipeline Setup" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Install yt-dlp (YouTube downloader)
Write-Host "`n[1/5] Installing yt-dlp..." -ForegroundColor Yellow
pip install yt-dlp --upgrade

# 2. Install ffmpeg-python wrapper (for video processing)
Write-Host "`n[2/5] Installing ffmpeg-python..." -ForegroundColor Yellow
pip install ffmpeg-python

# 3. Install huggingface_hub for model download
Write-Host "`n[3/5] Installing huggingface_hub..." -ForegroundColor Yellow
pip install huggingface_hub --upgrade

# 4. Install tqdm for progress bars
Write-Host "`n[4/5] Installing tqdm / opencv extras..." -ForegroundColor Yellow
pip install tqdm opencv-python imageio imageio-ffmpeg

# 5. Install torch if not present (RTX 4090 = CUDA 12.x)
Write-Host "`n[5/5] Checking PyTorch..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch not found! Installing PyTorch for CUDA 12.1..." -ForegroundColor Red
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

Write-Host "`n=====================================================" -ForegroundColor Green
Write-Host " Setup complete! Now run: python download_karate_video.py" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
