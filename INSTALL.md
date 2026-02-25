# Installation Guide

## Prerequisites

- **Python 3.11** (recommended via [conda](https://docs.conda.io/en/latest/miniconda.html) or [pyenv](https://github.com/pyenv/pyenv))
- **CUDA 12.x** compatible GPU with **16+ GB VRAM** (tested on RTX 4090)
- **ffmpeg** — required for video compilation (`winget install ffmpeg` on Windows, `brew install ffmpeg` on macOS)
- **Git** — for cloning dependencies

---

## 1. Create and Activate Environment

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
```

## 2. Install PyTorch

Install PyTorch with CUDA support following the [official instructions](https://pytorch.org/get-started/locally/).

```bash
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Python Dependencies

```bash
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset networkx==3.2.1 roma \
    joblib seaborn wandb appdirs ffmpeg cython jsonlines pytest loguru optree fvcore black \
    pycocotools tensorboard huggingface_hub tqdm imageio imageio-ffmpeg yt-dlp
```

## 4. Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

On Windows, you may need Visual Studio Build Tools. See `build_detectron2.bat` for a helper script.

## 5. Install MoGe (FOV Estimation)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

## 6. Install SAM3 (Optional — for SAM3 detector)

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install decord psutil
```

---

## Getting Model Checkpoints

We host model checkpoints on Hugging Face. **Available models:**
- [`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3) — DINOv3-H+ (840M params, best quality)
- [`facebook/sam-3d-body-vith`](https://huggingface.co/facebook/sam-3d-body-vith) — ViT-H (631M params, slightly faster)

> **Note:** You must **request access** on the HuggingFace repos above before downloading.

```bash
# Authenticate
huggingface-cli login

# Download all models (SAM 3D Body + ViTDet + MoGe2 + SAM2)
python download_models.py

# Or download only the primary model:
python download_models.py --skip-sam2 --skip-vitdet --skip-moge
```

---

## API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

- **GROQ_API_KEY** — Required for voice control in the Avatar Studio ([get one here](https://console.groq.com/))
- **HF_TOKEN** — Required to download gated model checkpoints ([get one here](https://huggingface.co/settings/tokens))

