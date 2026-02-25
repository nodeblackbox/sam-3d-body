# Karate Video → SAM 3D Body Mesh Pipeline

## Quick Summary

This pipeline takes the karate YouTube video and processes it through Meta's SAM 3D Body
model to produce full-body 3D human mesh reconstructions, frame by frame.

---

## Step 1: Download the Video

### Option A: Export cookies.txt (Most Reliable)
1. Install [**Get cookies.txt LOCALLY**](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) in Chrome or Edge
2. Go to **https://youtube.com** while **signed in to your Google account**
3. Click the extension icon → **"Export"** → **"For current site only"**
4. Save the file as **`cookies.txt`** in this folder (`sam-3d-body/`)
5. Run:
   ```powershell
   python download_karate_video.py
   ```

### Option B: Download Manually
1. Open this URL in your browser: `https://www.youtube.com/watch?v=1MrRmimBJoA`
2. Use a browser extension (e.g., Video DownloadHelper) to download as **mp4**
3. Save the file to: `input_video/karate_video.mp4`

### Option C: Use yt-dlp with your logged-in Edge/Chrome
```powershell
# Close Edge/Chrome first, then run:
yt-dlp --cookies-from-browser edge -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]" --merge-output-format mp4 -o "input_video/karate_video.mp4" https://www.youtube.com/watch?v=1MrRmimBJoA
```

---

## Step 2: Install ffmpeg (Required for video compilation)
```powershell
winget install ffmpeg
# OR
choco install ffmpeg
# Then restart the terminal
```

---

## Step 3: Install the Python Environment
```powershell
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset networkx==3.2.1 roma joblib seaborn wandb appdirs ffmpeg cython jsonlines pytest loguru optree fvcore black pycocotools tensorboard huggingface_hub tqdm imageio imageio-ffmpeg yt-dlp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install git+https://github.com/microsoft/MoGe.git
```

---

## Step 4: Download Model Checkpoints

⚠️ **You must first request access on HuggingFace:**
- https://huggingface.co/facebook/sam-3d-body-dinov3
- https://huggingface.co/facebook/sam-3d-body-vith

Then authenticate:
```powershell
huggingface-cli login
# OR set environment variable:
$env:HF_TOKEN = "hf_your_token_here"
```

Download:
```powershell
python download_models.py                    # Downloads dinov3 (recommended)
python download_models.py --model vith       # Download ViT-H instead
python download_models.py --model all        # Download both
```

Or manually:
```powershell
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

---

## Step 5: Run the 3D Body Pipeline

### Basic run (body-only, fastest):
```powershell
python process_karate_video.py --inference_type body
```

### Full run (body + hands, best quality):
```powershell
python process_karate_video.py --inference_type full
```

### Process every 2nd frame (2x faster, good for long videos):
```powershell
python process_karate_video.py --inference_type full --frame_skip 2
```

### Process a specific segment (frames 100-500):
```powershell
python process_karate_video.py --start_frame 100 --end_frame 500
```

### With mask conditioning (highest quality but requires SAM2):
```powershell
python process_karate_video.py --use_mask
```

### Full command with all paths specified:
```powershell
python process_karate_video.py `
    --video_path "input_video/karate_video.mp4" `
    --checkpoint_path "checkpoints/sam-3d-body-dinov3/model.ckpt" `
    --mhr_path "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt" `
    --inference_type full `
    --frame_skip 1 `
    --bbox_thresh 0.5
```

---

## Architecture Overview

```
Video Frame
    ↓
[ViTDet Detector] → person bounding boxes (auto-downloaded ~2.5GB)
    ↓
[MoGe2 FOV Estimator] → camera intrinsics K (auto-downloaded ~1.2GB)
    ↓
[DINOv3-H+ Backbone 840M] → image patch embeddings
    ↓
[CameraEncoder] → ray-conditioned features
    ↓
[PromptableDecoder × N layers] → iteratively refined pose tokens
    ↓
[MHR Head] → 18,439 mesh vertices, 70 joints
    ↓
[Renderer/pyrender] → front view + side view visualization
    ↓
Output JPEG (4 panels: original | 2D skeleton | 3D mesh front | 3D mesh side)
```

---

## GPU Requirements

| Mode | VRAM | Speed (RTX 4090) | Quality |
|------|------|-------------------|---------|
| Body-only, DINOv3-H+ | ~14 GB | ~0.8s/frame | Excellent |
| Full (body+hands), DINOv3-H+ | ~16 GB | ~1.5s/frame | Best |
| Full + SAM2 mask, DINOv3-H+ | ~20 GB | ~2.5s/frame | Maximum |

A GPU with at least **16 GB VRAM** is recommended. Use `--frame_skip 2` or `--inference_type body` to reduce requirements.

---

## Output Files

```
sam-3d-body/
├── input_video/
│   ├── karate_video.mp4          (downloaded video)
│   └── frames/                   (extracted frames)
│       └── frame_000000.jpg ...
└── output_video/
    ├── visualized_frames/        (processed frames with 3D mesh overlay)
    │   └── frame_000000.jpg ...
    └── karate_3d_body.mp4        (final compiled video)
```

Each output frame shows 4 panels side-by-side:
1. **Original** — untouched input frame
2. **2D Skeleton** — 70-joint MHR70 keypoints projected onto image
3. **3D Mesh (front)** — Semi-transparent blue 3D body mesh overlaid
4. **3D Mesh (side)** — 90° side view of the 3D mesh on white background
