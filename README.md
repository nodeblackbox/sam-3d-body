# Karate 3D Avatar Studio

**Turn a karate instructional video into an interactive 3D avatar you can control with your voice.**

This project builds an end-to-end pipeline on top of Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) model to extract 3D human meshes from video, label them with karate technique names, and present them in an interactive web-based 3D studio with agentic voice control.

<p align="left">
<a href="https://arxiv.org/abs/2602.15989"><img src="https://img.shields.io/badge/arXiv-2602.15989-b31b1b.svg" alt="arXiv"></a>
<a href="https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"><img src='https://img.shields.io/badge/Meta_AI-Paper-4A90E2?logo=meta&logoColor=white' alt='Paper'></a>
<a href="https://ai.meta.com/blog/sam-3d/"><img src='https://img.shields.io/badge/Project_Page-Blog-9B72F0?logo=googledocs&logoColor=white' alt='Blog'></a>
</p>

---

## What This Project Does

1. **Downloads a karate kata video** (Heian Sandan) from YouTube
2. **Extracts every frame** and labels them with karate technique names from a transcript
3. **Runs SAM 3D Body** on each frame to recover full 3D body meshes (18,439 vertices, 70 joints)
4. **Builds a pose library** — averaged 3D poses indexed by technique name (e.g., "kiba-dachi", "oizuki")
5. **Exports mesh data** to binary format for real-time web rendering
6. **Serves an interactive 3D Avatar Studio** (Three.js) with timeline, video sync, and segment labeling
7. **Voice-controlled agent** — say "show me a kick" and an LLM (Llama 3.3 70B via Groq) navigates to the matching 3D animation
8. **LLM Pose Controller** — programmatic JSON commands to interpolate between any two poses and render animated transitions

---

## Architecture Overview

```
YouTube Video (Karate Heian Sandan)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FRAME EXTRACTION                                       │
│  download_karate_video.py → karate_pose_pipeline.py     │
│  OpenCV: extract frames, resize, label with transcript  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  3D MESH RECOVERY (per frame)                           │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │ YOLOv8 /     │   │ MoGe2 FOV    │   │ SAM2       │  │
│  │ ViTDet       │──▶│ Estimator    │──▶│ Segmentor  │  │
│  │ (Detector)   │   │ (Camera K)   │   │ (optional) │  │
│  └──────┬───────┘   └──────┬───────┘   └─────┬──────┘  │
│         │                  │                  │         │
│         ▼                  ▼                  ▼         │
│  ┌─────────────────────────────────────────────────┐    │
│  │         SAM 3D Body (DINOv3-H+ 840M)           │    │
│  │                                                  │    │
│  │  Image Patches → DINOv3 Backbone → Embeddings   │    │
│  │  CameraEncoder → Ray-Conditioned Features        │    │
│  │  PromptableDecoder (N layers) → Pose Tokens     │    │
│  │  MHR Head → 18,439 vertices + 70 joints          │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────────┐
        ▼                 ▼                     ▼
 ┌──────────┐     ┌──────────────┐     ┌──────────────┐
 │ .npz per  │     │ Visualization│     │ Pose Library │
 │ frame     │     │ JPEG panels  │     │ (averaged    │
 │ (vertices,│     │ (original +  │     │ 3D poses per │
 │  joints,  │     │  skeleton +  │     │ technique)   │
 │  params)  │     │  mesh views) │     │              │
 └──────┬────┘     └──────┬───────┘     └──────┬───────┘
        │                 │                    │
        ▼                 ▼                    ▼
 ┌────────────────────────────────────────────────────────┐
 │  WEB EXPORT & AVATAR STUDIO                            │
 │                                                         │
 │  export_web_data.py → faces.bin + frame_*.bin           │
 │                                                         │
 │  avatar_studio.html (Three.js)                          │
 │  ├── 3D Mesh Viewer with orbit controls                 │
 │  ├── Video player synced to 3D frames                   │
 │  ├── Timeline scrubber with colored technique segments  │
 │  ├── Segment labeling / editing UI                      │
 │  ├── Voice Control Agent (Web Speech API → Groq LLM)   │
 │  └── Export (JSON labels, animation data)               │
 │                                                         │
 │  llm_pose_controller.py                                 │
 │  └── JSON command → interpolated 3D transition video    │
 └────────────────────────────────────────────────────────┘
```

---

## Models Used

This project orchestrates **5 different AI models** in a single pipeline:

| # | Model | Role | Size | Source |
|---|-------|------|------|--------|
| 1 | **SAM 3D Body (DINOv3-H+)** | Core 3D human mesh recovery from single images | 840M params (~3.5 GB) | [HuggingFace](https://huggingface.co/facebook/sam-3d-body-dinov3) |
| 2 | **ViTDet (Cascade Mask R-CNN)** | Human detection — finds person bounding boxes in frames | ~2.5 GB | [Detectron2](https://github.com/facebookresearch/detectron2) |
| 3 | **MoGe2** | Field-of-view estimation — predicts camera intrinsics K | ~1.2 GB | [HuggingFace](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| 4 | **SAM2.1 (Hiera-Large)** | Human segmentation masks (optional, highest quality) | ~900 MB | [Meta](https://github.com/facebookresearch/sam2) |
| 5 | **Llama 3.3 70B** | Voice command intent recognition (via Groq API) | Cloud API | [Groq](https://console.groq.com/) |

**Additional:** YOLOv8n is included as a lightweight alternative detector.

### Momentum Human Rig (MHR)

The output mesh uses Meta's [MHR](https://github.com/facebookresearch/MHR) parametric body model, which produces:
- **18,439 mesh vertices** per person per frame
- **70 3D joints** (MHR70 skeleton — body, hands, feet)
- **Body pose parameters** + **shape parameters**
- **Camera translation** and **focal length**

---

## Quick Start

### 1. Install Dependencies

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
```

See [INSTALL.md](INSTALL.md) for the full list of pip dependencies.

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env and add your Groq API key (for voice control)
# Add your HuggingFace token (for model downloads)
```

### 3. Download Model Checkpoints

```bash
# Authenticate with HuggingFace (you must request access first)
huggingface-cli login

# Download SAM 3D Body + ViTDet + MoGe2 + SAM2
python download_models.py
```

### 4. Download & Process the Karate Video

```bash
python download_karate_video.py
python karate_pose_pipeline.py
```

### 5. Export & Launch the 3D Studio

```bash
python export_web_data.py
python -m http.server 8080
# Open http://localhost:8080/avatar_studio.html
```

Or on Windows, just run:
```powershell
run_studio.bat
```

---

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `download_karate_video.py` | Downloads the karate Heian Sandan video from YouTube (with cookie auth support) |
| `download_models.py` | Downloads all model checkpoints from HuggingFace (SAM 3D Body, ViTDet, MoGe2, SAM2) |
| `karate_pose_pipeline.py` | **Main pipeline** — extracts frames, labels with transcript, runs SAM 3D Body, builds pose library |
| `process_karate_video.py` | Alternative video processing script (no transcript labeling, simpler) |
| `export_web_data.py` | Exports mesh data to binary format for the Three.js web viewer |
| `render_avatar_video.py` | Renders a standalone avatar-only video from the mesh data |
| `llm_pose_controller.py` | Accepts JSON pose commands and generates interpolated transition videos |
| `demo.py` | Original SAM 3D Body demo — single image inference |
| `run_dancing.py` | Quick test script on a sample image |

---

## 3D Avatar Studio (Web UI)

The `avatar_studio.html` file is a full-featured web application built with **Three.js** that provides:

- **3D Mesh Viewer** — orbit controls, PBR lighting, wireframe toggle, color presets
- **Video Sync** — source video synced frame-by-frame to the 3D mesh
- **Split View** — side-by-side 3D avatar and video
- **Timeline Scrubber** — color-coded technique segments, playback controls (0.25x–4x speed)
- **Segment Editor** — label frame ranges with technique names and categories
- **Smooth Morphing** — vertex interpolation between frames for fluid animation
- **Keyboard Shortcuts** — Space (play/pause), Arrow keys (step), Home/End (skip)
- **Export** — save labeled segments as JSON

### Voice Control Agent

The studio includes an **agentic voice control** system:

1. Click the **Voice Control** button (or use Chrome/Edge with microphone)
2. Speak a command like *"show me a horse stance"* or *"do a crescent kick"*
3. The **Web Speech API** transcribes your voice
4. The transcript is sent to **Llama 3.3 70B** (via Groq API) with the list of available techniques
5. The LLM returns a structured JSON action (`goto_move`, `play`, `stop`)
6. The studio navigates to the matching 3D animation and starts playback

This requires a [Groq API key](https://console.groq.com/) — the studio will prompt you on first use.

---

## LLM Pose Controller

The `llm_pose_controller.py` script enables **programmatic control** of 3D pose transitions:

```python
command = {
    "start_pose": "yoi ready stance",
    "end_pose": "kiba-dachi horse riding stance",
    "air_time": 2.0,       # transition duration in seconds
    "rotation": 15          # Y-axis rotation in degrees
}
```

This will:
1. Look up both poses in the pose library (averaged 3D vertices per technique)
2. Generate N interpolated frames between start and end
3. Apply optional Y-axis rotation with ease-in/out
4. Render front + side view video using the SAM 3D Body renderer

---

## Output Structure

After running the full pipeline:

```
sam-3d-body/
├── karate_frames/                  # Extracted video frames
│   └── frame_000000.jpg ...
├── karate_output/
│   ├── visualized_frames/          # 4-panel visualizations per frame
│   │   └── frame_000000.jpg ...    # (original | skeleton | mesh front | mesh side)
│   ├── mesh_data/                  # Per-frame 3D data (.npz)
│   │   └── frame_000000.npz ...    # (vertices, joints, pose params, camera)
│   ├── web/                        # Binary data for Three.js viewer
│   │   ├── faces.bin               # Mesh face topology (int32)
│   │   ├── frame_*.bin             # Per-frame vertices (float32)
│   │   └── manifest.json           # Frame metadata index
│   ├── pose_timeline.csv           # Frame → technique → 3D joint positions
│   ├── pose_library.npz            # Averaged 3D poses per technique
│   ├── pose_library_index.json     # Human-readable technique index
│   └── heian_sandan_3d.mp4         # Compiled output video
└── karate_transcript.json          # Technique labels with timestamps
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

## SAM 3D Body (Upstream Model)

This project is built on **SAM 3D Body** by Meta Superintelligence Labs.

**SAM 3D Body (3DB)** is a promptable model for single-image full-body 3D human mesh recovery (HMR). It uses an encoder-decoder architecture with a DINOv3-H+ backbone and supports auxiliary prompts (2D keypoints, masks). Trained on high-quality annotations from multi-view geometry and differentiable optimization.

### Checkpoints

|      **Backbone (size)**       | **3DPW (MPJPE)** |    **EMDB (MPJPE)**     | **RICH (PVE)** | **COCO (PCK@.05)** |  **LSPET (PCK@.05)** | **Freihand (PA-MPJPE)**
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :----------------: | :----------------: |
|  DINOv3-H+ (840M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/model.ckpt))   |      54.8      |          61.7         |       60.3        |       86.5        | 68.0 | 5.5
|   ViT-H  (631M) <br /> ([config](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model_config.yaml), [checkpoint](https://huggingface.co/facebook/sam-3d-body-vith/blob/main/model.ckpt))    |     54.8   |         62.9         |       61.7        |        86.8       | 68.9 |  5.5

### Quick Single-Image Demo

```python
import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
img_bgr = cv2.imread("path/to/image.jpg")
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

For the complete upstream demo, see [notebook/demo_human.ipynb](notebook/demo_human.ipynb).

---

## Project Structure

```
sam-3d-body/
├── sam_3d_body/                    # Core SAM 3D Body model (Meta upstream)
│   ├── models/                     # Encoder, decoder, heads
│   ├── visualization/              # Renderer, skeleton visualizer
│   ├── data/                       # Data loading utilities
│   ├── metadata/                   # MHR70 joint definitions
│   └── utils/                      # Model utilities
├── tools/                          # Detector, FOV estimator, segmentor builders
│   ├── build_detector.py           # YOLOv8 / ViTDet human detector
│   ├── build_fov_estimator.py      # MoGe2 field-of-view estimator
│   ├── build_sam.py                # SAM2 human segmentor
│   └── vis_utils.py                # Visualization helpers
├── notebook/                       # Jupyter demo notebook
├── data/                           # Dataset download scripts (upstream)
├── karate_pose_pipeline.py         # Main video → 3D mesh pipeline
├── llm_pose_controller.py          # LLM-driven pose interpolation
├── avatar_studio.html              # Interactive 3D web studio
├── export_web_data.py              # Mesh → binary for Three.js
├── download_karate_video.py        # YouTube video downloader
├── download_models.py              # Model checkpoint downloader
├── karate_transcript.json          # Karate technique labels
├── .env.example                    # API key template
├── INSTALL.md                      # Dependency installation guide
├── KARATE_PIPELINE_README.md       # Detailed pipeline documentation
└── LICENSE                         # SAM License
```

---

## License

The SAM 3D Body model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing SAM 3D Body

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}
```
