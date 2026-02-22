"""
Download SAM 3D Body model checkpoints from HuggingFace.

Models needed:
  1. facebook/sam-3d-body-dinov3   (primary - 840M DINOv3-H+)
  2. facebook/sam-3d-body-vith     (alternative - 631M ViT-H)
  3. ViTDet detector               (auto-downloaded by build_detector.py)
  4. MoGe2 FOV estimator           (auto-downloaded from HuggingFace Hub)
  5. SAM2 segmentor                 (optional - manual download)

NOTE: You MUST request access on HuggingFace first:
  https://huggingface.co/facebook/sam-3d-body-dinov3
  https://huggingface.co/facebook/sam-3d-body-vith

Usage:
    python download_models.py
    python download_models.py --model vith        # Download ViT-H instead
    python download_models.py --model all          # Download both
    python download_models.py --skip-sam2          # Skip SAM2 download
"""

import argparse
import os
import subprocess
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

MODEL_CONFIGS = {
    "dinov3": {
        "repo_id": "facebook/sam-3d-body-dinov3",
        "local_dir": os.path.join(CHECKPOINTS_DIR, "sam-3d-body-dinov3"),
        "checkpoint": "model.ckpt",
        "mhr_path": "assets/mhr_model.pt",
        "description": "DINOv3-H+ (840M params) - Best quality",
        "size_estimate": "~3.5 GB",
    },
    "vith": {
        "repo_id": "facebook/sam-3d-body-vith",
        "local_dir": os.path.join(CHECKPOINTS_DIR, "sam-3d-body-vith"),
        "checkpoint": "model.ckpt",
        "mhr_path": "assets/mhr_model.pt",
        "description": "ViT-H (631M params) - Slightly faster",
        "size_estimate": "~2.5 GB",
    },
}

SAM2_CONFIG = {
    "repo_id": "facebook/sam2.1",
    "checkpoint": "sam2.1_hiera_large.pt",
    "local_dir": os.path.join(BASE_DIR, "third_party", "sam2"),
    "description": "SAM2.1 Hiera-Large segmentor",
    "size_estimate": "~900 MB",
    "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
}


def check_hf_auth():
    """Check if HuggingFace authentication is set up."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print(f"[✔] HuggingFace token found in environment (length={len(token)})")
        return True

    # Check if logged in via huggingface-cli
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"[✔] Logged in as: {user.get('name', 'unknown')}")
        return True
    except Exception:
        print("\n[!] NOT authenticated with HuggingFace.")
        print("    The SAM 3D Body models require access request approval.")
        print("\n    Steps:")
        print("    1. Create account at https://huggingface.co/")
        print("    2. Request access at:")
        print("       https://huggingface.co/facebook/sam-3d-body-dinov3")
        print("       https://huggingface.co/facebook/sam-3d-body-vith")
        print("    3. Set your token:")
        print("       $env:HF_TOKEN='hf_your_token_here'")
        print("       OR run: huggingface-cli login")
        print()
        return False


def download_hf_model(model_key: str):
    """Download a SAM 3D Body model from HuggingFace."""
    config = MODEL_CONFIGS[model_key]
    repo_id = config["repo_id"]
    local_dir = config["local_dir"]

    # Check if already downloaded
    checkpoint_path = os.path.join(local_dir, config["checkpoint"])
    mhr_path = os.path.join(local_dir, config["mhr_path"])

    if os.path.exists(checkpoint_path) and os.path.exists(mhr_path):
        ckpt_size = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"[✔] {config['description']} already downloaded")
        print(f"    Checkpoint: {checkpoint_path} ({ckpt_size:.2f} GB)")
        return local_dir

    print(f"\n[→] Downloading: {config['description']}")
    print(f"    Repo: {repo_id}")
    print(f"    Destination: {local_dir}")
    print(f"    Estimated size: {config['size_estimate']}")
    print()

    os.makedirs(local_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.git*", "*.md", "*.txt"],
        )
        print(f"\n[✔] Downloaded to: {local_dir}")
        return local_dir

    except Exception as e:
        print(f"\n[✘] Download failed: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("\n    Access denied! Please:")
            print(f"    1. Request access at: https://huggingface.co/{repo_id}")
            print("    2. Wait for approval (usually instant)")
            print("    3. Make sure HF_TOKEN is set in your environment")
        elif "not found" in str(e).lower():
            print("\n    Repository not found. Check the repo ID.")
        raise


def download_moge2():
    """Verify MoGe2 will be auto-downloaded (it's public on HF Hub)."""
    print("\n[→] MoGe2 FOV Estimator")
    print("    Repo: Ruicheng/moge-2-vitl-normal")
    print("    This will be auto-downloaded on first run via moge.model.v2.MoGeModel.from_pretrained()")
    print("    Size: ~1.2 GB (cached in ~/.cache/huggingface/hub/)")

    # Pre-download it now to speed up first run
    try:
        print("    Pre-downloading MoGe2 now...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Ruicheng/moge-2-vitl-normal",
            ignore_patterns=["*.git*"],
        )
        print("    [✔] MoGe2 cached successfully")
    except Exception as e:
        print(f"    [!] Could not pre-download MoGe2: {e}")
        print("    MoGe2 will be downloaded automatically on first inference run.")


def download_sam2():
    """Download SAM2.1 large checkpoint."""
    cfg = SAM2_CONFIG
    checkpoint_path = os.path.join(cfg["local_dir"], "checkpoints", cfg["checkpoint"])

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024**2)
        print(f"[✔] SAM2.1 already downloaded: {checkpoint_path} ({size_mb:.0f} MB)")
        return checkpoint_path

    print(f"\n[→] Downloading SAM2.1 Hiera-Large checkpoint...")
    print(f"    URL: {cfg['url']}")
    print(f"    Destination: {checkpoint_path}")
    print(f"    Size: {cfg['size_estimate']}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Download using urllib
    import urllib.request
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                downloaded_mb = downloaded / (1024**2)
                total_mb = total_size / (1024**2)
                print(f"\r    Progress: {pct}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

        urllib.request.urlretrieve(cfg["url"], checkpoint_path, reporthook=progress_hook)
        print(f"\n[✔] SAM2.1 downloaded to: {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        print(f"\n[✘] SAM2.1 download failed: {e}")
        print(f"    Manual download from: {cfg['url']}")
        print(f"    Place at: {checkpoint_path}")
        return None


def download_vitdet():
    """ViTDet is auto-downloaded by detectron2, but inform user."""
    url = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    print(f"\n[→] ViTDet Human Detector")
    print(f"    URL: {url}")
    print(f"    This will be auto-downloaded on first run by detectron2.checkpoint.DetectionCheckpointer")
    print(f"    Size: ~2.5 GB (saved to detectron2 cache)")

    # Pre-download now
    try:
        checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", "vitdet")
        checkpoint_path = os.path.join(checkpoint_dir, "model_final_f05665.pkl")
        if os.path.exists(checkpoint_path):
            print(f"    [✔] Already downloaded: {checkpoint_path}")
            return checkpoint_path

        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"    Pre-downloading ViTDet to: {checkpoint_path}")

        import urllib.request
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                downloaded_mb = downloaded / (1024**2)
                total_mb = total_size / (1024**2)
                print(f"\r    Progress: {pct}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, checkpoint_path, reporthook=progress_hook)
        print(f"\n    [✔] ViTDet downloaded: {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        print(f"    [!] Pre-download failed: {e}. Will auto-download on first run.")
        return None


def print_summary(model_key: str):
    """Print the paths and commands to use for inference."""
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["dinov3"])
    local_dir = cfg["local_dir"]
    checkpoint = os.path.join(local_dir, cfg["checkpoint"])
    mhr_path = os.path.join(local_dir, cfg["mhr_path"])
    sam2_dir = os.path.join(BASE_DIR, "third_party", "sam2")

    print("\n" + "="*65)
    print("  DOWNLOAD SUMMARY")
    print("="*65)
    print(f"\n  Model      : {cfg['description']}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  MHR model  : {mhr_path}")
    print(f"\n  Environment variables to set:")
    print(f"    $env:SAM3D_MHR_PATH = '{mhr_path}'")
    print(f"\n  Run video processing:")
    print(f"    python process_karate_video.py")
    print(f"        --checkpoint_path '{checkpoint}'")
    print(f"        --mhr_path '{mhr_path}'")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download SAM 3D Body model checkpoints")
    parser.add_argument("--model", choices=["dinov3", "vith", "all"], default="dinov3",
                        help="Which SAM 3D Body model to download (default: dinov3)")
    parser.add_argument("--skip-sam2", action="store_true",
                        help="Skip SAM2 segmentor download")
    parser.add_argument("--skip-vitdet", action="store_true",
                        help="Skip ViTDet pre-download")
    parser.add_argument("--skip-moge", action="store_true",
                        help="Skip MoGe2 pre-download")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  SAM 3D Body - Model Downloader")
    print("="*65)
    print(f"  Checkpoints dir: {CHECKPOINTS_DIR}")
    print("="*65 + "\n")

    # Authentication check
    print("[STEP 1] Checking HuggingFace authentication...")
    auth_ok = check_hf_auth()
    if not auth_ok:
        print("\n[!] You can still try downloading - some may be accessible without auth.")

    # Download SAM 3D Body model(s)
    models_to_download = (
        ["dinov3", "vith"] if args.model == "all"
        else [args.model]
    )

    for model_key in models_to_download:
        print(f"\n[STEP 2] Downloading SAM 3D Body ({model_key})...")
        try:
            download_hf_model(model_key)
        except Exception as e:
            print(f"[✘] Failed to download {model_key}: {e}")

    # ViTDet
    if not args.skip_vitdet:
        print(f"\n[STEP 3] ViTDet detector...")
        download_vitdet()

    # SAM2
    if not args.skip_sam2:
        print(f"\n[STEP 4] SAM2.1 segmentor...")
        download_sam2()

    # MoGe2
    if not args.skip_moge:
        print(f"\n[STEP 5] MoGe2 FOV estimator...")
        download_moge2()

    # Summary
    print_summary(models_to_download[0])


if __name__ == "__main__":
    main()
