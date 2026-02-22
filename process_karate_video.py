"""
SAM 3D Body - Full Video Processing Pipeline
Processes a karate video frame by frame and generates 3D body mesh visualizations.

Usage:
    python process_karate_video.py \
        --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

    # Process with mask conditioning (higher quality, slower)
    python process_karate_video.py --use_mask

    # Process every 2nd frame (2x speedup)
    python process_karate_video.py --frame_skip 2

    # Control frame range
    python process_karate_video.py --start_frame 100 --end_frame 500

    # Use body-only mode (no hand refinement, faster)
    python process_karate_video.py --inference_type body
"""

import argparse
import os
import sys
import time
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Project root setup (needed for sam_3d_body imports) ──────────────────
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together


# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_VIDEO = os.path.join(root, "input_video", "karate_video.mp4")
DEFAULT_FRAMES_DIR = os.path.join(root, "input_video", "frames")
DEFAULT_OUTPUT_DIR = os.path.join(root, "output_video")
DEFAULT_CHECKPOINT = os.path.join(root, "checkpoints", "sam-3d-body-dinov3", "model.ckpt")
DEFAULT_MHR = os.path.join(root, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")


# ── Helpers ──────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    frame_skip: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    resize_width: int = -1,
) -> list[str]:
    """
    Extract video frames to JPEG files using OpenCV.
    
    Args:
        video_path: Path to input mp4
        output_dir: Where to save frames
        frame_skip: Only save every Nth frame (1=all, 2=every other, etc.)
        start_frame: First frame index to extract
        end_frame: Last frame index (-1 = all)
        resize_width: Resize to this width (maintaining aspect ratio). -1 = no resize.
    
    Returns:
        Sorted list of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if frames already extracted
    existing = sorted(glob(os.path.join(output_dir, "frame_*.jpg")))
    if existing:
        print(f"[✔] Found {len(existing)} existing frames in {output_dir}")
        print("    (Delete frames/ folder to re-extract)")
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame == -1:
        end_frame = total_frames

    print(f"\n[→] Extracting frames from: {video_path}")
    print(f"    Resolution: {orig_w}x{orig_h}  |  FPS: {fps:.2f}  |  Total: {total_frames}")
    print(f"    Range: frames [{start_frame} → {end_frame}]  |  Skip: every {frame_skip}")

    # Compute target resize
    if resize_width > 0 and resize_width < orig_w:
        scale = resize_width / orig_w
        target_w = resize_width
        target_h = int(orig_h * scale)
        print(f"    Resize: {orig_w}x{orig_h} → {target_w}x{target_h}")
    else:
        target_w, target_h = orig_w, orig_h

    frame_paths = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pbar = tqdm(
        total=(end_frame - start_frame) // frame_skip,
        desc="Extracting frames",
        unit="frame",
        dynamic_ncols=True,
    )

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_skip != 0:
            continue

        if target_w != orig_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"[✔] Extracted {len(frame_paths)} frames to: {output_dir}")
    return sorted(frame_paths)


def build_estimator(args) -> SAM3DBodyEstimator:
    """Build the full SAM3DBodyEstimator pipeline."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\n[→] Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    GPU: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")

    # Resolve paths (support env vars as fallback)
    checkpoint_path = args.checkpoint_path or os.environ.get("SAM3D_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", DEFAULT_MHR)
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")

    # Validate checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n[✘] Checkpoint not found: {checkpoint_path}")
        print("    Please run: python download_models.py")
        sys.exit(1)

    if not os.path.exists(mhr_path):
        print(f"\n[✘] MHR model not found: {mhr_path}")
        print("    Please run: python download_models.py")
        sys.exit(1)

    print(f"\n[→] Loading SAM 3D Body model...")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    MHR model:  {mhr_path}")
    t0 = time.time()
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    print(f"    [✔] Loaded in {time.time()-t0:.1f}s")

    # Human detector (ViTDet default)
    human_detector = None
    if args.detector_name:
        print(f"\n[→] Loading human detector: {args.detector_name}...")
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
        print(f"    [✔] Detector loaded")

    # Segmentor (SAM2 default, optional)
    human_segmentor = None
    if args.use_mask or args.segmentor_name:
        seg_name = args.segmentor_name or "sam2"
        # SAM2 requires a local path with the checkpoint
        if seg_name == "sam2" and not segmentor_path:
            sam2_candidate = os.path.join(root, "third_party", "sam2")
            if os.path.isdir(sam2_candidate):
                segmentor_path = sam2_candidate
                print(f"\n[→] Loading SAM2 segmentor from: {segmentor_path}...")
                from tools.build_sam import HumanSegmentor
                human_segmentor = HumanSegmentor(
                    name=seg_name, device=device, path=segmentor_path
                )
                print("    [✔] SAM2 loaded")
            else:
                print(f"\n[!] SAM2 not found at {sam2_candidate}")
                print("    Skipping mask conditioning. Run without --use_mask for now.")
                print("    Or run: python download_models.py  to get SAM2.")
                args.use_mask = False
        elif segmentor_path:
            from tools.build_sam import HumanSegmentor
            human_segmentor = HumanSegmentor(
                name=seg_name, device=device, path=segmentor_path
            )

    # FOV estimator (MoGe2 default)
    fov_estimator = None
    if args.fov_name:
        fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")
        print(f"\n[→] Loading FOV estimator: {args.fov_name}...")
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)
        print("    [✔] MoGe2 loaded")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    print("\n[✔] Full estimator pipeline ready!")
    return estimator


def process_frame_batch(
    estimator: SAM3DBodyEstimator,
    frame_paths: list[str],
    output_dir: str,
    use_mask: bool = False,
    inference_type: str = "full",
    bbox_thresh: float = 0.5,
) -> dict:
    """
    Process a list of frames and save visualizations.
    Returns timing stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "processed": 0,
        "skipped": 0,
        "no_person": 0,
        "errors": 0,
        "times": [],
    }

    pbar = tqdm(frame_paths, desc="Processing frames", unit="frame", dynamic_ncols=True)

    for frame_path in pbar:
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, frame_name)

        # Skip already-processed frames
        if os.path.exists(output_path):
            stats["skipped"] += 1
            pbar.set_postfix({"status": "cached", "total_done": stats["processed"] + stats["skipped"]})
            continue

        t0 = time.time()
        try:
            outputs = estimator.process_one_image(
                frame_path,
                bbox_thr=bbox_thresh,
                use_mask=use_mask,
                inference_type=inference_type,
            )

            if not outputs:
                # No person detected in this frame
                stats["no_person"] += 1
                # Still save the original frame (unchanged)
                img = cv2.imread(frame_path)
                cv2.imwrite(output_path, img)
                pbar.set_postfix({"status": "no_person"})
                continue

            # Render visualization
            img_bgr = cv2.imread(frame_path)
            rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
            cv2.imwrite(output_path, rend_img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

            elapsed = time.time() - t0
            stats["times"].append(elapsed)
            stats["processed"] += 1

            n_people = len(outputs)
            fps_proc = 1.0 / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "people": n_people,
                "fps": f"{fps_proc:.2f}",
                "time": f"{elapsed:.1f}s",
            })

        except Exception as e:
            stats["errors"] += 1
            pbar.set_postfix({"status": f"ERROR: {str(e)[:30]}"})
            print(f"\n[!] Error on {frame_name}: {e}")
            # Save original frame so video compilation still works
            try:
                img = cv2.imread(frame_path)
                if img is not None:
                    cv2.imwrite(output_path, img)
            except:
                pass

    return stats


def compile_video(
    frames_dir: str,
    output_video_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> str:
    """
    Compile processed frames back into a video.
    """
    frame_paths = sorted(glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_paths:
        print(f"[!] No frames found in {frames_dir}")
        return ""

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"[!] Could not read first frame: {frame_paths[0]}")
        return ""

    h, w = first_frame.shape[:2]

    print(f"\n[→] Compiling {len(frame_paths)} frames into video...")
    print(f"    Output: {output_video_path}")
    print(f"    Resolution: {w}x{h}  |  FPS: {fps}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for fp in tqdm(frame_paths, desc="Compiling video", unit="frame"):
        frame = cv2.imread(fp)
        if frame is not None:
            # Handle case where visualization is wider than expected
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(output_video_path) / (1024**2)
    print(f"[✔] Video saved: {output_video_path} ({size_mb:.1f} MB)")
    return output_video_path


def get_video_fps(video_path: str) -> float:
    """Get FPS from original video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def print_stats(stats: dict, frame_count: int):
    """Print processing statistics."""
    times = stats.get("times", [])
    print(f"\n{'='*55}")
    print(f"  PROCESSING STATISTICS")
    print(f"{'='*55}")
    print(f"  Frames processed  : {stats['processed']}")
    print(f"  Frames cached     : {stats['skipped']}")
    print(f"  No person found   : {stats['no_person']}")
    print(f"  Errors            : {stats['errors']}")
    if times:
        print(f"  Avg time/frame    : {np.mean(times):.2f}s")
        print(f"  Min time/frame    : {np.min(times):.2f}s")
        print(f"  Max time/frame    : {np.max(times):.2f}s")
        total_time = sum(times)
        print(f"  Total proc time   : {total_time//60:.0f}m {total_time%60:.0f}s")
        fps_avg = 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        print(f"  Avg throughput    : {fps_avg:.2f} FPS")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Body - Karate Video Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output
    parser.add_argument("--video_path", default=DEFAULT_VIDEO,
                        help=f"Path to input video (default: {DEFAULT_VIDEO})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--frames_dir", default=DEFAULT_FRAMES_DIR,
                        help=f"Temporary frames directory (default: {DEFAULT_FRAMES_DIR})")

    # Model
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT,
                        help="Path to SAM 3D Body .ckpt checkpoint")
    parser.add_argument("--mhr_path", default=DEFAULT_MHR,
                        help="Path to mhr_model.pt")
    parser.add_argument("--detector_name", default="vitdet",
                        help="Detector: 'vitdet' or 'sam3' (default: vitdet)")
    parser.add_argument("--detector_path", default="",
                        help="Path to ViTDet checkpoint folder")
    parser.add_argument("--segmentor_name", default="",
                        help="Segmentor: 'sam2' or 'sam3' (leave empty to disable)")
    parser.add_argument("--segmentor_path", default="",
                        help="Path to SAM2 folder")
    parser.add_argument("--fov_name", default="moge2",
                        help="FOV estimator: 'moge2' (default) or '' to disable")
    parser.add_argument("--fov_path", default="",
                        help="Path to FOV model (leave empty for auto HF download)")

    # Inference
    parser.add_argument("--inference_type", default="full",
                        choices=["full", "body", "hand"],
                        help="'full'=body+hands, 'body'=body only, 'hand'=hands only")
    parser.add_argument("--use_mask", action="store_true",
                        help="Enable mask-conditioned prediction (requires SAM2)")
    parser.add_argument("--bbox_thresh", default=0.5, type=float,
                        help="Person detection confidence threshold (default: 0.5)")

    # Frame control
    parser.add_argument("--frame_skip", default=1, type=int,
                        help="Process every Nth frame (1=all, 2=every other, etc.)")
    parser.add_argument("--start_frame", default=0, type=int,
                        help="Start from this frame index (default: 0)")
    parser.add_argument("--end_frame", default=-1, type=int,
                        help="Stop at this frame index (-1=all)")
    parser.add_argument("--resize_width", default=-1, type=int,
                        help="Resize frames to this width before processing (-1=original)")

    # Output
    parser.add_argument("--skip_compile", action="store_true",
                        help="Skip compiling frames back into a video")
    parser.add_argument("--output_fps", default=-1.0, type=float,
                        help="Output video FPS (-1=match input, default: -1)")

    args = parser.parse_args()

    print("\n" + "="*65)
    print("  SAM 3D Body - Karate Video 3D Mesh Processing Pipeline")
    print("="*65)
    print(f"  Input video    : {args.video_path}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Inference mode : {args.inference_type}")
    print(f"  Frame skip     : {args.frame_skip}")
    print(f"  Mask condition : {args.use_mask}")
    print(f"  FOV estimator  : {args.fov_name or 'disabled'}")
    print("="*65 + "\n")

    # Check video exists
    if not os.path.exists(args.video_path):
        print(f"[✘] Video not found: {args.video_path}")
        print("    Please run: python download_karate_video.py")
        sys.exit(1)

    # ── STEP 1: Extract frames ───────────────────────────────────────
    print("[ STEP 1 ] Extracting video frames...")
    frame_paths = extract_frames(
        video_path=args.video_path,
        output_dir=args.frames_dir,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        resize_width=args.resize_width,
    )

    if not frame_paths:
        print("[✘] No frames extracted!")
        sys.exit(1)

    # ── STEP 2: Build estimator ──────────────────────────────────────
    print("\n[ STEP 2 ] Loading SAM 3D Body model pipeline...")
    estimator = build_estimator(args)

    # ── STEP 3: Process frames ───────────────────────────────────────
    vis_dir = os.path.join(args.output_dir, "visualized_frames")
    print(f"\n[ STEP 3 ] Running 3D body mesh inference on {len(frame_paths)} frames...")
    print(f"    Output frames → {vis_dir}")
    print(f"    CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print()

    t_start = time.time()
    stats = process_frame_batch(
        estimator=estimator,
        frame_paths=frame_paths,
        output_dir=vis_dir,
        use_mask=args.use_mask,
        inference_type=args.inference_type,
        bbox_thresh=args.bbox_thresh,
    )
    total_time = time.time() - t_start
    print(f"\n[✔] All frames processed in {total_time/60:.1f} min")
    print_stats(stats, len(frame_paths))

    # ── STEP 4: Compile video ────────────────────────────────────────
    if not args.skip_compile:
        print("[ STEP 4 ] Compiling output video...")
        # Determine output FPS
        output_fps = args.output_fps
        if output_fps <= 0:
            output_fps = get_video_fps(args.video_path)
            if args.frame_skip > 1:
                output_fps = output_fps / args.frame_skip
            print(f"    Output FPS: {output_fps:.2f} (input FPS / frame_skip)")

        output_video = os.path.join(args.output_dir, "karate_3d_body.mp4")
        compile_video(vis_dir, output_video, fps=output_fps)

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print("="*65)
    print(f"  Visualized frames : {vis_dir}")
    if not args.skip_compile:
        print(f"  Output video      : {os.path.join(args.output_dir, 'karate_3d_body.mp4')}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
