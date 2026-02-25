"""
Karate Heian Sandan — Full 3D Mesh Recovery + Pose Labeling Pipeline
=====================================================================

What this does:
  1. Reads the karate video (karate_heian_sandan.mp4)
  2. Loads transcript labels from karate_transcript.json
  3. Extracts frames AND labels each frame with its karate technique
  4. Runs SAM 3D Body on every frame → 3D body mesh (18,439 vertices)
  5. Saves per-frame mesh data (.npz) + visualization JPEGs
  6. Compiles output video with pose labels overlaid
  7. Exports a pose timeline CSV: timestamp → technique → 3D joint positions

This gives you:
  - Full 3D mesh for every video frame
  - Each frame labeled with its karate technique name
  - 3D joint coordinates indexed by technique — ready for pose commands

Usage:
    python karate_pose_pipeline.py

    # Specific frame range:
    python karate_pose_pipeline.py --start_frame 0 --end_frame 500

    # Skip every 2nd frame (2x faster):
    python karate_pose_pipeline.py --frame_skip 2

    # Body-only mode (no hand decoder):
    python karate_pose_pipeline.py --inference_type body
"""

import os, sys, json, time, csv, argparse
import numpy as np
import cv2
from glob import glob
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH        = os.path.join(PROJECT_ROOT, "karate_heian_sandan.mp4")
TRANSCRIPT_PATH   = os.path.join(PROJECT_ROOT, "karate_transcript.json")
FRAMES_DIR        = os.path.join(PROJECT_ROOT, "karate_frames")
VIS_DIR           = os.path.join(PROJECT_ROOT, "karate_output", "visualized_frames")
MESH_DIR          = os.path.join(PROJECT_ROOT, "karate_output", "mesh_data")
OUTPUT_VIDEO      = os.path.join(PROJECT_ROOT, "karate_output", "heian_sandan_3d.mp4")
POSE_CSV          = os.path.join(PROJECT_ROOT, "karate_output", "pose_timeline.csv")
POSE_NPZ          = os.path.join(PROJECT_ROOT, "karate_output", "pose_library.npz")

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "sam-3d-body-dinov3", "model.ckpt")
MHR_PATH        = os.path.join(PROJECT_ROOT, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")

# Fallback: use the standalone MHR we downloaded
MHR_FALLBACK = os.path.join(PROJECT_ROOT, "checkpoints", "mhr", "assets", "mhr_model.pt")


# ── Transcript label lookup ────────────────────────────────────────────────────

class TranscriptLabeler:
    """Maps video timestamps → karate technique labels from the transcript JSON."""

    def __init__(self, transcript_path: str, fps: float):
        self.fps = fps
        self.segments = []
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                self.segments = json.load(f)
            print(f"[OK] Loaded {len(self.segments)} transcript segments")
        else:
            print(f"[WARN] Transcript not found: {transcript_path}")

    def label_for_second(self, t: float) -> dict:
        """Get technique label for a given timestamp in seconds."""
        best = None
        for seg in self.segments:
            if seg["start"] <= t <= seg["end"]:
                if best is None or seg["start"] > best["start"]:
                    best = seg
        if best:
            return {
                "pose_label":    best.get("pose_label", ""),
                "category":      best.get("category", ""),
                "karate_terms":  best.get("karate_terms", []),
                "start":         best["start"],
                "end":           best["end"],
            }
        return {"pose_label": "transition", "category": "unknown", "karate_terms": []}

    def label_for_frame(self, frame_idx: int) -> dict:
        return self.label_for_second(frame_idx / self.fps)


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path, out_dir, frame_skip=1, start_frame=0, end_frame=-1, resize_w=-1):
    from tqdm import tqdm
    os.makedirs(out_dir, exist_ok=True)

    # Delete existing frames if they exist so we get a clean extraction
    existing = glob(os.path.join(out_dir, "frame_*.jpg"))
    for f in existing:
        try: os.remove(f)
        except: pass

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if end_frame == -1: end_frame = total

    tgt_w = resize_w if (resize_w > 0 and resize_w < orig_w) else orig_w
    tgt_h = int(orig_h * (tgt_w / orig_w)) if tgt_w != orig_w else orig_h

    print(f"[→] Extracting {(end_frame - start_frame) // frame_skip} frames from {os.path.basename(video_path)}")
    print(f"    {orig_w}x{orig_h} @ {fps:.2f}fps  →  saving every {frame_skip} frame(s)")

    paths = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    with tqdm(total=(end_frame - start_frame) // frame_skip, desc="Extracting", unit="frame") as pbar:
        for idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            if (idx - start_frame) % frame_skip != 0: continue
            if tgt_w != orig_w:
                frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
            out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(out_path)
            pbar.update(1)
    cap.release()
    print(f"[OK] Extracted {len(paths)} frames")
    return sorted(paths)


def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


# ── Overlay helpers ────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_DUPLEX
COLORS = {
    "technique":    (0, 255, 120),   # green
    "intro":        (255, 200, 50),  # yellow
    "music":        (180, 180, 180), # grey
    "full_kata_demo": (80, 200, 255),# blue
    "command":      (255, 100, 100), # red
    "unknown":      (200, 200, 200),
    "transition":   (150, 150, 150),
    "instruction":  (255, 180, 50),
}

def draw_pose_overlay(frame: np.ndarray, label_info: dict, frame_idx: int, fps: float) -> np.ndarray:
    """Draw karate technique label + timestamp overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent banner at bottom
    banner_h = 72
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    pose  = label_info.get("pose_label", "")
    cat   = label_info.get("category", "unknown")
    terms = label_info.get("karate_terms", [])
    color = COLORS.get(cat, (200, 200, 200))

    t_sec = frame_idx / fps
    ts = f"{int(t_sec//60):02d}:{int(t_sec%60):02d}.{int((t_sec%1)*10):01d}"

    # Timestamp top-left
    cv2.putText(frame, ts, (12, 32), FONT, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame number top-right
    cv2.putText(frame, f"frame {frame_idx}", (w - 160, 32), FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Pose label
    if pose:
        cv2.putText(frame, pose.upper(), (14, h - banner_h + 26), FONT, 0.72, color, 1, cv2.LINE_AA)

    # Karate terms
    if terms:
        terms_str = "  •  ".join(terms)
        cv2.putText(frame, terms_str, (14, h - banner_h + 56), FONT, 0.52, (200, 230, 255), 1, cv2.LINE_AA)

    return frame


def draw_mesh_overlay(vis_frame: np.ndarray, outputs: list) -> np.ndarray:
    """Add person count and confidence overlay to the mesh viz."""
    if not outputs:
        return vis_frame
    h, w = vis_frame.shape[:2]
    n = len(outputs)
    cv2.putText(vis_frame, f"{n} person(s) detected", (12, h - 10),
                FONT, 0.55, (100, 255, 180), 1, cv2.LINE_AA)
    return vis_frame


# ── Core processing loop ───────────────────────────────────────────────────────

def process_frames(estimator, frame_paths, labeler, out_vis_dir, out_mesh_dir,
                   inference_type="full", bbox_thresh=0.4):
    from tqdm import tqdm
    from tools.vis_utils import visualize_sample_together

    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(out_mesh_dir, exist_ok=True)

    fps = labeler.fps
    pose_rows = []   # for CSV
    pose_library = {}  # technique_name → list of 3D joint arrays

    stats = {"processed": 0, "no_person": 0, "cached": 0, "errors": 0, "times": []}

    with tqdm(frame_paths, desc="3D Mesh Recovery", unit="frame", dynamic_ncols=True) as pbar:
        for frame_path in pbar:
            fname = os.path.basename(frame_path)
            frame_idx = int(fname.replace("frame_", "").replace(".jpg", ""))
            vis_path  = os.path.join(out_vis_dir, fname)
            mesh_path = os.path.join(out_mesh_dir, fname.replace(".jpg", ".npz"))

            # Get technique label for this frame
            label = labeler.label_for_frame(frame_idx)

            # ── Cache check ──────────────────────────────────────────────────
            if os.path.exists(vis_path) and os.path.exists(mesh_path):
                stats["cached"] += 1
                continue

            t0 = time.time()
            try:
                # ── Run SAM 3D Body ─────────────────────────────────────────
                outputs = estimator.process_one_image(
                    frame_path,
                    bbox_thr=bbox_thresh,
                    use_mask=False,
                    inference_type=inference_type,
                )

                img_bgr = cv2.imread(frame_path)

                if not outputs:
                    stats["no_person"] += 1
                    # Save original with label overlay but no mesh
                    labeled = draw_pose_overlay(img_bgr, label, frame_idx, fps)
                    cv2.imwrite(vis_path, labeled)
                    np.savez_compressed(mesh_path,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        pose_label=label["pose_label"],
                        n_persons=0)
                    pbar.set_postfix({"status": "no_person", "pose": label["pose_label"][:20]})
                    continue

                # ── Render 3D mesh visualization ─────────────────────────────
                rend = visualize_sample_together(img_bgr, outputs, estimator.faces)
                rend = draw_mesh_overlay(rend, outputs)
                rend = draw_pose_overlay(rend, label, frame_idx, fps)
                cv2.imwrite(vis_path, rend.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

                # ── Save mesh data (.npz) ─────────────────────────────────────
                # Store ALL 3D data for primary person
                p = outputs[0]
                np.savez_compressed(
                    mesh_path,
                    frame_idx        = frame_idx,
                    timestamp        = frame_idx / fps,
                    pose_label       = label["pose_label"],
                    category         = label["category"],
                    karate_terms     = np.array(label["karate_terms"]),
                    # 3D mesh
                    vertices         = p["pred_vertices"],       # (18439, 3) in meters
                    keypoints_3d     = p["pred_keypoints_3d"],   # (70, 3) in meters
                    keypoints_2d     = p["pred_keypoints_2d"],   # (70, 2) in pixels
                    # Pose params
                    body_pose        = p["body_pose_params"],
                    shape_params     = p["shape_params"],
                    # Camera
                    cam_t            = p["pred_cam_t"],
                    focal_length     = p["focal_length"],
                )

                # ── Accumulate for pose library ───────────────────────────────
                tech = label["pose_label"] or "unknown"
                if tech not in pose_library:
                    pose_library[tech] = {
                        "keypoints_3d": [],
                        "vertices": [],
                        "body_pose": [],
                        "category": label["category"],
                        "karate_terms": label["karate_terms"],
                        "frame_indices": [],
                    }
                pose_library[tech]["keypoints_3d"].append(p["pred_keypoints_3d"])
                pose_library[tech]["vertices"].append(p["pred_vertices"])
                pose_library[tech]["body_pose"].append(p["body_pose_params"])
                pose_library[tech]["frame_indices"].append(frame_idx)

                # ── CSV row ───────────────────────────────────────────────────
                kps = p["pred_keypoints_3d"]
                pose_rows.append({
                    "frame_idx":      frame_idx,
                    "timestamp_s":    round(frame_idx / fps, 3),
                    "pose_label":     label["pose_label"],
                    "category":       label["category"],
                    "karate_terms":   "|".join(label["karate_terms"]),
                    "n_persons":      len(outputs),
                    "focal_length_px": round(float(p["focal_length"]), 1),
                    # Key joint positions (head, spine, hips, knees, feet)
                    "head_x":     round(float(kps[0, 0]), 4),
                    "head_y":     round(float(kps[0, 1]), 4),
                    "head_z":     round(float(kps[0, 2]), 4),
                    "spine_x":    round(float(kps[3, 0]), 4),
                    "spine_y":    round(float(kps[3, 1]), 4),
                    "spine_z":    round(float(kps[3, 2]), 4),
                    "hip_l_x":   round(float(kps[11, 0]), 4),
                    "hip_l_y":   round(float(kps[11, 1]), 4),
                    "hip_l_z":   round(float(kps[11, 2]), 4),
                    "hip_r_x":   round(float(kps[12, 0]), 4),
                    "hip_r_y":   round(float(kps[12, 1]), 4),
                    "hip_r_z":   round(float(kps[12, 2]), 4),
                    "lfoot_x":   round(float(kps[17, 0]), 4),
                    "lfoot_y":   round(float(kps[17, 1]), 4),
                    "lfoot_z":   round(float(kps[17, 2]), 4),
                    "rfoot_x":   round(float(kps[18, 0]), 4),
                    "rfoot_y":   round(float(kps[18, 1]), 4),
                    "rfoot_z":   round(float(kps[18, 2]), 4),
                })

                elapsed = time.time() - t0
                stats["processed"] += 1
                stats["times"].append(elapsed)
                pbar.set_postfix({
                    "pose": label["pose_label"][:20] if label["pose_label"] else "?",
                    "fps":  f"{1/elapsed:.2f}",
                    "ppl":  len(outputs),
                })

            except Exception as e:
                stats["errors"] += 1
                pbar.set_postfix({"ERROR": str(e)[:30]})
                import traceback
                print(f"\n[!] Error frame {frame_idx}: {e}")
                # Still save original
                try:
                    img = cv2.imread(frame_path)
                    if img is not None:
                        labeled = draw_pose_overlay(img, label, frame_idx, fps)
                        cv2.imwrite(vis_path, labeled)
                except:
                    pass

    return stats, pose_rows, pose_library


# ── Save results ───────────────────────────────────────────────────────────────

def save_pose_csv(pose_rows, csv_path):
    if not pose_rows: return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pose_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pose_rows)
    print(f"[OK] Pose timeline CSV: {csv_path} ({len(pose_rows)} rows)")


def save_pose_library(pose_library, npz_path):
    """
    Save averaged 3D poses per technique.
    Later you can load this and say 'get into kiba-dachi' → look up the 3D joint positions.
    """
    if not pose_library: return
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    library_data = {}
    print("\n[→] Pose Library (averaged per technique):")
    print(f"{'Technique':<50} {'Frames':>6}")
    print("-" * 60)

    for tech, data in pose_library.items():
        kps_arr  = np.stack(data["keypoints_3d"], axis=0)  # (N, 70, 3)
        vert_arr = np.stack(data["vertices"], axis=0)        # (N, 18439, 3)
        pose_arr = np.stack(data["body_pose"], axis=0)       # (N, ...)
        n = len(kps_arr)

        # Average pose across all frames for this technique
        library_data[tech] = {
            "mean_keypoints_3d": kps_arr.mean(axis=0),
            "mean_vertices":     vert_arr.mean(axis=0),
            "mean_body_pose":    pose_arr.mean(axis=0),
            "n_frames":          n,
            "category":          data["category"],
            "karate_terms":      data["karate_terms"],
            "frame_indices":     data["frame_indices"],
        }
        print(f"  {tech[:48]:<50} {n:>6} frames")

    # Save as npz
    np.savez_compressed(npz_path, **{
        k.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", ""): v
        for k, v in library_data.items()
    })

    # Also save a human-readable index JSON
    index = {
        tech: {
            "n_frames":      data["n_frames"],
            "category":      data["category"],
            "karate_terms":  data["karate_terms"],
            "frame_indices": data["frame_indices"][:5],  # first 5
        }
        for tech, data in library_data.items()
    }
    with open(npz_path.replace(".npz", "_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[OK] Pose library: {npz_path}")
    print(f"[OK] Pose index:   {npz_path.replace('.npz', '_index.json')}")


def compile_output_video(frames_dir, output_path, fps):
    paths = sorted(glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not paths:
        print("[!] No output frames to compile")
        return

    first = cv2.imread(paths[0])
    if first is None: return
    h, w = first.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    from tqdm import tqdm
    for p in tqdm(paths, desc="Compiling video", unit="frame"):
        frame = cv2.imread(p)
        if frame is not None:
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    writer.release()

    sz = os.path.getsize(output_path) / 1e6
    print(f"[OK] Output video: {output_path} ({sz:.1f} MB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Karate Heian Sandan — SAM 3D Body Pose Pipeline")
    parser.add_argument("--video_path",    default=VIDEO_PATH)
    parser.add_argument("--checkpoint",    default=CHECKPOINT_PATH)
    parser.add_argument("--mhr_path",      default=MHR_PATH)
    parser.add_argument("--inference_type",default="full", choices=["full", "body"])
    parser.add_argument("--frame_skip",    default=1, type=int)
    parser.add_argument("--start_frame",   default=0, type=int)
    parser.add_argument("--end_frame",     default=-1, type=int)
    parser.add_argument("--bbox_thresh",   default=0.4, type=float)
    parser.add_argument("--resize_width",  default=1280, type=int)
    parser.add_argument("--skip_compile",  action="store_true")
    parser.add_argument("--use_detector",  action="store_true", default=True)
    args = parser.parse_args()

    # Use MHR fallback if main not found
    mhr_path = args.mhr_path
    if not os.path.exists(mhr_path) and os.path.exists(MHR_FALLBACK):
        mhr_path = MHR_FALLBACK
        print(f"[→] Using MHR fallback: {mhr_path}")

    print("\n" + "=" * 70)
    print("  Karate Heian Sandan — SAM 3D Body Full Pipeline")
    print("=" * 70)
    print(f"  Video      : {args.video_path}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  MHR model  : {mhr_path}")
    print(f"  Mode       : {args.inference_type}  |  skip={args.frame_skip}")
    print("=" * 70 + "\n")

    # Validate
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video not found: {args.video_path}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("  Run: python download_models.py  OR wait for current download to finish")
        sys.exit(1)

    fps = get_video_fps(args.video_path)
    print(f"[OK] Video FPS: {fps:.2f}")

    # ─ Step 1: Extract frames ───────────────────────────────────────────────
    print("\n[STEP 1] Extracting frames...")
    frame_paths = extract_frames(
        args.video_path, FRAMES_DIR,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        resize_w=args.resize_width,
    )

    # ─ Step 2: Load transcript labels ────────────────────────────────────────
    print("\n[STEP 2] Loading transcript labels...")
    labeler = TranscriptLabeler(TRANSCRIPT_PATH, fps)

    # ─ Step 3: Build estimator ───────────────────────────────────────────────
    print("\n[STEP 3] Loading SAM 3D Body model...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    t0 = time.time()
    model, model_cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=mhr_path)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Detector
    human_detector = None
    if args.use_detector:
        try:
            from tools.build_detector import HumanDetector
            human_detector = HumanDetector(name="yolov8", device=device)
            print("  [OK] YOLOv8 detector loaded")
        except Exception as e:
            print(f"  [WARN] YOLOv8 unavailable ({e}), using full-image bbox")

    # FOV estimator
    fov_estimator = None
    try:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name="moge2", device=device)
        print("  [OK] MoGe2 FOV estimator loaded")
    except Exception as e:
        print(f"  [WARN] MoGe2 unavailable: {e}")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {used_gb:.1f} / {total_gb:.1f} GB used")

    # ─ Step 4: Process all frames ────────────────────────────────────────────
    print(f"\n[STEP 4] Running 3D mesh recovery on {len(frame_paths)} frames...")
    t_start = time.time()
    stats, pose_rows, pose_library = process_frames(
        estimator, frame_paths, labeler,
        VIS_DIR, MESH_DIR,
        inference_type=args.inference_type,
        bbox_thresh=args.bbox_thresh,
    )
    total_time = time.time() - t_start

    # ─ Step 5: Save outputs ───────────────────────────────────────────────────
    print("\n[STEP 5] Saving pose data...")
    save_pose_csv(pose_rows, POSE_CSV)
    save_pose_library(pose_library, POSE_NPZ)

    # ─ Step 6: Compile video ─────────────────────────────────────────────────
    if not args.skip_compile:
        print("\n[STEP 6] Compiling output video...")
        out_fps = fps / args.frame_skip if args.frame_skip > 1 else fps
        compile_output_video(VIS_DIR, OUTPUT_VIDEO, out_fps)

    # ─ Done ───────────────────────────────────────────────────────────────────
    times = stats.get("times", [])
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Frames processed  : {stats['processed']}")
    print(f"  Cached            : {stats['cached']}")
    print(f"  No person found   : {stats['no_person']}")
    print(f"  Errors            : {stats['errors']}")
    if times:
        print(f"  Avg time/frame    : {np.mean(times):.2f}s  ({1/np.mean(times):.2f} FPS)")
        print(f"  Total time        : {total_time/60:.1f} min")
    print(f"\n  Visualized frames : {VIS_DIR}/")
    print(f"  Mesh data (.npz)  : {MESH_DIR}/")
    print(f"  Pose timeline CSV : {POSE_CSV}")
    print(f"  Pose library      : {POSE_NPZ}")
    if not args.skip_compile:
        print(f"  Output video      : {OUTPUT_VIDEO}")
    print("=" * 70 + "\n")

    # Print pose summary
    if pose_library:
        print("  TECHNIQUE SUMMARY:")
        print(f"  {'Technique':<48} {'Frames':>6}  {'Category'}")
        print("  " + "-" * 70)
        for tech, data in sorted(pose_library.items(), key=lambda x: -len(x[1]["frame_indices"])):
            print(f"  {tech[:48]:<48} {len(data['frame_indices']):>6}  {data['category']}")
        print()


if __name__ == "__main__":
    main()
