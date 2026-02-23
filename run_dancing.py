"""
Quick test of SAM 3D Body on dancing.jpg
Uses FULL IMAGE as bounding box (no ViTDet needed initially)
Then falls back to ViTDet if available.

Usage:
    C:\Python311\python.exe run_dancing.py
    C:\Python311\python.exe run_dancing.py --use_detector   # uses ViTDet
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Set PYOPENGL for headless rendering on Windows
# os.environ.setdefault("PYOPENGL_PLATFORM", "")

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="notebook/images/dancing.jpg")
    parser.add_argument("--output", default="output_dancing")
    parser.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path", default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--use_detector", action="store_true", help="Use ViTDet (requires detectron2)")
    parser.add_argument("--use_fov", action="store_true", default=True, help="Use MoGe2 FOV estimator")
    parser.add_argument("--bbox_thresh", default=0.3, type=float)
    args = parser.parse_args()

    image_path = os.path.join(PROJECT_ROOT, args.image) if not os.path.isabs(args.image) else args.image
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    mhr_path = os.path.join(PROJECT_ROOT, args.mhr_path) if not os.path.isabs(args.mhr_path) else args.mhr_path
    output_dir = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output

    print("\n" + "="*60)
    print("  SAM 3D Body - Dancing Image Test")
    print("="*60)
    print(f"  Image      : {image_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  MHR model  : {mhr_path}")
    print(f"  Output     : {output_dir}")
    print("="*60 + "\n")

    # Check files exist
    for path, name in [(image_path, "Image"), (checkpoint_path, "Checkpoint"), (mhr_path, "MHR model")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        size = os.path.getsize(path) / (1024*1024)
        print(f"[OK] {name}: {os.path.basename(path)} ({size:.1f} MB)")

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/5] Loading PyTorch + model...")
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "")

    # Load model
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    import time
    t0 = time.time()
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("\n[2/5] Setting up pipeline modules...")

    # Human detector (optional)
    human_detector = None
    if args.use_detector:
        try:
            from tools.build_detector import HumanDetector
            print("  Loading ViTDet detector...")
            human_detector = HumanDetector(name="vitdet", device=device)
            print("  [OK] ViTDet loaded")
        except Exception as e:
            print(f"  [WARN] ViTDet not available: {e}")
            print("  Using full-image bounding box instead")

    # FOV estimator (MoGe2)
    fov_estimator = None
    if args.use_fov:
        try:
            from tools.build_fov_estimator import FOVEstimator
            print("  Loading MoGe2 FOV estimator...")
            fov_estimator = FOVEstimator(name="moge2", device=device)
            print("  [OK] MoGe2 loaded")
        except Exception as e:
            print(f"  [WARN] MoGe2 not available: {e}")
            print("  Will use default FOV assumption")

    # Build estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    print("\n  [OK] Estimator pipeline ready!")

    print("\n[3/5] Running 3D body inference on dancing.jpg...")
    import cv2

    t0 = time.time()
    outputs = estimator.process_one_image(
        image_path,
        bbox_thr=args.bbox_thresh,
        use_mask=False,
        inference_type="full",
    )
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.2f}s")

    if not outputs:
        print("\n[ERROR] No person detected in the image!")
        print("  Try lowering bbox_thresh: --bbox_thresh 0.1")
        sys.exit(1)

    print(f"  [OK] Detected {len(outputs)} person(s)")
    for i, out in enumerate(outputs):
        print(f"    Person {i+1}: bbox={[int(x) for x in out['bbox']]}")

    print("\n[4/5] Rendering 3D mesh visualization...")
    from tools.vis_utils import visualize_sample_together, visualize_sample

    img_bgr = cv2.imread(image_path)

    # Full combined view (all people together - 4 panels)
    rend_combined = visualize_sample_together(img_bgr, outputs, estimator.faces)
    combined_path = os.path.join(output_dir, "dancing_3d_combined.jpg")
    cv2.imwrite(combined_path, rend_combined.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [OK] Combined view: {combined_path}")

    # Per-person views (4 panels each: original | keypoints | 3D front | 3D side)
    per_person = visualize_sample(img_bgr, outputs, estimator.faces)
    for i, rend in enumerate(per_person):
        person_path = os.path.join(output_dir, f"dancing_person_{i+1}.jpg")
        cv2.imwrite(person_path, rend.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [OK] Person {i+1}: {person_path}")

    # Also save keypoints-only version
    kp_img = img_bgr.copy()
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
    viz = SkeletonVisualizer(line_width=3, radius=6)
    viz.set_pose_meta(mhr70_pose_info)
    for out in outputs:
        kps = out["pred_keypoints_2d"]
        kps_with_conf = np.concatenate([kps, np.ones((kps.shape[0], 1))], axis=-1)
        kp_img = viz.draw_skeleton(kp_img, kps_with_conf)
    kp_path = os.path.join(output_dir, "dancing_keypoints.jpg")
    cv2.imwrite(kp_path, kp_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [OK] Keypoints: {kp_path}")

    print("\n[5/5] Printing 3D pose data...")
    for i, out in enumerate(outputs):
        print(f"\n  --- Person {i+1} ---")
        print(f"  Focal length   : {out['focal_length']:.1f} px")
        print(f"  Camera t (3D)  : {out['pred_cam_t']}")
        kps3d = out["pred_keypoints_3d"]
        print(f"  3D keypoints   : {kps3d.shape}  (range: {kps3d.min():.3f} to {kps3d.max():.3f} m)")
        verts = out["pred_vertices"]
        print(f"  Mesh vertices  : {verts.shape}  (18,439 vertices)")
        print(f"  Body pose dims : {out['body_pose_params'].shape}")
        print(f"  Shape params   : {out['shape_params'].shape}")

    print("\n" + "="*60)
    print("  SUCCESS! Results saved to:")
    print(f"  {output_dir}/")
    print("="*60)
    print(f"\n  Output files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        sz = os.path.getsize(fpath) / 1024
        print(f"    {f:40s}  ({sz:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
