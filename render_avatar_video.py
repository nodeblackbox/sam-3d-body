"""
Compile a proper avatar-only video from the mesh data.
This uses the SAM 3D Body renderer to draw JUST the mesh
on a clean background, preserving the original video's aspect ratio.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from tqdm import tqdm

def main():
    mesh_dir = "karate_output/mesh_data"
    output_path = "karate_output/avatar_only.mp4"

    # Get video dimensions
    cap = cv2.VideoCapture("karate_heian_sandan.mp4")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Source video: {vid_w}x{vid_h} @ {vid_fps}fps")

    # We render at the same aspect ratio
    render_w = vid_w
    render_h = vid_h

    # Load SAM model faces
    print("Loading SAM 3D Body model for face topology...")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_sam_3d_body(
        "checkpoints/sam-3d-body-dinov3/model.ckpt",
        device=device,
        mhr_path="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    )
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    faces = estimator.faces
    print(f"Faces: {faces.shape}")

    from sam_3d_body.visualization.renderer import Renderer

    # Gather frame files
    files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".npz")])
    print(f"Frames to render: {len(files)}")

    # Writer (same fps as original, adjusted for frame_skip)
    # frame_skip=6 means we sampled every 6th frame -> effective fps = vid_fps/6
    effective_fps = vid_fps / 6.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (render_w, render_h))

    for fn in tqdm(files, desc="Rendering avatar video"):
        d = np.load(os.path.join(mesh_dir, fn), allow_pickle=True)

        if "vertices" not in d:
            # Empty frame - write a blank
            blank = np.ones((render_h, render_w, 3), dtype=np.uint8) * 20
            writer.write(blank)
            continue

        verts = d["vertices"]
        cam_t = d["cam_t"] if "cam_t" in d else np.array([0.0, 0.5, 3.5])
        focal = d["focal_length"].item() if "focal_length" in d else 1000.0

        renderer = Renderer(focal_length=focal, faces=faces)
        bg = np.ones((render_h, render_w, 3), dtype=np.uint8) * 20  # Dark background
        img = renderer(
            verts,
            cam_t,
            bg,
            mesh_base_color=(0.65, 0.74, 0.86),
            scene_bg_color=(0.08, 0.08, 0.08),
        )
        frame = (img * 255).astype(np.uint8)

        # Add label overlay
        label = d["pose_label"].item() if "pose_label" in d else None
        if label:
            cv2.putText(frame, str(label), (20, render_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 220), 2)

        writer.write(frame)

    writer.release()
    print(f"\n[DONE] Avatar-only video: {output_path}")
    print(f"  Resolution: {render_w}x{render_h} (original aspect ratio)")
    sz = os.path.getsize(output_path) / (1024*1024)
    print(f"  Size: {sz:.1f} MB")

if __name__ == "__main__":
    main()
