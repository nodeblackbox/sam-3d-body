"""
Export mesh data to binary files for the Three.js web viewer.
Outputs:
  karate_output/web/faces.bin        - int32 face indices
  karate_output/web/frame_XXXXXX.bin - float32 vertices per frame
  karate_output/web/manifest.json    - metadata for the viewer
"""
import os, sys, json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    mesh_dir = "karate_output/mesh_data"
    web_dir = "karate_output/web"
    os.makedirs(web_dir, exist_ok=True)

    # 1. Export faces (only need to do this once)
    print("[1/3] Loading model to export face topology...")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    model, cfg = load_sam_3d_body(
        "checkpoints/sam-3d-body-dinov3/model.ckpt",
        device="cpu",
        mhr_path="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    )
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    faces = estimator.faces  # (F, 3) int array
    faces_path = os.path.join(web_dir, "faces.bin")
    faces.astype(np.int32).tofile(faces_path)
    print(f"  Saved {faces_path}  ({faces.shape[0]} triangles)")

    # 2. Export per-frame vertex positions
    print("[2/3] Exporting per-frame vertex data...")
    files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".npz")])
    manifest_frames = []

    for fn in tqdm(files, desc="Exporting"):
        d = np.load(os.path.join(mesh_dir, fn), allow_pickle=True)
        key = fn.replace(".npz", "")
        entry = {
            "file": key + ".bin",
            "frame_idx": int(d["frame_idx"]) if "frame_idx" in d else 0,
            "timestamp": float(d["timestamp"]) if "timestamp" in d else 0.0,
        }
        if "pose_label" in d:
            lbl = d["pose_label"]
            entry["label"] = str(lbl) if lbl is not None else ""
        if "vertices" in d:
            verts = d["vertices"].astype(np.float32)
            verts.tofile(os.path.join(web_dir, key + ".bin"))
            entry["has_mesh"] = True
            entry["n_verts"] = int(verts.shape[0])
        else:
            entry["has_mesh"] = False

        manifest_frames.append(entry)

    # 3. Write manifest
    print("[3/3] Writing manifest.json...")
    import cv2
    cap = cv2.VideoCapture("karate_heian_sandan.mp4")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    manifest = {
        "n_faces": int(faces.shape[0]),
        "n_verts": 18439,
        "video_width": vid_w,
        "video_height": vid_h,
        "video_fps": vid_fps,
        "frames": manifest_frames,
    }
    with open(os.path.join(web_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"  Manifest: {len(manifest_frames)} frames")

    # Also copy the transcript for the labeler
    if os.path.exists("karate_transcript.json"):
        import shutil
        shutil.copy("karate_transcript.json", os.path.join(web_dir, "transcript.json"))

    print("\n[DONE] Web data exported to karate_output/web/")

if __name__ == "__main__":
    main()
