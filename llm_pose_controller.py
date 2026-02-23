"""
LLM Pose Controller & Animator
================================

This script bridges the gap between the LLM JSON output (e.g. from Groq/Llama)
and the SAM 3D Body Momentum Human Rig (MHR).

How it works:
1. Loads the labeled 3D poses extracted dynamically from the whole video.
2. Accepts a JSON command containing a "start_pose" and an "end_pose".
3. Resolves these named poses from the library.
4. Generates a smooth, interpolated transition between the actual MHR body vertices.
5. Renders the frames to a video/GIF using PyRender!
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def slerp(p0, p1, t):
    """Linear Interpolation for vertices/poses."""
    return (1.0 - t) * p0 + t * p1

def interpolate_pose_sequence(llm_cmd, output_dir="test_output", fps=30, library_path="karate_output/pose_library.npz"):
    if not os.path.exists(library_path):
        print(f"[!] Library {library_path} not found. Ensure pipeline is finished or has generated data.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(" LLM COMMAND RECEIVED:")
    print(json.dumps(llm_cmd, indent=2))
    print("=" * 60)

    # 1. Load the pose library
    lib = np.load(library_path, allow_pickle=True)
    available_keys = list(lib.keys())
    if len(available_keys) == 0:
        print("[!] Pose library is empty.")
        return
        
    def match_key(k, avails):
        k_clean = k.lower().replace(" ", "_").replace("-", "_")
        # exact match
        for a in avails:
            if k_clean == a.lower(): return a
        # partial match
        for a in avails:
            if k_clean in a.lower(): return a
            if a.lower() in k_clean: return a
        # fallback
        print(f"    [WARN] Could not perfectly match '{k}', defaulting to {avails[0]}")
        return avails[0]
        
    actual_start = match_key(llm_cmd["start_pose"], available_keys)
    actual_end   = match_key(llm_cmd["end_pose"], available_keys)

    print(f"\n[+] Resolving Start Pose : '{llm_cmd['start_pose']}' -> Library Key: [{actual_start}]")
    print(f"[+] Resolving End Pose   : '{llm_cmd['end_pose']}' -> Library Key: [{actual_end}]")

    start_data = lib[actual_start].item()
    end_data   = lib[actual_end].item()

    # The actual 3D vertices configuration 
    start_vertices = start_data["mean_vertices"]
    end_vertices   = end_data["mean_vertices"]

    # 2. Setup interpolation frames
    air_time = float(llm_cmd.get("air_time", 1.0))
    steps = int(air_time * fps)
    print(f"\n[→] Generating transition ({steps} frames, representing {air_time}s)...")
    
    transition_frames_vertices = []
    modifiers = np.linspace(0, 1, steps)
    
    # Optional hip twist simulation
    rotation_deg = float(llm_cmd.get("rotation", 0.0))
    
    for i, t in enumerate(modifiers):
        # Morph the vertices 
        interp_verts = slerp(start_vertices, end_vertices, t)
        
        # Apply synthetic Y-axis rotation if requested by LLM
        if rotation_deg != 0:
            # Ease in/out rotation
            rot_t = (1 - np.cos(t * np.pi)) / 2 
            angle = np.radians(rotation_deg * rot_t)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R_y = np.array([
                [cos_a,  0, sin_a],
                [0,      1, 0],
                [-sin_a, 0, cos_a]
            ])
            # center at origin approximately via hip (assuming hip is around y=0 initially or we center it)
            center = interp_verts.mean(axis=0)
            interp_verts = (interp_verts - center) @ R_y.T + center
            
        transition_frames_vertices.append(interp_verts)

    # 3. Render the output
    print(f"\n[+] Rendering {len(transition_frames_vertices)} frames to video...")
    
    # We need the mesh faces to render. Best approach is loading SAM 3D body's estimator 
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.visualization.renderer import Renderer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Loading SAM 3D Model to extract accurate Faces mapping...")
    ckpt = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    mhr = "checkpoints/mhr/assets/mhr_model.pt"
    if not os.path.exists(mhr): mhr = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    
    model, cfg = load_sam_3d_body(ckpt, device=device, mhr_path=mhr)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    faces = estimator.faces
    target_focal_length = 1000.0 # generic portrait
    
    renderer = Renderer(focal_length=target_focal_length, faces=faces)
    
    out_video_path = os.path.join(output_dir, "llm_transition.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 1024, 1024
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w*2, h)) # Side-by-side
    
    # Generic wide shot camera translation
    cam_t = np.array([0.0, 0.5, 3.5]) 
    
    for i, verts in enumerate(tqdm(transition_frames_vertices, desc="Rendering")):
        # Front View
        bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        img_front = renderer(
            verts,
            cam_t,
            bg,
            mesh_base_color=(0.65, 0.74, 0.86),
            scene_bg_color=(1, 1, 1),
        ) * 255
        
        # Side View
        bg_side = np.ones((h, w, 3), dtype=np.uint8) * 255
        img_side = renderer(
            verts,
            cam_t,
            bg_side,
            mesh_base_color=(0.65, 0.74, 0.86),
            scene_bg_color=(1, 1, 1),
            side_view=True
        ) * 255
        
        frame = np.concatenate([img_front, img_side], axis=1).astype(np.uint8)
        
        # Add overlay text
        cv2.putText(frame, f"LLM Prompt: Transitioning {llm_cmd['start_pose']} -> {llm_cmd['end_pose']}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        cv2.putText(frame, f"Progress: {i/steps:.0%}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)
        cv2.putText(frame, f"Simulated Air Time: {air_time}s | Roation: {rotation_deg} deg", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        
        writer.write(frame)
        
    writer.release()
    print(f"\n[SUCCESS] Saved generated LLM transition video to: {out_video_path}")


if __name__ == "__main__":
    # Test JSON configuration from the prompt
    test_json = {
        "start_pose": "yoi ready stance",
        "end_pose": "morote uke augmented block from gedan barai", # A pose we know exists so far!
        "air_time": 2.0,  
        "rotation": 15    
    }

    interpolate_pose_sequence(test_json, output_dir="karate_output/llm_outputs", fps=30)
