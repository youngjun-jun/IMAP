import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import openai
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple

# --- 1. Visual Preparation (Grids) ---

def sample_frames(video_path: str, num_frames: int = 12) -> List[np.ndarray]:
    """Uniformly sample frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def create_grids(frames: List[np.ndarray], heatmap: np.ndarray, output_dir: Path) -> Dict[str, Path]:
    """
    Create 3 grid images: Raw (Original), Overlay (Heatmap on Image), Heatmap (Heatmap only).
    Returns paths to the saved images.
    """
    # Resize heatmap to match video dimensions
    h, w = frames[0].shape[:2]
    heatmap_resized = []
    overlays = []
    
    # Ensure heatmap has enough frames (simple alignment)
    if len(heatmap) != len(frames):
        # Simple resampling for demo purposes
        indices = np.linspace(0, len(heatmap)-1, len(frames), dtype=int)
        heatmap = heatmap[indices]

    for frame, map_slice in zip(frames, heatmap):
        # Normalize and resize map
        map_norm = (map_slice - map_slice.min()) / (map_slice.max() - map_slice.min() + 1e-8)
        map_resized = cv2.resize(map_norm, (w, h))
        
        # Threshold for overlay
        mask = map_resized > map_resized.mean()
        
        # Create Overlay (Apply mask to frame)
        overlay = frame.copy()
        overlay[~mask] = overlay[~mask] // 2 # Darken non-active areas
        overlays.append(overlay)
        
        # Create Heatmap visual (Apply colormap)
        heatmap_vis = cv2.applyColorMap((map_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heatmap_resized.append(cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB))

    # Helper to save grid
    paths = {}
    for name, imgs in [("raw", frames), ("overlay", overlays), ("heatmap", heatmap_resized)]:
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        for ax, img in zip(axes.flat, imgs):
            ax.imshow(img)
            ax.axis('off')
        save_path = output_dir / f"{name}_grid.jpg"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        paths[name] = save_path
        
    return paths

# --- 2. LLM Judge (Evaluation) ---

def call_judge(client, model: str, prompt: str, grid_paths: Dict[str, Path]) -> Dict:
    """
    Calls the LLM with the 3 grid images and the prompt.
    Parses the XML response to extract scores.
    """
    def encode_image(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    system_prompt = """
    Judge the activation maps for the given video and prompt.
    Rubric: SL (Spatial), TL (Temporal), PR (Prompt Relevance), SS (Sparseness), OBJ (Object Shape).
    Score 1-5. Return XML: <Assessment><Scores><SL>...</SL>...</Scores></Assessment>
    """
    
    user_content = [
        {"type": "text", "text": f"Prompt: {prompt}"},
        {"type": "text", "text": "Original Video"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(grid_paths['raw'])}"}},
        {"type": "text", "text": "Overlay (Activation)"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(grid_paths['overlay'])}"}},
        {"type": "text", "text": "Heatmap Only"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(grid_paths['heatmap'])}"}},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )
    
    # Simple XML parsing
    xml_str = response.choices[0].message.content
    root = ET.fromstring(xml_str)
    scores = {child.tag: int(child.text) for child in root.find("Scores")}
    return scores

# --- 3. Orchestration ---

def evaluate_sample(sample_id: int, video_path: str, heatmap_path: str, caption: str):
    # Setup
    output_dir = Path(f"results/{sample_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Visuals
    frames = sample_frames(video_path)
    heatmap = np.load(heatmap_path)
    grid_paths = create_grids(frames, heatmap, output_dir)
    
    # 2. Call Judge
    client = openai.OpenAI() # Assumes env var set
    scores = call_judge(client, "o3-pro", caption, grid_paths)
    
    print(f"Sample {sample_id} Scores: {scores}")
    return scores