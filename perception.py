"""
perception.py — PERCEPTION LAYER  (owned by the 4090 teammate).

Detection ONLY. This module produces FACTS about each frame: which objects are
present, how far apart they are, whether PPE is missing, whether there is fire /
smoke / spill / phone use. It must NOT decide a risk level — risk is the VLM's job
in vlm_judge.py, reasoning from safety_policy.txt. Keep it that way: there is no
"if hazard then risk" logic here, ever (that would lose the hackathon's #2 point).

Contract (full spec in PERCEPTION.md). For each frame you write:
    perception/<frame>.json        the facts (schema = EXAMPLE below)
    frames_annotated/<frame>.jpg   the frame with boxes drawn (optional but great for the demo)

The agent calls load_perception(frame_name) and folds the 'derived' facts into the
VLM prompt as DETECTOR FACTS. If no perception file exists, the agent still works
(VLM-only) — so you can ship incrementally.
"""
import json
from pathlib import Path

import config

PERCEPTION_DIR = config.PERCEPTION_DIR
ANNOTATED_DIR = config.ANNOTATED_DIR

# Canonical labels the agent understands. Map your YOLO class names onto these.
LABELS = [
    "person", "forklift",
    "hardhat", "no_hardhat", "hi_vis_vest", "no_vest",
    "fire", "smoke", "spill", "phone",
]

# ---- the JSON schema, by example -------------------------------------------
EXAMPLE = {
    "frame": "cam7_t3.jpg",
    "detections": [
        {"label": "person",    "bbox": [820, 300, 90, 210], "conf": 0.91},
        {"label": "forklift",  "bbox": [880, 360, 260, 300], "conf": 0.88},
        {"label": "no_hardhat","bbox": [840, 300, 40, 50],   "conf": 0.79},
    ],
    "derived": {
        "people": 3,
        "forklifts": 1,
        "min_person_forklift_dist_m": 1.2,    # null if not computable
        "ppe_missing": ["hardhat", "hi_vis_vest"],
        "fire": False, "smoke": False, "spill": False, "phone_in_use": False,
    },
    "annotated_image": "frames_annotated/cam7_t3.jpg",   # optional
}
# bbox = [x, y, w, h] in pixels (top-left corner + width + height).


def load_perception(frame_name, perception_dir=None):
    """Read perception/<frame>.json if present, else None.  (Used by the agent.)"""
    d = Path(perception_dir) if perception_dir else PERCEPTION_DIR
    p = d / (Path(frame_name).stem + ".json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_perception(result, perception_dir=None):
    """Write a perception dict to perception/<frame>.json.  (Used by you.)"""
    d = Path(perception_dir) if perception_dir else PERCEPTION_DIR
    d.mkdir(exist_ok=True)
    p = d / (Path(result["frame"]).stem + ".json")
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ===========================================================================
# TODO(4090 teammate): implement the two functions below with ultralytics YOLO.
# ===========================================================================
def detect_frame(image_path) -> dict:
    """
    Run detection on ONE frame and return a perception dict shaped like EXAMPLE.

    Sketch:
        from ultralytics import YOLO
        model = YOLO("weights/safety.pt")          # trained on the Roboflow sets
        r = model(image_path, verbose=False)[0]
        detections = []
        for b in r.boxes:
            name = model.names[int(b.cls)]          # map -> LABELS
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            detections.append({"label": MAP[name],
                               "bbox": [x1, y1, x2-x1, y2-y1],
                               "conf": float(b.conf)})
        derived = compute_derived(detections)       # counts, distances, ppe_missing, fire/smoke/...
        # draw boxes -> frames_annotated/<frame>.jpg, set annotated_image
        return {"frame": Path(image_path).name, "detections": detections,
                "derived": derived, "annotated_image": f"frames_annotated/{Path(image_path).name}"}

    person<->forklift distance: a simple pixel-distance + rough ground-plane scale
    is fine for the demo (document the assumption). Put null if you can't compute it.

    REMEMBER: facts only. Never put a risk level in here.
    """
    raise NotImplementedError(
        "Perception layer not implemented yet (owned by the 4090 teammate). See PERCEPTION.md.")


def run(frames_dir=None):
    """Batch: detect every frame in frames_dir, write perception JSON (+ annotated)."""
    frames_dir = Path(frames_dir or config.FRAMES_DIR)
    frames = sorted(p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for f in frames:
        res = detect_frame(str(f))
        save_perception(res)
        print(f"wrote perception for {f.name}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else None)
