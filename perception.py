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
# DETECTION  (ultralytics YOLO)
#
# Design: a small registry of YOLO models, each with a map from its own class
# names onto our canonical LABELS. We run every available model, merge + dedup
# the boxes, and report only the facts our active models can actually measure
# ("capabilities"). Dropping a new weights file into weights/ extends coverage —
# no code change. There is deliberately NO risk logic anywhere below.
# ===========================================================================
import math
import os

WEIGHTS_DIR = config.BASE_DIR / "weights"
CONF_THRES = float(os.getenv("PERCEPTION_CONF", "0.25"))   # base run threshold
# Per-LABEL confidence overrides (applied after class mapping). The forklift model
# (keremberke) is reliable on real forklifts at high conf but false-fires on press
# machines around 0.65-0.77, so require 0.8 for forklift while keeping person low.
LABEL_CONF = {"forklift": float(os.getenv("FORKLIFT_CONF", "0.8"))}
PERSON_HEIGHT_M = 1.7   # assumed average standing height, for the px->m scale
IOU_DEDUP = 0.6         # merge boxes of the same label across models above this IoU

# Each detector: a weights file (COCO model name auto-downloads) + a name map.
# A map value of None means "ignore this class". Add the forklift/PPE models here
# once their weights exist in weights/ — they are picked up automatically.
DETECTOR_SPECS = [
    # General COCO model — reliable person detection (no forklift/PPE in COCO).
    {"weights": "yolov8s.pt", "map": {"person": "person"}},
    # Trained safety model (forklift+person / PPE). Plugged in when present.
    {"weights": str(WEIGHTS_DIR / "forklift.pt"),
     "map": {"forklift": "forklift", "person": "person",
             "fork": "forklift", "worker": "person"}},
    {"weights": str(WEIGHTS_DIR / "ppe.pt"),
     "map": {"Hardhat": "hardhat", "NO-Hardhat": "no_hardhat", "helmet": "hardhat",
             "no_helmet": "no_hardhat", "no-helmet": "no_hardhat",
             "Safety Vest": "hi_vis_vest", "vest": "hi_vis_vest",
             "NO-Safety Vest": "no_vest", "no_safety_vest": "no_vest",
             "Person": "person"}},
]

_MODELS = None           # lazily-loaded list of (YOLO model, name_map)
_CAPABILITIES = None     # set of LABELS our active models can produce


def _load_models():
    """Load every detector whose weights are available. Cached after first call."""
    global _MODELS, _CAPABILITIES
    if _MODELS is not None:
        return _MODELS, _CAPABILITIES
    from ultralytics import YOLO
    models, caps = [], set()
    for spec in DETECTOR_SPECS:
        w = spec["weights"]
        # COCO names auto-download; local trained weights must exist on disk.
        is_coco = "/" not in w and w.startswith("yolov8")
        if not is_coco and not Path(w).exists():
            continue
        try:
            m = YOLO(w)
        except Exception as e:
            print(f"  (perception: could not load {w}: {e})")
            continue
        models.append((m, spec["map"]))
        caps.update(v for v in spec["map"].values() if v)
    if not models:
        raise RuntimeError("No detector weights available (not even COCO).")
    _MODELS, _CAPABILITIES = models, caps
    return _MODELS, _CAPABILITIES


# --- geometry helpers (pure measurement) -----------------------------------

def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedup(dets):
    """Per-label greedy NMS so two models detecting the same object count once."""
    out = []
    for label in {d["label"] for d in dets}:
        group = sorted([d for d in dets if d["label"] == label],
                       key=lambda d: d["conf"], reverse=True)
        kept = []
        for d in group:
            if all(_iou_xywh(d["bbox"], k["bbox"]) < IOU_DEDUP for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def _box_gap_px(b1, b2):
    """Shortest pixel distance between two boxes (0 if they overlap)."""
    ax, ay, aw, ah = b1
    bx, by, bw, bh = b2
    dx = max(0.0, max(bx - (ax + aw), ax - (bx + bw)))
    dy = max(0.0, max(by - (ay + ah), ay - (by + bh)))
    return math.hypot(dx, dy)


def _nearest_person_forklift(detections):
    """Return (dist_m, person_det, forklift_det) for the closest pair, scaling
    pixels to metres via a standing person's height. (None, None, None) if no pair.
    Shared by compute_derived and the annotation overlay so they agree exactly."""
    persons = [d for d in detections if d["label"] == "person"]
    forklifts = [d for d in detections if d["label"] == "forklift"]
    best, bp, bf = None, None, None
    for p in persons:
        ph = p["bbox"][3]
        if ph <= 0:
            continue
        mpp = PERSON_HEIGHT_M / ph
        for f in forklifts:
            dist_m = _box_gap_px(p["bbox"], f["bbox"]) * mpp
            if best is None or dist_m < best:
                best, bp, bf = dist_m, p, f
    return (round(best, 1) if best is not None else None), bp, bf


# --- derived facts (counts / distance / PPE / hazards) ---------------------

def compute_derived(detections, capabilities):
    """Turn raw detections into the 'derived' facts block.

    MEASUREMENTS ONLY. Every value here is something we counted or measured; none
    of it is a risk judgment. We only emit a fact if an active detector can
    actually assess it (so we never imply we checked for fire when we didn't).
    """
    derived = {}
    persons = [d for d in detections if d["label"] == "person"]
    forklifts = [d for d in detections if d["label"] == "forklift"]

    if "person" in capabilities:
        derived["people"] = len(persons)
    if "forklift" in capabilities:
        derived["forklifts"] = len(forklifts)

    # closest person<->forklift distance, in metres (rough ground-plane scale:
    # a standing person ~PERSON_HEIGHT_M tall sets metres-per-pixel locally).
    if "person" in capabilities and "forklift" in capabilities:
        dist_m, _, _ = _nearest_person_forklift(detections)
        derived["min_person_forklift_dist_m"] = dist_m

    # PPE missing: a no_hardhat / no_vest box overlapping a person.
    if {"no_hardhat", "no_vest"} & capabilities:
        missing = set()
        for d in detections:
            if d["label"] == "no_hardhat":
                missing.add("hardhat")
            elif d["label"] == "no_vest":
                missing.add("hi_vis_vest")
        derived["ppe_missing"] = sorted(missing)

    # boolean hazards — only if a detector for that class is active.
    label_to_key = {"fire": "fire", "smoke": "smoke", "spill": "spill",
                    "phone": "phone_in_use"}
    present = {d["label"] for d in detections}
    for label, key in label_to_key.items():
        if label in capabilities:
            derived[key] = label in present
    return derived


# --- annotation -------------------------------------------------------------

_COLORS = {  # BGR
    "person": (0, 200, 0), "forklift": (0, 140, 255),
    "hardhat": (0, 200, 0), "hi_vis_vest": (0, 200, 0),
    "no_hardhat": (0, 0, 255), "no_vest": (0, 0, 255),
    "fire": (0, 0, 255), "smoke": (0, 0, 255),
    "spill": (0, 0, 255), "phone": (0, 0, 255),
}


def _center(box):
    x, y, w, h = box
    return int(x + w / 2), int(y + h / 2)


def _annotate(image_path, detections, out_path, pair=None):
    """Draw labelled boxes for the dashboard, plus (if `pair` given) a person<->
    forklift distance line + "X.X m" label — the crown-jewel proximity fact made
    visible on the big screen. `pair` = (dist_m, person_det, forklift_det).
    Returns the output path, or None."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    for d in detections:
        x, y, w, h = (int(v) for v in d["bbox"])
        color = _COLORS.get(d["label"], (255, 255, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        tag = f"{d['label']} {d['conf']:.2f}"
        ty = max(0, y - 6)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, ty - th - 3), (x + tw, ty + 2), color, -1)
        cv2.putText(img, tag, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    # distance overlay: line between the nearest person & forklift + metre label
    if pair and pair[0] is not None and pair[1] and pair[2]:
        dist_m, p, f = pair
        pc, fc = _center(p["bbox"]), _center(f["bbox"])
        line_col = (0, 0, 255) if dist_m < 3.0 else (0, 215, 255)  # red if <3 m
        cv2.line(img, pc, fc, line_col, 2, cv2.LINE_AA)
        cv2.circle(img, pc, 4, line_col, -1)
        cv2.circle(img, fc, 4, line_col, -1)
        mid = ((pc[0] + fc[0]) // 2, (pc[1] + fc[1]) // 2)
        label = f"{dist_m} m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bx, by = mid[0] - tw // 2, mid[1]
        cv2.rectangle(img, (bx - 4, by - th - 6), (bx + tw + 4, by + 4), line_col, -1)
        cv2.putText(img, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


# --- the public entry points ------------------------------------------------

def detect_frame(image_path) -> dict:
    """Run detection on ONE frame; return a perception dict shaped like EXAMPLE.

    Facts only — counts, boxes, distances, PPE/hazard booleans. The VLM
    (vlm_judge.py) decides the risk level from safety_policy.txt; nothing here does.
    """
    models, caps = _load_models()
    frame_name = Path(image_path).name

    detections = []
    for model, name_map in models:
        r = model(str(image_path), conf=CONF_THRES, verbose=False)[0]
        for b in r.boxes:
            name = model.names[int(b.cls)]
            label = name_map.get(name)
            if not label:
                continue
            conf = float(b.conf)
            if conf < LABEL_CONF.get(label, CONF_THRES):   # per-label threshold
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            detections.append({
                "label": label,
                "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
                "conf": round(conf, 2),
            })
    detections = _dedup(detections)

    derived = compute_derived(detections, caps)

    pair = _nearest_person_forklift(detections)   # (dist_m, person, forklift)
    annotated_rel = f"{ANNOTATED_DIR.name}/{frame_name}"
    annotated = _annotate(image_path, detections, ANNOTATED_DIR / frame_name,
                          pair=pair)

    return {
        "frame": frame_name,
        "detections": detections,
        "derived": derived,
        "annotated_image": annotated_rel if annotated else None,
    }


def run(frames_dir=None):
    """Batch: detect every frame in frames_dir, write perception JSON (+ annotated)."""
    frames_dir = Path(frames_dir or config.FRAMES_DIR)
    frames = sorted(p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    _, caps = _load_models()
    print(f"perception: {len(frames)} frames | active capabilities: "
          f"{', '.join(sorted(caps)) or '(none)'}")
    for f in frames:
        res = detect_frame(str(f))
        save_perception(res)
        d = res["derived"]
        print(f"  {f.name}: {len(res['detections'])} dets | {d}")
    print(f"done -> {PERCEPTION_DIR}")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and Path(arg).is_file():           # single-frame mode for quick checks
        res = detect_frame(arg)
        save_perception(res)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        run(arg)
