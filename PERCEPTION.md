# PERCEPTION.md — the perception (CV) layer

**Owner:** the 4090 teammate.  **Goal:** give the safety agent better *eyes*.

The agent already works VLM-only (Qwen3-VL reads the frame + policy and judges
risk). Your job is to add an on-prem **YOLO detector** that outputs *facts* per
frame — objects, counts, distances, PPE-missing, fire/smoke/spill/phone. The VLM
folds those facts in and judges more accurately, on the big screen with boxes drawn.

## The one rule that wins or loses the hackathon
**Facts only. Never decide a risk level here.** No `if no_hardhat: risk = "medium"`.
Risk is reasoned by the VLM from `safety_policy.txt`. You output *measurements*
(boxes, counts, distances, booleans); the agent decides severity. Geometry (a
distance in metres) is a measurement and is fine.

## Contract (the only seam between us)
For every frame in `frames/` you produce:

1. `perception/<frame_stem>.json` — facts (schema below)
2. `frames_annotated/<frame_name>.jpg` — the frame with boxes drawn (optional, but
   it looks great in the demo)

The agent calls `perception.load_perception(frame_name)`; if the JSON exists it is
shown to the VLM and the dashboard shows your annotated frame. If it doesn't exist,
the agent runs VLM-only — so partial progress is always safe.

### JSON schema (`perception.EXAMPLE`)
```json
{
  "frame": "cam7_t3.jpg",
  "detections": [
    {"label": "person",     "bbox": [820, 300, 90, 210], "conf": 0.91},
    {"label": "forklift",   "bbox": [880, 360, 260, 300], "conf": 0.88},
    {"label": "no_hardhat", "bbox": [840, 300, 40, 50],   "conf": 0.79}
  ],
  "derived": {
    "people": 3,
    "forklifts": 1,
    "min_person_forklift_dist_m": 1.2,
    "ppe_missing": ["hardhat", "hi_vis_vest"],
    "fire": false, "smoke": false, "spill": false, "phone_in_use": false
  },
  "annotated_image": "frames_annotated/cam7_t3.jpg"
}
```
- `bbox` = `[x, y, w, h]` in pixels (top-left + width + height).
- `min_person_forklift_dist_m` = `null` if you can't estimate it.
- `label` must be one of `perception.LABELS`
  (`person, forklift, hardhat, no_hardhat, hi_vis_vest, no_vest, fire, smoke, spill, phone`).
  Map your YOLO class names onto these.

## What to implement
Fill in `detect_frame(image_path)` and reuse `run()` in `perception.py`:
```bash
python perception.py            # batch over frames/  -> perception/*.json (+ annotated)
```

### Model — try READY WEIGHTS first; datasets only if they underperform on our frames
COCO has `person` but **NOT `forklift`** — don't expect COCO alone to do the near-miss
line. Fastest path = load existing weights, no training, into `weights/` (git-ignored):
- **person** → `yolov8s.pt` (COCO, auto-downloads with ultralytics)
- **forklift** → `keremberke/yolov8s-forklift-detection` (Hugging Face, ready `.pt`, ~81 mAP@50)
- **PPE** (hardhat / no_hardhat / vest / person / cone) →
  `VoxDroid/Construction-Site-Safety-PPE-Detection` or
  `snehilsanyal/Construction-Site-Safety-PPE-Detection` (pretrained weights + 10 classes)
- **fire / smoke / spill** → P1 only; skip on the first pass.

License: ultralytics/YOLOv8 is AGPL-3.0 (fine for the hackathon).

Our frames are a grey Turkish press-shop CCTV — different domain from construction-site
weights. **Only if** a ready weight is clearly wrong on `frames/`, fine-tune on the
matching Roboflow set (export **YOLOv8 format WITH labels**), minutes on a 4090:
- forklift+person: https://universe.roboflow.com/test-gun7j/project-forklift01
- PPE: https://universe.roboflow.com/test-h7imi/hello-ogmw7
- `ultralytics` example:
  ```python
  from ultralytics import YOLO
  m = YOLO("weights/safety.pt")
  r = m(image_path, verbose=False)[0]
  for b in r.boxes:
      name = m.names[int(b.cls)]; x1,y1,x2,y2 = b.xyxy[0].tolist()
      ...  # map name -> LABELS, append {"label","bbox":[x1,y1,x2-x1,y2-y1],"conf"}
  ```

### Derived facts
- `people`, `forklifts` = counts.
- `min_person_forklift_dist_m`: pixel distance between nearest person/forklift boxes,
  scaled by a rough ground-plane factor. A documented approximation is fine for the demo.
- `ppe_missing`: e.g. a `person` box overlapping a `no_hardhat`/`no_vest` box → add
  `"hardhat"`/`"hi_vis_vest"`.
- `fire`/`smoke`/`spill`/`phone_in_use`: true if such a detection exists above threshold.

## Stretch (if time)
- Real-time: decode a CCTV clip (`cv2`/`decord`), run the detector live, write frames
  + perception as it goes. 4090 handles this easily.
- **Eval slide:** run on a labeled split and report detector mAP + how often the VLM's
  verdict agrees with ground truth. Great for the ops-manager judges.

## Git workflow
- `git pull --rebase origin main` before you push; commit small, prefix messages `[B]`.
- You own: `perception.py`, `frames_annotated/`, `perception/`, training scripts.
- Don't touch `vlm_judge.py` / `dashboard.py` (that's A). Coordinate before editing
  `requirements.txt` / `config.py`.
- Weights/datasets are git-ignored (`*.pt`, `datasets/`, `runs/`); JSON + annotated
  frames are committed.

## Definition of done (today)
`python perception.py` writes a valid `perception/*.json` (+ annotated jpg) for the
24 frames in `frames/`; then `python main.py` shows DETECTOR FACTS in the verdicts and
the dashboard draws boxes. Ship the first frame end-to-end early, then improve.
