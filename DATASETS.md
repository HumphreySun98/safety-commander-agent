# DATASETS — what we actually use (filtered from the brief's 29+ sets)

Decision key: **USE** = in the demo now · **FALLBACK** = only fine-tune if ready
weights are weak on our frames · **SKIP** = not worth the time/bandwidth.

---

## 1. Vision model — USE (the agent's brain, already wired)
OpenAI-compatible **Qwen3-VL** endpoint (`config.py` / `.env`). This does the
*reasoning* (risk level from the policy). Nothing to download.

## 2. Demo footage (A) — USE
- **Mendeley "Safe & unsafe behaviours" CCTV** — `data.mendeley.com/datasets/xjmtb22pff/1`,
  **CC BY 4.0**. Already sampled into `frames/` (+ `frames/SOURCES.md`). This is the
  real factory footage the demo runs on.
- *Optional (P1)* clear-hazard stills for a full none→critical spread — pull a few raw
  **images** (no labels, they feed the VLM) from `sf-0p9wv` (fire/smoke → critical) and
  `chemical-thray` (spill → medium).

## 3. Perception / YOLO (B) — READY WEIGHTS FIRST (probably zero dataset downloads)
Load existing weights, no training (see `PERCEPTION.md`):
- person → `yolov8s.pt` (COCO). **COCO has no `forklift`.**
- forklift → `keremberke/yolov8s-forklift-detection` (HF).
- PPE → `VoxDroid` / `snehilsanyal` `Construction-Site-Safety-PPE-Detection` (pretrained).

### FALLBACK datasets — only if a ready weight is wrong on our CCTV
Export **YOLOv8 format WITH labels**, fine-tune on the 4090 (minutes):

| purpose | dataset | classes |
|---|---|---|
| Forklift near-miss (primary) | `test-gun7j/project-forklift01` | forklift, **person** |
| Forklift near-miss (alt) | `helo-rgfls/ddd-xuf6i` | forklift, person |
| PPE (primary, richest) | `test-h7imi/hello-ogmw7` | Person, Hardhat, Mask, Vest, Cone, vehicle, machinery |
| PPE negatives (NO-hardhat) | `muhamad-ilham-8ppof/ppi-dev-safety` | Person, Hardhat, **NO-Hardhat**, Suit, NO-Suit |
| On-domain combined (small) | `bui-hung/factory1-dinob` | helmet, forklift, person, no_helmet, no_safety_vest, cone |
| Fire/smoke (P1) | `sensorikas-workspace/sf-0p9wv` | FIRE, SMOKE |
| Spill (P1) | `rashed-sawab/chemical-thray` | spill |
| Phone use (P1, clause 5.1) | `nmapogha/phone-use-detection` | phone |

### SKIP (waste of time for our demo)
- **Forklift-only, no person** → can't do near-miss: `forkliftcollision2`, `forklift-detect-pkyqw`.
- **Redundant helmet/vest-only**: `safetyhelmet-*`, `vest-elvfp`, `helmet-detection-hnsa1`,
  `helmet_detect_final`, `safe-check`, `find-helmet-safety-shoe-and-more`, `dmn_cctv`.
- **Noisy / irrelevant**: `camera-22` (noisy/dup), `geofencefoot`, `test-segmentation-qvngo`,
  `factory-origin-footage` / `factory-footage-dzngi` (plain person, no labels we need).

## 4. Video datasets
- **USE:** Mendeley CCTV (above) — CC BY 4.0, open.
- **SKIP:** `Egocentric-100K` (head-mounted manual labour, not a safety-officer view);
  `Human-Robot factory` (CC BY-**NC**, 13 GB, robot-cell — heavy + non-commercial);
  `NVIDIA SOP server-fan` (non-commercial; SOP-step, not safety — only a P2 LOTO stretch).

## 5. Licenses
Demo (non-shipping) is fine with CC BY / Apache / even NC — just **cite** sources.
Our primary footage (Mendeley) is **CC BY 4.0**; most Roboflow sets are CC BY 4.0.
Avoid the NC ones if you ever productize.
