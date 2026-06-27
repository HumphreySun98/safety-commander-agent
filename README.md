# 🦺 SafetyCommander — Autonomous Factory Safety Officer

An AI agent that **owns the safety officer's shift**: it watches the production
floor through camera frames, reasons about risk **by reading the site's written
safety policy**, takes the action that policy requires, and hands a written report
to the next shift.

Built for the **Zapdos Labs · AI Agents for the American Industrial Revolution**
hackathon (Role 01 — Safety officer / EHS coordinator).

---

## The core idea (and why it scores)

The hackathon's central judging question is:

> *"Is the **model** doing the reasoning, or the **developer**? Hardcoded rules do not count."*

SafetyCommander is built around that line:

- The **VLM (Qwen3-VL) decides the risk level** by reading `safety_policy.txt` and
  the camera frame together. It must **cite the specific policy clause** it relied on.
- The code **never** maps a hazard to a risk level. There is no
  `if "no_hardhat": risk = "high"` anywhere. Grep for it — it isn't there.
- `dispatch()` only **routes** the model's risk level to the actions the policy
  prescribes for that level (log / notify / corrective / escalate / flag area).
- **Edit the policy → the judgments change.** Make the welding bay exempt, and the
  same welding frame stops being a violation. That's the whole demo.

This shows all three things the brief asks an "agent" to be: a **reasoning model**,
**workflow tools** (safety log, supervisor notify, corrective actions, escalation,
shift report), and **autonomous** (it runs watch → decide → act → report on its own).

---

## Architecture

```
frames/                 demo camera frames (static; no live camera needed)
safety_policy.txt       the rules — single source of truth, editable by ops manager
knowledge/              RAG corpus — OSHA standards, plant SOPs, SDS (factory regs)
rag.py                  TF-IDF retrieval over knowledge/ (relevant regs per scene)
config.py               env + paths + policy loader
vlm_judge.py            ❤ judge_frame()/judge_clip(): VLM reads policy + frame (+ retrieved regs)
actions.py              guarded actions + dispatch() (routes risk level -> actions)
shift_report.py         accumulates events -> markdown handoff report
main.py                 the autonomous loop (headless)
dashboard.py            Flask live dashboard
templates/index.html    single-page dark UI
extract_frames.py       optional: sample frames from a demo video
```

### Grounding in the factory's regulations (RAG)

`safety_policy.txt` is the editable *house rules*. Behind it, `knowledge/` holds the
actual references — OSHA 1910 standards, the plant's SOPs, and chemical SDS. For each
frame/clip, [rag.py](rag.py) retrieves (TF-IDF) the regulations relevant to the scene and
feeds them to the VLM, which then cites the specific standard. Example: an overloaded
forklift the bare site policy passes is flagged **once OSHA `1910.178(n)(6)`
(obstructed-view) is retrieved** — and the model cites it. Retrieval supplies *knowledge*;
the VLM still decides the risk (no rules live in `rag.py`). Disable with `SC_RAG=0`.

Per frame, the verdict is structured JSON:

```json
{
  "observation": "A worker on the warehouse floor is not wearing a hard hat...",
  "hazard_type": "missing_hardhat",
  "risk_level": "medium",
  "policy_clause": "1.1 Hard hats are mandatory in all production, warehouse, and dock zones.",
  "reasoning": "Section 1.1 requires hard hats in the warehouse and the worker has none; per Section 8 a clear violation with a person exposed is MEDIUM.",
  "recommended_actions": ["Issue hard hat", "Coach worker", "Open corrective action"]
}
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then put the real VLLM_KEY in .env
```

The model endpoint (OpenAI-compatible Qwen3-VL) and key come from the hackathon brief.

## Run

**Headless** (process `frames/`, print + save the handoff report):

```bash
python main.py                              # static frames (single-frame judging)
python main.py demo_clips/forklift_overload.mp4   # VIDEO: one clip, temporal judging
python main.py demo_clips/                   # VIDEO: all 8 cameras as ONE shift
```

**Video mode** is the closed-loop monitor: it slides over a clip in short windows,
sends each window's frames to Qwen3-VL *together*, and judges the **behaviour over
time** (a pedestrian entering a forklift's path = a developing near-miss; an
overloaded load tilting in transit). Single frames miss these; the clip catches them.

**Live dashboard** (recommended for the demo — Flask):

```bash
python dashboard.py                                          # static frames
SC_VIDEO=demo_clips/forklift_overload.mp4 python dashboard.py   # video: one clip
SC_VIDEO=demo_clips python dashboard.py                         # video: 8-camera shift
# open http://localhost:8000
```

The dashboard shows, in real time: the current frame, the VLM verdict, the **cited
policy clause**, the risk level (colour-coded), the actions triggered, the live
event feed, and the final shift handoff report.

**Test just the judge** against the live endpoint:

```bash
python vlm_judge.py                 # judges every frame in frames/
python vlm_judge.py frames/01_forklift.jpg
```

---

## The killer demo: prove the model is reasoning, not the code

1. Run the dashboard — note how a welding frame outside a hot-work bay is flagged.
2. Edit `safety_policy.txt`: add the camera's zone to the **Zone W** welding-bay
   exemption (clause 3.4).
3. Click **Restart shift**. The same frame is now judged `none`/`low`, and the
   model cites the exemption clause. **No code changed.** That's the point.

---

## Outputs

- `logs/safety_log.json`, `corrective_actions.json`, `incidents.json`, `notifications.json`
- `reports/handoff_SHIFT-*.md` — the shift handoff report

## Notes

- Detection (YOLO) is intentionally **not** wired in — Qwen3-VL reads the raw frame
  directly. A YOLO overlay can be added later as a pre-filter if needed.
- Frames are downscaled before sending to respect the model's token budget.
- `VLM_TEMPERATURE=0` by default so verdicts are reproducible for the demo.

### Demo data

- `frames/` — **real factory CCTV**, sampled from the Mendeley *"Video Dataset for
  Safe and Unsafe Behaviours"* (Eskişehir press shop, CC BY 4.0). 8 camera views ×
  3 snapshots. See [frames/SOURCES.md](frames/SOURCES.md) for attribution and the
  camera→class mapping. Extracted with `extract_frames.py`.
- `frames_samples/` — openly-licensed stock imagery (Wikimedia Commons) that
  exercises the higher-severity / escalation path. Run it with
  `python main.py frames_samples`.

A note on honesty: on this low-res wide-angle CCTV the model's most common *true*
finding is missing PPE (clause 1.1/1.2/1.3); it also clears compliant scenes
(`none`) and flags distinct hazards (phone use 5.1, eye protection 1.3). A grounding
instruction in the prompt keeps it from inventing people/loads it cannot see — so it
does **not** raise false criticals. Edit `safety_policy.txt` to match a specific
site and the verdicts shift accordingly; that is the point.
