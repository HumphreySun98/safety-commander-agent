# SafetyCommander — Task Board

Role 01 **Safety officer** agent for the Zapdos Labs hackathon. Demo today, judged by
operations managers. Owners: **A = repo owner (agent/demo)**, **B = 4090 teammate
(perception/CV)**, **both = shared**.

> **The rule that wins or loses:** the **VLM** decides risk by reading
> `safety_policy.txt`; code only *routes* actions and *supplies facts*. Never hardcode
> `if hazard → risk`. (Judging question #2.)

---

## 0. Status snapshot

- ✅ Core agent pipeline works end-to-end on real CCTV, pushed to GitHub (private).
- ✅ Perception interface contract merged — A & B can now work in parallel.
- ⏳ Remaining = perception layer (B) + demo polish / integrations / pitch (A).

---

## 1. Done ✅

- [x] Repo scaffold: `config.py`, `vlm_judge.py`, `actions.py`, `shift_report.py`,
      `main.py`, `dashboard.py` (Flask), `templates/index.html`, `requirements.txt`,
      `.env(.example)`, `.gitignore`, `README.md`.
- [x] **Reasoning core**: `judge_frame()` calls live Qwen3-VL, returns structured JSON
      (observation, hazard_type, risk_level, policy_clause, reasoning, actions);
      tolerant JSON parsing; **grounding** instruction (no hallucinated criticals);
      `temperature=0` for reproducibility.
- [x] **Policy** `safety_policy.txt`: PPE / forklift / fire / spill / guarding / LOTO +
      **risk classification (none→critical)** + zone exemptions; clause 1.6 scoped.
- [x] **Guarded actions** + `dispatch()` routing (log / notify / corrective / escalate /
      flag-area) keyed off the VLM's risk level only.
- [x] **Shift handoff report** (markdown) with counts, by-hazard, open correctives, log.
- [x] **Dashboard**: live polling, current frame, verdict + cited clause, actions, feed,
      final report; serves `/frames` and `/annotated`.
- [x] **Real demo data**: Mendeley CCTV (CC BY 4.0) → `frames/` (8 cams × 3), attribution
      in `frames/SOURCES.md`; Wikimedia set in `frames_samples/` (escalation path).
- [x] `extract_frames.py` (imageio-ffmpeg, self-contained).
- [x] **Perception contract**: `perception.py` (schema + loaders + stub),
      `PERCEPTION.md`, `judge_frame(..., perception=)`, dashboard box overlay,
      gitignore weights. Verified: a "0.8 m" fact made the VLM escalate to HIGH citing 2.1.

---

## 2. To do

### One-time setup
- [ ] **[A]** Add B as collaborator:
      `gh api --method PUT repos/HumphreySun98/safety-commander-agent/collaborators/<B> -f permission=push`
- [ ] **[A]** Send `VLLM_KEY` to B privately (not in git).
- [ ] **[B]** Clone, `pip install -r requirements.txt ultralytics`, create `.env`, get datasets.
- [ ] **[A]** (optional) Get a free Roboflow API key for clear-hazard frames + eval.

### P0 — must have for the pitch
- [ ] **[B]** Implement `perception.detect_frame()` + train/run YOLO on the Roboflow sets;
      write `perception/*.json` + `frames_annotated/*.jpg` for every frame in `frames/`.
      (See `PERCEPTION.md`. Facts only — no risk levels.)
- [ ] **[A]** **"Edit policy → re-judge → verdict flips" live demo** (scripted, reliable) —
      the proof that the *model* reasons, not the code. (Judging #2.)
- [ ] **[A]** **Slack webhook** for `notify_supervisor` (visible real integration → pillar ②).
- [ ] **[A]** Dashboard polish for the big screen (font sizes, layout at demo resolution).
- [ ] **[both]** Pitch deck (~5 slides) + **3-minute run-of-show** + who-says-what.

### P1 — high value
- [ ] **[A/B]** Clear high-severity frames (fire / smoke / spill / forklift-proximity) for a
      full none→critical spread of *true positives* (Roboflow images, or dense-sample the
      CCTV forklift/intervention clips). Fire/smoke/spill only exist in Roboflow.
- [ ] **[B]** **Eval slide**: detector mAP + how often the VLM verdict agrees with ground-truth
      labels → credibility for ops-manager judges. (Judging #4.)
- [ ] **[A]** Corrective-action **"tickets" panel** in the dashboard + handoff report export.

### P2 — stretch (only if time)
- [ ] **[B]** Real-time video (decode a clip, detect live, stream frames + perception).
- [ ] **[B]** Better person↔forklift proximity (ground-plane calibration).
- [ ] **[B]** SOP / LOTO step-compliance using the NVIDIA SOP dataset.
- [ ] **[A]** Show detection confidence / fact-vs-verdict side-by-side on cards.

---

## 3. Demo run-of-show (target ~3 min)

1. **Problem (15s):** safety officer can't watch every camera every minute.
2. **Watch→decide→act (60s):** start `dashboard.py`; agent streams real CCTV, for each
   frame shows verdict + **cited policy clause** + auto actions; show a YOLO-boxed frame.
3. **Proof it reasons (45s):** edit `safety_policy.txt` live → Restart shift → same frame,
   different verdict, new clause cited. *No code changed.*
4. **It owns the shift (30s):** show the **handoff report** + the **Slack** alert + open
   corrective tickets.
5. **Evidence (20s):** the eval slide (agrees with ground truth X%).
6. **Close (10s):** "reasoning model + real tools + fully autonomous — built for your floor."

Answers the 4 judging questions and shows all 3 "agent" pillars (reasoning / tools / autonomous).

---

## 4. Checkpoints today
- **CP1 (late morning):** B produces 1 perception JSON; A shows DETECTOR FACTS + a box on the dashboard.
- **CP2 (early afternoon):** B → all frames + eval; A → Slack + live-policy demo + tickets.
- **CP3 (before 5:00):** full merge, big-screen rehearsal, slides locked. Buffer to 5:30.

## 5. Risk / fallback
- **`main` always runs (VLM-only is the safety net).** The YOLO layer is additive — if
  training overruns, the demo still works without it. Say this on stage (pragmatism = points).
- Keep weights/datasets out of git; commit small `perception/*.json` + annotated jpgs.
- `pull --rebase` before every push; A and B own disjoint files (see PERCEPTION.md).
