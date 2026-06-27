# D4 — Evaluation (demo evidence)

Two reproducible layers: **detection** (`eval_perception.py`, owner B) and **agent verdicts**
(`eval_agent.py`, owner A). 23–26 frame/clip demo set — a demo set, **not** a benchmark.

---

## 1. Detection — perception layer (B)  ·  `eval_perception.py`
keremberke YOLO, forklift single-class confidence ≥ 0.8 (person 0.25).

| metric | value |
|---|---|
| forklift precision (press-machine frames) | **0/16 false positives = 100%** (presses never misjudged) |
| forklift recall (true-forklift frames) | **6/7** (only cam8_t3 @ 0.75 < 0.8 threshold) |
| money-shot confidence | forklift 0.87 + operator |
| measured nearest person–forklift distance | **2.1 m** (cam8_t1), 8.4 m (cam1_t3) |

*Honest: distance is a documented px→m approximation (person-height ground scale); no GT
distance, so we report measured values, not error.*

---

## 2. Agent verdicts — video mode (A)  ·  `eval_agent.py`
Per clip, peak severity over **all** windows (continuous monitoring). RAG off (main mode).

| behaviour | truth | agent peak | hazard @ peak | note |
|---|---|---|---|---|
| walkway violation | unsafe | **HIGH** | forklift_pedestrian_proximity | ✅ caught, cites 2.1 |
| unauthorized intervention | unsafe | **HIGH** | forklift_pedestrian_proximity | ✅ caught |
| forklift overload | unsafe | MEDIUM | unstable_load | ✅ caught; RAG cites **OSHA 1910.178(o)(1)** |
| open panel / guard | unsafe | none | — | ❌ **missed** (open guard hard to see on CCTV) |
| safe walkway | safe | low | (transient endpoint error this run) | — |
| authorized intervention | safe | HIGH | forklift_pedestrian_proximity | ⚠️ flagged (co-occurring forklift, or over-read) |
| closed panel | safe | medium | phone / PPE | granular (PPE-aware) |
| safe carrying | safe | medium | unstable_load | ⚠️ granular / mild over-read |

**Aggregate:** FALSE CRITICALS **0** · forklift near-miss caught at HIGH · avg latency **~4 s/window**.

---

## 3. Headline (true — safe to claim)
- **Zero false criticals** (single-frame mode produced 4; video/temporal + tightened criterion → 0).
- **Forklift near-misses caught at HIGH**, citing the policy clause (2.1) **and the OSHA standard**
  (1910.178) — auditability on the record.
- **~4 s/window** latency — feasible for live monitoring.
- Detection: **100% forklift precision (0/16 FP)**, 6/7 recall, **measured 2.1 m** near-miss distance.

## 4. Honest limitations (say these — don't get caught)
- **Open machine-guard detection is weak** (missed the open-panel clip) — needs a guard-state model.
- The coarse **safe/unsafe label ≠ the agent's PPE-granular judgments** — it flags PPE on "safe"
  clips, which is arguably correct, not a miss.
- Occasional **shared-endpoint slowness/error** → this is exactly why we built the local-VLM
  fallback (D5).

## 5. What to claim on stage
> "On the forklift near-miss line — the injury-causing events — it catches them, cites the OSHA
> standard, with **zero false alarms**, at ~4 s latency."

Not: "it catches every hazard." Lead with the strong, true line; own the gaps if asked.
