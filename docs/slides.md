# SafetyCommander — pitch deck (5 slides)

Content + speaker notes. Pair with the live demo (`docs/demo_script.md`). Honest claims
only (see `docs/eval.md`). Judged by operations managers.

---

## Slide 1 — The problem
**Title:** SafetyCommander — an autonomous safety officer
**Bullets:**
- A safety officer can't watch every camera every second.
- Near-misses, PPE gaps, forklift conflicts go unseen — and unlogged.
- Today: all manual. Walking the floor, judging, writing it up.

*Notes:* "We built the agent that does the floor-watch slice of that job — autonomously."

---

## Slide 2 — What it is (an agent, not a chatbot)
**Title:** Watches · Reasons · Acts · Reports — on its own
**Bullets (the 3 "agent" pillars):**
- **Reasoning model** — Qwen3-VL reads the *written safety policy* + the video, and judges. Not regex, not if-then.
- **Workflow tools** — safety log, corrective tickets, incident escalation, **Slack alerts**, shift handoff.
- **Autonomous** — runs watch→decide→act→report with no human in the loop.

*Notes:* "No one types at it. It owns the shift." (Slack screenshot here.)

---

## Slide 3 — How it works (the grounding stack)
**Title:** Three inputs, one reasoned verdict — none hardcodes risk
**Diagram:**
```
 video → [① YOLO facts: person/forklift + distance]
         [② RAG: the matching OSHA / SOP / SDS]      → VLM reasons RISK + cites clause
         [③ editable site policy: the risk tiers]    → dispatch → actions → handoff report
```
**Bullets:**
- Perception *measures* (2.1 m). RAG *cites* (OSHA 1910.178). Policy *sets the tiers*.
- The **model** decides risk; code only routes the action. (Judging #2.)

*Notes:* "Like a real safety officer: see → consult the rulebook → judge."

---

## Slide 4 — The proof
**Title:** The model reasons from the rulebook — watch it
**Bullets:**
- **Live:** edit ONE line of the policy → the *same* forklift footage flips NONE → MEDIUM,
  citing the clause I just wrote. **Zero code changed.**
- **Evidence (`docs/eval.md`):**
  - **0 false criticals** (video/temporal; single-frame had 4)
  - forklift near-miss caught at HIGH, cites **OSHA 1910.178**, ~**4 s**/window
  - detection: **100% forklift precision (0/16 FP)**, measured **2.1 m** near-miss

*Notes:* This is the climax — run `demo_policy_flip.py`, pause, "I didn't touch a line of code."

---

## Slide 5 — Close
**Title:** An agent built for your floor
**Bullets:**
- Real problem ✓ · the **model** reasons (not the dev) ✓ · believable end-to-end on real CCTV ✓ · safe to show an ops manager ✓
- Honest scope: the **real-time safety-monitor + handoff** slice (not permits/training/KPI).
- Roadmap: scale RAG to full OSHA + every machine's SOP; add guard/PPE/fire detectors; on-prem VLM fallback.

*Notes:* "It watches, reasons, acts, and reports — and it tells you, on the record, why."

---

### Backup slide (if asked about accuracy/limits)
- Eval is a 23–26 frame demo set, not a benchmark; distance is a documented px→m approximation.
- Open machine-guard detection is the known gap; PPE-granular ≠ the dataset's coarse safe/unsafe label.
- Shared endpoint can stall → on-prem Qwen2.5-VL fallback (one `.env` line).
