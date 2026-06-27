# SafetyCommander — 3-minute demo run-of-show

Judged by operations managers. Lead with the strong, true lines (see `docs/eval.md`).
**It's an agent, not a chatbot: no one types — it watches, reasons, acts, reports.**

## Pre-flight (do BEFORE you go on)
- [ ] Warm-run everything once on the **real 30B endpoint** (so first calls are cached/fast).
- [ ] `SC_VIDEO=demo_clips python dashboard.py` → http://localhost:8000 loads, loop streams.
- [ ] `python demo_policy_flip.py` prints NONE→MEDIUM (climax works). **Run it STANDALONE**
      (not while the dashboard loop is hammering the endpoint — concurrent load can make
      Policy A flag the operator's PPE → MEDIUM→MEDIUM instead of the clean NONE→MEDIUM).
      Reliable in isolation (3/3). **Record a clean run as backup.** If it ever comes up
      MEDIUM→MEDIUM live, pivot: "and it now cites the exact clause I added — 2.6" (always true).
- [ ] Have ready in tabs/windows: dashboard · a terminal · annotated `frames_annotated/cam8_t1.jpg` (red 2.1 m line) · `docs/eval.md`.
- [ ] Backup: a screen recording of the full run (in case the shared endpoint stalls; B's local-VLM fallback = break-glass, switch via one `.env` line).
- [ ] Roles: one **drives** (clicks/types), one **narrates**.

---

## The script (≈3:00)

**0:00–0:20 · Problem + what it is**
> "A safety officer can't watch every camera every second. This is an autonomous safety
> officer — it watches the floor on video, judges each event against the plant's written
> rules, takes the action, and hands a report to the next shift. No one is chatting with it."

**0:20–1:15 · Live video closed loop** *(dashboard, SC_VIDEO=demo_clips)*
- Point at the **violation banner burned on the video** as windows stream.
- Land the forklift near-miss: *"HIGH — forklift-pedestrian proximity, citing clause 2.1."*
- Point at the feed: *"It just opened a corrective ticket and escalated the near-miss —
  automatically. Watch → decide → act → no human in the loop."*

**1:15–1:40 · Precision + auditability** *(show `frames_annotated/cam8_t1.jpg`)*
> "Detection measures what the model can't eyeball — the operator is **2.1 m** from the
> forklift (red line). And when it flags the overload, it retrieves and **cites OSHA
> 1910.178** — so the incident log says, on the record, why it acted."

**1:40–2:30 · The climax: the model reasons from the rulebook** *(terminal)*
- `python demo_policy_flip.py` — same forklift footage, judged twice.
> "Our rulebook says nothing about load height — it passes: **NONE**. Now I add **one line**
> to the policy — 'max 2 stacked bins' — I touch **zero code** — and the same footage is now
> a **MEDIUM violation, citing the clause I just wrote**."
- **PAUSE. Look at the judges.** *"The reasoning isn't in my code. It's reading the rulebook."*

**2:30–2:50 · It owns the shift** *(dashboard report / handoff)*
> "At end of shift it hands this report: violations, near-misses, open corrective tickets,
> each with the clause cited — the page a supervisor signs every day." *(+ Slack alert if wired)*

**2:50–3:00 · Close (answer the 4 judging questions in one breath)**
> "Real problem, the **model** reasons (not hardcoded — you saw the policy flip), believable
> end-to-end on real CCTV, zero false alarms on the high-severity line. An agent that
> watches, reasons, acts, and reports — built for your floor."

---

## Maps to judging
| Question | Beat |
|---|---|
| Real operational problem? | 0:00 + the handoff report |
| **Model reasoning, not the dev?** | **1:40 climax (policy flip)** + cited clauses/OSHA |
| Believable end-to-end + feasible? | 0:20 live video loop + ~4 s/window latency |
| Show to an ops manager? | handoff report + 0 false criticals (`docs/eval.md`) |

## Honest lines (if asked — don't overclaim)
- "On the **forklift near-miss line** it catches them, cites OSHA, **zero false alarms**." Not "catches everything."
- Open machine-guard detection is a known gap; weekly inspections/permits/KPI are out of scope (this is the **real-time safety-monitor** slice of the role).
- Eval is a 23–26 frame demo set, not a benchmark; distance is a documented approximation.

## If the endpoint stalls
Switch `.env` `VLLM_BASE_URL` to B's local Qwen2.5-VL (one line), or play the screen recording. Keep talking.
