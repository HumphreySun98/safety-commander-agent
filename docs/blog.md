---
title: "SafetyCommander: an AI safety officer where the model reasons and the code never decides"
published: true
tags: ai, llm, python, architecture
canonical_url: https://dev.to/humphreysun98/safetycommander-an-ai-safety-officer-where-the-model-reasons-and-the-code-never-decides-4765
---

> 📝 Published on dev.to → https://dev.to/humphreysun98/safetycommander-an-ai-safety-officer-where-the-model-reasons-and-the-code-never-decides-4765

Every factory floor already has cameras. The problem is that nobody is watching them. A
safety officer can't be on every camera, on every shift, on every floor — so the footage
gets recorded and reviewed *after* someone is already hurt.

The obvious fix — "detect no hardhat → fire an alarm" — is worse than it sounds. It's
hardcoded, it can't read *your* site's rules, and it drowns the ops team in false alarms
until they mute it. What's actually missing is something that **reasons about risk the way a
human EHS officer does — and can prove why.**

So I built **SafetyCommander**: an autonomous EHS agent that watches the floor, reasons about
hazards from your written policy, grounds every alert in OSHA law, routes it to the right
worker, and hands off a report. It started at the Zapdos Labs × Antler hackathon (*AI Agents
for the American Industrial Revolution*) and I've kept building on it since.

This post is about the one design decision that made it work, and two engineering war-stories
that were more interesting than I expected.

---

## The one decision that matters: risk lives in exactly one module

The hardest question in this whole space — and the one the hackathon judged on — is:

> *Is the **model** doing the reasoning, or is the **developer** hardcoding it? Rules don't count.*

It's an easy thing to *claim* and a hard thing to *build*, because the temptation to sneak a
rule in is everywhere. The moment you write `if hazard == "no_hardhat": risk = "high"`, you no
longer have an agent — you have a rulebook with a language-model bolted on top, and it breaks
the instant the site's policy differs from your assumptions.

So I made one rule for myself: **risk is decided in exactly one place.**

```
THINK    vlm_judge.py   ← reads safety_policy.txt + the footage together,
                          returns a verdict AND the clause it relied on
ACT      actions.py     ← dispatch() only ROUTES the risk level to actions
GROUND   perception.py  ← YOLO measures facts (distance, counts) — no risk
         rag.py         ← retrieves the OSHA clause to cite — no risk
REPORT   shift_report.py, kpi_report.py, planner.py
```

Every module except `vlm_judge.py` is deliberately **decision-free**. `dispatch()` takes the
model's `risk_level` and maps it to actions (log, notify, corrective ticket, escalate). The
perception layer *measures* — it will tell you a forklift is 2.1 m from a person, but it will
never tell you that's dangerous. The retriever *cites* — it hands the model OSHA 1910.178, but
it doesn't decide anything. Risk is the model's job, and only the model's.

The payoff is a property you can *demonstrate live*: **edit one line of the policy and the
verdict flips**, with zero code changes. A hardcoded system physically cannot do that. More on
the demo below.

The verdict the model returns is structured, and it has to cite its work:

```json
{
  "observation": "A forklift moves with a raised load while a worker stands in the aisle...",
  "hazard_type": "forklift_pedestrian_proximity",
  "risk_level": "high",
  "policy_clause": "2.1 A minimum 3-meter separation must be kept between a moving forklift and any pedestrian.",
  "reasoning": "Section 2.1 requires 3 m; perception measured 2.1 m with the load raised → per Section 8 this is HIGH.",
  "recommended_actions": ["Sound horn", "Stop forklift", "Open corrective action"]
}
```

Note the `policy_clause` field. If the model can't point to the rule it applied, the alert
isn't trustworthy — and an ops manager will (correctly) ignore it.

---

## The loop: Watch → Decide → Act → Report

The agent runs a plain, auditable control loop over video — not single stills. It slides over
each clip in short temporal windows and sends the frames in a window to the VLM *together*, so
it reasons about **behaviour over time**: a pedestrian *entering* a forklift's path is a
developing near-miss; a single frame misses it.

- **Watch** — an on-prem YOLO detector measures people, forklifts, and the distance between them.
- **Decide** — Qwen3-VL (served on vLLM) reads the policy, the frames, and those facts, and returns the verdict above.
- **Act** — `dispatch()` routes the risk level to guarded actions.
- **Report** — events accumulate into a written shift-handoff for the next crew, and roll up into weekly plans and monthly KPIs.

Deliberately boring. Boring is auditable, and auditable is what a safety tool needs.

---

## War-story #1: teaching YOLO to ignore a press machine

The perception layer exists for one honest reason: **a VLM can't reliably measure distance on
low-res, wide-angle CCTV.** So YOLO supplies hard numbers *before* the model ever sees the
frame — and outputs facts only.

Getting the forklift detector right was the first place reality bit back. Off-the-shelf
forklift models **false-fired on our press machines** — grey factory CCTV looks nothing like
the construction-site data those models were trained on.

The instinct is "just fine-tune it." I tried. A plain fine-tune fixed the false alarms and
then went *blind to real forklifts* — a classic domain overfit. I tried negative mining
(oversampling confirmed forklift-free press frames as background). It didn't ship either; it
kept costing recall on the money-shot frames.

What actually shipped was much less glamorous and much more effective: **use the pretrained
model and raise the confidence threshold for the forklift class to 0.8.** The presses fire at
0.65–0.77; real forklifts fire at 0.83–0.92. A single threshold cleanly separates them.

```python
LABEL_CONF = {"forklift": float(os.getenv("FORKLIFT_CONF", "0.8"))}
```

Honest result: **18/21 precision**, clean on the demo's money-shot frames, with the remaining
false positives isolated to one camera's viewing angle. The lesson I keep relearning: the
boring knob often beats the clever retrain.

---

## Grounded in real regulation (RAG)

An alert is only as good as its justification. Behind the editable house rules
(`safety_policy.txt`) sits a corpus of the *real* references — OSHA 1910 standards, plant SOPs,
chemical safety sheets. For each scene the agent retrieves the relevant regulation, and the
model cites the specific standard in its verdict.

The nice part: retrieval supplies *knowledge*, not *decisions*. An overloaded forklift your
house policy happens to pass gets flagged the moment OSHA 1910.178 (obstructed view) is
retrieved — but it's still the model that decides, and cites, the risk.

---

## The killer demo: edit one line, watch the verdict flip

This is the whole thesis in twenty seconds.

1. A forklift carries a tall, view-blocking load. Under the current policy, the agent watches it and says: **no violation**.
2. I add **one line** to `safety_policy.txt` — clause 2.6: *a load must not block the operator's view.*
3. Re-run. The **same footage** is now flagged **MEDIUM**, and the model cites **2.6** — the clause I just wrote.
4. No code changed.

That's the difference between an agent and a rulebook. The behaviour came from the policy,
routed through the model's reasoning — not from a branch in my code.

---

## War-story #2: making the boxes actually track the video

The live dashboard plays the CCTV clip and draws YOLO boxes on it. My first version ran
detection once per analysis window (~every 3 s) and painted those boxes on the playing video.
It looked wrong immediately: the video moves at 25 fps, the boxes were frozen at the moment of
analysis, so they lagged the people by seconds. Boxes that don't track are worse than no
boxes — they read as a broken product.

True per-frame detection over a network tunnel isn't feasible at video framerate. The fix was
to flip *who* drives detection: instead of the server pushing stale boxes, **the browser
samples the frame it is currently showing**, sends it to a `/api/detect` endpoint (proxied to
the GPU), and draws the returned boxes. Now the boxes line up with what's on screen, within
one round-trip, and refresh a few times a second.

```js
// grab the frame the <video> is showing right now, detect, draw
cap.getContext('2d').drawImage(video, 0, 0, cap.width, cap.height);
const { boxes } = await (await fetch('/api/detect', {
  method: 'POST', body: JSON.stringify({ image_b64: cap.toDataURL('image/jpeg', 0.6).split(',')[1] })
})).json();
drawBoxes(boxes);
```

The insight that generalizes: when you can't make the producer fast enough, move the sampling
to the consumer that already knows the ground truth (here, the exact frame on screen).

---

## From an agent to a product

An alert nobody receives is useless, so SafetyCommander isn't just the loop — it's a small
web app for the two people who actually use it:

- **Workers** get a Slack/DingTalk-style **inbox**: alerts in *their* zone are pushed to them to Acknowledge / Resolve / Escalate, and their scheduled inspections and training arrive as tasks.
- **Managers** get a console: KPIs, hazard and zone charts, the corrective-action backlog, the AI weekly plan, and an alert-delivery table that closes the loop — who got each alert, and who acted on it.

And it works across three horizons from the same data: **DAY** (live monitor + handoff),
**WEEK** (an AI-proposed inspection/training plan grounded in retrieved industry cadence), and
**MONTH** (a KPI roll-up). The agent proposes; humans confirm.

---

## Honest numbers

- **Reasoning:** verdicts cite the controlling clause; flipping a policy line flips the verdict.
- **Detection:** 18/21 forklift precision, 3/4 recall, measured 2.1 m near-miss distance. (About half of person↔forklift pairs read 0.0 m on 2-D box overlap — reported, not hidden.)
- **Robustness:** in temporal (video) mode, **0 false criticals** across the demo clips; the grounding prompt plus real distances stop the model inventing hazards it can't see.

None of these are "100%." Ops managers respect an honest number far more than a suspicious one.

---

## What's next

This started as a hackathon build and it's still moving: a served remote-perception layer
(the GPU can live anywhere), a local-VLM fallback for air-gapped sites, and richer per-worker
routing. But the core won't change — **the model reasons, and the code never decides.**

Code: [github.com/HumphreySun98/safety-commander-agent](https://github.com/HumphreySun98/safety-commander-agent)
Stack: Qwen3-VL on vLLM · YOLO · TF-IDF RAG over an OSHA/SOP corpus · Flask.

*Built at the Zapdos Labs × Antler hackathon; in active development since.*
