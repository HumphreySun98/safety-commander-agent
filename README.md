# 🦺 SafetyCommander — Autonomous Factory Safety Officer

An AI agent that **owns the safety officer's shift**. It watches the production floor on
camera, reasons about risk **by reading the site's written safety policy**, takes the
action that policy requires, routes the alert to the right worker, and rolls the shift up
into reports and a forward-looking inspection/training plan.

## Origin & status

Built at the **Zapdos Labs × Antler** hackathon — *AI Agents for the American Industrial
Revolution* (Role 01 — safety officer / EHS coordinator). [Antler](https://www.antler.co) is a
global early-stage venture firm. The first working version shipped during the hackathon, and
it has been in **active development since** — the YOLO perception layer, the two-role web app,
the worker-notification inbox, and the WEEK/MONTH planning were all added after the initial build.

**Stack:** Qwen3-VL served on **vLLM** (OpenAI-compatible) · **YOLO** perception ·
**TF-IDF RAG** over an OSHA / SOP corpus · **Flask** web app.

---

## The core idea (and why it scores)

The hackathon's central judging question is:

> *"Is the **model** doing the reasoning, or the **developer**? Hardcoded rules do not count."*

SafetyCommander is built around that line:

- The **VLM decides the risk level** by reading `safety_policy.txt` and the camera footage
  *together*, and must **cite the specific policy clause** it relied on.
- The code **never** maps a hazard to a risk level. There is no `if "no_hardhat": risk="high"`
  anywhere — grep for it, it isn't there.
- Every other component is deliberately **decision-free**: `actions.dispatch()` only *routes*
  the model's risk level to actions; `perception.py` only *measures* (distances, counts);
  `rag.py` only *retrieves* regulations to cite; `notify.py` only *delivers* alerts to people.
- **Edit one line of policy → the verdict flips.** That is the demo, and it is impossible
  for a hardcoded system.

That makes it a real **agent**, not a chatbot: a reasoning model in a closed
**sense → think → act → report** loop that runs autonomously.

---

## What it does — one agent, three horizons

| Horizon | What the agent does |
|---|---|
| **DAY** (live) | Watches each camera over time (temporal, multi-frame), reasons risk from policy, fires risk-graded actions (log · notify · corrective ticket · escalate · Slack), routes each alert to the right worker, and writes a **shift-handoff report**. |
| **WEEK** | `planner.py` reads the period's data + policy + retrieved industrial EHS practice and proposes a **preventive plan** — inspections & training with when / who / content / target clause. |
| **MONTH** | `kpi_report.py` rolls logs into a **KPI report** — violation rate, near-misses, leading/lagging indicators, top hazards, open corrective-action backlog. |

The agent **proposes**; humans confirm.

---

## Architecture — four layers

```
THINK        vlm_judge.py     ❤ judge_frame()/judge_clip(): VLM reads policy + footage
                              (+ retrieved regs + perception facts) → verdict + cited clause
ACT          actions.py       dispatch(): routes the model's risk level → guarded actions
             notify.py        routes each alert to the right worker (by role/zone) + ack state
REPORT       shift_report.py  accumulates events → markdown handoff
             kpi_report.py    MONTH roll-up   ·   planner.py  WEEK plan

GROUND       perception.py    YOLO measures facts only (person/forklift, distance) — no risk
             rag.py           TF-IDF retrieval over knowledge/ — cites OSHA/SOP — no risk
             safety_policy.txt  the editable house rules (single source of truth)
             knowledge/         OSHA 1910 standards, plant SOPs, SDS

LOOP / UI    main.py          the autonomous loop (headless; static frames or video)
             dashboard.py     Flask app + JSON APIs
             templates/ static/  the web product (below)
```

**Risk is decided in exactly one place — `vlm_judge.py`.** Everything else measures, routes,
retrieves, or reports. That separation is the whole point.

### Codebase map

| File | Key entry points | Role |
|---|---|---|
| `vlm_judge.py` | `judge_frame()` · `judge_clip()` · `_format_user_text()` · `_extract_json()` | **THINK** — assembles the prompt (policy + retrieved regs + perception facts) and parses the VLM's structured verdict. **The only module that produces a risk level.** |
| `actions.py` | `dispatch()` · `notify_supervisor()` · `_post_webhooks()` | **ACT** — routes the verdict's risk level to guarded actions (log/notify/corrective/escalate/Slack). No risk logic. |
| `main.py` | `run_videos()` · `run_shift()` · `sample_windows()` · `_detect_remote()` | **LOOP** — slides over clips in temporal windows; per window calls judge → dispatch → report; optional remote-YOLO client. |
| `perception.py` | `detect_for_frame()` · `compute_derived()` · `_nearest_person_forklift()` | **GROUND** — YOLO person/forklift + nearest distance; `FORKLIFT_CONF=0.8` to reject press-machine false fires. Facts only. |
| `rag.py` | `retrieve_for_hazard()` · `should_use_rag()` · `relevant_enough()` | **GROUND** — TF-IDF over `knowledge/`; gated, citation-only (judge-time OSHA + plan-time cadence). |
| `notify.py` | `route()` · `inbox()` · `set_state()` · `deliveries()` | routes alerts to worker inboxes by role/zone; tracks acknowledge/resolve/escalate. |
| `kpi_report.py` | `summarize()` · `generate_rollup()` | **MONTH** roll-up (rates, leading/lagging, backlog). |
| `planner.py` | `generate_week_plan()` · `render()` | **WEEK** AI inspection/training plan, grounded in retrieved cadence. |
| `shift_report.py` | `ShiftReport.add()` · `generate_handoff()` | **REPORT** — accumulates events into the markdown handoff. |
| `dashboard.py` | Flask routes + `/api/state·kpi·correctives·plan·inbox·deliveries·detect·clip` | the web app + JSON APIs; serves the four views. |
| `demo_policy_flip.py` | — | the climax: judges `cam7_overload` under base policy vs base + clause 2.6. |
| `safety_policy.txt` · `knowledge/` | — | the editable house rules + the OSHA 1910 / SOP / SDS corpus. |

---

## The product — a two-role web app

`python dashboard.py` → `http://localhost:8000`

- **`/` role chooser** → Worker or Manager.
- **`/monitor` — live big-screen:** the camera **video plays in real time** with **YOLO
  boxes + person↔forklift distance** drawn on it, a burned-in violation banner, the VLM
  verdict with the cited clause, and a live event feed.
- **`/worker` — Slack/DingTalk-style inbox:** pick your identity; safety alerts in *your*
  zone are pushed to you with **Acknowledge / Resolve / Escalate**, and your scheduled
  inspections/training arrive as tasks. Mobile-friendly.
- **`/manager` — operations console:** KPI cards, hazard & zone charts, open-CA backlog, the
  AI weekly plan, and an **alert delivery & acknowledgment** table that closes the loop
  (who got each alert, who acted).
- **Trilingual:** English (default) · 中文 · Español, switchable on every page.

---

## Grounding (so the judgment is trustworthy)

- **Perception (YOLO, on a GPU):** `perception.py` detects people & forklifts and measures
  the nearest person↔forklift **distance** — facts the VLM uses instead of guessing (it does
  not invent a "0.0 m" it cannot see). Facts only; never a risk level.
- **RAG (two domain-targeted paths):** at **judge time** it retrieves the relevant **OSHA
  1910** standard for the hazard so the model cites it; at **plan time** it retrieves
  industrial inspection/training cadence to ground the weekly plan. Citation/knowledge only —
  no rules live in `rag.py`. Toggle with `SC_RAG`.

Per window the verdict is structured JSON:

```json
{
  "observation": "A forklift is moving with a raised load while a worker stands in the aisle...",
  "hazard_type": "forklift_pedestrian_proximity",
  "risk_level": "high",
  "policy_clause": "2.1 A minimum 3-meter separation must be kept between a moving forklift and any pedestrian.",
  "reasoning": "Section 2.1 requires 3 m; perception measured 2.1 m with the load raised → per Section 8 this is HIGH.",
  "recommended_actions": ["Sound horn", "Stop forklift", "Open corrective action"]
}
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # set VLLM_BASE_URL / VLLM_KEY (the Qwen3-VL endpoint)
```

---

## Run

**Recommended demo — on the GPU machine** (local YOLO, real-time boxes + distance, no tunnel):

```bash
SC_VIDEO=demo_live python dashboard.py        # then open http://localhost:8000
```
`demo_live/` is the curated camera set (`mkdir demo_live && cp demo_clips/cam{2..8}*.mp4 demo_live`).
With no `SC_VIDEO`, the dashboard defaults to real-time video over `demo_clips/`.

**Drive from a laptop, compute YOLO on a remote GPU** (the perception layer is served over HTTP,
exactly like the VLM):

```bash
export YOLO_URL=https://<gpu-host>/      # a yolo_server.py on the GPU box
SC_VIDEO=demo_live YOLO_URL=$YOLO_URL python dashboard.py
```

**Headless end-to-end** (prints + saves every artifact; good for verifying the whole agent):

```bash
python main.py demo_clips/      # DAY: watch 8 cameras → handoff report
python kpi_report.py            # MONTH: KPI roll-up
python planner.py               # WEEK: AI preventive plan
python demo_policy_flip.py      # the climax (below)
```

---

## The killer demo — prove the model is reasoning, not the code

`python demo_policy_flip.py` judges the **same** overloaded-forklift clip twice:

1. Under the base policy → the model returns its verdict and cites the clause it used.
2. Add **one clause** (`2.6` — max 2 stacked bins / load must not block the operator's view)
   and re-judge → the **same footage now reads as a violation**, and the model **cites the new
   clause `2.6`**.

No code changed — only the policy text. That is the entire thesis in 20 seconds.

---

## Evaluation (honest numbers)

- **Reasoning:** verdicts cite the controlling clause; flipping a policy line flips the verdict.
- **Detection (YOLO):** **18/21 forklift precision** (the money-shot frames are clean; the 3
  false positives are one camera's press at an angle that scores like a forklift), **3/4 recall**,
  measured **2.1 m** near-miss distance. ~half of person↔forklift pairs read 0.0 m on 2D box
  overlap — reported honestly.
- **Robustness:** in video (temporal) mode, **0 false criticals** across the demo clips; the
  grounding prompt + real distances stop the model inventing hazards it cannot see.
- **Latency:** ~4–5 s per analysis window (window-cadence monitoring, not 30 fps).

See [docs/eval.md](docs/eval.md), [docs/system_design.html](docs/system_design.html), and
[docs/demo_script.md](docs/demo_script.md).

---

## Outputs

- `logs/` — `safety_log.json`, `corrective_actions.json`, `incidents.json`
- `reports/` — `handoff_SHIFT-*.md`, `kpi_rollup.md`, `weekly_plan.md`

## Notes

- `VLM_TEMPERATURE=0` by default so verdicts are reproducible for the demo.
- Graceful degradation: with no GPU/`YOLO_URL` the agent still runs (VLM-only, no boxes); with
  no Slack webhook the notify action is a no-op. The agent never crashes on a missing optional.
- Demo footage is **real factory CCTV** (Mendeley *"Video Dataset for Safe and Unsafe
  Behaviours"*, Eskişehir press shop, CC BY 4.0). See [frames/SOURCES.md](frames/SOURCES.md).
