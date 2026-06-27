"""
planner.py — AI weekly preventive plan.

The agent reads THIS period's KPIs (top hazards, hot zones, open corrective actions)
and reasons out NEXT week's preventive actions — scheduled inspections, toolbox talks,
and targeted training — each with WHEN / WHO / WHAT and the hazard + policy/OSHA clause
it addresses. The MODEL decides the plan from the data; nothing here is a hardcoded
schedule. This covers the Role-01 WEEK tasks (scheduled inspections + training).

    python planner.py
"""
import config
import kpi_report
import rag
from vlm_judge import _get_client, _extract_json

SYSTEM = (
    "You are an experienced EHS / safety officer planning next week's PREVENTIVE actions "
    "for a factory floor. You schedule inspections, toolbox talks, and targeted training "
    "based on what actually happened this period — most frequent and most severe hazards "
    "first. Be specific, realistic, and tie each item to the hazard it addresses and the "
    "relevant policy clause or OSHA standard."
)


def generate_week_plan():
    s = kpi_report.summarize()
    hazards = ", ".join(f"{h} ({n})" for h, n in s["by_hazard"]) or "none recorded"
    zones = "; ".join(f"{z} ({n})" for z, n in s["by_zone"]) or "none"
    ca = s["open_correctives"]
    ca_txt = "\n".join(f"- {c['hazard']} @ {c['frame']}: {c['instruction']}"
                       for c in ca[:12]) or "none"
    policy = config.load_policy()

    # domain-targeted RAG: retrieve real industrial EHS cadences/KPIs to ground the plan
    hz_words = " ".join(h for h, _ in s["by_hazard"][:4])
    prac = rag.retrieve(
        "safety inspection frequency schedule toolbox talk training cadence corrective "
        f"action closure KPI leading lagging indicator {hz_words}", k=3)
    practice = "\n".join(f"- [{c['source']}] {c['text'][:320].strip()}" for c in prac) or "none"

    user = (
        "THIS PERIOD'S SAFETY DATA (from the autonomous shift monitoring):\n"
        f"- observations: {s['observations']}, violations: {s['violations']}, "
        f"near-misses: {s['near_miss']}, critical: {s['critical']}\n"
        f"- top hazards (count): {hazards}\n"
        f"- hot zones (violations): {zones}\n"
        f"- open corrective actions:\n{ca_txt}\n\n"
        f"SITE POLICY (for clause references):\n{policy}\n\n"
        f"INDUSTRY EHS PRACTICE & STANDARDS (align cadence/owners/KPIs to these):\n{practice}\n\n"
        "Plan NEXT WEEK's preventive actions that DIRECTLY target the data above, "
        "using the real cadences above (e.g. daily forklift pre-shift checks, weekly "
        "toolbox talks, operator re-eval, ~85% corrective-action closure target). "
        "Prioritise by frequency and severity; cover the open corrective actions. "
        "Output ONLY JSON, no prose:\n"
        '{ "week_of": "<e.g. Mon Jun 30>", "focus": "<one-line theme>", "items": [ '
        '{ "type": "inspection|toolbox_talk|training", "title": "...", '
        '"day": "Mon..Fri", "time": "HH:MM", "owner": "responsible role", '
        '"audience": "who attends", "content": "what to cover, concretely", '
        '"addresses": "which hazard/data it targets", '
        '"clause": "policy clause / OSHA standard" } ] }'
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    resp = _get_client().chat.completions.create(
        model=config.VLLM_MODEL, messages=messages, temperature=0.2, max_tokens=1300)
    plan = _extract_json(resp.choices[0].message.content or "") or {}
    return plan, s


def render(plan, s) -> str:
    L = [f"# Weekly Preventive Plan — week of {plan.get('week_of', 'next week')}"]
    L.append(f"\n*AI-planned by SafetyCommander from this period: {s['violations']} violations, "
             f"{s['near_miss']} near-misses, {len(s['open_correctives'])} open corrective actions.*")
    if plan.get("focus"):
        L.append(f"\n**Focus this week:** {plan['focus']}")
    L.append("")
    for it in plan.get("items", []):
        L.append(f"## {str(it.get('type','')).replace('_',' ').title()} — {it.get('title','')}")
        L.append(f"- **When:** {it.get('day','')} {it.get('time','')}")
        L.append(f"- **Owner:** {it.get('owner','')}  ·  **Audience:** {it.get('audience','')}")
        L.append(f"- **Content:** {it.get('content','')}")
        L.append(f"- **Targets:** {it.get('addresses','')}  ·  _{it.get('clause','')}_")
        L.append("")
    L.append("---\n*The model reasoned this plan from the observed data + policy — "
             "not a fixed schedule. Owners/times are proposals for the safety committee to confirm.*")
    return "\n".join(L)


if __name__ == "__main__":
    plan, s = generate_week_plan()
    md = render(plan, s)
    out = config.REPORTS_DIR / "weekly_plan.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[saved {out}]  ({len(plan.get('items', []))} planned items)")
