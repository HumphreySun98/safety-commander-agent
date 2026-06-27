"""
kpi_report.py — weekly / monthly safety KPI roll-up.

shift_report.py covers ONE shift (the DAY task). This aggregates the persisted logs
across MANY shifts into the WEEK / MONTH view a safety officer owns: violation rate,
near-miss trend, top hazards & zones, and the open corrective-action backlog.

Reuses what the autonomous loop already wrote to logs/ — no new data collection.

    python kpi_report.py              # roll up everything in logs/
    python kpi_report.py --period "June 2026"
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import config

VIOLATION = {"medium", "high", "critical"}


def _load(name):
    p = config.LOGS_DIR / name
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def summarize() -> dict:
    """Aggregated stats dict (used by planner.py to reason a weekly plan)."""
    log = _load("safety_log.json")
    correctives = _load("corrective_actions.json")
    incidents = _load("incidents.json")
    violations = [e for e in log if str(e.get("risk_level")) in VIOLATION]
    return {
        "observations": len(log),
        "violations": len(violations),
        "by_hazard": Counter(e.get("hazard_type", "unknown") for e in violations).most_common(8),
        "by_zone": Counter((e.get("zone") or "unspecified") for e in violations).most_common(6),
        "near_miss": sum(1 for i in incidents if i.get("incident_type") == "near_miss"),
        "critical": sum(1 for i in incidents if i.get("incident_type") == "incident"),
        "open_correctives": [
            {"hazard": c.get("hazard_type"), "frame": c.get("frame"),
             "clause": c.get("policy_clause"), "instruction": c.get("instruction")}
            for c in correctives if c.get("status") == "open"
        ],
    }


def generate_rollup(period="all logged shifts") -> str:
    log = _load("safety_log.json")            # one record per observed frame
    correctives = _load("corrective_actions.json")
    incidents = _load("incidents.json")

    by_level = Counter(str(e.get("risk_level", "?")) for e in log)
    violations = [e for e in log if str(e.get("risk_level")) in VIOLATION]
    by_hazard = Counter(e.get("hazard_type", "unknown") for e in violations)
    by_zone = Counter((e.get("zone") or "unspecified") for e in violations)
    near_miss = sum(1 for i in incidents if i.get("incident_type") == "near_miss")
    crit = sum(1 for i in incidents if i.get("incident_type") == "incident")
    open_ca = [c for c in correctives if c.get("status") == "open"]
    by_day = Counter((e.get("ts") or e.get("timestamp") or "")[:10] for e in log)
    by_day.pop("", None)

    obs = len(log)
    rate = (len(violations) / obs * 100) if obs else 0

    L = []
    L.append(f"# Safety KPI Roll-up — {period}")
    L.append(f"\n- **Generated:** {datetime.now():%Y-%m-%d %H:%M}")
    L.append(f"- **Frames observed:** {obs}")
    L.append(f"- **Shifts/days covered:** {len(by_day)}")
    L.append("\n## Headline KPIs")
    L.append(f"- **Violations (medium+):** {len(violations)}  ({rate:.0f}% of observations)")
    L.append(f"- **Near-misses:** {near_miss}")
    L.append(f"- **Critical incidents:** {crit}")
    L.append(f"- **Open corrective actions (backlog):** {len(open_ca)}")

    L.append("\n## Indicators (industry framing)")
    L.append(f"- **Leading:** near-misses reported = {near_miss} (rising = stronger reporting "
             f"culture → fewer recordables); violation rate {rate:.0f}%; corrective-action "
             f"backlog {len(open_ca)} (industry target: ≥85% closed on time, ~30-day resolution).")
    L.append(f"- **Lagging:** critical incidents = {crit}. (TRIR/DART need hours-worked + "
             f"recordable counts — outside this demo dataset; the framework is ready to compute them.)")

    L.append("\n## By risk level")
    for lvl in ("critical", "high", "medium", "low", "none"):
        if by_level.get(lvl):
            L.append(f"- {lvl}: {by_level[lvl]}")

    L.append("\n## Top hazard types (violations)")
    for h, c in by_hazard.most_common(8):
        L.append(f"- {h}: {c}")

    L.append("\n## By zone")
    for z, c in by_zone.most_common(6):
        L.append(f"- {c:>3}  {z}")

    if len(by_day) > 1:
        L.append("\n## Trend (violations by day)")
        viol_day = Counter((e.get("ts") or e.get("timestamp") or "")[:10] for e in violations)
        for d in sorted(by_day):
            L.append(f"- {d}: {viol_day.get(d, 0)} violations / {by_day[d]} observed")

    L.append("\n## Open corrective-action backlog")
    if open_ca:
        for c in open_ca[:15]:
            L.append(f"- [ ] **{c.get('hazard_type')}** @ {c.get('frame')} "
                     f"({c.get('ts','')[:10]}) — {c.get('instruction', '')}")
        if len(open_ca) > 15:
            L.append(f"- … and {len(open_ca) - 15} more")
    else:
        L.append("- None open ✅")

    L.append("\n---\n*Auto-generated by SafetyCommander from the autonomous shift logs. "
             "Same engine as the daily handoff, rolled up to the week/month KPI view.*")
    return "\n".join(L)


if __name__ == "__main__":
    period = "all logged shifts"
    if "--period" in sys.argv:
        period = sys.argv[sys.argv.index("--period") + 1]
    md = generate_rollup(period)
    out = config.REPORTS_DIR / "kpi_rollup.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[saved {out}]")
