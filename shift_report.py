"""
shift_report.py — accumulate a shift's events and produce a handoff report.

The report is what a safety officer hands to the next shift: how many violations
and near-misses were caught, broken down by type, which corrective actions are
still open, and what was escalated.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path

import config


class ShiftReport:
    def __init__(self, shift_id=None, context=None):
        self.started_at = datetime.now()
        self.shift_id = shift_id or self.started_at.strftime("SHIFT-%Y%m%d-%H%M")
        self.context = context or {}
        self.events = []  # list of {frame, ts, judgment, actions}

    def add(self, judgment, actions):
        self.events.append({
            "frame": judgment.get("frame"),
            "ts": judgment.get("timestamp"),
            "judgment": judgment,
            "actions": actions or [],
        })

    # --- stats helpers ---------------------------------------------------------
    def _by_level(self):
        return Counter(e["judgment"].get("risk_level", "?") for e in self.events)

    def _by_hazard(self):
        c = Counter()
        for e in self.events:
            j = e["judgment"]
            if j.get("risk_level") in ("medium", "high", "critical"):
                c[j.get("hazard_type", "unknown")] += 1
        return c

    def _open_correctives(self):
        return [a for e in self.events for a in e["actions"]
                if a.get("action") == "corrective_action" and a.get("status") == "open"]

    def _escalations(self):
        return [a for e in self.events for a in e["actions"]
                if a.get("action") in ("escalation", "flag_area")]

    def _count_incident(self, kind):
        return sum(1 for e in self.events for a in e["actions"]
                   if a.get("incident_type") == kind)

    def stats(self):
        by_level = self._by_level()
        return {
            "frames": len(self.events),
            "violations": sum(by_level.get(l, 0) for l in ("medium", "high", "critical")),
            "near_misses": self._count_incident("near_miss"),
            "incidents": self._count_incident("incident"),
            "open_correctives": len(self._open_correctives()),
            "by_level": dict(by_level),
            "by_hazard": dict(self._by_hazard()),
        }

    # --- markdown handoff ------------------------------------------------------
    def generate_handoff(self) -> str:
        ended = datetime.now()
        s = self.stats()
        by_level = self._by_level()
        opens = self._open_correctives()
        esc = self._escalations()

        L = []
        L.append(f"# Shift Safety Handoff — {self.shift_id}")
        L.append("")
        L.append(f"- **Generated:** {ended:%Y-%m-%d %H:%M}")
        L.append(f"- **Shift start:** {self.started_at:%Y-%m-%d %H:%M}")
        for k, v in self.context.items():
            L.append(f"- **{str(k).capitalize()}:** {v}")
        L.append(f"- **Frames analyzed:** {s['frames']}")
        L.append("")

        L.append("## Summary")
        L.append(f"- **Violations (medium+):** {s['violations']}")
        L.append(f"- **Near-misses:** {s['near_misses']}")
        L.append(f"- **Critical incidents:** {s['incidents']}")
        L.append(f"- **Open corrective actions:** {s['open_correctives']}")
        L.append("")

        L.append("### By risk level")
        for lvl in ("critical", "high", "medium", "low", "none"):
            if by_level.get(lvl):
                L.append(f"- {lvl}: {by_level[lvl]}")
        L.append("")

        if s["by_hazard"]:
            L.append("### Violations by hazard type")
            for h, c in self._by_hazard().most_common():
                L.append(f"- {h}: {c}")
            L.append("")

        if esc:
            L.append("## Escalations & near-misses")
            for a in esc:
                kind = a.get("incident_type", a.get("action"))
                L.append(f"- `{a.get('ts')}` **{kind}** — {a.get('hazard_type')} "
                         f"@ {a.get('frame')} — _{a.get('policy_clause')}_")
            L.append("")

        L.append("## Open corrective actions (carry to next shift)")
        if opens:
            for a in opens:
                L.append(f"- [ ] **{a.get('hazard_type')}** @ {a.get('frame')} — "
                         f"{a.get('instruction')}  \n  _(clause: {a.get('policy_clause')})_")
        else:
            L.append("- None open. ✅")
        L.append("")

        L.append("## Full event log")
        for e in self.events:
            j = e["judgment"]
            lvl = str(j.get("risk_level", "?")).upper()
            L.append(f"- `{j.get('timestamp')}` **[{lvl}]** {j.get('hazard_type')} "
                     f"@ {j.get('frame')}")
            L.append(f"    - Observation: {j.get('observation')}")
            L.append(f"    - Policy: {j.get('policy_clause')}")
            L.append(f"    - Reasoning: {j.get('reasoning')}")
            acts = ", ".join(a.get("action") for a in e["actions"]) or "none"
            L.append(f"    - Actions taken: {acts}")
        L.append("")

        L.append("---")
        L.append("*Generated autonomously by SafetyCommander. Risk levels were "
                 "reasoned by the VLM from the site safety policy; actions were "
                 "routed by risk level. No hardcoded hazard rules.*")
        return "\n".join(L)

    def save(self, path=None) -> Path:
        path = Path(path) if path else (config.REPORTS_DIR / f"handoff_{self.shift_id}.md")
        path.write_text(self.generate_handoff(), encoding="utf-8")
        return path
