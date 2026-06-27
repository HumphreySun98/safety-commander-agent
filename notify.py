"""
notify.py — worker-targeted notification inbox (Slack / DingTalk-style workflow).

Routes safety ALERTS and scheduled PLAN TASKS to specific workers by role/zone,
tracks acknowledge / resolve / done state, and exposes the manager-side delivery
status (the closed loop: alert → delivered to the right worker → acknowledged).

This module makes NO risk decisions. Alerts come from the agent's judgment; this
only ROUTES them to people and tracks who acted. Risk is still reasoned by the VLM.
"""
import re
import threading
from datetime import datetime

import config

# --- demo worker roster (no real identities; routing is by ROLE/ZONE) ---
WORKERS = [
    {"id": "maria", "name": "Maria Lopez", "role": "Press Operator",       "icon": "👩‍🏭"},
    {"id": "chen",  "name": "Chen Wei",    "role": "Forklift Driver",      "icon": "🚜"},
    {"id": "ana",   "name": "Ana Ruiz",    "role": "Walkway / Logistics",  "icon": "🦺"},
    {"id": "sam",   "name": "Sam Okoro",   "role": "Shift Supervisor",     "icon": "🧑‍💼"},
]
_BY_ID = {w["id"]: w for w in WORKERS}

VIOLATION = {"medium", "high", "critical"}

_lock = threading.Lock()
_alerts = []        # [{id,type,ts,to[],level,hazard,observation,clause,frame,state{worker:state}}]
_task_state = {}    # (worker, task_id) -> "done"
_seq = [0]


def route(hazard_type, observation=""):
    """Which workers should receive this hazard. Supervisor always included."""
    h = (str(hazard_type) + " " + str(observation)).lower()
    ids = set()
    if any(k in h for k in ("forklift", "load", "truck")):            ids.add("chen")
    if any(k in h for k in ("hardhat", "ppe", "helmet", "vest")):     ids.add("maria")
    if any(k in h for k in ("pedestrian", "walkway", "crossing")):    ids.add("ana")
    if not ids:                                                       ids.add("maria")
    ids.add("sam")                                                    # supervisor sees all
    return ids


def route_audience(audience, title=""):
    """Which workers a weekly-plan task is addressed to (by its Audience text)."""
    a = (str(audience) + " " + str(title)).lower()
    ids = set()
    if any(k in a for k in ("forklift", "operator", "material handler")): ids.add("chen")
    if "supervisor" in a:                                                 ids.add("sam")
    if any(k in a for k in ("ppe", "press")):                            ids.add("maria")
    if any(k in a for k in ("all", "floor", "personnel", "production", "warehouse")):
        ids.update(["maria", "chen", "ana"])
    if "ehs" in a:                                                       ids.add("sam")
    if not ids:                                                          ids.update(_BY_ID)
    return ids


def push_alert(judgment, actions=None):
    """Create a targeted alert from a VIOLATION judgment (medium+). Returns it, or None."""
    lvl = str(judgment.get("risk_level", "none")).lower()
    if lvl not in VIOLATION:
        return None
    to = sorted(route(judgment.get("hazard_type"), judgment.get("observation", "")))
    _seq[0] += 1
    n = {
        "id": f"a{_seq[0]}", "type": "alert",
        "ts": judgment.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
        "to": to, "level": lvl,
        "hazard": judgment.get("hazard_type"),
        "observation": judgment.get("observation"),
        "clause": judgment.get("policy_clause"),
        "frame": judgment.get("label") or "",
        "state": {w: "unread" for w in to},
    }
    with _lock:
        _alerts.append(n)
        if len(_alerts) > 60:
            del _alerts[:-60]
    return n


def _plan_tasks():
    """Parse reports/weekly_plan.md into task cards routed by audience."""
    p = config.REPORTS_DIR / "weekly_plan.md"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    out = []
    for i, b in enumerate(txt.split("\n## ")[1:], 1):
        title = b.splitlines()[0].strip()
        grab = lambda key: (re.search(r"\*\*" + key + r":\*\*\s*(.+)", b) or [None, ""])[1].strip()
        out.append({
            "id": f"t{i}", "type": "task", "title": title,
            "when": grab("When"), "audience": grab("Audience"), "content": grab("Content"),
            "to": sorted(route_audience(grab("Audience"), title)),
        })
    return out


def inbox(worker):
    """This worker's messages (alerts + tasks), newest first, with unread count."""
    with _lock:
        msgs = []
        for n in _alerts:
            if worker in n["to"]:
                m = {k: v for k, v in n.items() if k != "state"}
                m["mine_state"] = n["state"].get(worker, "unread")
                msgs.append(m)
    msgs.sort(key=lambda m: m["ts"], reverse=True)
    tasks = [dict(t, mine_state=_task_state.get((worker, t["id"]), "open"))
             for t in _plan_tasks() if worker in t["to"]]
    unread = sum(1 for m in msgs if m["mine_state"] == "unread")
    return {"alerts": msgs, "tasks": tasks, "unread": unread}


def set_state(worker, msg_id, action):
    """action: acknowledged | resolved | escalated | done."""
    with _lock:
        if msg_id.startswith("t"):
            _task_state[(worker, msg_id)] = "done"
            return True
        for n in _alerts:
            if n["id"] == msg_id and worker in n["state"]:
                n["state"][worker] = action
                return True
    return False


def deliveries():
    """Manager view: recent alerts + who got them + their ack state (closed loop)."""
    with _lock:
        out = []
        for n in reversed(_alerts[-20:]):
            out.append({
                "id": n["id"], "ts": n["ts"], "level": n["level"], "hazard": n["hazard"],
                "to": [{"worker": _BY_ID.get(w, {}).get("name", w), "state": st}
                       for w, st in n["state"].items()],
            })
    return out


def reset():
    with _lock:
        _alerts.clear()
        _task_state.clear()
        _seq[0] = 0
