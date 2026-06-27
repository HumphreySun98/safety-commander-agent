"""
actions.py — risk-graded GUARDED ACTIONS.

Every action here is triggered BY the risk level the VLM produced
(judgment['risk_level']). This module does NOT decide risk and contains no
hazard reasoning. dispatch() is pure routing: it maps a model-given label to the
set of actions the policy prescribes for that label.

    VLM reads the policy  ->  decides risk level   (vlm_judge.py)
    dispatch routes that level  ->  executes actions (this file)
"""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import config

_lock = threading.Lock()

SAFETY_LOG    = config.LOGS_DIR / "safety_log.json"
CORRECTIVE    = config.LOGS_DIR / "corrective_actions.json"
INCIDENTS     = config.LOGS_DIR / "incidents.json"
NOTIFICATIONS = config.LOGS_DIR / "notifications.json"


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _append(path: Path, record: dict):
    """Append a record to a JSON-array file (thread-safe)."""
    with _lock:
        data = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = []
        data.append(record)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _base(judgment, kind):
    return {
        "id": _id(kind),
        "ts": _now(),
        "action": kind,
        "frame": judgment.get("frame"),
        "zone": judgment.get("zone"),
        "risk_level": judgment.get("risk_level"),
        "hazard_type": judgment.get("hazard_type"),
        "policy_clause": judgment.get("policy_clause"),
    }


# --- individual guarded actions -------------------------------------------------

def create_safety_log(judgment) -> dict:
    """Write an observation to the persistent safety log (every frame)."""
    rec = _base(judgment, "safety_log")
    rec["observation"] = judgment.get("observation")
    rec["reasoning"] = judgment.get("reasoning")
    _append(SAFETY_LOG, rec)
    print(f"  📝 safety_log     [{rec['risk_level']}] {rec['hazard_type']} @ {rec['frame']}")
    return rec


def notify_supervisor(judgment, channel="dashboard") -> dict:
    """Notify the shift supervisor (prints + persists; optional Discord webhook)."""
    rec = _base(judgment, "notify_supervisor")
    rec["channel"] = channel
    rec["message"] = (f"[{str(judgment.get('risk_level','')).upper()}] "
                      f"{judgment.get('hazard_type')} — {judgment.get('observation')}")
    _append(NOTIFICATIONS, rec)
    print(f"  📣 notify_super   {rec['message']}")
    _maybe_discord(rec["message"])
    return rec


def assign_corrective_action(judgment) -> dict:
    """Open a corrective-action ticket (stays open until closed by a human)."""
    rec = _base(judgment, "corrective_action")
    rec["status"] = "open"
    rec["due"] = "end_of_shift"
    rec["instruction"] = ("; ".join(judgment.get("recommended_actions") or [])
                          or "Address the cited policy violation.")
    _append(CORRECTIVE, rec)
    print(f"  🛠️  corrective     #{rec['id']} (open): {rec['instruction']}")
    return rec


def escalate_incident(judgment, kind="near_miss") -> dict:
    """Escalate to supervisor + EHS. kind = 'near_miss' (high) or 'incident' (critical)."""
    rec = _base(judgment, "escalation")
    rec["incident_type"] = kind
    rec["status"] = "escalated"
    rec["detail"] = judgment.get("reasoning")
    _append(INCIDENTS, rec)
    icon = "🚨" if kind == "incident" else "⚠️"
    print(f"  {icon} escalate       {kind.upper()} -> supervisor + EHS")
    return rec


def flag_area(judgment) -> dict:
    """Mark the camera's zone as unsafe and request immediate clearing (critical)."""
    rec = _base(judgment, "flag_area")
    rec["status"] = "area_flagged"
    rec["zone_flagged"] = judgment.get("zone") or "current camera zone"
    _append(INCIDENTS, rec)
    print(f"  ⛔ flag_area      {rec['zone_flagged']} marked — clear personnel")
    return rec


def _maybe_discord(message):
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        return
    try:
        import requests
        requests.post(url, json={"content": f"**SafetyCommander** {message}"}, timeout=5)
    except Exception as e:
        print(f"  (discord webhook failed: {e})")


# --- the router -----------------------------------------------------------------
#
# IMPORTANT: this is action routing, not risk classification. The VLM already
# classified the risk by reading the policy. Here we only execute the response
# the policy prescribes for that level. There is deliberately no hazard logic.

def dispatch(judgment) -> list[dict]:
    """Execute the actions that correspond to judgment['risk_level']."""
    level = str(judgment.get("risk_level") or "").lower()
    actions: list[dict] = []

    # If the judge itself failed, get a human to look — never fabricate a level.
    if not judgment.get("ok", True):
        actions.append(create_safety_log(judgment))
        actions.append(notify_supervisor(judgment, channel="ops-review"))
        return actions

    if level == "none":
        actions.append(create_safety_log(judgment))

    elif level == "low":
        actions.append(create_safety_log(judgment))

    elif level == "medium":
        actions.append(create_safety_log(judgment))
        actions.append(notify_supervisor(judgment))
        actions.append(assign_corrective_action(judgment))

    elif level == "high":
        actions.append(create_safety_log(judgment))
        actions.append(escalate_incident(judgment, kind="near_miss"))
        actions.append(notify_supervisor(judgment))

    elif level == "critical":
        actions.append(create_safety_log(judgment))
        actions.append(escalate_incident(judgment, kind="incident"))
        actions.append(notify_supervisor(judgment))
        actions.append(flag_area(judgment))

    else:
        # Unrecognized label from the model: log + ask a human to review.
        actions.append(create_safety_log(judgment))
        actions.append(notify_supervisor(judgment, channel="ops-review"))

    return actions
