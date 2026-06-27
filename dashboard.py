"""
dashboard.py — live web dashboard for SafetyCommander (Flask).

Run:
    python dashboard.py            # open http://localhost:8000

It starts the shift loop in a background thread and the single-page UI polls
/api/state to show, in real time: the current frame, the VLM's verdict, the cited
policy clause, the risk level, the actions triggered, and the running handoff report.
"""
import os
import threading
from copy import deepcopy
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, Response

import config
import kpi_report
from main import run_shift, run_video, run_videos, DEFAULT_CONTEXT, VIDEO_CONTEXT, VIDEO_EXT

app = Flask(__name__)

TPL_DIR = Path(__file__).resolve().parent / "templates"


def _tpl(name):
    return Response((TPL_DIR / name).read_text(encoding="utf-8"), mimetype="text/html")

_state_lock = threading.Lock()
_thread = None
STATE = {
    "status": "idle",          # idle | running | done | error
    "index": 0, "total": 0,
    "current_frame": None,
    "current_annotated": None, # annotated frame name (boxes) if perception ran
    "latest": None,            # latest judgment dict
    "latest_actions": [],
    "events": [],              # [{frame, judgment, actions}, ...]
    "context": DEFAULT_CONTEXT,
    "report_md": None,
    "error": None,
}


def _on_update(payload):
    with _state_lock:
        STATE["status"] = "running"
        STATE["index"] = payload["index"]
        STATE["total"] = payload["total"]
        STATE["current_frame"] = payload["frame"]
        STATE["current_annotated"] = payload.get("annotated")
        STATE["latest"] = payload["judgment"]
        STATE["latest_actions"] = payload["actions"]
        STATE["events"].append({
            "frame": payload["frame"],
            "annotated": payload.get("annotated"),
            "judgment": payload["judgment"],
            "actions": payload["actions"],
        })


def _on_done(report):
    with _state_lock:
        STATE["status"] = "done"
        STATE["report_md"] = report.generate_handoff()


def _run():
    try:
        video = os.getenv("SC_VIDEO")          # SC_VIDEO=clip.mp4 (or a folder of clips)
        if not video:                          # default to REAL-TIME VIDEO over demo_clips/ if present
            d = Path(__file__).resolve().parent / "demo_clips"
            if d.is_dir() and any(q.suffix.lower() in VIDEO_EXT for q in d.iterdir()):
                video = str(d)
        if video:
            with _state_lock:
                STATE["context"] = VIDEO_CONTEXT
            vp = Path(video)
            if vp.is_dir():
                clips = sorted(str(q) for q in vp.iterdir() if q.suffix.lower() in VIDEO_EXT)
                run_videos(clips, context=VIDEO_CONTEXT, on_update=_on_update, on_done=_on_done)
            else:
                run_video(video, context=VIDEO_CONTEXT, on_update=_on_update, on_done=_on_done)
        else:
            run_shift(context=STATE["context"], on_update=_on_update, on_done=_on_done)
    except Exception as e:
        with _state_lock:
            STATE["status"] = "error"
            STATE["error"] = str(e)


def start_shift():
    global _thread
    with _state_lock:
        if STATE["status"] == "running":
            return False
        STATE.update({"status": "running", "index": 0, "total": 0,
                      "current_frame": None, "current_annotated": None,
                      "latest": None, "latest_actions": [],
                      "events": [], "report_md": None, "error": None})
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    return True


@app.get("/")
def home():
    return _tpl("chooser.html")        # role selector: 员工端 / 管理端


@app.get("/worker")
def worker():
    return _tpl("worker.html")         # frontline view


@app.get("/manager")
def manager():
    return _tpl("manager.html")        # operations / safety-manager console


@app.get("/monitor")
def monitor():
    return _tpl("index.html")          # the live real-time monitor (unchanged)


@app.get("/frames/<path:filename>")
def frames(filename):
    return send_from_directory(str(config.FRAMES_DIR), filename)


@app.get("/annotated/<path:filename>")
def annotated(filename):
    return send_from_directory(str(config.ANNOTATED_DIR), filename)


@app.get("/api/state")
def api_state():
    with _state_lock:
        return jsonify(deepcopy(STATE))


@app.post("/api/restart")
def api_restart():
    return jsonify({"restarted": start_shift()})


@app.get("/api/report")
def api_report():
    with _state_lock:
        return Response(STATE.get("report_md") or "Report not ready yet.",
                        mimetype="text/plain")


@app.get("/api/kpi")
def api_kpi():
    return jsonify(kpi_report.summarize())          # MONTH roll-up stats


@app.get("/api/correctives")
def api_correctives():
    return jsonify(kpi_report.summarize().get("open_correctives", []))


@app.get("/api/plan")
def api_plan():
    p = config.REPORTS_DIR / "weekly_plan.md"
    txt = p.read_text(encoding="utf-8") if p.exists() else \
        "No weekly plan yet. Generate it with:  python planner.py"
    return Response(txt, mimetype="text/plain")


if __name__ == "__main__":
    start_shift()  # auto-start the shift when the server boots
    # threaded=True so polling + frame serving work while the shift runs.
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)
