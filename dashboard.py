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

from flask import Flask, jsonify, send_from_directory, Response, request

import config
import kpi_report
import notify
import main  # module ref needed by /api/detect (main.YOLO_URL, main.perception)
from main import run_shift, run_video, run_videos, DEFAULT_CONTEXT, VIDEO_CONTEXT, VIDEO_EXT

app = Flask(__name__)

TPL_DIR = Path(__file__).resolve().parent / "templates"
CLIPS_DIR = None        # set when running in video mode → served at /clip/<name>


def _tpl(name):
    return Response((TPL_DIR / name).read_text(encoding="utf-8"), mimetype="text/html")

_state_lock = threading.Lock()
_thread = None
STATE = {
    "status": "idle",          # idle | running | done | error
    "index": 0, "total": 0,
    "current_frame": None,
    "current_annotated": None, # annotated frame name (boxes) if perception ran
    "current_clip": None,      # mp4 being analysed now (for live video playback)
    "current_boxes": [],       # YOLO detections (bbox+label) to overlay on the video
    "current_derived": {},     # derived facts (people/forklifts/distance)
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
        STATE["current_clip"] = payload.get("clip") or STATE.get("current_clip")
        STATE["current_boxes"] = payload.get("boxes") or []
        STATE["current_derived"] = payload.get("derived") or {}
        STATE["latest"] = payload["judgment"]
        STATE["latest_actions"] = payload["actions"]
        STATE["events"].append({
            "frame": payload["frame"],
            "annotated": payload.get("annotated"),
            "judgment": payload["judgment"],
            "actions": payload["actions"],
        })
    notify.push_alert(payload["judgment"], payload.get("actions"))  # route to worker inboxes


def _on_done(report):
    with _state_lock:
        STATE["status"] = "done"
        STATE["report_md"] = report.generate_handoff()


def _run():
    try:
        video = os.getenv("SC_VIDEO")          # SC_VIDEO=clip.mp4 (or a folder of clips)
        if video and not Path(video).exists():  # bad path → don't hard-error, fall back
            print(f"  (SC_VIDEO={video!r} not found — falling back to demo_clips/)")
            video = None
        if not video:                          # default to REAL-TIME VIDEO over demo_clips/ if present
            d = Path(__file__).resolve().parent / "demo_clips"
            if d.is_dir() and any(q.suffix.lower() in VIDEO_EXT for q in d.iterdir()):
                video = str(d)
        if video:
            with _state_lock:
                STATE["context"] = VIDEO_CONTEXT
            vp = Path(video)
            global CLIPS_DIR
            CLIPS_DIR = vp if vp.is_dir() else vp.parent
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
                      "current_clip": None, "current_boxes": [], "current_derived": {},
                      "latest": None, "latest_actions": [],
                      "events": [], "report_md": None, "error": None})
    notify.reset()   # clear worker inboxes for the new shift
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


@app.get("/clip/<path:filename>")
def clip(filename):
    if CLIPS_DIR is None:
        return ("no clips", 404)
    return send_from_directory(str(CLIPS_DIR), filename)


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


@app.get("/api/workers")
def api_workers():
    return jsonify(notify.WORKERS)


@app.get("/api/inbox")
def api_inbox():
    return jsonify(notify.inbox(request.args.get("worker", "sam")))


@app.post("/api/inbox/ack")
def api_inbox_ack():
    d = request.get_json(force=True, silent=True) or {}
    ok = notify.set_state(d.get("worker"), d.get("id", ""), d.get("action", "acknowledged"))
    return jsonify({"ok": ok})


@app.get("/api/deliveries")
def api_deliveries():
    return jsonify(notify.deliveries())


@app.post("/api/detect")
def api_detect():
    """On-demand YOLO for the live overlay: the browser sends the frame it is currently
    showing, we run detection (remote on B's 4090 if YOLO_URL, else local) and return the
    boxes — so the overlay tracks the playing video instead of a stale per-window frame."""
    d = request.get_json(force=True, silent=True) or {}
    img_b64 = d.get("image_b64")
    if not img_b64:
        return jsonify({"boxes": [], "derived": {}})
    perc = {}
    try:
        if main.YOLO_URL:                                   # detect on B's 4090
            import requests
            # high-frequency overlay path: we only draw from `detections`, so ask the
            # service to SKIP the ~400KB annotated image (saves ~95% of tunnel bandwidth)
            data = requests.post(main.YOLO_URL.rstrip("/") + "/detect",
                                 json={"image_b64": img_b64, "annotate": False}, timeout=15).json()
            perc = data.get("perception") if isinstance(data.get("perception"), dict) \
                else (data if "derived" in data else {})
        elif hasattr(main.perception, "detect_for_frame"):  # detect locally
            import base64
            tmp = config.ANNOTATED_DIR / "_detect_tmp.jpg"
            tmp.write_bytes(base64.b64decode(img_b64))
            perc = main.perception.detect_for_frame(str(tmp)) or {}
    except Exception as e:
        return jsonify({"boxes": [], "derived": {}, "error": str(e)})
    return jsonify({"boxes": perc.get("detections") or [], "derived": perc.get("derived") or {}})


if __name__ == "__main__":
    start_shift()  # auto-start the shift when the server boots
    # threaded=True so polling + frame serving work while the shift runs.
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)
