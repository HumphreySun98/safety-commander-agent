"""
yolo_server.py — remote perception (YOLO) service on the 4090  (owned by B).

Exposes the perception layer over HTTP so the agent (A's machine) can offload
detection to this GPU: POST a frame, get back the DETECTOR FACTS + an annotated
image (boxes + person<->forklift distance line). It reuses perception.detect_for_frame
unchanged — so it is FACTS ONLY, never a risk level (risk stays with the VLM).

Run:
    python yolo_server.py                 # serves on 0.0.0.0:8077
    cloudflared tunnel --url http://localhost:8077   # -> public https URL for A
    # (or, same LAN: A uses http://<this-4090-ip>:8077)

Endpoints:
    GET  /health   -> {"ok": true, "capabilities": [...], "detectors": N}
    POST /detect   <- {"image_b64": "<base64 jp` or data-URL>"}
                   -> {"perception": <perception.EXAMPLE dict>, "annotated_b64": "<jpg b64>"}

A sets YOLO_URL=<url> and the agent POSTs each frame here.
Weights live on this 4090 and are git-ignored; nothing about risk is decided here.
"""
import base64
import os
import tempfile

from flask import Flask, jsonify, request

import perception

app = Flask(__name__)
PORT = int(os.getenv("YOLO_PORT", "8077"))


def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("empty image_b64")
    if "," in s and s.strip().lower().startswith("data:"):
        s = s.split(",", 1)[1]            # strip data-URL prefix
    return base64.b64decode(s)


@app.get("/health")
def health():
    try:
        _, caps = perception._load_models()
        return jsonify({"ok": True,
                        "capabilities": sorted(caps),
                        "detectors": len(perception._MODELS or [])})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/detect")
def detect():
    data = request.get_json(force=True, silent=True) or {}
    try:
        raw = _decode_b64(data.get("image_b64", ""))
    except Exception as e:
        return jsonify({"error": f"bad image_b64: {e}"}), 400

    # unique temp files so concurrent (threaded) requests never collide
    fd_in, in_path = tempfile.mkstemp(suffix=".jpg", prefix="yolo_in_")
    fd_out, out_path = tempfile.mkstemp(suffix=".jpg", prefix="yolo_out_")
    os.close(fd_in)
    os.close(fd_out)
    try:
        with open(in_path, "wb") as f:
            f.write(raw)
        # facts only — same function the local video loop uses
        perc = perception.detect_for_frame(in_path, annotate_to=out_path)
        annotated_b64 = ""
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                annotated_b64 = base64.b64encode(f.read()).decode("utf-8")
        return jsonify({"perception": perc, "annotated_b64": annotated_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    # warm up the detectors so the first request isn't slow
    try:
        _, caps = perception._load_models()
        print(f"yolo_server: models loaded, capabilities = {sorted(caps)}")
    except Exception as e:
        print(f"yolo_server: WARNING could not preload models: {e}")
    print(f"yolo_server: listening on 0.0.0.0:{PORT}  (POST /detect, GET /health)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
