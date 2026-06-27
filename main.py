"""
main.py — the autonomous loop: WATCH -> DECIDE -> ACT -> REPORT.

    WATCH   read the next camera frame from frames/
    DECIDE  judge_frame()  -> VLM reasons risk level from the policy
    ACT     dispatch()     -> run the actions that level requires
    REPORT  accumulate into a ShiftReport, save a handoff at the end

Headless usage:
    python main.py                 # process frames/ with the default context
    python main.py path/to/frames  # process a different folder
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import os

import config
from vlm_judge import judge_frame, judge_clip
from actions import dispatch
from shift_report import ShiftReport
import perception
from perception import load_perception
import rag

# Main loop is RAG-FREE by default (image + policy + perception). RAG is hazard-driven
# and only enriches the citation once a hazard is already established. Turn on with SC_RAG=1.
RAG_ENABLED = os.getenv("SC_RAG", "0") == "1"
YOLO_URL = os.getenv("YOLO_URL", "")   # if set, run perception on a remote service (B's 4090)


def _detect_remote(frame_path, annotate_to=None):
    """Send a frame to a remote YOLO service (B's 4090) and get perception facts back.
    Mirrors how the VLM runs on a served endpoint. Returns the perception dict."""
    import base64
    import requests
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(YOLO_URL.rstrip("/") + "/detect",
                      json={"image_b64": img_b64}, timeout=20)
    r.raise_for_status()
    data = r.json()
    # tolerate naming variants for the boxed image
    b64 = data.get("annotated_b64") or data.get("annotated") or data.get("image_b64")
    if annotate_to and b64:                                 # save the boxed frame for the dashboard
        with open(annotate_to, "wb") as f:
            f.write(base64.b64decode(b64))
    # tolerate the perception payload being nested, top-level derived, or the dict itself
    if isinstance(data.get("perception"), dict):
        return data["perception"]
    if "derived" in data:
        return data
    return {}


def _rag_enrich(judgment, frame_or_frames, policy, context, perception,
                is_clip=False, window_sec=2.0, label=None):
    """Hazard-driven, gated RAG. The visual pass already decided the hazard + risk tier;
    here we ONLY retrieve the matching regulation and enrich the citation/reasoning.
    The risk tier is never changed by retrieval — visual evidence owns the hazard."""
    if not RAG_ENABLED:
        return judgment
    if not rag.should_use_rag(judgment.get("hazard_type"), judgment.get("risk_level")):
        return judgment
    chunks = rag.retrieve_for_hazard(judgment.get("hazard_type"), context.get("zone", ""))
    if not chunks:
        return judgment
    kn = rag.format_for_prompt(chunks)
    if is_clip:
        j2 = judge_clip(frame_or_frames, policy, context, window_sec=window_sec,
                        label=label, knowledge=kn)
    else:
        j2 = judge_frame(frame_or_frames, policy, context, perception=perception, knowledge=kn)
    judgment["policy_clause"] = j2.get("policy_clause", judgment.get("policy_clause"))
    judgment["reasoning"] = j2.get("reasoning", judgment.get("reasoning"))
    judgment["retrieved"] = [c["source"] for c in chunks]
    judgment["rag_used"] = True
    return judgment

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CONTEXT = {
    "zone": "Production press shop — fixed CCTV over the power-press machine cells, "
            "the green pedestrian walkway, and the forklift aisle",
    "shift": "Day shift (06:00–14:00)",
    "operations": "Metal stamping / pressing; material movement by forklift; "
                  "operators working at the press cells",
}


def list_frames(frames_dir):
    from pathlib import Path
    d = Path(frames_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT)


def run_shift(frames_dir=None, context=None, on_update=None, on_done=None, interval=None):
    """Run one full shift over the frames in frames_dir.

    on_update(payload) is called after each frame (used by the dashboard).
    on_done(report) is called once at the end.
    """
    frames_dir = frames_dir or config.FRAMES_DIR
    context = context or DEFAULT_CONTEXT
    interval = config.FRAME_INTERVAL_SEC if interval is None else interval

    problems = config.check()
    if problems:
        raise RuntimeError("Config problems:\n  - " + "\n  - ".join(problems))

    policy = config.load_policy()
    report = ShiftReport(context=context)
    frames = list_frames(frames_dir)

    print(f"=== SafetyCommander · shift {report.shift_id} · {len(frames)} frames ===")
    print(f"    zone: {context.get('zone')}")

    for i, frame in enumerate(frames, 1):
        print(f"\n[{i}/{len(frames)}] {frame.name}")
        perc = load_perception(frame.name)   # YOLO facts if the perception layer ran; else None
        judgment = judge_frame(str(frame), policy, context, perception=perc)   # visual + policy
        judgment.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        judgment = _rag_enrich(judgment, str(frame), policy, context, perc)    # cite reg if hazard
        print(f"  👁️  {str(judgment.get('risk_level','?')).upper():8} "
              f"{judgment.get('hazard_type')} | "
              f"clause: {str(judgment.get('policy_clause'))[:70]}")
        actions = dispatch(judgment)
        report.add(judgment, actions)

        annotated = None
        if perc and perc.get("annotated_image"):
            cand = config.ANNOTATED_DIR / Path(perc["annotated_image"]).name
            if cand.exists():
                annotated = cand.name

        if on_update:
            on_update({"index": i, "total": len(frames), "frame": frame.name,
                       "annotated": annotated, "judgment": judgment,
                       "actions": actions, "report": report})

        if interval and i < len(frames):
            time.sleep(interval)

    path = report.save()
    print(f"\n=== Shift complete. Handoff report saved to: {path} ===")
    if on_done:
        on_done(report)
    return report


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

VIDEO_CONTEXT = {
    "zone": "Production press shop — fixed CCTV over the forklift aisle and machine cells",
    "shift": "Day shift (06:00–14:00)",
    "operations": "Forklift material moves and operators at the press cells",
}


def sample_windows(video_path, window_sec=1.6, stride_sec=3.0, k=4):
    """Slide over a video, returning [(t_start_sec, [k frames], rep_frame), ...]."""
    import imageio.v2 as imageio
    rdr = imageio.get_reader(str(video_path), "ffmpeg")
    meta = rdr.get_meta_data()
    fps = meta.get("fps", 25) or 25
    nfr = max(1, int(fps * (meta.get("duration", 0) or 0)))
    win = max(1, int(fps * window_sec))
    stride = max(1, int(fps * stride_sec))
    out, start = [], 0
    while True:
        idxs = [min(nfr - 1, start + int(j * win / max(1, k - 1))) for j in range(k)]
        frames = []
        for ix in idxs:
            try:
                frames.append(rdr.get_data(ix))
            except Exception:
                pass
        if frames:
            out.append((start / fps, frames, frames[len(frames) // 2]))
        start += stride
        if start >= nfr:
            break
    rdr.close()
    return out


def _count_windows(video_path, window_sec, stride_sec):
    """Cheap pre-count of sliding windows (reads only metadata), for progress totals."""
    import imageio.v2 as imageio
    rdr = imageio.get_reader(str(video_path), "ffmpeg")
    meta = rdr.get_meta_data()
    fps = meta.get("fps", 25) or 25
    nfr = max(1, int(fps * (meta.get("duration", 0) or 0)))
    rdr.close()
    stride = max(1, int(fps * stride_sec))
    c, start = 0, 0
    while True:
        c += 1
        start += stride
        if start >= nfr:
            break
    return c


def run_videos(video_paths, context=None, on_update=None, on_done=None,
               interval=None, window_sec=1.6, stride_sec=3.0, k=4):
    """Closed loop over one OR MANY video clips, as a single shift (multi-camera).
    Each sliding window is judged TEMPORALLY (motion / behaviour over ~window_sec),
    dispatched, and accumulated into one handoff. The agent actually 'watches' the
    footage instead of isolated stills."""
    from PIL import Image
    import numpy as np

    paths = [Path(p) for p in video_paths]
    context = context or VIDEO_CONTEXT
    interval = config.FRAME_INTERVAL_SEC if interval is None else interval
    problems = config.check()
    if problems:
        raise RuntimeError("Config problems:\n  - " + "\n  - ".join(problems))

    policy = config.load_policy()
    report = ShiftReport(context=context)
    total = sum(_count_windows(p, window_sec, stride_sec) for p in paths)
    print(f"=== SafetyCommander · VIDEO shift · {len(paths)} clip(s) · "
          f"{total} windows · {report.shift_id} ===")

    idx = 0
    for p in paths:
        name = p.stem
        for (t, frames, rep) in sample_windows(p, window_sec, stride_sec, k):
            if not frames:
                continue
            idx += 1
            label = f"{name} @ {int(t)}s"
            print(f"\n[{idx}/{total}] {label}  ({len(frames)} frames)")
            # Representative frame; if the perception layer supports video (B's
            # detect_for_frame), run YOLO on it -> real distance facts for the VLM
            # (no visual over-read) + boxes/distance line drawn for the dashboard.
            rep_name, perc = None, None
            if rep is not None:
                rep_name = f"_live_{name}_{idx:03d}.jpg"   # transient (git-ignored)
                rep_path = config.ANNOTATED_DIR / rep_name
                Image.fromarray(np.asarray(rep)).convert("RGB").save(rep_path, "JPEG", quality=85)
                try:
                    if YOLO_URL:                               # YOLO on B's 4090, results streamed back
                        perc = _detect_remote(str(rep_path), annotate_to=str(rep_path))
                    elif hasattr(perception, "detect_for_frame"):   # YOLO local (same machine)
                        perc = perception.detect_for_frame(str(rep_path), annotate_to=str(rep_path))
                except Exception as e:
                    print(f"  (perception on window failed: {e})")

            judgment = judge_clip(frames, policy, context, perception=perc,
                                  window_sec=window_sec, label=label)
            judgment.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
            judgment = _rag_enrich(judgment, frames, policy, context, perc,
                                   is_clip=True, window_sec=window_sec, label=label)
            print(f"  👁️  {str(judgment.get('risk_level','?')).upper():8} "
                  f"{judgment.get('hazard_type')} | clause: {str(judgment.get('policy_clause'))[:60]}")

            actions = dispatch(judgment)
            report.add(judgment, actions)
            if on_update:
                on_update({"index": idx, "total": total, "frame": rep_name or label,
                           "annotated": rep_name, "clip": p.name,
                           "boxes": (perc or {}).get("detections") or [],
                           "derived": (perc or {}).get("derived") or {},
                           "judgment": judgment, "actions": actions, "report": report})
            if interval and idx < total:
                time.sleep(interval)

    path = report.save()
    print(f"\n=== Video shift complete. Handoff saved to: {path} ===")
    if on_done:
        on_done(report)
    return report


def run_video(video_path, **kwargs):
    """Single-clip convenience wrapper around run_videos()."""
    return run_videos([video_path], **kwargs)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    p = Path(arg) if arg else None
    if p and p.is_dir() and any(q.suffix.lower() in VIDEO_EXT for q in p.iterdir()):
        clips = sorted(str(q) for q in p.iterdir() if q.suffix.lower() in VIDEO_EXT)
        report = run_videos(clips)                 # multi-camera video shift
    elif p and p.suffix.lower() in VIDEO_EXT:
        report = run_video(arg)                     # single clip, temporal
    else:
        report = run_shift(arg)                     # static frames
    print("\n" + "=" * 72)
    print(report.generate_handoff())
