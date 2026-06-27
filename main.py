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

import config
from vlm_judge import judge_frame
from actions import dispatch
from shift_report import ShiftReport
from perception import load_perception

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
        judgment = judge_frame(str(frame), policy, context, perception=perc)
        judgment.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
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
    from vlm_judge import judge_clip
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
            judgment = judge_clip(frames, policy, context, window_sec=window_sec, label=label)
            judgment.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
            print(f"  👁️  {str(judgment.get('risk_level','?')).upper():8} "
                  f"{judgment.get('hazard_type')} | clause: {str(judgment.get('policy_clause'))[:60]}")

            rep_name = None
            if rep is not None:
                rep_name = f"_live_{name}_{idx:03d}.jpg"   # transient (git-ignored)
                Image.fromarray(np.asarray(rep)).convert("RGB").save(
                    config.ANNOTATED_DIR / rep_name, "JPEG", quality=85)

            actions = dispatch(judgment)
            report.add(judgment, actions)
            if on_update:
                on_update({"index": idx, "total": total, "frame": rep_name or label,
                           "annotated": rep_name, "judgment": judgment,
                           "actions": actions, "report": report})
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
