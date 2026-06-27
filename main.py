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

import config
from vlm_judge import judge_frame
from actions import dispatch
from shift_report import ShiftReport

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CONTEXT = {
    "zone": "Warehouse & dock — active forklift area",
    "shift": "Day shift (06:00–14:00)",
    "operations": "Inbound receiving, put-away, and order picking",
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
        judgment = judge_frame(str(frame), policy, context)
        judgment.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        print(f"  👁️  {str(judgment.get('risk_level','?')).upper():8} "
              f"{judgment.get('hazard_type')} | "
              f"clause: {str(judgment.get('policy_clause'))[:70]}")
        actions = dispatch(judgment)
        report.add(judgment, actions)

        if on_update:
            on_update({"index": i, "total": len(frames), "frame": frame.name,
                       "judgment": judgment, "actions": actions, "report": report})

        if interval and i < len(frames):
            time.sleep(interval)

    path = report.save()
    print(f"\n=== Shift complete. Handoff report saved to: {path} ===")
    if on_done:
        on_done(report)
    return report


if __name__ == "__main__":
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else None
    report = run_shift(frames_dir)
    print("\n" + "=" * 72)
    print(report.generate_handoff())
