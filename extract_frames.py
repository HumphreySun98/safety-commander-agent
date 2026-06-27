"""
extract_frames.py — sample frames from a demo video into frames/.

For the demo we feed static frames (not a live camera). Use this to turn one of
the hackathon CCTV / factory video clips into a frames/ folder.

    python extract_frames.py path/to/video.mp4
    python extract_frames.py path/to/video.mp4 --every 1.5 --out frames --max 20

Tries OpenCV first; falls back to ffmpeg if it is on PATH.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def with_opencv(video, out, every, mx):
    try:
        import cv2
    except ImportError:
        return False
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"OpenCV could not open {video}")
        return True  # handled (but failed); don't try ffmpeg on a bad file
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * every))
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            p = out / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(p), frame)
            saved += 1
            print(f"  saved {p.name}")
            if mx and saved >= mx:
                break
        i += 1
    cap.release()
    print(f"Done: {saved} frames -> {out}")
    return True


def with_ffmpeg(video, out, every, mx):
    if not shutil.which("ffmpeg"):
        return False
    fps = 1.0 / every
    pattern = str(out / "frame_%04d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}"]
    if mx:
        cmd += ["-frames:v", str(mx)]
    cmd += [pattern]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done -> {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--max", type=int, default=0, help="max frames (0 = all)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if with_opencv(a.video, out, a.every, a.max):
        return
    if with_ffmpeg(a.video, out, a.every, a.max):
        return
    print("Neither OpenCV nor ffmpeg is available.\n"
          "  pip install opencv-python-headless   (or install ffmpeg)\n"
          "Or just drop .jpg/.png images straight into the frames/ folder.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
