"""
train_forklift.py — fine-tune a forklift+person YOLO on the 4090 (perception layer).

WHY: off-the-shelf forklift weights (keremberke) false-positive on press machines
in our CCTV. Fine-tuning on a labelled forklift+person set fixes that. This trains
a small yolov8s and exports weights to weights/forklift.pt, where perception.py
picks it up automatically (DETECTOR_SPECS).

This is a TRAINING script only. It produces detection weights — it has nothing to
do with risk classification (that stays in the VLM / safety_policy.txt).

DATA — two ways to provide it:
  A) Roboflow (recommended). Free API key from roboflow.com -> Settings -> API.
       export ROBOFLOW_API_KEY=...
       python train_forklift.py                 # pulls project-forklift01 (YOLOv8)
  B) Local dataset dir already in YOLOv8 format (has data.yaml):
       python train_forklift.py /path/to/dataset

Roboflow source (forklift, person): https://universe.roboflow.com/test-gun7j/project-forklift01
Weights/datasets are git-ignored; only the resulting perception JSON + annotated
frames get committed.
"""
import os
import shutil
import sys
from pathlib import Path

import config

WEIGHTS_OUT = config.BASE_DIR / "weights" / "forklift.pt"
DATASETS_DIR = config.BASE_DIR / "datasets"   # git-ignored
EPOCHS = int(os.getenv("FORKLIFT_EPOCHS", "60"))
IMGSZ = int(os.getenv("FORKLIFT_IMGSZ", "832"))   # CCTV is wide; keep detail
BASE_MODEL = os.getenv("FORKLIFT_BASE", "yolov8s.pt")

# Roboflow project to pull when no local dataset is given.
RF_WORKSPACE = os.getenv("RF_WORKSPACE", "test-gun7j")
RF_PROJECT = os.getenv("RF_PROJECT", "project-forklift01")
RF_VERSION = int(os.getenv("RF_VERSION", "1"))


def fetch_roboflow() -> Path:
    """Download the Roboflow dataset in YOLOv8 format; return its data.yaml dir."""
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit(
            "No dataset dir given and ROBOFLOW_API_KEY is not set.\n"
            "  -> export ROBOFLOW_API_KEY=... (free at roboflow.com), or\n"
            "  -> pass a local YOLOv8 dataset dir: python train_forklift.py <dir>")
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow first.")
    DATASETS_DIR.mkdir(exist_ok=True)
    rf = Roboflow(api_key=key)
    proj = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    ds = proj.version(RF_VERSION).download("yolov8", location=str(DATASETS_DIR / RF_PROJECT))
    return Path(ds.location)


def find_data_yaml(d: Path) -> Path:
    if (d / "data.yaml").exists():
        return d / "data.yaml"
    hits = list(d.rglob("data.yaml"))
    if not hits:
        raise SystemExit(f"No data.yaml under {d} (need YOLOv8 format).")
    return hits[0]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv and Path(argv[0]).exists():
        data_yaml = find_data_yaml(Path(argv[0]))
    else:
        data_yaml = find_data_yaml(fetch_roboflow())
    print(f"training on: {data_yaml}")

    from ultralytics import YOLO
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ,
        device=0, patience=15, project=str(config.BASE_DIR / "runs"),
        name="forklift", exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    WEIGHTS_OUT.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"\n✅ best weights -> {WEIGHTS_OUT}")
    print("Now run: python perception.py   (forklift facts now active)")


if __name__ == "__main__":
    main()
