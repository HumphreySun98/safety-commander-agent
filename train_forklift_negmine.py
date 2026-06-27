"""
train_forklift_negmine.py — negative-mining fine-tune for the forklift detector.

PROBLEM: the off-the-shelf keremberke forklift model has good recall on OUR
forklifts (money-shot frame: 0.87) but false-fires on the press machines in our
press-shop CCTV. A plain fine-tune on project-forklift01 fixed the false-positives
but went blind to our forklifts (domain overfit).

FIX (negative mining): start FROM keremberke (keep its recall), FREEZE the backbone
(keep its learned forklift features), and train the head on:
  - project-forklift01 positives  (keep detecting forklifts/people)
  - OUR confirmed forklift-FREE press frames as background negatives, oversampled
    (teach: these press machines are NOT forklifts)

Output -> weights/forklift.pt (auto-loaded by perception.py). Training only; no risk
logic. Datasets/weights are git-ignored.
"""
import os
import shutil
from pathlib import Path

import config

SRC = config.BASE_DIR / "datasets" / "project-forklift01"
DST = config.BASE_DIR / "datasets" / "negmine"
BASE_WEIGHTS = config.BASE_DIR / "weights" / "forklift_keremberke.pt"
OUT_WEIGHTS = config.BASE_DIR / "weights" / "forklift.pt"

# Frames VISUALLY confirmed to contain NO forklift (press shop). Used as the
# hard negatives that teach the model to stop firing on press machines.
NEG_FRAMES = [
    "cam2_t1", "cam2_t2", "cam2_t3", "cam3_t1", "cam3_t2", "cam3_t3",
    "cam4_t1", "cam4_t2", "cam4_t3", "cam5_t1", "cam5_t2", "cam5_t3",
    "cam6_t1", "cam6_t2", "cam6_t3", "cam7_t3",
]
OVERSAMPLE = int(os.getenv("NEG_OVERSAMPLE", "25"))
EPOCHS = int(os.getenv("NEG_EPOCHS", "30"))
IMGSZ = int(os.getenv("NEG_IMGSZ", "832"))


def _copy_split(split_src, split_dst):
    (split_dst / "images").mkdir(parents=True, exist_ok=True)
    (split_dst / "labels").mkdir(parents=True, exist_ok=True)
    for img in (split_src / "images").iterdir():
        shutil.copy(img, split_dst / "images" / img.name)
    lbl_src = split_src / "labels"
    if lbl_src.exists():
        for lbl in lbl_src.iterdir():
            shutil.copy(lbl, split_dst / "labels" / lbl.name)


def build():
    if DST.exists():
        shutil.rmtree(DST)
    # positives from project-forklift01
    _copy_split(SRC / "train", DST / "train")
    _copy_split(SRC / "valid", DST / "valid")

    # our press-shop negatives, oversampled, with EMPTY label files (= background)
    n = 0
    for stem in NEG_FRAMES:
        src_img = config.FRAMES_DIR / f"{stem}.jpg"
        if not src_img.exists():
            print(f"  warn: {src_img} missing, skipping")
            continue
        for k in range(OVERSAMPLE):
            name = f"neg_{stem}_{k}"
            shutil.copy(src_img, DST / "train" / "images" / f"{name}.jpg")
            (DST / "train" / "labels" / f"{name}.txt").write_text("")  # no objects
            n += 1
    print(f"added {n} negative instances ({len(NEG_FRAMES)} frames x{OVERSAMPLE})")

    (DST / "data.yaml").write_text(
        "names:\n- forklift\n- person\nnc: 2\n"
        f"train: {DST/'train'/'images'}\n"
        f"val: {DST/'valid'/'images'}\n", encoding="utf-8")
    return DST / "data.yaml"


def main():
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"Base weights {BASE_WEIGHTS} not found "
                         "(keremberke). Download it first.")
    data_yaml = build()
    print(f"training (negative mining) from {BASE_WEIGHTS.name}, frozen backbone...")
    from ultralytics import YOLO
    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, device=0,
        freeze=10,            # freeze backbone -> keep recall, only adapt head
        lr0=0.001,            # gentle: don't forget keremberke
        patience=12, project=str(config.BASE_DIR / "runs"),
        name="forklift_negmine", exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy(best, OUT_WEIGHTS)
    print(f"\n✅ negative-mined weights -> {OUT_WEIGHTS}")
    print("Now run: python perception.py")


if __name__ == "__main__":
    main()
