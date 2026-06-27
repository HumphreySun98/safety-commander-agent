"""
eval_perception.py — reproducible DETECTION-side evaluation (B / perception layer).

Scores the shipped YOLO facts (committed perception/*.json) against a hand-labelled
set of our demo CCTV. Pairs with eval_agent.py (A, agent judgments) behind
docs/eval.md.

What it measures (forklift detector = keremberke yolov8s @ per-label conf 0.8):
  - PRECISION: false forklift detections on press-only frames (target: 0)
  - RECALL:    real-forklift frames in which a forklift was detected
  - DISTANCE:  person<->forklift distances measured (the crown-jewel fact)

Eval / facts only — no risk logic here (risk is the VLM's job).

HONESTY (keep these caveats in any slide):
  * This is our 25-frame demo CCTV set, hand-labelled — NOT a general benchmark.
  * Distance uses a documented ground-plane approximation (person height -> m/px);
    we have no ground-truth distance, so we report MEASURED values, not an error.

Usage:
    python perception.py        # (re)generate perception/*.json first, if needed
    python eval_perception.py            # print the report
    python eval_perception.py --md docs/eval_perception.md   # also write a table
"""
import argparse
import json
from pathlib import Path

import config

# Hand-verified labels for our 25-frame demo set (frame stem -> has a real forklift).
REAL_FORKLIFT = {
    "cam1_t1", "cam1_t2", "cam1_t3",          # forklift in the background
    "cam7_overload", "cam8_t1", "cam8_t2", "cam8_t3",
}
PRESS_ONLY = {  # press shop, NO forklift present (the false-positive trap)
    "cam2_t1", "cam2_t2", "cam2_t3", "cam3_t1", "cam3_t2", "cam3_t3",
    "cam4_t1", "cam4_t2", "cam4_t3", "cam5_t1", "cam5_t2", "cam5_t3",
    "cam6_t1", "cam6_t2", "cam6_t3", "cam7_t1", "cam7_t2", "cam7_t3",
}


def _load():
    out = {}
    for fp in sorted(config.PERCEPTION_DIR.glob("*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[fp.stem] = d
    return out


def evaluate(data):
    def n_forklift(stem):
        return (data.get(stem, {}).get("derived", {}) or {}).get("forklifts", 0)

    def forklift_conf(stem):
        confs = [x["conf"] for x in data.get(stem, {}).get("detections", [])
                 if x["label"] == "forklift"]
        return max(confs) if confs else None

    real = sorted(s for s in REAL_FORKLIFT if s in data)
    press = sorted(s for s in PRESS_ONLY if s in data)
    recalled = [s for s in real if n_forklift(s) >= 1]
    fp = [s for s in press if n_forklift(s) >= 1]

    dists = []
    for s, d in data.items():
        dm = (d.get("derived", {}) or {}).get("min_person_forklift_dist_m")
        if dm is not None:
            dists.append((s, dm))
    dists.sort(key=lambda x: x[1])

    return {
        "real_frames": real, "press_frames": press,
        "recalled": recalled, "false_positives": fp,
        "recall": (len(recalled), len(real)),
        "precision_clean": (len(press) - len(fp), len(press)),
        "money_shot_conf": forklift_conf("cam7_overload"),
        "distances": dists,
    }


def print_report(m):
    rc, rt = m["recall"]
    pc, pt = m["precision_clean"]
    print("=" * 64)
    print("SafetyCommander — perception (detection) eval   [B]")
    print("=" * 64)
    print(f"Forklift recall (real-forklift frames detected): {rc}/{rt}")
    print(f"Forklift precision (press-only frames clean):    {pc}/{pt}  "
          f"-> {len(m['false_positives'])} false positives")
    if m["false_positives"]:
        print(f"   ! false positives on: {', '.join(m['false_positives'])}")
    ms = m["money_shot_conf"]
    print(f"Money shot (cam7_overload) forklift conf:        "
          f"{ms if ms is not None else 'n/a'}")
    print("person<->forklift distances measured (m):")
    for s, dm in m["distances"]:
        print(f"   {s:16} {dm} m")
    print("-" * 64)
    print("NOTE: 25-frame demo set, hand-labelled (not a general benchmark).")
    print("NOTE: distance is a ground-plane approximation; measured values shown,")
    print("      no ground-truth distance so no error is claimed.")


def to_md(m):
    rc, rt = m["recall"]
    pc, pt = m["precision_clean"]
    ms = m["money_shot_conf"]
    nearest = min((d for _, d in m["distances"]), default=None)
    L = ["# Perception (detection) eval — B", "",
         "_Forklift detector: keremberke yolov8s @ per-label conf 0.8. "
         "25-frame hand-labelled demo CCTV; reproduce with `python eval_perception.py`._",
         "", "| Metric | Value |", "|---|---|",
         f"| Forklift precision (press-only frames clean) | {pc}/{pt} "
         f"({len(m['false_positives'])} false positives) |",
         f"| Forklift recall (real-forklift frames) | {rc}/{rt} |",
         f"| Money shot (cam7_overload) forklift conf | {ms} |",
         f"| Nearest person↔forklift distance (measured) | {nearest} m |",
         "",
         "_Distance is a documented ground-plane approximation (person height → m/px); "
         "measured values, no ground-truth distance so no error is claimed._", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detection-side eval (perception).")
    ap.add_argument("--md", help="also write a markdown table to this path")
    args = ap.parse_args(argv)
    data = _load()
    if not data:
        print(f"No perception/*.json found in {config.PERCEPTION_DIR}. "
              "Run `python perception.py` first.")
        return 1
    m = evaluate(data)
    print_report(m)
    if args.md:
        Path(args.md).write_text(to_md(m), encoding="utf-8")
        print(f"\nmarkdown table -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
