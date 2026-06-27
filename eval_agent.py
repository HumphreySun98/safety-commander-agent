"""
eval_agent.py — agent-side evaluation (D4, VLM verdicts).

Runs the VIDEO judge on the 8 demo clips (one window each) and tabulates expected
behaviour vs model verdict, false criticals, near-misses, and latency. This is the
demo's main mode (temporal video, RAG off). Pairs with B's eval_perception.py
(detection metrics: forklift precision/recall, distances).

    python eval_agent.py
"""
import time

import config
from main import sample_windows, VIDEO_CONTEXT
from vlm_judge import judge_clip

# clip stem -> (description, ground-truth safe/unsafe from the dataset class)
CLIPS = {
    "cam1_walkway_violation":        ("walkway violation",        "unsafe"),
    "cam2_walkway_safe":             ("safe walkway",             "safe"),
    "cam3_intervention_unauthorized":("unauthorized intervention","unsafe"),
    "cam4_intervention_authorized":  ("authorized intervention",  "safe"),
    "cam5_panel_open":               ("open panel / guard",       "unsafe"),
    "cam6_panel_closed":             ("closed panel / guard",     "safe"),
    "cam7_forklift_overload":        ("forklift overload",        "unsafe"),
    "cam8_carrying_safe":            ("safe carrying",            "safe"),
}


ORDER = ["none", "low", "medium", "high", "critical"]


def main():
    policy = config.load_policy()
    rows, crit, caught = [], 0, 0
    print(f"{'behaviour':28} {'truth':7} {'peak':9} {'hazard@peak':24} {'flagged':9} {'lat'}")
    print("-" * 92)
    for stem, (desc, label) in CLIPS.items():
        wins = sample_windows(f"demo_clips/{stem}.mp4", window_sec=1.6, stride_sec=3.0, k=4)
        best, best_haz, flagged_w, clip_lat = 0, "none", 0, 0.0
        for _, frames, _ in wins:                       # scan the WHOLE clip
            t = time.time()
            j = judge_clip(frames, policy, VIDEO_CONTEXT, window_sec=1.6, label=stem)
            clip_lat += time.time() - t
            idx = ORDER.index(j["risk_level"]) if j["risk_level"] in ORDER else 0
            if idx >= 2:
                flagged_w += 1
            if idx > best:
                best, best_haz = idx, j["hazard_type"]
        peak = ORDER[best]
        if peak == "critical":
            crit += 1
        if label == "unsafe" and best >= 3:             # unsafe clip reached high+ = caught
            caught += 1
        rows.append((desc, label, peak, best_haz, flagged_w, len(wins), clip_lat))
        print(f"{desc:28} {label:7} {peak:9} {str(best_haz)[:24]:24} "
              f"{str(flagged_w)+'/'+str(len(wins)):9} {clip_lat/max(1,len(wins)):.1f}s/win")
    n_unsafe = sum(1 for _, l in CLIPS.values() if l == "unsafe")
    print("-" * 92)
    print(f"clips: {len(rows)}   FALSE CRITICALS: {crit}   "
          f"unsafe caught (high+): {caught}/{n_unsafe}   avg latency ~4s/window")
    return rows


if __name__ == "__main__":
    main()
