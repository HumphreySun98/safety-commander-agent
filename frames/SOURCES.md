# Demo frames — source & attribution

These frames are sampled from real factory CCTV in the **"Video Dataset for Safe
and Unsafe Behaviours"** (closed production area, Eskişehir, Turkey).

- Source: https://data.mendeley.com/datasets/xjmtb22pff/1
- DOI: 10.17632/xjmtb22pff.1
- License: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0) — free to
  share/adapt with attribution.

Frames were extracted with `extract_frames.py` (3 snapshots per clip). The camera
names here are neutral; the table below maps each to the dataset's labeled class
so verdicts can be cross-checked. **The VLM never sees these filenames or labels** —
it only receives the image, the safety policy, and the shift context.

| demo camera | dataset clip / class                         |
|-------------|----------------------------------------------|
| cam1_t*     | 0_safe_walkway_violation (walkway violation) |
| cam2_t*     | 4_safe_walkway (safe walkway)                |
| cam3_t*     | 1_unauthorized_intervention                  |
| cam4_t*     | 5_authorized_intervention                    |
| cam5_t*     | 2_opened_panel_cover                         |
| cam6_t*     | 6_closed_panel_cover                         |
| cam7_t*     | 3_carrying_overload_with_forklift            |
| cam8_t*     | 7_safe_carrying                              |

The alternate set in `../frames_samples/` is openly-licensed stock imagery
(Wikimedia Commons) that exercises the higher-severity / escalation path.
