"""
demo_policy_flip.py — THE MONEY SHOT.

Proves the MODEL reasons from the written policy, not from code. The same camera
frame is judged twice; the only difference between the two runs is ONE line of
safety_policy.txt. The verdict flips. No code changes.

    python demo_policy_flip.py

Use this as the climax of the demo (reliable + fast: two calls, ~10s), or run it
live alongside editing safety_policy.txt.
"""
import textwrap

import config
from vlm_judge import judge_frame

FRAME = "frames/cam7_overload.jpg"
# NOTE: this exact wording is pinned — it was verified to give a clean, repeatable
# NONE -> MEDIUM flip (3/3 each). Don't reword it casually; the verdict is sensitive
# to context phrasing on this frame (it has an incidental background worker).
CTX = {
    "zone": "Forklift aisle / press shop CCTV",
    "shift": "Day",
    "operations": "forklift moving material",
}

# The rule we add. It lives ONLY in the policy text — no code path knows about it.
ANCHOR = ("2.5 An unattended forklift must have forks lowered, controls in neutral, and the\n"
          "    key removed.")
OVERLOAD_RULE = (
    "2.6 A forklift load must be stable and must NOT be stacked so high that it\n"
    "    blocks the operator's forward view. No more than 2 stacked stillage bins /\n"
    "    containers may be carried at once."
)


def show(title, j):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  RISK LEVEL : {str(j['risk_level']).upper()}")
    print(f"  HAZARD     : {j['hazard_type']}")
    print(f"  CLAUSE     : {j['policy_clause']}")
    wrapped = "\n               ".join(textwrap.wrap(str(j['reasoning']), 56))
    print(f"  REASONING  : {wrapped}")


def main():
    base = config.load_policy()                       # current rulebook (no 2.6)
    with_rule = base.replace(ANCHOR, ANCHOR + "\n" + OVERLOAD_RULE)

    print(f"\nSAME FRAME: {FRAME}")
    print("Judged twice — the ONLY difference is one line in safety_policy.txt. No code changes.")

    j1 = judge_frame(FRAME, base, CTX)
    show("POLICY A — the rulebook says nothing about load height", j1)

    j2 = judge_frame(FRAME, with_rule, CTX)
    show("POLICY B — added ONE line:  2.6 'no more than 2 stacked bins / must not block view'", j2)

    print("\n" + "-" * 72)
    print(f"  {str(j1['risk_level']).upper()}  ->  {str(j2['risk_level']).upper()}"
          "      Same footage. One policy line. Zero code changes.")
    print("  The agent didn't run new code — it read the rule I just wrote.")
    print("-" * 72)


if __name__ == "__main__":
    main()
