"""
config.py — central configuration for SafetyCommander.

All secrets / endpoints come from environment variables (loaded from a local
.env file that is NOT committed). The safety policy is plain text on disk so an
operations manager can edit the rules at any time without touching code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- VLM endpoint (OpenAI-compatible Qwen3-VL) ----------------------------------
# Defaults point at the hackathon shared endpoint; the KEY must come from .env.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://89.169.112.191:8080/v1")
VLLM_KEY      = os.getenv("VLLM_KEY", "")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")

# --- Paths ----------------------------------------------------------------------
FRAMES_DIR     = Path(os.getenv("FRAMES_DIR", str(BASE_DIR / "frames")))
LOGS_DIR       = BASE_DIR / "logs"
REPORTS_DIR    = BASE_DIR / "reports"
POLICY_PATH    = Path(os.getenv("SAFETY_POLICY_PATH", str(BASE_DIR / "safety_policy.txt")))
PERCEPTION_DIR = BASE_DIR / "perception"        # YOLO facts (written by perception.py)
ANNOTATED_DIR  = BASE_DIR / "frames_annotated"  # frames with boxes drawn

# --- Runtime tuning -------------------------------------------------------------
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "2.0"))   # demo pacing
MAX_IMAGE_DIM      = int(os.getenv("MAX_IMAGE_DIM", "1024"))         # token budget
VLM_MAX_TOKENS     = int(os.getenv("VLM_MAX_TOKENS", "800"))
VLM_TEMPERATURE    = float(os.getenv("VLM_TEMPERATURE", "0.0"))  # 0 = reproducible verdicts

# --- Optional integrations ------------------------------------------------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Make sure output dirs exist.
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
PERCEPTION_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


def load_policy() -> str:
    """Read the site safety policy. This file is the single source of truth the
    VLM reasons against; an ops manager can edit it anytime and judgments change."""
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Safety policy not found at {POLICY_PATH}")
    return POLICY_PATH.read_text(encoding="utf-8")


def check() -> list[str]:
    """Return a list of human-readable config problems (empty == OK)."""
    problems = []
    if not VLLM_KEY:
        problems.append("VLLM_KEY is empty — copy .env.example to .env and set it.")
    if not POLICY_PATH.exists():
        problems.append(f"Policy file missing: {POLICY_PATH}")
    return problems
