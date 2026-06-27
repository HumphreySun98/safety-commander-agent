"""
vlm_judge.py — THE HEART of SafetyCommander.

A vision-language model (Qwen3-VL) looks at a camera frame, reads the site's
written safety policy, and returns a structured safety judgment.

DESIGN PRINCIPLE (this is the whole point of the project):
    The RISK LEVEL is reasoned by the model from the policy text. It is NEVER
    computed by an if/then rule here or anywhere else in this codebase. Edit
    safety_policy.txt and the judgments change accordingly.
"""
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from PIL import Image

import config

# ---------------------------------------------------------------------------
# The agent's brain. This system prompt enforces policy-grounded reasoning.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous factory safety officer (EHS coordinator) monitoring a production floor through camera frames. You are NOT a generic object detector. You reason like a trained safety officer: you look at what is happening, you consult the site's written safety policy, and you make a judgment grounded in that policy.

You will be given:
1. A camera frame from the factory floor.
2. The site's SAFETY POLICY (plain text). This is the single source of truth for what is and isn't a violation, and how severe it is. Different sites have different rules — you must judge strictly against THIS policy, not generic assumptions.
3. The current SHIFT CONTEXT (zone, time, ongoing operations).

Your job:
- Observe what is actually happening in the frame.
- Identify any safety hazard or violation BY CHECKING IT AGAINST THE POLICY TEXT.
- Assign a risk level by reasoning from the policy — NOT from a fixed lookup. If the policy says a zone is exempt from a requirement, then there is NO violation there, even if it would be one elsewhere.
- Cite the specific policy clause your judgment rests on.
- Recommend actions appropriate to the risk level.

Risk levels (interpret severity using the policy):
- "none": no hazard, or behavior explicitly permitted by policy.
- "low": minor issue; log only, no human needed.
- "medium": real violation in an active area; log + notify supervisor + corrective action.
- "high": near-miss or serious hazard; escalate, flag as near-miss.
- "critical": immediate danger to life (fire, smoke, person in lifted-load path); emergency escalation.

Output ONLY valid JSON, no prose, no markdown fences. Schema:
{
  "observation": "what you literally see in the frame",
  "hazard_type": "short label, e.g. 'missing_hardhat', 'forklift_pedestrian_proximity', 'smoke', 'none'",
  "risk_level": "none | low | medium | high | critical",
  "policy_clause": "quote or paraphrase the specific policy line you based this on; if no policy applies, say 'no applicable clause'",
  "reasoning": "one or two sentences: why this risk level FOLLOWS FROM the policy and the scene",
  "recommended_actions": ["action1", "action2"]
}

Critical rule: your risk_level must be derivable from the policy text you were given. If the policy changes, your judgment must change accordingly. Never apply a hardcoded rule that contradicts the provided policy."""

ALLOWED_RISK = {"none", "low", "medium", "high", "critical"}
SCHEMA_KEYS = ["observation", "hazard_type", "risk_level",
               "policy_clause", "reasoning", "recommended_actions"]

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.VLLM_KEY:
            raise RuntimeError(
                "VLLM_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_KEY,
                         timeout=90.0, max_retries=2)
    return _client


def _encode_image(image_path, max_dim=None) -> str:
    """Load, downscale (to respect the model's token budget), and base64-encode
    the frame as a JPEG data URL."""
    max_dim = max_dim or config.MAX_IMAGE_DIM
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def encode_frame(src, max_dim=None) -> str:
    """Encode a frame (file path, PIL.Image, or numpy array) to a JPEG data URL.
    Used for multi-frame clip judging."""
    max_dim = max_dim or config.MAX_IMAGE_DIM
    if isinstance(src, (str, Path)):
        img = Image.open(src)
    elif isinstance(src, Image.Image):
        img = src
    else:                                  # numpy array from imageio
        img = Image.fromarray(src)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _extract_json(text):
    """Tolerant JSON parse: strip ```json fences, then grab the outermost {...}."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            return None
    return None


def _format_perception(perception) -> str:
    """Render the on-prem YOLO 'derived' facts. These corroborate what the model
    sees; the model still decides risk from the policy (no risk level is passed in)."""
    if not perception:
        return ""
    d = perception.get("derived", {}) or {}
    lines = []
    if d.get("people") is not None:
        lines.append(f"- people detected: {d['people']}")
    if d.get("forklifts") is not None:
        lines.append(f"- forklifts detected: {d['forklifts']}")
    if d.get("min_person_forklift_dist_m") is not None:
        lines.append(f"- closest person-to-forklift distance: {d['min_person_forklift_dist_m']} m")
    if d.get("ppe_missing"):
        lines.append(f"- PPE missing on someone: {', '.join(d['ppe_missing'])}")
    for key, label in [("fire", "fire"), ("smoke", "smoke"),
                       ("spill", "spill"), ("phone_in_use", "phone in use")]:
        if key in d:
            lines.append(f"- {label}: {'yes' if d[key] else 'no'}")
    if not lines:
        return ""
    return (
        "\nDETECTOR FACTS (from an on-prem YOLO model on the cameras — these are "
        "measurements to corroborate what you see, NOT a verdict. You still decide "
        "the risk level yourself, strictly from the policy):\n"
        + "\n".join(lines) + "\n"
    )


def _format_user_text(policy_text, shift_context, perception=None, knowledge="") -> str:
    ctx = shift_context or {}
    ctx_lines = "\n".join(f"- {k}: {v}" for k, v in ctx.items()) or "- (none provided)"
    return (
        "SAFETY POLICY (single source of truth — judge strictly against THIS text):\n"
        "----------------------------------------------------------------------\n"
        f"{policy_text}\n"
        "----------------------------------------------------------------------\n"
        f"{knowledge or ''}"
        "\nCURRENT SHIFT CONTEXT:\n"
        f"{ctx_lines}\n"
        f"{_format_perception(perception)}\n"
        "GROUNDING: Base every claim ONLY on what is clearly visible in THIS frame. "
        "CCTV is wide-angle and low-resolution — do NOT assume a person, vehicle, or "
        "action you cannot actually see (e.g. do not claim someone is 'under a load' "
        "unless a person is visibly there). If presence, proximity, or an action is "
        "ambiguous, choose the lower-severity reading and state the uncertainty in "
        "'reasoning'. Reserve 'critical' ONLY for a clear, immediate danger to a person "
        "you can actually see — e.g. a person CLEARLY directly beneath or in the swing "
        "path of an elevated load, visible fire/smoke, or a person struck or fallen. If a "
        "person is merely NEAR a forklift and you cannot clearly confirm they are under "
        "the load, use 'high' or 'medium' — never 'critical'.\n\n"
        "PRIORITY OF EVIDENCE: (1) what you SEE decides whether a hazard exists; "
        "(2) the SITE POLICY sets the risk tier; (3) RETRIEVED REGULATIONS are only for "
        "citing the applicable standard once a hazard is already established. NEVER infer "
        "a hazard from retrieved text alone — if the regulations mention a hazard you "
        "cannot see in the frame, ignore it. If a retrieved regulation conflicts with the "
        "site policy, follow the site policy and note the conflict.\n\n"
        "Now analyze the attached camera frame. Return ONLY the JSON object."
    )


def _chat(messages, max_tokens):
    """Call the endpoint. Prefer JSON mode; fall back if the server rejects it."""
    client = _get_client()
    kwargs = dict(model=config.VLLM_MODEL, messages=messages,
                  temperature=config.VLM_TEMPERATURE, max_tokens=max_tokens)
    try:
        return client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs)
    except Exception:
        return client.chat.completions.create(**kwargs)


def _fallback(frame_name, error, t0, raw="", zone=""):
    """Return a safe, well-formed record when the judge itself fails (so the
    pipeline never crashes and a human still gets pinged for review)."""
    return {
        "observation": "Judge unavailable for this frame.",
        "hazard_type": "judge_error",
        "risk_level": "low",
        "policy_clause": "no applicable clause",
        "reasoning": f"The VLM judge could not produce a verdict ({error}).",
        "recommended_actions": ["Manually review this frame"],
        "frame": frame_name, "zone": zone,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latency_ms": int((time.time() - t0) * 1000),
        "ok": False, "error": error, "raw": raw, "model": config.VLLM_MODEL,
    }


def _normalize(parsed, frame_name, t0, raw, zone=""):
    out = {}
    out["observation"] = str(parsed.get("observation") or "").strip()
    out["hazard_type"] = str(parsed.get("hazard_type") or "none").strip()

    rl = str(parsed.get("risk_level") or "").strip().lower()
    # Keep the model's label as-is when valid. We do NOT re-grade it; if it is
    # unrecognized we surface that for human review rather than inventing a level.
    out["risk_level"] = rl if rl in ALLOWED_RISK else "low"
    out["risk_level_valid"] = rl in ALLOWED_RISK
    if not out["risk_level_valid"]:
        out["raw_risk_level"] = rl

    out["policy_clause"] = str(parsed.get("policy_clause") or "no applicable clause").strip()
    out["reasoning"] = str(parsed.get("reasoning") or "").strip()

    acts = parsed.get("recommended_actions")
    if isinstance(acts, str):
        acts = [acts]
    elif not isinstance(acts, list):
        acts = []
    out["recommended_actions"] = [str(a).strip() for a in acts if str(a).strip()]

    out.update({
        "frame": frame_name, "zone": zone,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latency_ms": int((time.time() - t0) * 1000),
        "ok": True, "error": None, "raw": raw, "model": config.VLLM_MODEL,
    })
    return out


def judge_frame(image_path, policy_text, shift_context=None, perception=None,
                knowledge="") -> dict:
    """
    Feed (camera frame + safety policy + shift context [+ optional YOLO facts]) to
    Qwen3-VL and return a structured, policy-grounded safety judgment.

    `perception` is the optional dict from the perception layer (perception.py);
    its 'derived' facts are shown to the model as corroborating measurements. The
    model still decides the risk level from the policy — perception never sets it.

    Returns a dict with the schema keys (observation, hazard_type, risk_level,
    policy_clause, reasoning, recommended_actions) plus metadata
    (frame, zone, timestamp, latency_ms, ok, error, raw, model).
    """
    t0 = time.time()
    frame_name = Path(image_path).name
    zone = (shift_context or {}).get("zone", "")

    try:
        data_url = _encode_image(image_path)
    except Exception as e:
        return _fallback(frame_name, f"image_error: {e}", t0, zone=zone)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text",
             "text": _format_user_text(policy_text, shift_context, perception, knowledge)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    raw = ""
    try:
        resp = _chat(messages, config.VLM_MAX_TOKENS)
        raw = resp.choices[0].message.content or ""
        parsed = _extract_json(raw)
        if parsed is None:
            # One corrective retry: ask for clean JSON only.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "That was not valid JSON. Reply with ONLY the JSON object — "
                "no code fences, no prose."})
            resp = _chat(messages, config.VLM_MAX_TOKENS)
            raw = resp.choices[0].message.content or ""
            parsed = _extract_json(raw)
    except Exception as e:
        return _fallback(frame_name, f"api_error: {e}", t0, raw=raw, zone=zone)

    if parsed is None:
        return _fallback(frame_name, "json_parse_failed", t0, raw=raw, zone=zone)

    return _normalize(parsed, frame_name, t0, raw, zone=zone)


def _format_clip_text(policy_text, shift_context, perception, n_frames, window_sec,
                      knowledge="") -> str:
    preamble = (
        f"You are watching {n_frames} CONSECUTIVE frames from ONE camera, in time order, "
        f"spanning about {window_sec:.1f} seconds. Judge the BEHAVIOUR OVER TIME, not a single "
        "still: look at motion and trajectory. A pedestrian moving INTO a forklift's path is a "
        "developing near-miss; a worker reaching into running machinery; an unstable or "
        "overloaded load in transit; someone leaving a hazard zone is the situation improving. "
        "Decide the risk from how the scene EVOLVES across the frames.\n\n"
    )
    return preamble + _format_user_text(policy_text, shift_context, perception, knowledge)


def judge_clip(frames, policy_text, shift_context=None, perception=None,
               window_sec=2.0, label=None, max_dim=512, knowledge="") -> dict:
    """
    Temporal version of judge_frame: feed a SEQUENCE of consecutive frames (a short
    clip) to Qwen3-VL so it can reason about motion / behaviour over time (near-miss
    approach, reaching into a machine, an overloaded load in transit). Returns the
    same structured schema as judge_frame, so dispatch()/ShiftReport are unchanged.

    `frames` = list of file paths, PIL images, or numpy arrays (in time order).
    """
    t0 = time.time()
    name = label or f"clip[{len(frames)}f]"
    zone = (shift_context or {}).get("zone", "")
    try:
        urls = [encode_frame(f, max_dim) for f in frames]
    except Exception as e:
        return _fallback(name, f"image_error: {e}", t0, zone=zone)

    text = _format_clip_text(policy_text, shift_context, perception, len(urls),
                             window_sec, knowledge)
    content = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]

    raw = ""
    try:
        resp = _chat(messages, config.VLM_MAX_TOKENS)
        raw = resp.choices[0].message.content or ""
        parsed = _extract_json(raw)
        if parsed is None:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "That was not valid JSON. Reply with ONLY the JSON object — no fences, no prose."})
            resp = _chat(messages, config.VLM_MAX_TOKENS)
            raw = resp.choices[0].message.content or ""
            parsed = _extract_json(raw)
    except Exception as e:
        return _fallback(name, f"api_error: {e}", t0, raw=raw, zone=zone)

    if parsed is None:
        return _fallback(name, "json_parse_failed", t0, raw=raw, zone=zone)

    out = _normalize(parsed, name, t0, raw, zone=zone)
    out["n_frames"] = len(urls)
    out["temporal"] = True
    return out


if __name__ == "__main__":
    # Quick smoke test: judge every frame in frames/ and print the verdicts.
    import sys
    policy = config.load_policy()
    ctx = {"zone": "Warehouse & dock — active forklift area",
           "shift": "Day shift", "operations": "Receiving & picking"}
    target = sys.argv[1] if len(sys.argv) > 1 else config.FRAMES_DIR
    target = Path(target)
    frames = [target] if target.is_file() else sorted(
        p for p in target.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for f in frames:
        j = judge_frame(str(f), policy, ctx)
        print(f"\n=== {f.name}  ({j['latency_ms']}ms, ok={j['ok']}) ===")
        print(json.dumps({k: j.get(k) for k in SCHEMA_KEYS}, indent=2, ensure_ascii=False))
