"""
rag.py — retrieval over the factory's REGULATIONS & KNOWLEDGE base (knowledge/).

The site policy (safety_policy.txt) is the editable house rulebook. This module
retrieves the specific OSHA standards, equipment SOPs, and SDS excerpts behind it,
so the VLM can ground its verdict in the actual regulation for the scene and cite it.

DESIGN: retrieval supplies KNOWLEDGE only — the VLM still reasons the risk. There is no
if/then risk logic here. (Keeps the hackathon's #2 principle intact.)

Uses TF-IDF + cosine (scikit-learn): no model download, fast, deterministic. Degrades to
a no-op (returns []) if the knowledge dir is empty or sklearn is unavailable, so the rest
of the pipeline is never blocked.
"""
import os
import re

import config

_index = None  # cached: (vectorizer | "empty", matrix | None, [(source, text), ...])


def _chunk(text, source, max_chars=750):
    """Split a document into prompt-sized chunks on blank lines / headings."""
    chunks, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 1 > max_chars:
            chunks.append((source, buf.strip()))
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append((source, buf.strip()))
    return chunks


def _build():
    """Load + chunk knowledge/, fit a TF-IDF index. Cached."""
    global _index
    docs = []
    kd = config.KNOWLEDGE_DIR
    if kd.exists():
        for f in sorted(kd.glob("*.md")) + sorted(kd.glob("*.txt")):
            docs += _chunk(f.read_text(encoding="utf-8"), f.stem)
    if not docs:
        _index = ("empty", None, [])
        return _index
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([c[1] for c in docs])
        _index = (vec, mat, docs)
    except Exception:
        _index = ("empty", None, docs)   # sklearn missing -> no-op
    return _index


def retrieve(query, k=3):
    """Return the top-k knowledge chunks for a query: [{source, text, score}, ...]."""
    if os.getenv("SC_RAG", "1") == "0":
        return []
    vec, mat, docs = _index or _build()
    if not docs or vec == "empty" or mat is None or not str(query).strip():
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    order = sims.argsort()[::-1][:k]
    return [{"source": docs[i][0], "text": docs[i][1], "score": float(sims[i])}
            for i in order if sims[i] > 0.02]


def format_for_prompt(chunks):
    """Render retrieved chunks as a prompt block (empty string if none)."""
    if not chunks:
        return ""
    out = ["\nRETRIEVED REGULATIONS & KNOWLEDGE (factory reference library — cite the "
           "specific standard / SOP / SDS when it applies; these SUPPORT the site policy, "
           "they do not replace it):"]
    for c in chunks:
        out.append(f"--- [{c['source']}] ---\n{c['text']}")
    return "\n".join(out) + "\n"


def retrieve_text(query, k=3):
    """Convenience: retrieve + format in one call (used by the orchestrator)."""
    return format_for_prompt(retrieve(query, k))


def sources(query, k=3):
    """Just the source names retrieved (for logging / the dashboard)."""
    return [c["source"] for c in retrieve(query, k)]


# ===========================================================================
# HAZARD-DRIVEN retrieval (the redesign).
#
# RAG is NOT a perception aid. It runs ONLY after the VLM has already decided a
# hazard exists from the image, and only for hazard types we have regulations for.
# The query is built from the *hazard*, not the generic zone. A keyword relevance
# gate then drops chunks that aren't about that hazard — so retrieved text can never
# induce a hazard the model didn't see (no over-grounding / false HIGHs).
# ===========================================================================

HAZARD_QUERIES = {
    "forklift_near_miss": [
        "forklift pedestrian near miss travel path warehouse aisle",
        "powered industrial truck pedestrian separation distance",
        "forklift operator obstructed view pedestrian exposure",
        "forklift load stability tiered load rated capacity",
    ],
    "ppe_violation": [
        "hard hat head protection required production zone",
        "high visibility vest powered industrial truck area",
        "eye protection safety glasses grinding machining welding",
    ],
    "smoke_fire": [
        "smoke fire emergency response evacuation factory",
        "hot work welding cutting fire watch permit",
        "fire extinguisher exit route unobstructed",
    ],
    "spill": [
        "liquid spill slip hazard contain absorbent sign",
        "chemical spill response SDS isolate area",
    ],
    "machine_guarding": [
        "machine guarding point of operation running equipment",
        "lockout tagout energized maintenance servicing",
        "power press guard die area two-hand control",
    ],
}

HAZARD_KEYWORDS = {
    "forklift_near_miss": ["forklift", "powered industrial truck", "pedestrian", "travel",
                           "aisle", "load", "truck"],
    "ppe_violation": ["hard hat", "hardhat", "helmet", "vest", "head protection", "high-vis"],
    "smoke_fire": ["smoke", "fire", "evacuation", "emergency", "hot work", "extinguisher"],
    "spill": ["spill", "slip", "absorbent", "sds", "chemical"],
    "machine_guarding": ["guard", "guarding", "lockout", "tagout", "loto",
                         "point of operation", "press", "die"],
}


def map_hazard(hazard_type):
    """Map the VLM's free-text hazard_type onto a RAG hazard class (or None)."""
    h = (hazard_type or "").lower()
    if any(w in h for w in ("forklift", "truck", "load", "aisle", "pedestrian", "obstruct")):
        return "forklift_near_miss"
    if any(w in h for w in ("hardhat", "hard_hat", "ppe", "vest", "glasses", "helmet", "eye")):
        return "ppe_violation"
    if any(w in h for w in ("smoke", "fire")):
        return "smoke_fire"
    if any(w in h for w in ("spill", "slip")):
        return "spill"
    if any(w in h for w in ("guard", "loto", "intervention", "machine", "press")):
        return "machine_guarding"
    return None


def should_use_rag(hazard_type, risk_level):
    """RAG fires only on a suspected, regulable hazard — never on safe frames."""
    if risk_level not in ("medium", "high", "critical"):
        return False
    return map_hazard(hazard_type) is not None


def relevant_enough(chunk_text, rag_hazard, min_hits=2):
    """Keyword relevance gate: a chunk must clearly be about THIS hazard to be injected."""
    t = chunk_text.lower()
    keys = HAZARD_KEYWORDS.get(rag_hazard, [])
    return sum(k in t for k in keys) >= min_hits


def retrieve_for_hazard(hazard_type, zone="", k=2):
    """Hazard-specific, relevance-gated retrieval. Returns [] when RAG shouldn't fire."""
    rh = map_hazard(hazard_type)
    if rh is None:
        return []
    picked = {}
    for q in HAZARD_QUERIES[rh]:
        for c in retrieve(f"{q} {zone}".strip(), k=4):
            if not relevant_enough(c["text"], rh):
                continue
            key = (c["source"], c["text"][:48])
            if key not in picked or c["score"] > picked[key]["score"]:
                picked[key] = c
    return sorted(picked.values(), key=lambda c: c["score"], reverse=True)[:k]


if __name__ == "__main__":
    import sys
    hz = sys.argv[1] if len(sys.argv) > 1 else "forklift_pedestrian_proximity"
    print(f"hazard_type: {hz}  ->  rag class: {map_hazard(hz)}\n")
    for c in retrieve_for_hazard(hz, "warehouse aisle"):
        print(f"[{c['score']:.3f}] {c['source']}\n   {c['text'][:150].strip()}...\n")
