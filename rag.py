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


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "forklift carrying a tall stack of bins blocking view"
    print(f"query: {q}\n")
    for c in retrieve(q, 3):
        print(f"[{c['score']:.3f}] {c['source']}")
        print("   " + c["text"][:160].replace("\n", " ") + "...\n")
