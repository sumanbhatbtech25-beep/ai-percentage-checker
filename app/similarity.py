"""Similarity check: does the report reuse text from published papers?

Scope (stated in the UI): the report is compared with the abstracts of papers
found for its topic via OpenAlex / Semantic Scholar - not the whole web or
other students' submissions, which only services like Turnitin can search.

For every prose sentence in the report we look for the closest abstract sentence:
  * copied            - shares a long run of identical words (8-word shingles)
  * close paraphrase  - same sentence reworded (embedding similarity >= 0.90)
  * similar idea      - a common idea in common words (0.85-0.90): shown, not counted
and note whether the sentence carries a citation, e.g. (Smith, 2019), [3], et al.
"""
from __future__ import annotations

import re

from .compare import _sentences
from .detector import classify_qualifying, split_sentences
from .papers import embed, find_similar

SHINGLE = 8               # words in a row that count as copying
PARAPHRASE_SIM = 0.90     # close paraphrase: same sentence reworded (see tune_similarity)
SIMILAR_SIM = 0.85        # similar idea in common words: shown, but not counted as a problem
CITATION_RE = re.compile(
    r"\(([A-Z][A-Za-z'\-]+( et al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,? (19|20)\d{2}[a-z]?|(19|20)\d{2}[a-z]?)(;[^)]*)?\)"
    r"|\[\d+(?:[,–\-]\s*\d+)*\]|\bet al\.?|\baccording to [A-Z]", re.I)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _shingles(words: list[str]) -> set[tuple]:
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def check(text: str, top_papers: int = 10) -> dict:
    sents = split_sentences(text)
    classify_qualifying(text, sents)
    report = [s for s in sents if s.kind == "prose" and s.words >= 8]
    if len(report) < 3:
        return {"error": "Not enough prose to check. Paste or upload your full report."}

    found = find_similar(text, top=top_papers)
    papers = found.get("papers", [])
    if not papers:
        return {"error": "Couldn't find related papers to compare with. " + " ".join(found.get("errors", []))}

    # every abstract sentence, remembering which paper it came from
    src = []
    for pi, p in enumerate(papers):
        for s in re.split(r"(?<=[.!?])\s+", p.get("abstract_full") or p.get("abstract") or ""):
            if len(s.split()) >= 6:
                src.append((pi, s.strip()))
    if not src:
        return {"error": "The related papers had no usable abstracts."}

    rep_vecs = embed([s.text for s in report])
    src_vecs = embed([s for _, s in src])
    sims = rep_vecs @ src_vecs.T
    best_sim, best_idx = sims.max(dim=1)
    src_shingles = [_shingles(_words(s)) for _, s in src]

    matches = []
    for k, s in enumerate(report):
        j = int(best_idx[k])
        sim = float(best_sim[k])
        sh = _shingles(_words(s.text))
        copied = max(((len(sh & src_shingles[i]) / max(1, len(sh)), i) for i in range(len(src))), default=(0, j))
        kind = None
        if copied[0] >= 0.3:
            kind, j = "copied", copied[1]
        elif sim >= PARAPHRASE_SIM:
            kind = "paraphrase"
        elif sim >= SIMILAR_SIM:
            kind = "similar"
        if not kind:
            continue
        p = papers[src[j][0]]
        matches.append({
            "start": s.start, "end": s.end, "sentence": s.text, "words": s.words,
            "kind": kind, "similarity": round(max(sim, copied[0]), 3),
            "cited": bool(CITATION_RE.search(s.text)),
            "source_sentence": src[j][1],
            "paper": {"title": p["title"], "year": p.get("year"), "url": p.get("url"),
                      "authors": p.get("authors", [])[:3]},
        })

    total = sum(s.words for s in report)
    serious = [m for m in matches if m["kind"] != "similar"]  # "similar idea" is informational only
    matched = sum(m["words"] for m in serious)
    uncited = [m for m in serious if not m["cited"]]
    return {
        "matched_percent": round(100 * matched / total) if total else 0,
        "uncited_count": len(uncited),
        "papers_checked": len(papers),
        "sentences_checked": len(report),
        "matches": sorted(matches, key=lambda m: (m["kind"] == "similar", m["cited"], -m["similarity"])),
        "notes": found.get("errors", []),
    }


def tune_similarity():
    """Print similarities for paraphrases vs. merely related sentences (sanity check for PARAPHRASE_SIM)."""
    src = "Government spending on infrastructure raises aggregate demand and creates jobs in construction and transport."
    pairs = {
        "close paraphrase": "When the government spends on infrastructure, aggregate demand rises and jobs are created in construction and transport.",
        "light paraphrase": "Public investment in infrastructure boosts overall demand and generates employment in building and transportation.",
        "related, not reused": "Infrastructure projects in India such as PM Gati Shakti aim to cut logistics costs for companies.",
        "same topic, different claim": "Higher taxes on businesses can reduce investment and slow down hiring.",
    }
    a = embed([src])
    for name, s in pairs.items():
        print(f"{name:28} {float(embed([s]) @ a.T):.3f}")


if __name__ == "__main__":
    tune_similarity()
