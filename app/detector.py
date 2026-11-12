"""Turnitin-style document scoring + rewrite suggestions.

Pipeline
--------
1. Split the document into sentences (character offsets kept).
2. Build overlapping windows of ~200 words (stride ~100) - the same idea
   Turnitin documents: segments of a few hundred words that overlap so every
   sentence is judged in context.
3. Score every window with three independent signals:
     a. supervised classifier (DeBERTa-v3-large, desklib)
     b. zero-shot Binoculars (observer/performer LLM pair)
     c. stylometry (AI-vocabulary density + "burstiness" of sentence lengths)
4. Fuse the signals according to the strictness mode.
5. Each sentence inherits the mean score of the windows that contain it,
   lightly blended with a sentence-level classifier pass.
6. AI % = share of words that sit in flagged sentences.
7. Suggestions: AI-typical phrases (curated lexicon) + highly predictable
   words (per-token probability from the performer LM), with context-aware
   replacements from a masked LM.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass

from . import config
from .lexicon import STOPWORDS, alternatives_for, find_lexicon_hits
from .models import get_models

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]")
ABBREV = {"e.g", "i.e", "etc", "mr", "mrs", "ms", "dr", "prof", "vs", "fig", "al", "no", "st", "jr", "sr", "inc", "ltd", "co"}

MODES = {
    # name: (weights, flag threshold)
    "standard": {"threshold": 0.55},
    "strict": {"threshold": 0.45},
    "very_hard": {"threshold": 0.38},
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- text utils
@dataclass
class Sentence:
    start: int
    end: int
    text: str
    words: int


def split_sentences(text: str) -> list[Sentence]:
    sents: list[Sentence] = []
    # paragraphs first, then sentence boundaries inside each paragraph
    for para in re.finditer(r"[^\n]+", text):
        p_start, chunk = para.start(), para.group(0)
        last = 0
        for m in re.finditer(r"[.!?]+[\"')\]]*(?=\s+|$)", chunk):
            before = chunk[last : m.start()].split()
            tail = before[-1].lower().rstrip(".") if before else ""
            if tail in ABBREV or (len(tail) == 1 and tail.isalpha()):
                continue
            _add(sents, text, p_start + last, p_start + m.end())
            last = m.end()
        if last < len(chunk):
            _add(sents, text, p_start + last, p_start + len(chunk))
    return sents


def _add(sents: list[Sentence], text: str, a: int, b: int) -> None:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    if b > a:
        s = text[a:b]
        n = len(WORD_RE.findall(s))
        if n:
            sents.append(Sentence(a, b, s, n))


def build_windows(sents: list[Sentence]) -> list[tuple[int, int]]:
    """Return windows as (first_sentence_idx, last_sentence_idx_exclusive)."""
    if not sents:
        return []
    windows = []
    i = 0
    while i < len(sents):
        j, words = i, 0
        while j < len(sents) and (words < config.SEGMENT_WORDS or j == i):
            words += sents[j].words
            j += 1
        windows.append((i, j))
        if j >= len(sents):
            break
        # advance by ~stride words
        k, adv = i, 0
        while k < j - 1 and adv < config.SEGMENT_STRIDE_WORDS:
            adv += sents[k].words
            k += 1
        i = max(k, i + 1)
    return windows


# --------------------------------------------------------------------------- stylometry
def style_score(sents: list[Sentence]) -> float:
    text = " ".join(s.text for s in sents)
    words = sum(s.words for s in sents) or 1
    hits = len(find_lexicon_hits(text))
    density = hits / words * 100  # AI-typical phrases per 100 words
    lens = [s.words for s in sents]
    if len(lens) >= 3 and statistics.mean(lens) > 0:
        cv = statistics.pstdev(lens) / statistics.mean(lens)  # "burstiness"
    else:
        cv = 0.5
    dashes = text.count("—") / words * 100
    z = 0.9 * (density - 1.2) + 5.0 * (0.42 - cv) + 0.6 * dashes
    return sigmoid(z)


# --------------------------------------------------------------------------- fusion
def fuse(pc: float, pb: float, ps: float, mode: str) -> float:
    if mode == "standard":
        return 0.5 * pc + 0.4 * pb + 0.1 * ps
    hi, lo = max(pc, pb), min(pc, pb)
    if mode == "strict":
        return 0.55 * hi + 0.35 * lo + 0.10 * ps
    # very_hard: whichever detector is most suspicious dominates
    return 0.70 * hi + 0.18 * lo + 0.12 * ps


def bino_prob(score: float) -> float:
    return sigmoid((config.BINO_THRESHOLD - score) / config.BINO_SCALE)


# --------------------------------------------------------------------------- main entry
def analyze(text: str, mode: str = "very_hard", with_suggestions: bool = True) -> dict:
    if mode not in MODES:
        mode = "very_hard"
    text = text.replace("\r\n", "\n")[: config.MAX_CHARS]
    sents = split_sentences(text)
    total_words = sum(s.words for s in sents)
    warnings = []
    if total_words < config.MIN_RELIABLE_WORDS:
        warnings.append(
            f"Only {total_words} words. Detectors (including Turnitin, which needs 300+) are unreliable on short text."
        )
    if not sents:
        return {"ai_percent": 0, "mode": mode, "word_count": 0, "sentences": [], "suggestions": [], "warnings": warnings, "signals": {}}

    m = get_models()
    windows = build_windows(sents)
    win_texts = [text[sents[a].start : sents[b - 1].end] for a, b in windows]
    win_offsets = [sents[a].start for a, _ in windows]

    with m.lock:
        pcs = m.classifier_probs(win_texts)
        bino = m.binoculars(win_texts)
        long_idx = [i for i, s in enumerate(sents) if s.words >= 12]
        sent_pc = dict(zip(long_idx, m.classifier_probs([sents[i].text for i in long_idx])))

    win_scores, win_detail = [], []
    token_probs: dict[int, list[float]] = {}  # word char start -> probs
    for wi, (a, b) in enumerate(windows):
        pc = pcs[wi]
        bscore, toks = bino[wi]
        pb = bino_prob(bscore)
        ps = style_score(sents[a:b])
        win_scores.append(fuse(pc, pb, ps, mode))
        win_detail.append({"classifier": pc, "binoculars_score": bscore, "binoculars": pb, "style": ps})
        base = win_offsets[wi]
        for s, e, p in toks:
            token_probs.setdefault(base + s, []).append(p)

    # sentence pooling
    thr = MODES[mode]["threshold"]
    sent_scores = []
    for i, s in enumerate(sents):
        covering = [win_scores[w] for w, (a, b) in enumerate(windows) if a <= i < b]
        pooled = sum(covering) / len(covering)
        if i in sent_pc:
            local = sent_pc[i]
            pooled = 0.75 * pooled + 0.25 * (max(local, pooled) if mode == "very_hard" else local)
        sent_scores.append(pooled)

    flagged_words = sum(s.words for s, sc in zip(sents, sent_scores) if sc >= thr)
    ai_percent = round(100 * flagged_words / total_words) if total_words else 0

    def avg(key):
        return sum(d[key] for d in win_detail) / len(win_detail)

    result = {
        "ai_percent": ai_percent,
        "mode": mode,
        "threshold": thr,
        "word_count": total_words,
        "warnings": warnings,
        "signals": {
            "classifier": round(avg("classifier"), 3),
            "binoculars": round(avg("binoculars"), 3),
            "binoculars_raw": round(avg("binoculars_score"), 4),
            "style": round(avg("style"), 3),
            "document_score": round(sum(win_scores) / len(win_scores), 3),
        },
        "sentences": [
            {"start": s.start, "end": s.end, "score": round(sc, 3), "flagged": sc >= thr}
            for s, sc in zip(sents, sent_scores)
        ],
        "suggestions": [],
    }
    if with_suggestions:
        result["suggestions"] = build_suggestions(text, sents, sent_scores, thr, token_probs)
    return result


# --------------------------------------------------------------------------- suggestions
def _match_case(src: str, cand: str) -> str:
    if not cand:
        return cand
    if src.isupper() and len(src) > 1:
        return cand.upper()
    if src[0].isupper():
        return cand[0].upper() + cand[1:]
    return cand


def _too_similar(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    n = min(len(a), len(b), 5)
    return n >= 4 and a[:n] == b[:n]


def build_suggestions(text, sents, sent_scores, thr, token_probs, max_predictable_per_sent=3, max_total=250):
    m = get_models()
    sugg = []
    taken: list[tuple[int, int]] = []

    def overlaps(a, b):
        return any(a < y and x < b for x, y in taken)

    # 1. curated AI-phrase lexicon (whole document; higher priority in flagged sentences)
    for s, sc in zip(sents, sent_scores):
        for a, b, phrase in find_lexicon_hits(s.text):
            alts = [_match_case(phrase, x) for x in alternatives_for(phrase)]
            alts = [x if x else "(remove)" for x in alts]
            A, B = s.start + a, s.start + b
            taken.append((A, B))
            sugg.append({
                "start": A, "end": B, "original": text[A:B],
                "kind": "ai_phrase", "reason": "Word/phrase that AI models overuse",
                "severity": "high" if sc >= thr else "medium",
                "alternatives": alts,
            })

    # 2. highly predictable content words in flagged / borderline sentences
    targets = []
    for s, sc in zip(sents, sent_scores):
        if sc < thr * 0.8:
            continue
        cands = []
        for wm in WORD_RE.finditer(s.text):
            w = wm.group(0)
            A, B = s.start + wm.start(), s.start + wm.end()
            if len(w) < 4 or w.lower() in STOPWORDS or "'" in w or overlaps(A, B):
                continue
            if wm.start() > 0 and w[0].isupper():  # likely a proper noun
                continue
            p = _word_prob(token_probs, A, B)
            if p is not None and p >= 0.30:
                cands.append((p, A, B, w))
        cands.sort(reverse=True)
        for p, A, B, w in cands[:max_predictable_per_sent]:
            ctx_start = max(0, s.start - 250)
            ctx = text[ctx_start : s.end]
            targets.append((p, A, B, w, ctx, A - ctx_start, B - ctx_start))

    targets = targets[:max_total]
    if targets:
        with m.lock:
            cand_lists = m.mlm_candidates([(ctx, a, b) for *_, ctx, a, b in targets])
        for (p, A, B, w, *_), cands in zip(targets, cand_lists):
            alts = []
            for c, _prob in cands:
                if not c.isalpha() or len(c) < 3 or c.lower() in STOPWORDS or _too_similar(c, w):
                    continue
                alts.append(_match_case(w, c))
                if len(alts) >= 4:
                    break
            if alts:
                taken.append((A, B))
                sugg.append({
                    "start": A, "end": B, "original": w,
                    "kind": "predictable", "reason": f"Very predictable word ({round(p * 100)}% likely to an AI model)",
                    "severity": "high" if p >= 0.6 else "medium",
                    "alternatives": alts,
                })

    sugg.sort(key=lambda x: x["start"])
    return sugg


def _word_prob(token_probs, A, B):
    """Probability of the first token of the word at [A, B)."""
    # Qwen tokens usually include the leading space, so the token may start at A-1.
    for k in (A, A - 1):
        if k in token_probs:
            ps = token_probs[k]
            return sum(ps) / len(ps)
    return None
