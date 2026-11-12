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
    # Tuned with app/tune.py on pre-ChatGPT human writing (student essays, academic
    # essays, arXiv abstracts) vs AI essays / ChatGPT answers.
    #   cls_center: the DeBERTa classifier only counts as "AI" once it's more confident
    #               than this. Raw 0.5 flagged ~48% of real student-essay sections.
    #   w_cls / w_style: weights in the fused score (Binoculars gets the remainder).
    #   threshold: a sentence is flagged when its fused score reaches this.
    "standard": {"cls_center": 0.99, "w_cls": 0.80, "w_style": 0.05, "threshold": 0.54},
    "strict": {"cls_center": 0.95, "w_cls": 0.60, "w_style": 0.05, "threshold": 0.48},
    "very_hard": {"cls_center": 0.95, "w_cls": 0.50, "w_style": 0.05, "threshold": 0.36},
}
DEFAULT_MODE = "standard"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- text utils
@dataclass
class Sentence:
    start: int
    end: int
    text: str
    words: int
    para: int = 0  # index of the paragraph (line) the sentence belongs to
    kind: str = "prose"  # "prose" counts toward the AI %; anything else is excluded (see classify_qualifying)


def split_sentences(text: str) -> list[Sentence]:
    sents: list[Sentence] = []
    # paragraphs first, then sentence boundaries inside each paragraph
    for pi, para in enumerate(re.finditer(r"[^\n]+", text)):
        p_start, chunk = para.start(), para.group(0)
        first = len(sents)
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
        for s in sents[first:]:
            s.para = pi
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


MIN_GROUP_WORDS = 25  # headings / one-liners get scored together with the paragraph after them


def build_paragraph_windows(sents: list[Sentence]) -> list[tuple[int, int]]:
    """Windows that never cross a paragraph boundary, so one paragraph's score
    can't bleed into its neighbours. Very short paragraphs (headings, captions)
    are joined to the following paragraph so they still get a usable window."""
    groups: list[list[int]] = []
    pending: list[int] = []
    for pi in dict.fromkeys(s.para for s in sents):
        idx = [i for i, s in enumerate(sents) if s.para == pi]
        pending += idx
        if sum(sents[i].words for i in pending) >= MIN_GROUP_WORDS:
            groups.append(pending)
            pending = []
    if pending:
        if groups:
            groups[-1] += pending
        else:
            groups.append(pending)
    windows = []
    for g in groups:
        base = g[0]
        windows += [(base + a, base + b) for a, b in build_windows(sents[g[0] : g[-1] + 1])]
    return windows


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
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def calibrated_cls(pc: float, mode: str) -> float:
    """Shift the classifier so that its own `cls_center` confidence maps to 0.5."""
    return sigmoid(_logit(pc) - _logit(MODES[mode]["cls_center"]))


def fuse(pc: float, pb: float, ps: float, mode: str) -> float:
    m = MODES[mode]
    return m["w_cls"] * calibrated_cls(pc, mode) + (1 - m["w_cls"] - m["w_style"]) * pb + m["w_style"] * ps


def bino_prob(score: float) -> float:
    return sigmoid((config.BINO_THRESHOLD - score) / config.BINO_SCALE)


# --------------------------------------------------------------------------- qualifying text
# Turnitin only scores "qualifying text": prose sentences in long-form writing.
# Lists, tables, code, headings etc. are left out, and so are text you quote or
# cite (quoting and referencing sources is allowed; it isn't your AI writing).
_LIST_RE = re.compile(r"^\s*(?:[-*•·▪◦●○‣⁃]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[ivxIVX]{1,4}[.)])\s+")
_CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?\.?|table|chart|graph|source|note|equation)\s*[\dIVX]*\s*[:.\-–]", re.I)
_REF_HEAD_RE = re.compile(r"^\s*(?:\d+[.)]?\s*)?(?:references?|bibliography|works cited|reference list|sources|citations|literature cited)\s*:?\s*$", re.I)
_BACK_HEAD_RE = re.compile(r"^\s*(?:\d+[.)]?\s*)?(?:appendix|appendices|acknowledg(?:e)?ments?)\b", re.I)
_REF_ENTRY_RE = re.compile(r"(\(\d{4}[a-z]?\)|\b(19|20)\d{2}[a-z]?\.)|doi\.org|https?://|\bvol\.|\bpp\.|^\s*\[\d+\]", re.I)
_CODE_RE = re.compile(r"[{};=<>]|^\s*(def|class|import|for|while|if|return|public|private|#include)\b")
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,})[\"”]")

EXCLUDED_LABELS = {
    "heading": "Heading or title",
    "list": "List item",
    "caption": "Figure or table caption",
    "reference": "Reference list",
    "code": "Code",
    "table": "Table row",
    "quote": "Direct quotation (cited source)",
    "short": "Too short to judge",
}


def classify_qualifying(text: str, sents: list[Sentence]) -> None:
    """Set Sentence.kind: 'prose' (scored) or the reason it's excluded."""
    lines = list(re.finditer(r"[^\n]+", text))
    line_kind: dict[int, str] = {}
    in_refs = False
    for pi, m in enumerate(lines):
        line = m.group(0).strip()
        words = len(WORD_RE.findall(line))
        if _REF_HEAD_RE.match(line):
            in_refs, kind = True, "heading"
        elif in_refs and _BACK_HEAD_RE.match(line):
            in_refs, kind = False, "heading"
        elif in_refs:
            kind = "reference"
        elif _CAPTION_RE.match(line):
            kind = "caption"
        elif (line.count("\t") >= 2 or line.count("|") >= 2) and words / (max(line.count("\t"), line.count("|")) + 1) < 8:
            kind = "table"  # several short tab/pipe-separated cells; long prose with stray tabs still counts
        elif _LIST_RE.match(line) and words < 60:
            kind = "list"
        elif words <= 12 and not re.search(r"[.!?]\s*[\"')\]]*$", line):
            kind = "heading"  # short line with no full stop: a title or heading
        elif _CODE_RE.search(line) and words < 25 and len(re.findall(r"[{};=()<>]", line)) >= 3:
            kind = "code"
        elif _REF_ENTRY_RE.search(line) and words < 60 and line.count(",") >= 2 and re.match(r"^\s*([A-Z][A-Za-z'\-]+,\s*[A-Z]\.|\[\d+\])", line):
            kind = "reference"  # a lone reference entry outside a References section
        else:
            kind = "prose"
        line_kind[pi] = kind

    for s in sents:
        kind = line_kind.get(s.para, "prose")
        if kind == "prose":
            quoted = sum(len(WORD_RE.findall(q)) for q in _QUOTE_RE.findall(s.text))
            if quoted and quoted >= 0.6 * s.words:
                kind = "quote"
            elif s.words < 4:
                kind = "short"
        s.kind = kind


# --------------------------------------------------------------------------- main entry
@dataclass
class Raw:
    """Everything the models produce for one text; scoring rules run on top of this."""
    text: str
    sents: list
    q: list            # indices (into sents) of qualifying sentences
    windows: list      # (a, b) index ranges into q
    pcs: list          # classifier prob per window
    bino: list         # raw Binoculars score per window
    styles: list       # style score per window
    sent_pc: dict      # sents index -> sentence-level classifier prob
    token_probs: dict  # char offset -> list of token probabilities


def collect_raw(text: str) -> Raw:
    sents = split_sentences(text)
    classify_qualifying(text, sents)
    q = [i for i, s in enumerate(sents) if s.kind == "prose"]
    qs = [sents[i] for i in q]
    windows = build_paragraph_windows(qs) if qs else []
    if not windows:
        return Raw(text, sents, q, [], [], [], [], {}, {})
    def window_text(a: int, b: int) -> tuple[str, bool]:
        """Contiguous slice of the document when nothing excluded sits in between
        (so token offsets map back), otherwise the prose sentences joined."""
        gap_has_text = any(text[qs[k].end:qs[k + 1].start].strip() for k in range(a, b - 1))
        if gap_has_text:
            return " ".join(s.text for s in qs[a:b]), False
        return text[qs[a].start:qs[b - 1].end], True

    built = [window_text(a, b) for a, b in windows]
    win_texts = [t for t, _ in built]
    m = get_models()
    with m.lock:
        pcs = m.classifier_probs(win_texts)
        bino = m.binoculars(win_texts)
        long_idx = [i for i in q if sents[i].words >= 12]
        sent_pc = dict(zip(long_idx, m.classifier_probs([sents[i].text for i in long_idx])))
    token_probs: dict[int, list[float]] = {}
    for wi, (a, b) in enumerate(windows):
        if not built[wi][1]:
            continue  # offsets only map back for contiguous windows
        base = qs[a].start
        for s, e, p in bino[wi][1]:
            token_probs.setdefault(base + s, []).append(p)
    return Raw(text, sents, q, windows, pcs, [b[0] for b in bino],
               [style_score(qs[a:b]) for a, b in windows], sent_pc, token_probs)


def score_raw(raw: Raw, mode: str) -> tuple[list, list]:
    """Return (sentence scores aligned with raw.sents, None for excluded), window scores."""
    win_scores = [fuse(pc, bino_prob(bs), ps, mode) for pc, bs, ps in zip(raw.pcs, raw.bino, raw.styles)]
    scores: list = [None] * len(raw.sents)
    for qi, si in enumerate(raw.q):
        covering = [win_scores[w] for w, (a, b) in enumerate(raw.windows) if a <= qi < b]
        if not covering:
            continue
        pooled = sum(covering) / len(covering)
        if si in raw.sent_pc:
            pooled = 0.75 * pooled + 0.25 * calibrated_cls(raw.sent_pc[si], mode)
        scores[si] = pooled
    return scores, win_scores


def analyze(text: str, mode: str = DEFAULT_MODE, with_suggestions: bool = True) -> dict:
    if mode not in MODES:
        mode = DEFAULT_MODE
    text = text.replace("\r\n", "\n")[: config.MAX_CHARS]
    raw = collect_raw(text)
    sents = raw.sents
    thr = MODES[mode]["threshold"]
    q_words = sum(sents[i].words for i in raw.q)
    excluded_words = sum(s.words for s in sents) - q_words
    warnings = []
    if q_words < config.MIN_RELIABLE_WORDS:
        warnings.append(
            f"Only {q_words} words of prose to score. Turnitin won't give an AI score below 300 words, so treat this result as a rough guide."
        )
    if not raw.windows:
        return {"ai_percent": 0, "mode": mode, "word_count": 0, "excluded_words": excluded_words,
                "sentences": [], "paragraphs": [], "suggestions": [], "warnings": warnings,
                "signals": {}, "threshold": thr, "low_confidence": False}

    sent_scores, win_scores = score_raw(raw, mode)
    flagged_words = sum(s.words for s, sc in zip(sents, sent_scores) if sc is not None and sc >= thr)
    ai_percent = round(100 * flagged_words / q_words) if q_words else 0

    n = len(raw.windows)
    result = {
        "ai_percent": ai_percent,
        # Turnitin shows 1-19% as "*%" because false positives are more likely in that range
        "low_confidence": 0 < ai_percent < 20,
        "mode": mode,
        "threshold": thr,
        "word_count": q_words,
        "excluded_words": excluded_words,
        "warnings": warnings,
        "signals": {
            "classifier": round(sum(raw.pcs) / n, 3),
            "binoculars": round(sum(bino_prob(b) for b in raw.bino) / n, 3),
            "binoculars_raw": round(sum(raw.bino) / n, 4),
            "style": round(sum(raw.styles) / n, 3),
            "document_score": round(sum(win_scores) / n, 3),
        },
        "sentences": [
            {"start": s.start, "end": s.end, "kind": s.kind,
             "score": None if sc is None else round(sc, 3),
             "flagged": sc is not None and sc >= thr}
            for s, sc in zip(sents, sent_scores)
        ],
        "paragraphs": paragraph_scores(text, sents, sent_scores, thr),
        "suggestions": [],
    }
    if with_suggestions:
        result["suggestions"] = build_suggestions(text, sents, sent_scores, thr, raw.token_probs)
    return result


# --------------------------------------------------------------------------- paragraphs
def paragraph_scores(text, sents, sent_scores, thr):
    """Per-paragraph AI % (share of words in flagged sentences) and mean score."""
    paras = []
    for pm in re.finditer(r"[^\n]+", text):
        a, b = pm.start(), pm.end()
        inside_all = [(s, sc) for s, sc in zip(sents, sent_scores) if s.start >= a and s.end <= b]
        if not inside_all:
            continue
        # trim surrounding whitespace so offsets match sentence spans
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        inside = [(s, sc) for s, sc in inside_all if sc is not None]
        words = sum(s.words for s, _ in inside)
        if not words:
            # nothing scoreable here: a heading, list item, reference, quote...
            kind = max((s.kind for s, _ in inside_all), key=[s.kind for s, _ in inside_all].count)
            paras.append({"start": a, "end": b, "words": 0, "ai_percent": None, "score": None,
                          "excluded": EXCLUDED_LABELS.get(kind, "Not counted"), "reasons": []})
            continue
        flagged = sum(s.words for s, sc in inside if sc >= thr)
        mean = sum(s.words * sc for s, sc in inside) / words
        paras.append({
            "start": a, "end": b, "words": words,
            "ai_percent": round(100 * flagged / words), "score": round(mean, 3),
            "reasons": paragraph_reasons(text[a:b], [s for s, _ in inside]),
        })
    return paras


TRANSITIONS = re.compile(
    r"^(moreover|furthermore|additionally|however|ultimately|overall|in conclusion|in summary|"
    r"consequently|therefore|thus|nevertheless|nonetheless|notably|importantly|in addition|"
    r"as a result|similarly|likewise|firstly|secondly|lastly|finally)\b",
    re.I,
)


def paragraph_reasons(ptext: str, psents: list[Sentence]) -> list[str]:
    """Plain-language explanations of what makes a paragraph read as AI, each with a fix."""
    out = []
    hits = [h[2] for h in find_lexicon_hits(ptext)]
    if hits:
        uniq = list(dict.fromkeys(h.lower() for h in hits))[:5]
        out.append(f"Uses {len(hits)} AI-typical word(s)/phrase(s): {', '.join(uniq)}. Swap them for plain words.")
    openers = [s.text.split(",")[0] for s in psents if TRANSITIONS.match(s.text)]
    if len(openers) >= 2 or (openers and len(psents) <= 3):
        out.append(f"{len(openers)} sentence(s) open with stock transitions ({', '.join(dict.fromkeys(openers))}). Start some sentences with the subject instead, or drop the transition.")
    lens = [s.words for s in psents]
    if len(lens) >= 3 and statistics.mean(lens) > 0:
        cv = statistics.pstdev(lens) / statistics.mean(lens)
        if cv < 0.35:
            out.append(f"Sentences are all a similar length ({min(lens)}-{max(lens)} words). Human writing varies more: add a short sentence or split a long one.")
    if not re.search(r"\d", ptext) and not re.search(r"(?<=[a-z,;] )[A-Z][a-z]+", ptext):
        out.append("No specific examples, numbers, names or sources. Add a concrete detail from your own research or experience.")
    if re.search(r"\b(not only|both)\b.+\bbut also\b", ptext, re.I) or len(re.findall(r"\b\w+, \w+,? and \w+\b", ptext)) >= 2:
        out.append("Uses formulaic patterns (\"not only... but also\", lists of three). Rephrase one in your own way.")
    if psents and statistics.mean(lens) > 24:
        out.append("Long, evenly balanced sentences. Break one up.")
    return out


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
        if sc is None:
            continue  # headings, references, quotes etc. are not suggested
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
        if sc is None or sc < thr * 0.8:
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
