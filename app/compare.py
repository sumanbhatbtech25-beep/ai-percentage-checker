"""Compare the user's project with each similar paper.

For every paper we work out, grounded in the actual text:
  * what the paper covers that the project doesn't (abstract sentences with no
    close match anywhere in the project, plus key terms only the paper uses)
  * what the project covers that the paper doesn't
Those facts are handed to the local instruction model (Qwen2.5-1.5B-Instruct,
already loaded for Binoculars), which writes short "How your project differs"
and "What you can improve" bullets. If the model's answer can't be parsed, the
grounded facts are shown directly instead.

Only abstracts are available (full papers are usually paywalled), so every
comparison is labelled as abstract-based.
"""
from __future__ import annotations

import json
import re

import torch

from .detector import classify_qualifying, split_sentences
from .lexicon import STOPWORDS
from .models import DEVICE, get_models
from .papers import embed, keywords

SYSTEM = (
    "You help a university student compare their own project report with a published research paper. "
    "Use only the information provided. Be specific: name the paper's actual methods, data, setting or findings. "
    "Never invent numbers, results or claims that are not in the text. Write plainly, in second person."
)


# Front-matter boilerplate (declarations, certificates) says nothing about the project's content.
_BOILERPLATE = re.compile(
    r"\b(hereby declare|certify that|bona ?fide|partial fulfil+ment|award of (the )?degree|submitted (by|to|in)|"
    r"academic year|to the best of (my|our) knowledge|carried out by)\b", re.I)


def _sentences(text: str, limit: int = 220) -> list[str]:
    sents = split_sentences(text)
    classify_qualifying(text, sents)
    return [s.text for s in sents if s.kind == "prose" and s.words >= 6 and not _BOILERPLATE.search(s.text)][:limit]


_CLAIM_WORDS = set("""focuses focus examines examine discusses discuss covers cover primarily mainly various including
specifically aspects analyzes analyses analyse analyze addresses address emphasizes emphasises explores explore narrows
broader range topics while whereas project paper studies looks""".split())


def _misattributes(sentence: str, project_text: str, paper_text: str) -> bool:
    """True when the 'your project ...' part of a difference uses terms that only the paper contains."""
    parts = re.split(r"\b(?:while|whereas|but|unlike)\b", sentence, flags=re.I)
    proj = [p for p in parts if re.search(r"\b(your|the) project\b", p, re.I)]
    if not proj:
        return False
    pt, pp = project_text.lower(), paper_text.lower()
    for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", proj[0]):
        lw = w.lower()
        if lw in _CLAIM_WORDS or lw in _VAGUE or lw in STOPWORDS:
            continue
        if lw in pp and lw not in pt:
            return True
    return False


def _project_digest(doc_sents: list[str], doc_vecs: torch.Tensor, k: int = 7) -> list[str]:
    """The k most central sentences of the project, in their original order."""
    if len(doc_sents) <= k:
        return doc_sents
    centroid = torch.nn.functional.normalize(doc_vecs.mean(0), dim=-1)
    top = torch.topk(doc_vecs @ centroid, k).indices.sort().values.tolist()
    return [doc_sents[i] for i in top]


# Words too general to describe how two pieces of writing differ.
_VAGUE = set("""people economy economic economics businesses business increasing increase increases decrease
evidence economists government governments things way ways time times example examples result results
paper project study research analysis approach effect effects impact impacts role level levels""".split())


def _topics(text: str, k: int) -> list[str]:
    return [t for t in keywords(text, k=k + 10) if t.lower() not in _VAGUE and len(t) > 2][:k]


def _grounding(doc_text, doc_sents, doc_vecs, paper):
    abstract = paper.get("abstract_full") or paper.get("abstract") or ""
    p_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.split()) >= 6][:14]
    paper_only, shared = [], []
    if p_sents and len(doc_sents):
        p_vecs = embed(p_sents)
        best = (p_vecs @ doc_vecs.T).max(dim=1).values.tolist()
        for s, b in zip(p_sents, best):
            (shared if b >= 0.80 else paper_only).append((b, s))
        paper_only.sort()  # least covered first
    doc_kw = {k.lower() for k in _topics(doc_text, 25)}
    pap_kw = _topics(paper["title"] + ". " + abstract, 14)
    paper_terms = [k for k in pap_kw if k.lower() not in doc_kw and k.lower() not in doc_text.lower()][:6]
    blob = (paper["title"] + " " + abstract).lower()
    project_terms = [k for k in _topics(doc_text, 14) if k.lower() not in blob][:6]
    return {
        "paper_only": [s for _, s in paper_only[:3]],
        "shared": [s for _, s in sorted(shared, reverse=True)[:2]],
        "paper_terms": paper_terms,
        "project_terms": project_terms,
        "abstract": abstract,
    }


def _prompt(tok, digest, paper, g, project_topics):
    hints = []
    if g["paper_only"]:
        hints.append("Parts of the paper the project does NOT cover:\n" + "\n".join(f"- {s}" for s in g["paper_only"]))
    if g["paper_terms"]:
        hints.append("Topics only the paper mentions: " + ", ".join(g["paper_terms"]))
    if g["project_terms"]:
        hints.append("Topics only the student's project mentions: " + ", ".join(g["project_terms"]))
    user = (
        "STUDENT PROJECT\nMain topics: " + ", ".join(project_topics) + "\nKey sentences:\n" + "\n".join(f"- {s}" for s in digest) +
        f"\n\nRESEARCH PAPER: {paper['title']} ({paper.get('year') or 'n.d.'})\nAbstract: {g['abstract'][:1600]}\n\n" +
        "\n\n".join(hints) +
        "\n\nReply with JSON only, exactly in this format:\n"
        '{"differences": ["...", "..."], "improvements": ["...", "..."]}\n'
        "Rules:\n"
        "- differences: 2-3 real differences in focus, method, data, country or time period. Begin each sentence with the words Your project or The paper. "
        "Only state what the text above shows; never guess what the project 'might' or 'should' do.\n"
        "- improvements: 2-3 specific ideas the student can take from THIS paper. Each must name a concrete method, variable, "
        "finding or concept from the abstract (for example a model, a measure, a comparison group). Start each with a verb.\n"
        "- Do NOT give generic advice such as 'cite examples', 'compare your findings', 'use graphs', 'add more data'.\n"
        "- Each item is one sentence under 30 words."
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# Bullets that add nothing (generic study tips) or that guess about the project.
_GENERIC = re.compile(
    r"\b(cite (specific )?(examples|studies|the paper)|compare your findings|compare (the|your) (findings|results)|"
    r"use (graphs|charts|visuals)|add more (data|examples|details)|provide more (data|examples|details)|"
    r"strengthen your (understanding|arguments?)|support your arguments?|(missed|overlooked) (key|potential))\b",
    re.I,
)
_GUESS = re.compile(r"\byour project (might|may|could|should|would|probably|likely)\b|\bwhereas your project (might|may|should)\b", re.I)
_ECHO = re.compile(r"\b(methods, data,? and settings?|setting and findings|focus, method, data)\b", re.I)


def _clean(items, kind: str) -> list[str]:
    out = []
    for x in items:
        x = str(x).strip().strip("-• ").strip()
        if len(x.split()) < 5 or _ECHO.search(x) or (kind == "diff" and _GUESS.search(x)) or (kind == "imp" and _GENERIC.search(x)):
            continue
        out.append(x)
    return out[:3]


def _lenient_lists(out: str) -> dict:
    """Recover the two lists from almost-JSON (unbalanced quotes, missing commas, cut off)."""
    res = {}
    for key in ("differences", "improvements"):
        m = re.search(rf'"?{key}"?\s*:\s*\[(.*?)(\]|$|"?(differences|improvements)"?\s*:)', out, re.S)
        items = []
        if m:
            for line in re.split(r"\n|\",\s*\"", m.group(1)):
                line = line.strip().strip(",").strip()
                line = re.sub(r'^"(Your project|The paper)"\s*', r"\1 ", line)  # "Your project" focuses -> Your project focuses
                line = line.strip('"').replace('" ', " ").strip()
                if len(line.split()) >= 5:
                    items.append(line)
        res[key] = items
    return res


def _parse(out: str) -> dict | None:
    out = re.sub(r"```(?:json)?", "", out)
    out = re.sub(r'"(Your project|The paper)"\s+', r'"\1 ', out)  # repair the most common slip
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        raw = m.group(0)
        for attempt in (raw, re.sub(r",\s*([\]}])", r"\1", raw)):  # tolerate trailing commas
            try:
                d = json.loads(attempt)
                return {
                    "differences": _clean(d.get("differences", []), "diff"),
                    "improvements": _clean(d.get("improvements", []), "imp"),
                }
            except (json.JSONDecodeError, AttributeError):
                continue
    d = _lenient_lists(out)
    if not (d["differences"] or d["improvements"]):
        return None
    return {"differences": _clean(d["differences"], "diff"), "improvements": _clean(d["improvements"], "imp")}


def _fallback(g) -> dict:
    diff, imp = [], []
    if g["project_terms"]:
        diff.append("Your project covers " + ", ".join(g["project_terms"][:4]) + ", which this paper's abstract doesn't mention.")
    if g["paper_terms"]:
        diff.append("The paper also looks at " + ", ".join(g["paper_terms"][:4]) + ".")
    for s in g["paper_only"][:2]:
        imp.append(f"Consider covering this point from the paper: “{s}”")
    if g["paper_terms"]:
        imp.append("Look into " + ", ".join(g["paper_terms"][:3]) + " and cite this paper where relevant.")
    return {"differences": diff or ["Your project and this paper cover very similar ground."],
            "improvements": imp[:3] or ["Cite this paper to support your main argument."]}


@torch.inference_mode()
def compare(text: str, papers: list[dict]) -> list[dict]:
    doc_sents = _sentences(text)
    if len(doc_sents) < 3:
        return [{"error": "Paste or upload your project text to compare it with papers."} for _ in papers]
    doc_vecs = embed(doc_sents)
    digest = _project_digest(doc_sents, doc_vecs)
    prose = " ".join(doc_sents)  # cover page, headings, references left out
    grounds = [_grounding(prose, doc_sents, doc_vecs, p) for p in papers]
    project_topics = _topics(prose, 12)

    m = get_models()
    tok, model = m.lm_tok, m.performer
    prompts = [_prompt(tok, digest, p, g, project_topics) for p, g in zip(papers, grounds)]
    with m.lock:
        tok.padding_side = "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        finally:
            tok.padding_side = "right"
        gen = model.generate(
            **enc, do_sample=False, max_new_tokens=300, repetition_penalty=1.08,
            pad_token_id=tok.pad_token_id,
        )
        outs = [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True) for s in gen]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    results = []
    for out, g in zip(outs, grounds):
        parsed = _parse(out) or {"differences": [], "improvements": []}
        paper_text = g["abstract"]
        parsed["differences"] = [d for d in parsed["differences"] if not _misattributes(d, prose, paper_text)]
        fb = _fallback(g)
        # top up from the grounded facts when filtering removed too much
        res = {
            "differences": parsed["differences"] or fb["differences"],
            "improvements": parsed["improvements"] or fb["improvements"],
        }
        res["source"] = "ai" if parsed["differences"] or parsed["improvements"] else "extracted"
        res["paper_only_terms"] = g["paper_terms"]
        res["shared"] = g["shared"][:1]
        results.append(res)
    return results
