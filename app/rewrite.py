"""Paragraph rewrite suggestions.

1. The instruction-tuned LLM (already loaded as the Binoculars "performer")
   writes several rewrites of the paragraph, each with a different style brief.
   The style briefs steer it away from AI-typical vocabulary.
2. Every candidate is scored with the same detector pipeline the user sees.
3. Candidates come back ranked by AI %. Rewrites that don't beat the original
   are still returned but marked so the UI can say so.
"""
from __future__ import annotations

import re

import torch

from .detector import analyze
from .models import DEVICE, get_models

SYSTEM = (
    "You are an editor helping a university student revise their own paragraph. "
    "You keep every fact, claim, name, number and citation exactly, and you never add new information."
)

STYLES = [
    (
        "plain",
        "Rewrite the paragraph in plain, direct English, the way a thoughtful student explains "
        "something in their own words. Mix short sentences with longer ones. Use everyday words "
        "instead of fancy ones. Do not start sentences with transition words like Moreover, "
        "Furthermore, Additionally, Overall or In conclusion.",
    ),
    (
        "concrete",
        "Rewrite the paragraph so it sounds less generic: prefer concrete, specific wording and the "
        "active voice, cut filler phrases, and vary the rhythm (at least one short sentence). "
        "Change the sentence structure, not just the words. Keep a natural academic tone.",
    ),
    (
        "restructure",
        "Rewrite the paragraph by reordering and combining ideas: merge some sentences, split "
        "others, and change how each sentence opens. Keep it formal but natural, avoid buzzwords "
        "and stock phrases, and do not use em dashes.",
    ),
]

_PREAMBLE = re.compile(r"^(here(?:'s| is)[^:\n]*:|sure[^:\n]*:|rewritten paragraph:|revised paragraph:)\s*", re.I)


def _clean(out: str) -> str:
    out = out.strip()
    out = _PREAMBLE.sub("", out).strip()
    out = out.strip('"“”').strip()
    out = re.sub(r"\s*\n+\s*", " ", out)  # a paragraph stays one paragraph
    out = re.sub(r"\s{2,}", " ", out)
    return out


@torch.inference_mode()
def _generate(paragraph: str, per_style: int) -> list[str]:
    m = get_models()
    tok, model = m.lm_tok, m.performer
    n_tokens = len(tok(paragraph).input_ids)
    prompts = []
    for _, brief in STYLES:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{brief}\nReply with only the rewritten paragraph.\n\nParagraph:\n{paragraph}"},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    outs = []
    with m.lock:
        tok.padding_side = "left"  # all styles in one batched generate call
        try:
            enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        finally:
            tok.padding_side = "right"
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=0.95,
            top_p=0.95,
            repetition_penalty=1.05,
            max_new_tokens=int(n_tokens * 1.6) + 40,
            num_return_sequences=per_style,
            # Note: blocking lexicon words via bad_words_ids made the model emit
            # broken fragments ("fosts"), so we let the detector ranking handle them.
            pad_token_id=tok.pad_token_id,
        )
        for seq in gen:
            outs.append(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return outs


def rewrite_paragraph(paragraph: str, mode: str = "very_hard", per_style: int = 1) -> dict:
    paragraph = " ".join(paragraph.split())
    orig = analyze(paragraph, mode, with_suggestions=False)
    n_words = orig["word_count"]

    seen, candidates = set(), []
    for raw in _generate(paragraph, per_style):
        text = _clean(raw)
        n = len(text.split())
        if not text or text in seen or n < 0.5 * n_words or n > 1.8 * n_words + 10:
            continue
        seen.add(text)
        res = analyze(text, mode, with_suggestions=False)
        candidates.append({
            "text": text,
            "ai_percent": res["ai_percent"],
            "score": res["signals"].get("document_score", 0),
            "signals": res["signals"],
        })

    candidates.sort(key=lambda c: (c["ai_percent"], c["score"]))
    for c in candidates:
        c["note"] = (
            "AI-generated rewrite: the deep classifier still recognises it as AI text. "
            "Use it as a starting point and put it in your own words."
            if c["signals"].get("classifier", 0) >= 0.5 else ""
        )
    for c in candidates:
        c["better"] = (c["ai_percent"], c["score"]) < (orig["ai_percent"], orig["signals"].get("document_score", 1))
    return {
        "original": {
            "ai_percent": orig["ai_percent"],
            "score": orig["signals"].get("document_score", 0),
            "signals": orig["signals"],
        },
        "candidates": candidates[:4],
    }
