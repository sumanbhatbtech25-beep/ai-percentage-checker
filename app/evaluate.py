"""Measure false positives / detection on real human and AI academic writing.

    python -m app.evaluate collect   # run the models once, cache raw signals (GPU, ~10 min)
    python -m app.evaluate report    # score the cache under every mode (instant)

Human sets (written before ChatGPT existed, so they're reliably human):
  - persuade student essays     (dmitva/human_ai_generated_text, human_text)
  - IvyPanda academic essays    (qwedsacf/ivypanda-essays)
  - arXiv abstracts, 2007-2014  (gfissore/arxiv-abstracts-2021)
AI sets:
  - essays written to the same prompts by LLMs (dmitva/human_ai_generated_text, ai_text)
  - ChatGPT answers             (Hello-SimpleAI/HC3)

A "false positive" follows Turnitin's definition: a human document reported
at 20% AI or more (below 20% Turnitin only shows an asterisk).
"""
from __future__ import annotations

import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "eval_cache.json"
ROWS = "https://datasets-server.huggingface.co/rows?"


def _rows(dataset: str, offset: int, length: int = 100) -> list[dict]:
    import time

    q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": length})
    for attempt in range(4):  # the datasets server sometimes answers 502 under load
        try:
            with urllib.request.urlopen(ROWS + q, timeout=60) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except Exception as e:  # noqa: BLE001
            print(f"  retry {dataset}@{offset}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return []


def _trim(text: str, max_words: int = 550) -> str:
    """Keep whole paragraphs up to ~max_words, so documents look like real submissions."""
    out, n = [], 0
    for para in re.split(r"\n\s*\n|\n", text.strip()):
        w = len(para.split())
        if not w:
            continue
        if n and n + w > max_words:
            break
        out.append(para.strip())
        n += w
    return "\n\n".join(out)


def load_sets(n: int = 60) -> dict[str, list[str]]:
    rnd = random.Random(7)
    sets: dict[str, list[str]] = {}

    rows = []
    for off in rnd.sample(range(0, 900000, 100), 3):
        rows += _rows("dmitva/human_ai_generated_text", off)
    rnd.shuffle(rows)
    sets["human_student_essays"] = [_trim(r["human_text"]) for r in rows if len(r["human_text"].split()) >= 200][:n]
    sets["ai_essays"] = [_trim(r["ai_text"]) for r in rows if len(r["ai_text"].split()) >= 200][:n]

    rows = []
    for off in rnd.sample(range(0, 120000, 100), 3):
        rows += _rows("qwedsacf/ivypanda-essays", off)
    sets["human_academic_essays"] = [
        _trim(re.sub(r"^.*?\n\n", "", r["TEXT"], count=1)) for r in rows if len(r["TEXT"].split()) >= 350
    ][:n]

    rows = []
    for off in rnd.sample(range(0, 600000, 100), 4):
        rows += _rows("gfissore/arxiv-abstracts-2021", off)
    sets["human_arxiv_abstracts"] = [
        " ".join(r["abstract"].split()) for r in rows
        if re.match(r"^(07|08|09|10|11|12|13|14)\d{2}\.", r["id"]) and len(r["abstract"].split()) >= 120
    ][:n]

    url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{}.jsonl"
    hc3 = []
    for split in ("finance", "wiki_csai", "open_qa"):
        for line in urllib.request.urlopen(url.format(split), timeout=60).read().decode().splitlines():
            for a in json.loads(line).get("chatgpt_answers", []):
                if len(a.split()) >= 150:
                    hc3.append(a.strip())
    rnd.shuffle(hc3)
    sets["ai_chatgpt_answers"] = hc3[:n]
    return sets


def collect(n: int = 60) -> None:
    from .detector import collect_raw

    sets = load_sets(n)
    cache = {}
    for name, texts in sets.items():
        print(f"{name}: {len(texts)} texts", flush=True)
        items = []
        for t in texts:
            raw = collect_raw(t)
            items.append({
                "words": [s.words for s in raw.sents],
                "q": raw.q,
                "windows": raw.windows,
                "pcs": raw.pcs,
                "bino": raw.bino,
                "styles": raw.styles,
                "sent_pc": {str(k): v for k, v in raw.sent_pc.items()},
            })
        cache[name] = items
    CACHE.write_text(json.dumps(cache))
    print(f"saved {CACHE}")


# ----------------------------------------------------------------------------- offline scoring
def doc_percent(item: dict, mode: str) -> int:
    """Same scoring as detector.score_raw/analyze, run on cached raw signals."""
    from .detector import MODES, bino_prob, calibrated_cls, fuse

    thr = MODES[mode]["threshold"]
    win = [fuse(pc, bino_prob(b), ps, mode) for pc, b, ps in zip(item["pcs"], item["bino"], item["styles"])]
    flagged = total = 0
    for qi, si in enumerate(item["q"]):
        cov = [win[w] for w, (a, b) in enumerate(item["windows"]) if a <= qi < b]
        if not cov:
            continue
        sc = sum(cov) / len(cov)
        loc = item["sent_pc"].get(str(si))
        if loc is not None:
            sc = 0.75 * sc + 0.25 * calibrated_cls(loc, mode)
        w = item["words"][si]
        total += w
        flagged += w if sc >= thr else 0
    return round(100 * flagged / total) if total else 0


def summarize(cache: dict, mode: str) -> dict:
    out = {}
    for name, items in cache.items():
        pcts = [doc_percent(it, mode) for it in items]
        out[name] = {
            "avg": sum(pcts) / len(pcts),
            "ge20": sum(p >= 20 for p in pcts) / len(pcts),
            "ge50": sum(p >= 50 for p in pcts) / len(pcts),
        }
    return out


def report() -> None:
    from .detector import MODES

    cache = json.loads(CACHE.read_text())
    print(f"{'mode':10} {'set':24} {'avg AI%':>8} {'>=20%':>7} {'>=50%':>7}")
    for mode in MODES:
        s = summarize(cache, mode)
        for name, v in s.items():
            print(f"{mode:10} {name:24} {v['avg']:8.1f} {v['ge20']:7.0%} {v['ge50']:7.0%}")
        print()


if __name__ == "__main__":
    {"collect": collect, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
