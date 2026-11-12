"""Calibrate the Binoculars threshold and measure the full pipeline.

Uses the HC3 dataset (same questions answered by humans and by ChatGPT).
    python -m app.calibrate [--n 60]
Prints the threshold that gives ~1% false positives on human text, then the
AI % the full pipeline reports for human vs AI texts in each strictness mode.
"""
import argparse
import json
import random
import statistics
import urllib.request

from . import config, detector
from .models import get_models

SPLITS = ["finance", "open_qa", "wiki_csai", "medicine", "reddit_eli5"]
URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{}.jsonl"


def load(n: int):
    human, ai = [], []
    for split in SPLITS:
        raw = urllib.request.urlopen(URL.format(split), timeout=60).read().decode("utf-8")
        for line in raw.splitlines():
            row = json.loads(line)
            for h in row.get("human_answers", []):
                if 120 <= len(h.split()) <= 400:
                    human.append(h.strip())
            for a in row.get("chatgpt_answers", []):
                if 120 <= len(a.split()) <= 400:
                    ai.append(a.strip())
    random.seed(0)
    random.shuffle(human)
    random.shuffle(ai)
    return human[:n], ai[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    human, ai = load(args.n)
    print(f"loaded {len(human)} human / {len(ai)} AI samples")
    m = get_models()

    hb = [s for s, _ in m.binoculars(human)]
    ab = [s for s, _ in m.binoculars(ai)]
    hb_sorted = sorted(hb)
    fpr1 = hb_sorted[max(0, int(len(hb_sorted) * 0.01) - 1)]
    print(f"Binoculars human mean={statistics.mean(hb):.4f} sd={statistics.pstdev(hb):.4f} min={min(hb):.4f}")
    print(f"Binoculars AI    mean={statistics.mean(ab):.4f} sd={statistics.pstdev(ab):.4f} max={max(ab):.4f}")
    mid = (statistics.mean(hb) + statistics.mean(ab)) / 2
    print(f"suggested AIC_BINO_THRESHOLD (1% FPR) ~ {fpr1:.4f}; midpoint {mid:.4f}")
    caught = sum(s < fpr1 for s in ab) / len(ab)
    print(f"  -> AI recall at that threshold: {caught:.0%}")

    hc = m.classifier_probs(human)
    ac = m.classifier_probs(ai)
    print(f"Classifier: human>0.5 = {sum(p > .5 for p in hc)/len(hc):.0%}, AI>0.5 = {sum(p > .5 for p in ac)/len(ac):.0%}")

    for mode in detector.MODES:
        hp = [detector.analyze(t, mode, with_suggestions=False)["ai_percent"] for t in human]
        ap_ = [detector.analyze(t, mode, with_suggestions=False)["ai_percent"] for t in ai]
        print(
            f"[{mode:9}] human avg AI%={statistics.mean(hp):5.1f}  (flagged >=50%: {sum(x >= 50 for x in hp)/len(hp):.0%})"
            f" | AI avg AI%={statistics.mean(ap_):5.1f}  (flagged >=50%: {sum(x >= 50 for x in ap_)/len(ap_):.0%})"
        )


if __name__ == "__main__":
    main()
