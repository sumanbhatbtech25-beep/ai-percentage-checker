"""Grid-search the fusion settings on eval_cache.json (see app/evaluate.py).

Tunes on even-indexed documents, reports on odd-indexed (held-out) ones.
Targets follow Turnitin's published approach: a human document counts as a
false positive when it's reported at >= 20% AI.
    python -m app.tune
"""
import itertools
import json
import math

from .evaluate import CACHE
from .detector import bino_prob, sigmoid


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def make(center, wc, ws):
    lc = logit(center)

    def cal(p):  # classifier only "counts" once it is more confident than `center`
        return sigmoid(logit(p) - lc)

    def fuse(pc, pb, ps):
        return wc * cal(pc) + (1 - wc - ws) * pb + ws * ps

    return fuse, cal


def doc_pct(it, fuse, cal, thr):
    win = [fuse(pc, bino_prob(b), ps) for pc, b, ps in zip(it["pcs"], it["bino"], it["styles"])]
    flagged = total = 0
    for qi, si in enumerate(it["q"]):
        cov = [win[w] for w, (a, b) in enumerate(it["windows"]) if a <= qi < b]
        if not cov:
            continue
        sc = sum(cov) / len(cov)
        loc = it["sent_pc"].get(str(si))
        if loc is not None:
            sc = 0.75 * sc + 0.25 * cal(loc)
        w = it["words"][si]
        total += w
        flagged += w if sc >= thr else 0
    return 100 * flagged / total if total else 0


def evaluate(cache, fuse, cal, thr, part):
    res = {}
    for name, items in cache.items():
        sel = items[part::2]
        pcts = [doc_pct(it, fuse, cal, thr) for it in sel]
        res[name] = (sum(p >= 20 for p in pcts) / len(pcts), sum(pcts) / len(pcts))
    human = [k for k in res if k.startswith("human")]
    ai = [k for k in res if k.startswith("ai")]
    fpr = max(res[k][0] for k in human)          # worst human set
    tpr = min(res[k][0] for k in ai)             # worst AI set
    ai_avg = min(res[k][1] for k in ai)
    return fpr, tpr, ai_avg, res


def main():
    cache = json.loads(CACHE.read_text())
    grid = itertools.product([0.9, 0.95, 0.97, 0.98, 0.99], [0.5, 0.6, 0.7, 0.8], [0.0, 0.05], [x / 100 for x in range(30, 80, 2)])
    best = {}
    targets = {"standard": 0.02, "strict": 0.05, "very_hard": 0.10}
    for center, wc, ws, thr in grid:
        fuse, cal = make(center, wc, ws)
        fpr, tpr, ai_avg, _ = evaluate(cache, fuse, cal, thr, part=0)
        for mode, t in targets.items():
            if fpr <= t:
                key = (tpr, ai_avg)
                if mode not in best or key > best[mode][0]:
                    best[mode] = (key, (center, wc, ws, thr))
    for mode, (_, (center, wc, ws, thr)) in best.items():
        fuse, cal = make(center, wc, ws)
        fpr, tpr, ai_avg, res = evaluate(cache, fuse, cal, thr, part=1)  # held-out half
        print(f"\n{mode}: classifier_center={center} w_cls={wc} w_style={ws} threshold={thr}")
        print(f"  held-out: worst human FPR(>=20%)={fpr:.0%}  worst AI detection(>=20%)={tpr:.0%}  AI avg={ai_avg:.0f}%")
        for k, (r20, avg) in res.items():
            print(f"    {k:24} >=20%: {r20:5.0%}   avg AI%: {avg:5.1f}")


if __name__ == "__main__":
    main()
