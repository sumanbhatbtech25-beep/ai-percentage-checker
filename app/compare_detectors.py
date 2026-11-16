"""Head-to-head test of AI-text classifiers on the evaluation sets.

    python -m app.compare_detectors

Every text is cut into the same ~200-word paragraph windows the app uses. Each
detector scores each window; a document's AI % is the share of its words in
windows above the detector's threshold. The threshold is set per detector so
that at most 3% of documents in ANY human set reach 20% AI (the Balanced-mode
bar), and we report how much AI text is caught at that setting.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .detector import build_paragraph_windows, classify_qualifying, split_sentences

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ROOT / "eval_texts.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FP_BUDGET = 0.03
FILES = ["*.json", "*.safetensors", "*.txt", "spm.model", "vocab.json", "merges.txt"]


def load_texts() -> dict[str, list[str]]:
    if TEXTS.exists():
        return json.loads(TEXTS.read_text(encoding="utf-8"))
    from .evaluate import load_sets

    sets = load_sets(60)
    TEXTS.write_text(json.dumps(sets), encoding="utf-8")
    return sets


def windows(text: str) -> list[tuple[str, int]]:
    sents = split_sentences(text)
    classify_qualifying(text, sents)
    qs = [s for s in sents if s.kind == "prose"]
    return [(" ".join(s.text for s in qs[a:b]), sum(s.words for s in qs[a:b])) for a, b in build_paragraph_windows(qs)]


class _MeanPoolHead(nn.Module):
    """desklib-style model: encoder + mean pooling + linear head (1 or 4 outputs)."""

    def __init__(self, path: str, n_out: int):
        super().__init__()
        from safetensors.torch import load_file

        cfg = AutoConfig.from_pretrained(path)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, n_out)
        state = {}
        for f in Path(path).glob("*.safetensors"):
            state.update(load_file(str(f)))
        missing, _ = self.load_state_dict(state, strict=False)
        assert not [k for k in missing if "position_ids" not in k], missing[:5]

    def forward(self, ids, mask):
        h = self.model(input_ids=ids, attention_mask=mask)[0]
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


def scorer(name: str):
    """Return (fn(list[str]) -> list[P(AI)], cleanup)."""
    dtype = torch.bfloat16 if DEV == "cuda" else torch.float32
    if name in ("desklib (current)", "tropa-mini"):
        repo = "desklib/ai-text-detector-v1.01" if name.startswith("desklib") else "wasitaigeneratedcom/tropa-mini-ai-text-detector"
        path = snapshot_download(repo, allow_patterns=FILES)
        tok = AutoTokenizer.from_pretrained(path)
        n_out = 1 if name.startswith("desklib") else 4
        model = _MeanPoolHead(path, n_out).to(DEV, dtype).eval()

        @torch.inference_mode()
        def fn(texts):
            out = []
            for i in range(0, len(texts), 8):
                enc = tok(texts[i:i + 8], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
                logits = model(enc["input_ids"], enc["attention_mask"]).float()
                out += (torch.sigmoid(logits[:, 0]) if n_out == 1 else 1 - torch.softmax(logits, -1)[:, 0]).tolist()
            return out
        return fn, model

    repo = {"vanguard": "ShantanuT01/vanguard-ai-text-detector", "tmr": "Oxidane/tmr-ai-text-detector",
            "rasbt-modernbert": "rasbt/ai-text-detector-modernbert"}[name]
    path = snapshot_download(repo, allow_patterns=FILES)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype).to(DEV).eval()
    temp = 1.0
    if name == "rasbt-modernbert":
        meta = json.loads(Path(path, "detector-config.json").read_text())
        temp = float(meta.get("temperature", 1.0))

    @torch.inference_mode()
    def fn(texts):
        out = []
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            logits = model(**enc).logits.float()
            out += (torch.sigmoid(logits[:, 0]) if logits.shape[1] == 1 else torch.softmax(logits / temp, -1)[:, 1]).tolist()
        return out
    return fn, model


def doc_percent(win_scores, win_words, thr):
    total = sum(win_words)
    return 100 * sum(w for s, w in zip(win_scores, win_words) if s >= thr) / total if total else 0


def main():
    sets = load_texts()
    extra = {}
    for f in os.environ.get("AIC_EXTRA_DOCS", "").split(";"):
        if f.strip():
            from .extract import extract_text

            extra[Path(f).name] = extract_text(f, Path(f).read_bytes())
    wins = {k: [windows(t) for t in v] for k, v in sets.items()}
    xwins = {k: windows(t) for k, t in extra.items()}
    results = {}
    names = [n.strip() for n in os.environ.get("AIC_DETECTORS", "desklib (current),tropa-mini,vanguard,tmr,rasbt-modernbert").split(",")]
    dump = {"words": {k: [[w for _, w in d] for d in docs] for k, docs in wins.items()}, "extra_words": {k: [w for _, w in d] for k, d in xwins.items()}, "scores": {}}
    for name in names:
        fn, model = scorer(name)
        flat = [w for docs in wins.values() for d in docs for w, _ in d] + [w for d in xwins.values() for w, _ in d]
        scores = iter(fn(flat))
        per = {k: [[next(scores) for _ in d] for d in docs] for k, docs in wins.items()}
        xper = {k: [next(scores) for _ in d] for k, d in xwins.items()}
        # lowest threshold that keeps every human set within the false-positive budget
        thr = 0.999
        for t in [x / 1000 for x in range(500, 1000, 5)]:
            ok = all(sum(doc_percent(s, [w for _, w in d], t) >= 20 for s, d in zip(per[k], wins[k])) / len(per[k]) <= FP_BUDGET
                     for k in per if k.startswith("human"))
            if ok:
                thr = t
                break
        row = {"threshold": thr}
        for k in per:
            pcts = [doc_percent(s, [w for _, w in d], thr) for s, d in zip(per[k], wins[k])]
            row[k] = (sum(p >= 20 for p in pcts) / len(pcts), sum(pcts) / len(pcts))
        row["extra"] = {k: round(doc_percent(s, [w for _, w in xwins[k]], thr)) for k, s in xper.items()}
        results[name] = row
        dump["scores"][name] = {"sets": per, "extra": xper}
        print(f"\n### {name}  (threshold {thr})", flush=True)
        for k, v in row.items():
            if k not in ("threshold", "extra"):
                print(f"   {k:26} flagged>=20%: {v[0]:5.0%}   avg AI%: {v[1]:5.1f}")
        for k, v in row["extra"].items():
            print(f"   [your file] {k:40} AI%: {v}")
        del model, fn
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()
    (ROOT / "detector_comparison.json").write_text(json.dumps(results, indent=1))
    (ROOT / "detector_windows.json").write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
