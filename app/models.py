"""Model loading and the raw scoring primitives.

Detectors (chosen in Nov 2026 by app/compare_detectors.py on human writing vs
2022-2026 AI text, including Claude Opus 5, GPT-4.1, gpt-oss and Qwen 3.8):
  * tropa-mini - DeBERTa-v3-large, 4 classes: human / ai / ai_edited / humanized
  * Vanguard   - ModernBERT-large, 2nd place at PAN-CLEF 2026 AI-text detection
Their scores are averaged (in logit space); see detector.py.

Helpers:
  * writer (Qwen2.5-1.5B-Instruct) - per-token predictability for word
    suggestions, paragraph rewrites, paper comparisons
  * masked LM (RoBERTa)            - context-aware replacement words
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# The paper-matching embedder runs on the CPU (it's not on the analysis path),
# which keeps the 8 GB GPU free for the detectors and avoids out-of-memory errors.
HELPER_DEVICE = "cpu"
MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "spm.model", "vocab.json", "merges.txt"]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class MeanPoolDetector(nn.Module):
    """desklib-style detector: DeBERTa encoder + mean pooling + linear head."""

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
        real_missing = [k for k in missing if "position_ids" not in k]
        if real_missing:
            raise RuntimeError(f"Detector weights missing: {real_missing[:5]}")

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.classifier((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))


class Models:
    def __init__(self):
        self.lock = threading.Lock()
        print(f"[models] device={DEVICE} dtype={DTYPE}")

        print(f"[models] loading detector {config.TROPA_MODEL}")
        path = snapshot_download(config.TROPA_MODEL, allow_patterns=MODEL_FILES)
        self.tropa_tok = AutoTokenizer.from_pretrained(path)
        self.tropa = MeanPoolDetector(path, 4).to(DEVICE, DTYPE).eval()  # human, ai, ai_edited, humanized

        print(f"[models] loading detector {config.VANGUARD_MODEL}")
        path = snapshot_download(config.VANGUARD_MODEL, allow_patterns=MODEL_FILES)
        self.vanguard_tok = AutoTokenizer.from_pretrained(path)
        self.vanguard = AutoModelForSequenceClassification.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()

        print(f"[models] loading writer model {config.WRITER_MODEL}")
        self.lm_tok = AutoTokenizer.from_pretrained(config.WRITER_MODEL)
        if self.lm_tok.pad_token is None:
            self.lm_tok.pad_token = self.lm_tok.eos_token
        self.lm_tok.padding_side = "right"
        self.performer = AutoModelForCausalLM.from_pretrained(config.WRITER_MODEL, dtype=DTYPE).to(DEVICE).eval()

        print(f"[models] loading masked LM {config.MLM_MODEL}")
        self.mlm_tok = AutoTokenizer.from_pretrained(config.MLM_MODEL)
        # small (~250 MB) and used on every analysis, so it stays on the GPU for speed
        self.mlm = AutoModelForMaskedLM.from_pretrained(config.MLM_MODEL, dtype=DTYPE).to(DEVICE).eval()
        print("[models] ready")

    # ------------------------------------------------------------------ detectors
    @torch.inference_mode()
    def detect(self, texts: list[str], batch_size: int = 8) -> list[dict]:
        """Per text: P(AI) from each detector, tropa's sub-classes, and the fused logit z."""
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tropa_tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            tp = torch.softmax(self.tropa(enc["input_ids"], enc["attention_mask"]).float(), -1)
            enc = self.vanguard_tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            vl = self.vanguard(**enc).logits.float()
            vp = torch.sigmoid(vl[:, 0]) if vl.shape[1] == 1 else torch.softmax(vl, -1)[:, 1]
            for t, v in zip(tp.tolist(), vp.tolist()):
                p_tropa = 1 - t[0]
                out.append({
                    "tropa": p_tropa, "vanguard": v,
                    "ai": t[1], "ai_edited": t[2], "humanized": t[3],
                    "z": (_logit(p_tropa) + _logit(v)) / 2,
                })
        return out

    # ------------------------------------------------------------------ word predictability
    @torch.inference_mode()
    def token_probs(self, texts: list[str], batch_size: int = 2, max_tokens: int = 512):
        """[(char_start, char_end, probability), ...] per text: how predictable each token is."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.lm_tok(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=max_tokens, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            enc = enc.to(DEVICE)
            logits = self.performer(**enc).logits[:, :-1].float()
            targets = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:]
            probs = F.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).exp().cpu()
            for b in range(len(batch)):
                n = int(mask[b].sum().item())
                results.append([(*offsets[b, t + 1].tolist(), float(probs[b, t])) for t in range(n)])
            del logits
        return results

    # ------------------------------------------------------------------ masked LM
    @torch.inference_mode()
    def mlm_candidates(self, items: list[tuple[str, int, int]], top_k: int = 60, batch_size: int = 32):
        """items = [(context, word_start, word_end)] -> list of [(candidate, score)]

        Lexical substitution: a candidate must fit the gap (masked prediction)
        AND be close in meaning to the original word (prediction at the
        unmasked position, which concentrates on synonyms of the original).
        """
        out = []
        mask_tok = self.mlm_tok.mask_token
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            masked = [s[:a] + mask_tok + s[b:] for s, a, b in chunk]
            enc_m = self.mlm_tok(masked, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
            enc_u = self.mlm_tok(
                [s for s, _, _ in chunk], return_tensors="pt", padding=True, truncation=True,
                max_length=256, return_offsets_mapping=True,
            )
            offs = enc_u.pop("offset_mapping")
            enc_u = enc_u.to(DEVICE)
            lm = F.log_softmax(self.mlm(**enc_m).logits.float(), dim=-1)
            lu = F.log_softmax(self.mlm(**enc_u).logits.float(), dim=-1)
            for j, (_, a, b) in enumerate(chunk):
                pos = (enc_m["input_ids"][j] == self.mlm_tok.mask_token_id).nonzero()
                upos = [t for t, (s, e) in enumerate(offs[j].tolist()) if e > a and s < b]
                if len(pos) == 0 or not upos:
                    out.append([])
                    continue
                score = lm[j, pos[0, 0]] + 0.6 * lu[j, upos[0]]
                top = torch.topk(score, top_k)
                cands = []
                for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
                    tok = self.mlm_tok.convert_ids_to_tokens(idx)
                    if not tok.startswith("Ġ"):  # must be a whole word preceded by space
                        continue
                    cands.append((tok[1:], sc))
                out.append(cands)
        return out


_models: Models | None = None
_load_lock = threading.Lock()


def get_models() -> Models:
    global _models
    with _load_lock:
        if _models is None:
            _models = Models()
    return _models


def free_gpu() -> None:
    """Return cached GPU memory after a heavy job so the next one has room."""
    if DEVICE == "cuda":
        import gc

        gc.collect()
        torch.cuda.empty_cache()


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
