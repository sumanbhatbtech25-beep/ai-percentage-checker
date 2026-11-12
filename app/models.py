"""Model loading and the three raw scoring primitives.

1. classifier_probs  - supervised DeBERTa-v3-large detector (desklib)
2. binoculars        - zero-shot Binoculars score + per-token probabilities
3. mlm_candidates    - masked-LM replacement candidates for a word in context
"""
from __future__ import annotations

import glob
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class DesklibDetector(nn.Module):
    """DeBERTa encoder + mean pooling + single-logit head (matches the model card)."""

    def __init__(self, model_id: str):
        super().__init__()
        path = snapshot_download(model_id)
        cfg = AutoConfig.from_pretrained(path)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, 1)
        files = glob.glob(os.path.join(path, "*.safetensors"))
        if files:
            from safetensors.torch import load_file

            state = {}
            for f in files:
                state.update(load_file(f))
        else:
            state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if "position_ids" not in k]
        if real_missing:
            raise RuntimeError(f"Classifier weights missing: {real_missing[:5]}")

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled).squeeze(-1)


class Models:
    def __init__(self):
        self.lock = threading.Lock()
        print(f"[models] device={DEVICE} dtype={DTYPE}")

        print(f"[models] loading classifier {config.CLASSIFIER_MODEL}")
        self.cls_tok = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
        self.cls = DesklibDetector(config.CLASSIFIER_MODEL).to(DEVICE, DTYPE).eval()

        print(f"[models] loading Binoculars pair {config.OBSERVER_MODEL} / {config.PERFORMER_MODEL}")
        self.lm_tok = AutoTokenizer.from_pretrained(config.OBSERVER_MODEL)
        if self.lm_tok.pad_token is None:
            self.lm_tok.pad_token = self.lm_tok.eos_token
        self.lm_tok.padding_side = "right"
        self.observer = AutoModelForCausalLM.from_pretrained(config.OBSERVER_MODEL, dtype=DTYPE).to(DEVICE).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(config.PERFORMER_MODEL, dtype=DTYPE).to(DEVICE).eval()

        print(f"[models] loading masked LM {config.MLM_MODEL}")
        self.mlm_tok = AutoTokenizer.from_pretrained(config.MLM_MODEL)
        self.mlm = AutoModelForMaskedLM.from_pretrained(config.MLM_MODEL, dtype=DTYPE).to(DEVICE).eval()
        print("[models] ready")

    # ------------------------------------------------------------------ classifier
    @torch.inference_mode()
    def classifier_probs(self, texts: list[str], batch_size: int = 8) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            enc = self.cls_tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=768,
                return_tensors="pt",
            ).to(DEVICE)
            logits = self.cls(enc["input_ids"], enc["attention_mask"]).float()
            out.extend(torch.sigmoid(logits).tolist())
        return out

    # ------------------------------------------------------------------ binoculars
    @torch.inference_mode()
    def binoculars(self, texts: list[str], batch_size: int = 2, max_tokens: int = 512):
        """Return list of (score, token_info) where token_info is
        [(char_start, char_end, prob_under_performer), ...] for each predicted token."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.lm_tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            enc = enc.to(DEVICE)
            obs_logits = self.observer(**enc).logits[:, :-1].float()
            perf_logits = self.performer(**enc).logits[:, :-1].float()
            targets = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()

            perf_logp = F.log_softmax(perf_logits, dim=-1)
            tok_logp = perf_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            log_ppl = -(tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)

            obs_p = F.softmax(obs_logits, dim=-1)
            x_ent = -(obs_p * perf_logp).sum(-1)
            log_xppl = (x_ent * mask).sum(1) / mask.sum(1).clamp(min=1)
            scores = (log_ppl / log_xppl).tolist()

            probs = tok_logp.exp().cpu()
            for b in range(len(batch)):
                info = []
                n = int(mask[b].sum().item())
                for t in range(n):
                    s, e = offsets[b, t + 1].tolist()
                    info.append((s, e, float(probs[b, t])))
                results.append((scores[b], info))
            del obs_logits, perf_logits, perf_logp, obs_p, x_ent
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
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
