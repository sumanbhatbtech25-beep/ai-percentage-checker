"""Central configuration. Every value can be overridden with an environment variable."""
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Models -----------------------------------------------------------------
# Supervised classifier: DeBERTa-v3-large fine-tuned for AI-text detection
# (one of the strongest open detectors on the RAID benchmark).
CLASSIFIER_MODEL = _env("AIC_CLASSIFIER", "desklib/ai-text-detector-v1.01")

# Zero-shot "Binoculars" detector (Hans et al., ICML 2024). It needs a base
# model (observer) and its instruction-tuned sibling (performer) that share a
# tokenizer. The paper used Falcon-7B; we default to Qwen2.5-1.5B so it fits an
# 8 GB laptop GPU. Use Qwen/Qwen2.5-0.5B(-Instruct) if you run out of VRAM.
OBSERVER_MODEL = _env("AIC_OBSERVER", "Qwen/Qwen2.5-1.5B")
PERFORMER_MODEL = _env("AIC_PERFORMER", "Qwen/Qwen2.5-1.5B-Instruct")

# Masked LM used to propose context-aware replacement words.
MLM_MODEL = _env("AIC_MLM", "roberta-base")

# --- Binoculars calibration ---------------------------------------------------
# Binoculars score = log-perplexity / cross-perplexity. LOWER = more AI-like.
# These defaults were produced by `python -m app.calibrate` for the Qwen pair;
# re-run calibration if you change the observer/performer models.
BINO_THRESHOLD = float(_env("AIC_BINO_THRESHOLD", "0.92"))
BINO_SCALE = float(_env("AIC_BINO_SCALE", "0.045"))

# --- Segmentation (mirrors Turnitin's published approach) --------------------
# Text is cut into overlapping windows of a few hundred words; every sentence
# inherits the pooled score of all windows that contain it.
SEGMENT_WORDS = int(_env("AIC_SEGMENT_WORDS", "200"))
SEGMENT_STRIDE_WORDS = int(_env("AIC_SEGMENT_STRIDE", "100"))
MIN_RELIABLE_WORDS = 150

MAX_CHARS = 120_000  # ~20k words
