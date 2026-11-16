"""Central configuration. Every value can be overridden with an environment variable."""
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Detectors ----------------------------------------------------------------
# Chosen by app/compare_detectors.py (Nov 2026): on held-out tests the average of
# these two caught 100% of 2026-model AI text (Claude Opus 5, GPT-4.1, gpt-oss,
# Claude Haiku 4.5, Qwen 3.8) and 100% of ChatGPT text while flagging 0% of
# human writing. The previous detectors (desklib + Binoculars) caught 55-61%.
TROPA_MODEL = _env("AIC_TROPA", "wasitaigeneratedcom/tropa-mini-ai-text-detector")  # Apache-2.0
VANGUARD_MODEL = _env("AIC_VANGUARD", "ShantanuT01/vanguard-ai-text-detector")      # MIT

# Instruction-tuned writer: word predictability, paragraph rewrites, paper comparisons.
WRITER_MODEL = _env("AIC_WRITER", "Qwen/Qwen2.5-1.5B-Instruct")

# Masked LM used to propose context-aware replacement words.
MLM_MODEL = _env("AIC_MLM", "roberta-base")

# --- Segmentation (mirrors Turnitin's published approach) --------------------
# Text is cut into overlapping windows of a few hundred words; every sentence
# inherits the pooled score of all windows that contain it.
SEGMENT_WORDS = int(_env("AIC_SEGMENT_WORDS", "200"))
SEGMENT_STRIDE_WORDS = int(_env("AIC_SEGMENT_STRIDE", "100"))
MIN_RELIABLE_WORDS = 150

MAX_CHARS = 120_000  # ~20k words
