# Research notes: why AI checkers disagree, and how this one is built

## 1. Why online checkers give different numbers (and differ from Turnitin)

| Reason | What it means |
|---|---|
| **Different models** | Every checker trains its own classifier on its own data. GPTZero, Copyleaks, ZeroGPT, Originality, and Turnitin are all different models. Turnitin's is proprietary and can't be downloaded or copied. |
| **Different units** | Turnitin reports the **% of qualifying sentences** that it thinks are AI. Many free sites report a **document-level probability** ("87% likely AI"). Those two numbers aren't measuring the same thing. |
| **Different segmentation** | Turnitin splits text into **overlapping segments of a few hundred words**. Each segment gets a 0–1 score, and each sentence inherits the pooled score of the segments that contain it. Sites that score the whole text at once, or score sentence by sentence, get different results. |
| **Different thresholds** | Turnitin tunes for a **<1% false-positive rate** at the document level, which means it holds back. Many free sites flag far more aggressively. |
| **Model updates** | Turnitin changes its model over time. In July 2026 it moved from a multi-model ensemble to a single model. The same essay can score differently from one month to the next. |
| **Length** | Detection is unreliable on short texts. Turnitin needs about 300 words. Binoculars works best at about 256 tokens. |

**Bottom line:** no tool outside Turnitin can promise to match Turnitin's number. What we *can* do is (a) copy the method Turnitin publishes (overlapping segments, sentence-level percentage), (b) use the strongest open detectors available, and (c) be **stricter** than Turnitin. If a text passes here in "Very hard" mode, it is much less likely to be flagged elsewhere.

## 2. The detectors used

1. **Supervised classifier: `desklib/ai-text-detector-v1.01`**. DeBERTa-v3-large fine-tuned for AI detection, one of the top open models on the RAID benchmark. It catches the "style fingerprint" of modern chat models.
2. **Binoculars (Hans et al., ICML 2024)**. A zero-shot detector that compares how surprising the text is to one LLM (perplexity) with how surprising one LLM's predictions are to another (cross-perplexity). It needs no training data, so it generalises to models the classifier never saw. The paper reports >90% detection at 0.01% FPR with Falcon-7B. Independent tests (Mozilla) measured about 0.7% FPR. We run it with **Qwen2.5-1.5B / -Instruct** so it fits an 8 GB GPU.
3. **Stylometry**. Two signals: how dense the "LLM-overused vocabulary" is (*delve, tapestry, pivotal, underscore, showcase…*, from Kobak et al. 2024 and Liang et al. 2024), and **burstiness**, i.e. how much sentence length varies. Human writing varies more.

## 3. How "Very hard" mode fuses them

For each ~200-word window: `score = 0.70·max(classifier, binoculars) + 0.18·min(...) + 0.12·style`. A window is treated as AI if **either** strong detector is suspicious. A sentence is flagged when its pooled score is ≥ 0.38 (Strict: 0.45, Standard: 0.55). The AI % is the share of words in flagged sentences, which is the same unit Turnitin reports.

## 4. Rewrite suggestions

* **AI-phrase lexicon**: about 200 words and phrases that LLMs overuse, each with plainer alternatives. These are flagged across the whole document.
* **Predictable words**: Binoculars already gives the probability of every token. In flagged sentences, content words that the LLM found ≥30% likely are marked. A masked language model (RoBERTa) then proposes replacements that fit the context. Swapping a highly predictable word for a less predictable one that still fits raises perplexity, which is the signal detectors depend on.

## 5. Measured results (this machine, RTX 4070 Laptop)

Calibration on 60 human and 60 ChatGPT answers from HC3, 120–400 words each:

| Mode | Human text: avg AI % | Human docs flagged ≥50% | AI text: avg AI % | AI docs flagged ≥50% |
|---|---|---|---|---|
| Standard | 0.7% | 0% | 100% | 100% |
| Strict | 2.1% | 2% | 100% | 100% |
| **Very hard** | 8.1% | 5% | 100% | 100% |

Binoculars (Qwen2.5-1.5B pair): human mean 1.018, AI mean 0.870. The threshold at 1% FPR is about 0.912, so the default is 0.92. Analysing about 160 words takes about 2.6 s.

**Word-swap experiment:** applying every suggested replacement to a 158-word ChatGPT paragraph lowered Binoculars from 29% → 11% and style from 100% → 48%. The **deep classifier stayed at 100%**. Swapping vocabulary alone doesn't beat a strong supervised detector, and probably doesn't beat Turnitin either. To get a lower score, sentence structure and voice have to change, not only individual words. Calibration data is 2022-era ChatGPT, so newer models may score differently.

**Paragraph-rewrite experiment:** for each paragraph, the Qwen2.5-1.5B-Instruct model writes 3 rewrites with different style briefs, and each is re-scored. On a ChatGPT paragraph, the rewrites cut Binoculars from 51% to 1–10% and style from 100% to about 40–50%. The classifier stayed at about 100%, because the rewrites are AI text too. The UI labels them honestly as starting points. Blocking AI-typical words during generation (`bad_words_ids`) produced broken words ("fosts"), so that was dropped.

## 6. Recalibration against how the big checkers actually work (Nov 2026)

### What the major checkers do
| Checker | Key rules |
|---|---|
| **Turnitin** | Scores only *qualifying text*: prose sentences in long-form writing. Lists, tables, code and poetry are excluded. Needs 300+ words. Shows 1–19% as `*%` because false positives are more common there. Targets <1% false positives at document level and says the score must not be the only basis for action. Grammar tools like Grammarly aren't the target, but heavy AI paraphrasing can be flagged. |
| **GPTZero** | Classifies sentences, then gives a whole-document class: human, mixed or AI. Its % is *confidence in that class*, not "share of text". Recommends treating uncertain results as no clear signal. |
| **Grammarly** | Its AI Detector scores sections for AI patterns. Authorship separately records *how* text was produced (typed, pasted, AI-generated, grammar-edited) and treats grammar edits as edits, not AI writing. |
| **Originality.ai** | Its "Lite" model explicitly allows light AI editing (spelling and grammar). |
| **University policies** | Most allow support uses (brainstorming, grammar, spellcheck, properly cited sources). Submitting AI-generated text as your own is misconduct. Quoting and citing sources is fine; that falls under plagiarism rules, not AI rules. |

### What was wrong here
Measured on real pre-ChatGPT human writing (60 student essays, 60 academic essays, 60 arXiv abstracts) against 120 AI texts:

| Mode (old) | Student essays flagged ≥20% | Academic essays | arXiv | AI caught |
|---|---|---|---|---|
| Standard | 47% | 28% | 5% | 100% |
| Very hard (old default) | **83%** | 60% | 30% | 100% |

The cause was the DeBERTa classifier, which rated 48% of human student-essay sections above 0.5. On AI text it is above 0.9 in 94% of cases. The style signal also fires on formal academic prose.

### What changed
1. **Qualifying-text rules** (Turnitin-style). Headings, list items, captions, table rows, code, direct quotations and the reference list are excluded and labelled "Not counted".
2. **Classifier recalibrated.** It only counts once it is more confident than 0.99 (Balanced) or 0.95 (Strict, Very hard). Style weight is cut to 5%.
3. **Balanced is the default**, with 1–19% shown as low-confidence `N*%`, like Turnitin.
4. **Scoring windows stay inside paragraphs**, so one paragraph can't drag its neighbours' scores up.

### Results after (all 300 texts; settings tuned on half and checked on the held-out half)
| Mode | Student essays ≥20% | Academic | arXiv | AI essays caught | ChatGPT caught |
|---|---|---|---|---|---|
| **Balanced** | **3%** (avg 1.9%) | 0% | 0% | 98% (avg 81%) | 98% (avg 88%) |
| Strict | 13% | 2% | 0% | 98% | 100% |
| Very hard | 18% | 10% | 0% | 100% | 100% |

Re-run with `python -m app.evaluate collect` then `python -m app.evaluate report`, and tune with `python -m app.tune`.

## 7. Honest limitations

* No detector is proof. Turnitin itself says its score shouldn't be the only basis for academic action.
* False positives hit non-native English writers and very formulaic writing harder.
* Swapping words lowers the *score*. It doesn't make anyone the author of text they didn't write. The intended use is to help students who wrote their own work understand *why* it's being flagged.

## Sources
- [Turnitin – AI writing detection FAQs](https://guides.turnitin.com/hc/en-us/articles/28477544839821-Turnitin-s-AI-writing-detection-capabilities-FAQs)
- [Turnitin – AI writing detection capabilities](https://www.turnitin.com/products/features/ai-writing-detection/)
- [Hans et al. – Spotting LLMs with Binoculars (arXiv 2401.12070)](https://arxiv.org/abs/2401.12070)
- [Mozilla Foundation – Who wrote that? Evaluating AI-text detectors](https://www.mozillafoundation.org/en/blog/who-wrote-that-evaluating-tools-to-detect-ai-generated-text/)
- [RAID benchmark (Dugan et al., ACL 2024)](https://arxiv.org/abs/2405.07940) · [GitHub](https://github.com/liamdugan/raid)
- [desklib/ai-text-detector-v1.01 model card](https://huggingface.co/desklib/ai-text-detector-v1.01)
- Kobak et al. 2024, *Delving into ChatGPT usage in academic writing through excess vocabulary*
- HC3 dataset (Guo et al. 2023), used for calibration
