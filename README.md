# AI Percentage Checker

A Turnitin-style AI-writing checker that runs locally on your GPU. It:

* takes **.docx / .pdf / .txt** uploads or pasted text
* shows an **AI %** using Turnitin's unit: the share of text in sentences flagged as AI
* highlights likely-AI sentences (red) and borderline ones (yellow)
* underlines words and phrases that push the score up. **Click one to get a dropdown of replacements**, edit right on the page, then **Re-check**
* scores **every paragraph** (¶1 · 87% AI…), ranks paragraphs from most to least AI-like, and explains **why** each one is flagged (AI-typical words, stock transitions, uniform sentence length, no concrete details…)
* **suggests rewritten paragraphs**: 3 versions, each re-scored by the same detectors, which you can drop in with one click
* lets you download the revised text as .docx or .txt
* **Similar Papers** page (`/papers`): finds the 5 research papers closest to your project. It searches OpenAlex and Semantic Scholar with locally extracted keywords, then ranks candidates by meaning with a local embedding model. Only short keyword queries leave your computer.

> **Honest note on rewrites:** AI-written rewrites lower the Binoculars and style signals a lot, but the deep classifier still recognises them as AI text. Use them as a starting point and put the ideas in your own words. That is what actually lowers the score.

See [RESEARCH.md](RESEARCH.md) for how detectors work, why their numbers differ, and how this one is designed.

## Run

```bash
run.bat
```
Then open http://localhost:8000. The first start downloads about 8 GB of models into `~/.cache/huggingface`. After that, models take about 30 s to load.

### Manual setup (if `.venv` is missing)
```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

## Calibrate
```bash
.venv\Scripts\python -m app.calibrate --n 60
```
This scores human and ChatGPT answers from the HC3 dataset, then prints a Binoculars threshold and the AI % each mode gives for human text and for AI text. Put the threshold in `app/config.py` (or the `AIC_BINO_THRESHOLD` environment variable).

## Low VRAM (<8 GB)
```bash
set AIC_OBSERVER=Qwen/Qwen2.5-0.5B
set AIC_PERFORMER=Qwen/Qwen2.5-0.5B-Instruct
```
Re-run calibration after changing models.

## Paper search limits
Without a key, OpenAlex gives each network a small free daily budget (about 30 searches), and Semantic Scholar's shared pool is often rate-limited. For more:
```bash
set OPENALEX_API_KEY=your_free_key   &:: from https://openalex.org/settings/api
set S2_API_KEY=your_key              &:: optional, from semanticscholar.org/product/api
```

## Layout
| File | Purpose |
|---|---|
| `app/detector.py` | sentence splitting, overlapping windows, signal fusion, AI %, suggestions |
| `app/models.py` | DeBERTa classifier, Binoculars, RoBERTa masked-LM |
| `app/lexicon.py` | about 200 LLM-overused words and phrases with alternatives |
| `app/rewrite.py` | paragraph rewrites (Qwen2.5-1.5B-Instruct), each re-scored |
| `app/papers.py` | keyword extraction, OpenAlex + Semantic Scholar search, embedding re-rank |
| `app/main.py` | FastAPI server |
| `static/` | web UI (`index.html` checker, `papers.html` similar papers) |
