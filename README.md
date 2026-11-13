# REPORT PROGRAM

Three tools for getting a report ready to hand in, all running locally on your GPU:

| Page | URL | What it does |
|---|---|---|
| **Home** | `/` | Welcome page explaining each tool, with a button to open it |
| **AI Checker** | `/checker` | AI % score, highlighted sentences, per-paragraph scores and "Why AI?" reasons |
| **Editor** | `/editor` | Everything in the checker, plus tap-to-change words, paragraph rewrites, re-check, confetti under 5%, and .docx/.txt export |
| **Research Papers** | `/papers` | Top 5 research papers most similar to your report |

"Fix it in the Editor →" on the checker carries your text straight into the editor, and the papers page picks up your last text automatically.

**Saved reports:** "Save report" on the checker or editor stores the report, encrypted, on this computer (see Security below). After the first save, every change is saved automatically. The home page lists saved reports with **search**, **sort**, **Rename**, **Continue this report** and **Delete**, and each one stays until you delete it.

**Editor extras:** **↶ / ↷ undo and redo** (Ctrl+Z / Ctrl+Y) for every single change, and **Fix all** / **Undo all**.

**Similarity check:** a card on the checker and editor finds sentences **copied** or **closely paraphrased** from research papers on your topic, and whether you cited them. It compares against the abstracts of the top 10 papers from OpenAlex and Semantic Scholar, not the whole web or other students' work.

**Research Papers:** each of the top 5 papers has two buttons, **How your project differs** and **What you can improve**. Each comparison is grounded in the parts of the paper's abstract your report doesn't cover, then summarised by the local Qwen model. Only abstracts are used.

Together the tools:

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

Double-click the **Report Program** shortcut on the Desktop, or run `run.bat`. It starts the server in a minimised window and opens http://localhost:8000. If the server is already running, it just opens the browser. Close the window to stop the server. The first start downloads about 8 GB of models into `~/.cache/huggingface`. After that, models take about 30 s to load.

To recreate the Desktop shortcut:
```bash
powershell -ExecutionPolicy Bypass -File tools\create_shortcut.ps1
```

### Use it from your phone (same Wi-Fi)
Run `run_phone.bat` instead. It asks you to choose a password (8+ characters) and shows an address such as `http://192.168.1.20:8000` to open on your phone. The phone logs in once and stays logged in for 12 hours; this computer never needs the password. If Windows asks, allow Python on **Private networks**. Only use this on your home Wi-Fi or another network you trust, never public Wi-Fi.

## Security
Users' reports are private work, so the app is locked down:

| Risk | Protection |
|---|---|
| Reports synced to the cloud (this project folder is in OneDrive) | Saved reports live **outside** the project, in `%LOCALAPPDATA%\ReportProgram\reports`, which isn't synced. Reports saved by older versions are moved there automatically. |
| Someone reading the files on disk | Encrypted with **Windows DPAPI**: only your Windows account on this computer can decrypt them. |
| Text leaving the computer | Detection, editing, comparison and similarity all run locally. Only short **keyword** searches go to the paper databases, never your text. |
| Other computers on the network | The server only listens on this computer (`127.0.0.1`), except in phone mode, which requires a password. |
| Malicious websites you visit in your browser | A host allow-list blocks DNS-rebinding, and Origin / Sec-Fetch-Site checks block cross-site saves, renames and deletes. |
| Injected scripts (XSS) | A strict Content-Security-Policy (only the site's own scripts), all text escaped, outside links restricted to http(s). |
| Clickjacking, leaks, caching | `frame-ancestors 'none'`, `no-referrer`, and `Cache-Control: no-store` on every API response. |
| Phone login guessing | Constant-time password check, 5 tries per minute per device, sessions expire after 12 h, `/logout`, strict cookie. |
| Bad uploads | Only .docx / .pdf / .txt, size capped while reading (20 MB), corrupt files fail safely. |
| Path tricks | Report ids are random and strictly validated. The API docs pages are switched off. |

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
| `app/papers.py` | keyword extraction, OpenAlex + Semantic Scholar search (cached), embedding re-rank |
| `app/compare.py` | "how your project differs / what to improve" for each paper |
| `app/similarity.py` | copied / paraphrased sentence check against paper abstracts |
| `app/reports.py`, `app/vault.py` | saved reports, encrypted with DPAPI, stored outside OneDrive |
| `app/security.py`, `app/access.py` | host/origin checks, security headers, phone-mode password |
| `app/evaluate.py`, `app/tune.py` | accuracy measurement and tuning (2022–2026 AI vs human writing) |
| `app/main.py` | FastAPI server |
| `static/` | web UI (`home.html`, `index.html` checker/editor, `papers.html`) |
| `run.bat`, `run_phone.bat`, `tools/` | launchers, Desktop shortcut, icon |
