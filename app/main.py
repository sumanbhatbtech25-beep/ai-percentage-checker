"""FastAPI server: python -m uvicorn app.main:app --port 8000"""
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, reports
from .detector import analyze
from .extract import extract_text, text_to_docx
from .models import DEVICE, get_models
from .compare import compare
from .papers import find_similar
from .rewrite import rewrite_paragraph

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="AI Percentage Checker")


@app.on_event("startup")
def _warm():
    # load models in the background so the page opens immediately
    threading.Thread(target=get_models, daemon=True).start()


class AnalyzeIn(BaseModel):
    text: str
    mode: str = "standard"


class ExportIn(BaseModel):
    text: str


@app.get("/api/health")
def health():
    from . import models

    return {"ready": models._models is not None, "device": DEVICE}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large (20 MB max)")
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"text": text[: config.MAX_CHARS], "filename": file.filename}


@app.post("/api/analyze")
def analyze_ep(body: AnalyzeIn):
    if not body.text.strip():
        raise HTTPException(400, "No text")
    return analyze(body.text, body.mode)


class RewriteIn(BaseModel):
    text: str
    mode: str = "standard"


@app.post("/api/rewrite")
def rewrite_ep(body: RewriteIn):
    if len(body.text.split()) < 8:
        raise HTTPException(400, "Paragraph too short to rewrite")
    return rewrite_paragraph(body.text[:6000], body.mode)


class PapersIn(BaseModel):
    text: str = ""
    query: str = ""


@app.post("/api/papers")
def papers_ep(body: PapersIn):
    if len(body.text.split()) < 5 and not body.query.strip():
        raise HTTPException(400, "Paste your project text or type a topic to search.")
    return find_similar(body.text[: config.MAX_CHARS] or body.query, query=body.query or None)


class CompareIn(BaseModel):
    text: str
    papers: list[dict]


@app.post("/api/papers/compare")
def compare_ep(body: CompareIn):
    if not body.papers:
        raise HTTPException(400, "No papers to compare")
    papers = [
        {"title": str(p.get("title", ""))[:400], "year": p.get("year"),
         "abstract_full": str(p.get("abstract_full") or p.get("abstract") or "")[:4000]}
        for p in body.papers[:5]
    ]
    return {"comparisons": compare(body.text[: config.MAX_CHARS], papers)}


# ----------------------------------------------------------------------------- saved reports
class ReportIn(BaseModel):
    text: str
    mode: str = "standard"
    id: str | None = None
    title: str | None = None
    ai_percent: int | None = None


@app.get("/api/reports")
def reports_list():
    return {"reports": reports.list_reports()}


@app.get("/api/reports/{report_id}")
def reports_get(report_id: str):
    try:
        return reports.get_report(report_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Report not found")


@app.post("/api/reports")
def reports_save(body: ReportIn):
    if not body.text.strip():
        raise HTTPException(400, "Nothing to save")
    try:
        return reports.save_report(body.text, body.mode, body.id, body.title, body.ai_percent)
    except ValueError:
        raise HTTPException(400, "Invalid report id")


@app.delete("/api/reports/{report_id}")
def reports_delete(report_id: str):
    try:
        reports.delete_report(report_id)
    except ValueError:
        raise HTTPException(404, "Report not found")
    return {"deleted": report_id}


@app.get("/papers")
def papers_page():
    return FileResponse(STATIC / "papers.html")


@app.post("/api/export-docx")
def export_docx(body: ExportIn):
    return Response(
        text_to_docx(body.text),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="revised.docx"'},
    )


@app.get("/")
def home():
    return FileResponse(STATIC / "home.html")


@app.get("/checker")
@app.get("/editor")
def checker_editor():
    # one page, two modes: static/app.js reads the path to decide
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
