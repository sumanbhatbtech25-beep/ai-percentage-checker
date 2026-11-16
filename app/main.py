"""FastAPI server: python -m uvicorn app.main:app --port 8000"""
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import access, config, reports, security, similarity
from .detector import analyze
from .extract import extract_text, text_to_docx
from .models import DEVICE, get_models
from .compare import compare
from .papers import find_similar
from .rewrite import rewrite_paragraph

STATIC = Path(__file__).resolve().parent.parent / "static"
access.check_password_strength()  # refuse to start phone mode with a weak password
# API docs pages are switched off: nobody needs them, and they map out the API.
app = FastAPI(title="Report Program", docs_url=None, redoc_url=None, openapi_url=None)
# Middleware runs last-added-first: host/origin checks and headers wrap everything,
# then the phone-mode password check.
app.middleware("http")(access.middleware)
app.middleware("http")(security.middleware)


@app.exception_handler(Exception)
async def _friendly_errors(request: Request, exc: Exception):
    """Never show a raw crash page: give the user a plain message (details stay in the server log)."""
    from .models import free_gpu, is_oom

    if is_oom(exc):
        free_gpu()
        return JSONResponse({"detail": "The graphics card ran out of memory. Please try again; if it keeps happening, "
                                       "close other GPU-heavy apps or restart Report Program."}, status_code=503)
    import traceback

    traceback.print_exception(exc)
    return JSONResponse({"detail": "Something went wrong on the server. Please try again."}, status_code=500)


@app.get("/login")
def login_form(next: str = "/"):
    return access.login_page(next)


@app.post("/login")
async def login_post(request: Request):
    return await access.login_submit(request)


@app.get("/logout")
def logout(request: Request):
    return access.logout(request)


@app.on_event("startup")
def _warm():
    moved = reports.migrate_legacy()  # move any reports out of the OneDrive project folder
    if moved:
        print(f"[reports] moved {moved} saved report(s) to {reports.DIR} (encrypted)")
    # load models in the background so the page opens immediately
    threading.Thread(target=get_models, daemon=True).start()


class AnalyzeIn(BaseModel):
    text: str
    mode: str = "standard"


class ExportIn(BaseModel):
    text: str


@app.get("/api/health")
def health(request: Request):
    from . import models

    # "remote" tells a logged-in phone to show a Log out link
    return {"ready": models._models is not None, "device": DEVICE,
            "remote": access.enabled() and not access.is_local(request)}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    limit = 20 * 1024 * 1024
    name = (file.filename or "").lower()
    if not name.endswith((".docx", ".pdf", ".txt", ".md")):
        raise HTTPException(400, "Upload a .docx, .pdf or .txt file")
    data = await file.read(limit + 1)  # never read more than the limit into memory
    if len(data) > limit:
        raise HTTPException(413, "File too large (20 MB max)")
    try:
        text = extract_text(name, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:  # corrupt or malicious file: fail safely, reveal nothing
        raise HTTPException(400, "Couldn't read that file. Is it a valid .docx, .pdf or .txt?")
    return {"text": text[: config.MAX_CHARS], "filename": (file.filename or "")[:200]}


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


class SimilarityIn(BaseModel):
    text: str


@app.post("/api/similarity")
def similarity_ep(body: SimilarityIn):
    if len(body.text.split()) < 40:
        raise HTTPException(400, "Paste or upload your full report first.")
    return similarity.check(body.text[: config.MAX_CHARS])


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
    original_b64: str | None = None   # the uploaded .docx, sent once so downloads keep its formatting
    original_name: str | None = None


def _decode_docx(b64: str | None) -> bytes | None:
    """Decode and sanity-check an uploaded original .docx (size-limited, must open as Word)."""
    if not b64:
        return None
    import base64
    import binascii
    import io as _io

    if len(b64) > 28_000_000:
        raise HTTPException(413, "Original file too large (20 MB max)")
    try:
        data = base64.b64decode(b64, validate=True)
        from docx import Document

        Document(_io.BytesIO(data))
    except (binascii.Error, ValueError, Exception):
        raise HTTPException(400, "The original file isn't a valid .docx")
    return data


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
        return reports.save_report(body.text, body.mode, body.id, body.title, body.ai_percent,
                                   _decode_docx(body.original_b64), body.original_name)
    except ValueError:
        raise HTTPException(400, "Invalid report id")


class RenameIn(BaseModel):
    title: str


@app.patch("/api/reports/{report_id}")
def reports_rename(report_id: str, body: RenameIn):
    try:
        return reports.rename_report(report_id, body.title)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Report not found")


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


class ExportOriginalIn(BaseModel):
    text: str
    report_id: str | None = None      # use the original stored with a saved report...
    original_b64: str | None = None   # ...or the one the page still holds
    filename: str | None = None


@app.post("/api/export-original")
def export_original(body: ExportOriginalIn):
    """Write the edited wording back into the user's own .docx, keeping all formatting."""
    import re as _re
    from urllib.parse import quote

    from .docx_edit import apply_edits_to_docx

    data = _decode_docx(body.original_b64)
    if data is None and body.report_id:
        try:
            data = reports.get_original(body.report_id)
        except ValueError:
            data = None
    if data is None:
        raise HTTPException(404, "The original Word file isn't available. Upload the .docx again to download with its formatting.")
    out, stats = apply_edits_to_docx(data, body.text[: config.MAX_CHARS])
    stem = _re.sub(r"\.docx$", "", body.filename or "report", flags=_re.I)
    name = f"{stem} (revised).docx"
    return Response(out, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={
        "Content-Disposition": f"attachment; filename=\"revised.docx\"; filename*=UTF-8''{quote(name)}",
        "X-Changes": str(stats["changes"]),
        "X-Paragraphs-Changed": str(stats["paragraphs_changed"]),
        "X-Structure-Changes": str(stats["structure_changes"]),
    })


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
