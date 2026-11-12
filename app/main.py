"""FastAPI server: python -m uvicorn app.main:app --port 8000"""
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .detector import analyze
from .extract import extract_text, text_to_docx
from .models import DEVICE, get_models

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="AI Percentage Checker")


@app.on_event("startup")
def _warm():
    # load models in the background so the page opens immediately
    threading.Thread(target=get_models, daemon=True).start()


class AnalyzeIn(BaseModel):
    text: str
    mode: str = "very_hard"


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


@app.post("/api/export-docx")
def export_docx(body: ExportIn):
    return Response(
        text_to_docx(body.text),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="revised.docx"'},
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
