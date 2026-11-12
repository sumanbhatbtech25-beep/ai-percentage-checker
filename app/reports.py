"""Saved reports: stored as JSON files on this computer until the user deletes them.

Location: <project>/data/reports/<id>.json  (git-ignored; never uploaded anywhere)
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

DIR = Path(os.environ.get("AIC_REPORTS_DIR", Path(__file__).resolve().parent.parent / "data" / "reports"))
_ID_RE = re.compile(r"^[a-f0-9]{12}$")
MAX_TEXT = 400_000


def _path(report_id: str) -> Path:
    if not _ID_RE.match(report_id or ""):
        raise ValueError("Invalid report id")
    return DIR / f"{report_id}.json"


def _derive_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if 3 <= len(line) <= 120 and len(line.split()) <= 16:
            return line
    words = text.split()
    return (" ".join(words[:8]) + "…") if words else "Untitled report"


def list_reports() -> list[dict]:
    if not DIR.exists():
        return []
    out = []
    for f in DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: d.get(k) for k in ("id", "title", "created", "updated", "ai_percent", "words", "mode")})
    out.sort(key=lambda d: d.get("updated") or 0, reverse=True)
    return out


def get_report(report_id: str) -> dict:
    p = _path(report_id)
    if not p.exists():
        raise FileNotFoundError(report_id)
    return json.loads(p.read_text(encoding="utf-8"))


def save_report(text: str, mode: str = "standard", report_id: str | None = None,
                title: str | None = None, ai_percent: int | None = None) -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    existing = {}
    if report_id:
        try:
            existing = get_report(report_id)
        except FileNotFoundError:
            existing = {}
    else:
        report_id = uuid.uuid4().hex[:12]
    text = text[:MAX_TEXT]
    d = {
        "id": report_id,
        "title": (title or existing.get("title") or _derive_title(text)).strip()[:120],
        "created": existing.get("created", now),
        "updated": now,
        "text": text,
        "mode": mode,
        "ai_percent": ai_percent if ai_percent is not None else existing.get("ai_percent"),
        "words": len(text.split()),
    }
    p = _path(report_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic: a crash never leaves a half-written report
    return {k: d[k] for k in ("id", "title", "updated", "ai_percent", "words")}


def delete_report(report_id: str) -> None:
    p = _path(report_id)
    if p.exists():
        p.unlink()
