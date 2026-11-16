"""Saved reports, kept on this computer until the user deletes them.

Security:
  * Stored OUTSIDE the project folder, in the user's local app-data folder
    (%LOCALAPPDATA%\\ReportProgram\\reports on Windows). The project lives in
    OneDrive, which would otherwise sync every saved report to the cloud.
  * Encrypted at rest with Windows DPAPI (see vault.py): only this Windows
    account on this computer can read them.
  * Report ids are random and strictly validated (no path tricks).
  * Writes are atomic, and deleted reports are removed from disk.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

from . import vault


def _default_dir() -> Path:
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ReportProgram" / "reports"
    return Path.home() / ".local" / "share" / "report-program" / "reports"


DIR = Path(os.environ.get("AIC_REPORTS_DIR") or _default_dir())
LEGACY_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"  # old location (inside OneDrive)
_ID_RE = re.compile(r"^[a-f0-9]{12}$")
MODES = {"standard", "strict", "very_hard"}
MAX_TEXT = 400_000
MAX_ORIGINAL = 20 * 1024 * 1024


def _path(report_id: str) -> Path:
    if not _ID_RE.match(report_id or ""):
        raise ValueError("Invalid report id")
    return DIR / f"{report_id}.json"


def _ensure_dir() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(DIR, 0o700)


def _write(d: dict) -> None:
    _ensure_dir()
    p = _path(d["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(vault.protect(json.dumps(d, ensure_ascii=False).encode("utf-8")))
    if sys.platform != "win32":
        os.chmod(tmp, 0o600)
    tmp.replace(p)  # atomic: a crash never leaves a half-written report


def _read(p: Path) -> dict:
    return json.loads(vault.unprotect(p.read_bytes()).decode("utf-8"))


def migrate_legacy() -> int:
    """Move reports saved by older versions out of the OneDrive project folder, encrypting them."""
    if not LEGACY_DIR.exists():
        return 0
    moved = 0
    for f in LEGACY_DIR.glob("*.json"):
        try:
            d = _read(f)
            if _ID_RE.match(str(d.get("id", ""))):
                _write(d)
                f.unlink()
                moved += 1
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    for d in (LEGACY_DIR, LEGACY_DIR.parent):  # rmdir only ever removes EMPTY folders
        try:
            d.rmdir()
        except OSError:
            break
    return moved


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
            d = _read(f)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        out.append({k: d.get(k) for k in ("id", "title", "created", "updated", "ai_percent", "words", "mode", "original_name")})
    out.sort(key=lambda d: d.get("updated") or 0, reverse=True)
    return out


def get_report(report_id: str) -> dict:
    p = _path(report_id)
    if not p.exists():
        raise FileNotFoundError(report_id)
    return _read(p)


def save_report(text: str, mode: str = "standard", report_id: str | None = None,
                title: str | None = None, ai_percent: int | None = None,
                original: bytes | None = None, original_name: str | None = None) -> dict:
    now = time.time()
    existing = {}
    if report_id:
        try:
            existing = get_report(report_id)
        except FileNotFoundError:
            existing = {}
    else:
        report_id = secrets.token_hex(6)
    text = text[:MAX_TEXT]
    d = {
        "id": report_id,
        "title": (title or existing.get("title") or _derive_title(text)).strip()[:120],
        "created": existing.get("created", now),
        "updated": now,
        "text": text,
        "mode": mode if mode in MODES else "standard",
        "ai_percent": ai_percent if isinstance(ai_percent, int) and 0 <= ai_percent <= 100 else existing.get("ai_percent"),
        "words": len(text.split()),
        "original_name": existing.get("original_name"),
    }
    if original is not None:  # the uploaded .docx, kept so downloads can keep its formatting
        _write_original(report_id, original)
        d["original_name"] = (original_name or "report.docx")[:200]
    _write(d)
    return {k: d[k] for k in ("id", "title", "updated", "ai_percent", "words", "original_name")}


def rename_report(report_id: str, title: str) -> dict:
    d = get_report(report_id)
    d["title"] = (title.strip() or d["title"])[:120]
    _write(d)
    return {"id": d["id"], "title": d["title"]}


def delete_report(report_id: str) -> None:
    for p in (_path(report_id), _orig_path(report_id)):
        if p.exists():
            p.unlink()


def _orig_path(report_id: str) -> Path:
    return _path(report_id).with_suffix(".orig")


def _write_original(report_id: str, data: bytes) -> None:
    if len(data) > MAX_ORIGINAL:
        raise ValueError("Original file too large")
    _ensure_dir()
    p = _orig_path(report_id)
    tmp = p.with_suffix(".otmp")
    tmp.write_bytes(vault.protect(data))  # encrypted like the report itself
    tmp.replace(p)


def get_original(report_id: str) -> bytes | None:
    p = _orig_path(report_id)
    return vault.unprotect(p.read_bytes()) if p.exists() else None
