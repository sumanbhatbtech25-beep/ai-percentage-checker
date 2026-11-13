"""Password protection for using the site from a phone on the same Wi-Fi.

Off unless the server is started with AIC_PASSWORD set (run_phone.bat does this).
  * Requests from this computer (127.0.0.1 / ::1) never need the password.
  * Any other device must log in; a login lasts 12 hours (or until the
    server restarts), and /logout ends it.
  * Repeated wrong passwords from one address are locked out for a minute.
"""
from __future__ import annotations

import asyncio
import hmac
import html
import os
import secrets
import time
from urllib.parse import quote, urlparse

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

PASSWORD = os.environ.get("AIC_PASSWORD", "")
COOKIE = "aic_session"
_sessions: dict[str, float] = {}  # token -> expiry time
SESSION_HOURS = 12
MIN_PASSWORD = 8
_failures: dict[str, list[float]] = {}
LOCAL = {"127.0.0.1", "::1", "localhost"}
OPEN_PATHS = {"/login", "/static/style.css", "/static/icon.png", "/favicon.ico"}


def enabled() -> bool:
    return bool(PASSWORD)


def is_local(request: Request) -> bool:
    return (request.client.host if request.client else "") in LOCAL


def _safe_next(target: str) -> str:
    """Only allow redirects back into this site."""
    u = urlparse(target or "/")
    return target if (not u.scheme and not u.netloc and target.startswith("/")) else "/"


async def middleware(request: Request, call_next):
    if not enabled() or is_local(request) or request.url.path in OPEN_PATHS:
        return await call_next(request)
    token = request.cookies.get(COOKIE, "")
    exp = _sessions.get(token) if token else None
    if exp and exp > time.time():
        return await call_next(request)
    if exp:
        _sessions.pop(token, None)  # expired
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Log in first"}, status_code=401)
    nxt = request.url.path + (("?" + request.url.query) if request.url.query else "")
    return RedirectResponse(f"/login?next={quote(nxt)}", status_code=303)


def _locked(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _failures.get(ip, []) if now - t < 60]
    _failures[ip] = recent
    return len(recent) >= 5


LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Log in · Report Program</title><link rel="icon" href="/static/icon.png">
<link rel="stylesheet" href="/static/style.css">
<style>
.login {{ max-width: 380px; margin: 12vh auto 0; padding: 0 16px; }}
.login .card {{ padding: 26px 24px; }}
.login h1 {{ font-size: 22px; margin: 0 0 6px; }}
.login input {{ width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); margin: 14px 0 12px; font-size: 16px; }}
.login button {{ width: 100%; padding: 10px; }}
.login .err {{ color: #d92d20; font-size: 14px; margin-top: 10px; }}
</style></head><body>
<main class="login"><form class="card" method="post" action="/login">
<h1>◎ REPORT PROGRAM</h1>
<p class="muted">Enter the password shown in the Report Program window on the computer.</p>
<input type="password" name="password" autocomplete="current-password" placeholder="Password" required autofocus>
<input type="hidden" name="next" value="{next}">
<button class="primary" type="submit">Log in</button>
{error}
</form></main></body></html>"""


def login_page(next_url: str = "/", error: str = "") -> HTMLResponse:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return HTMLResponse(LOGIN_HTML.format(next=html.escape(_safe_next(next_url)), error=err))


async def login_submit(request: Request):
    form = await request.form()
    ip = request.client.host if request.client else "?"
    nxt = _safe_next(str(form.get("next", "/")))
    if _locked(ip):
        return login_page(nxt, "Too many wrong attempts. Wait a minute and try again.")
    if enabled() and hmac.compare_digest(str(form.get("password", "")).encode(), PASSWORD.encode()):
        token = secrets.token_urlsafe(32)
        _sessions[token] = time.time() + SESSION_HOURS * 3600
        _failures.pop(ip, None)
        resp = RedirectResponse(nxt, status_code=303)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=SESSION_HOURS * 3600)
        return resp
    _failures.setdefault(ip, []).append(time.time())
    await asyncio.sleep(0.8)  # slow down guessing (without blocking other requests)
    return login_page(nxt, "That password isn't right.")


def check_password_strength() -> None:
    """Refuse to start phone mode with a short password."""
    if enabled() and len(PASSWORD) < MIN_PASSWORD:
        raise SystemExit(f"Phone-access password must be at least {MIN_PASSWORD} characters. Start run_phone.bat again.")


def logout(request: Request):
    _sessions.pop(request.cookies.get(COOKIE, ""), None)
    resp = RedirectResponse("/login" if enabled() else "/", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp
