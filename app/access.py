"""Who may open Report Program.

This computer - launch key (AIC_LOCAL_KEY, set by run.bat / run_phone.bat):
  * Each launch creates a random key and saves it in the user's own
    %LOCALAPPDATA%/ReportProgram/launch.key, which other Windows accounts
    can't read. The launcher opens the browser once with ?key=..., and the
    server swaps it for a login cookie. Without key or cookie the site is
    locked, so another account on a shared PC can't open your reports.
  * Started without a key (e.g. by a developer running uvicorn directly),
    this computer is let in as before.

Phone on the same Wi-Fi - password (AIC_PASSWORD, set by run_phone.bat):
  * Other devices must log in; a login lasts 12 hours (or until the server
    restarts), and /logout ends it.
  * Repeated wrong passwords from one address are locked out for a minute.
"""
from __future__ import annotations

import asyncio
import hmac
import html
import os
import secrets
import time
from urllib.parse import quote, urlencode, urlparse

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

PASSWORD = os.environ.get("AIC_PASSWORD", "")
LOCAL_KEY = os.environ.get("AIC_LOCAL_KEY", "")
LOCAL_SESSION_HOURS = 24 * 30  # until the server restarts, at most 30 days
COOKIE = "aic_session"
_sessions: dict[str, float] = {}  # token -> expiry time
SESSION_HOURS = 12
MIN_PASSWORD = 8
_failures: dict[str, list[float]] = {}
LOCAL = {"127.0.0.1", "::1", "localhost"}
OPEN_PATHS = {"/login", "/static/style.css", "/static/icon.png", "/favicon.ico", "/api/health"}  # health: "models ready?" only, nothing private


def enabled() -> bool:
    return bool(PASSWORD)


def is_local(request: Request) -> bool:
    return (request.client.host if request.client else "") in LOCAL


def _safe_next(target: str) -> str:
    """Only allow redirects back into this site."""
    u = urlparse(target or "/")
    return target if (not u.scheme and not u.netloc and target.startswith("/")) else "/"


def _has_session(request: Request) -> bool:
    token = request.cookies.get(COOKIE, "")
    exp = _sessions.get(token) if token else None
    if exp and exp <= time.time():
        _sessions.pop(token, None)  # expired
        return False
    return bool(exp)


def _start_session(resp, hours: float):
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + hours * 3600
    resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=int(hours * 3600))
    return resp


LOCKED_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Locked · Report Program</title><link rel="icon" href="/static/icon.png">
<link rel="stylesheet" href="/static/style.css"></head><body>
<main style="max-width:440px;margin:14vh auto 0;padding:0 16px"><div class="card">
<h1 style="font-size:22px;margin:0 0 8px">◎ REPORT PROGRAM</h1>
<p>Open Report Program with the <b>Report Program</b> shortcut on your Desktop (or <code>run.bat</code>).</p>
<p class="muted tiny">For privacy, only the Windows account that started Report Program can open it.</p>
</div></main></body></html>"""


async def middleware(request: Request, call_next):
    path = request.url.path
    if path in OPEN_PATHS or _has_session(request):
        return await call_next(request)
    if is_local(request):
        if not LOCAL_KEY:  # started without the launcher: this computer is trusted as before
            return await call_next(request)
        key = request.query_params.get("key", "")
        if key and hmac.compare_digest(key.encode(), LOCAL_KEY.encode()):
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
            clean = path + ("?" + urlencode(rest) if rest else "")  # drop the key from the address bar
            return _start_session(RedirectResponse(clean, status_code=303), LOCAL_SESSION_HOURS)
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Open Report Program from its Desktop shortcut."}, status_code=401)
        return HTMLResponse(LOCKED_HTML, status_code=401)
    if not enabled():  # another device, but phone mode is off
        return JSONResponse({"detail": "Not available from other devices"}, status_code=403)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "Log in first"}, status_code=401)
    nxt = path + (("?" + request.url.query) if request.url.query else "")
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
        _failures.pop(ip, None)
        return _start_session(RedirectResponse(nxt, status_code=303), SESSION_HOURS)
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
