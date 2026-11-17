"""HTTP hardening for the local server.

1. Host allow-list - only answer requests addressed to this computer
   (localhost / 127.0.0.1 / ::1, plus its own LAN address in phone mode).
   Blocks DNS-rebinding, where a malicious website points its own domain
   name at 127.0.0.1 to read your saved reports.
2. Cross-site write protection - POST/PATCH/DELETE must come from this site
   (checked with the browser's Origin and Sec-Fetch-Site headers), so other
   websites can't save, rename or delete reports behind your back.
3. Security headers - a strict Content-Security-Policy (only this site's own
   scripts run), no framing, no referrer leaks, and no caching of API
   responses that contain report text.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse

from . import access

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}

CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",                    # no inline or third-party scripts
    "style-src 'self' 'unsafe-inline'",     # style attributes only (can't run code)
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])
HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _lan_addresses() -> set[str]:
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0].split("%")[0]
            if not ipaddress.ip_address(ip).is_loopback:
                addrs.add(ip)
    except OSError:
        pass
    return addrs


ALLOWED_HOSTS = set(LOCAL_HOSTS) | {h.strip().lower() for h in os.environ.get("AIC_ALLOWED_HOSTS", "").split(",") if h.strip()}
_lan: tuple[float, set[str]] = (0.0, set())


def _current_lan() -> set[str]:
    """This computer's network addresses, re-read at most every 30 s. The address can
    change while the server runs (sleep, new Wi-Fi), and phone mode must keep working."""
    global _lan
    now = time.time()
    if now - _lan[0] > 30:
        _lan = (now, _lan_addresses())
    return _lan[1]


def host_allowed(host: str) -> bool:
    return host in ALLOWED_HOSTS or (access.enabled() and host in _current_lan())


def _hostname(host_header: str) -> str:
    h = (host_header or "").strip().lower()
    if h.startswith("["):  # [::1]:8000
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


async def middleware(request: Request, call_next):
    host = _hostname(request.headers.get("host", ""))
    if not host_allowed(host):
        return PlainTextResponse("Unknown host", status_code=421)

    if request.method in UNSAFE:
        site = request.headers.get("sec-fetch-site")
        origin = request.headers.get("origin")
        if site and site not in ("same-origin", "none"):
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
        if origin and origin != "null" and _hostname(urlparse(origin).netloc) != host:
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)

    response = await call_next(request)
    for k, v in HEADERS.items():
        response.headers.setdefault(k, v)
    if request.url.path.startswith("/api/") or request.url.path == "/login":
        response.headers["Cache-Control"] = "no-store"  # report text never lands in the browser cache
    return response
