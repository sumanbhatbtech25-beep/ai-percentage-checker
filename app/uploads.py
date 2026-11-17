"""Uploaded Word files, kept in server memory so downloads can keep their formatting.

The page only holds a short random id, not the file itself. That avoids the
browser's storage limits (a 2.3 MB .docx used to be dropped when moving from
the Checker to the Editor). Files live in memory only, never on disk: at most
MAX_ITEMS of them, each for TTL seconds. Saving a report copies the file into
the encrypted report store, so it outlives this cache.
"""
from __future__ import annotations

import secrets
import threading
import time

TTL = 6 * 3600
MAX_ITEMS = 8
_items: dict[str, tuple[float, bytes]] = {}
_lock = threading.Lock()


def put(data: bytes) -> str:
    uid = secrets.token_hex(12)
    with _lock:
        now = time.time()
        for k in [k for k, (exp, _) in _items.items() if exp < now]:
            del _items[k]
        while len(_items) >= MAX_ITEMS:  # drop the oldest
            del _items[min(_items, key=lambda k: _items[k][0])]
        _items[uid] = (now + TTL, data)
    return uid


def get(uid: str | None) -> bytes | None:
    if not uid:
        return None
    with _lock:
        item = _items.get(uid)
    return item[1] if item and item[0] > time.time() else None
