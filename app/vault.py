"""Encrypt saved reports at rest.

On Windows we use DPAPI (CryptProtectData), the same mechanism browsers use for
saved passwords: the data can only be decrypted by the same Windows user account
on the same computer. Nothing to remember, no key file to leak.

Elsewhere (no DPAPI) data is stored unencrypted but with owner-only permissions.
"""
from __future__ import annotations

import sys

MAGIC = b"RPDPAPI1"
ENCRYPTED = sys.platform == "win32"

if ENCRYPTED:
    import ctypes
    from ctypes import wintypes

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.windll.crypt32
    _kernel32 = ctypes.windll.kernel32
    _UI_FORBIDDEN = 0x1

    def _in(data: bytes):
        buf = ctypes.create_string_buffer(data, len(data))
        return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    def _out(blob: _Blob) -> bytes:
        try:
            return ctypes.string_at(blob.pbData, blob.cbData)
        finally:
            _kernel32.LocalFree(blob.pbData)

    def protect(data: bytes) -> bytes:
        src, _keep = _in(data)
        dst = _Blob()
        if not _crypt32.CryptProtectData(ctypes.byref(src), ctypes.c_wchar_p("ReportProgram"), None, None, None,
                                         _UI_FORBIDDEN, ctypes.byref(dst)):
            raise OSError("Windows could not encrypt the report")
        return MAGIC + _out(dst)

    def unprotect(data: bytes) -> bytes:
        if not data.startswith(MAGIC):
            return data  # older unencrypted file; re-encrypted next time it's saved
        src, _keep = _in(data[len(MAGIC):])
        dst = _Blob()
        if not _crypt32.CryptUnprotectData(ctypes.byref(src), None, None, None, None, _UI_FORBIDDEN, ctypes.byref(dst)):
            raise OSError("This report was encrypted by a different Windows account")
        return _out(dst)
else:
    def protect(data: bytes) -> bytes:
        return data

    def unprotect(data: bytes) -> bytes:
        return data
