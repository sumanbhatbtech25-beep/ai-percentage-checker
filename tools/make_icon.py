"""Draw the app icon (a ring with a tick, in the site's blue) as static/icon.ico.

Pure standard library: renders 256x256 RGBA pixels, wraps them in PNG, and
stores the PNG inside an .ico container (supported since Windows Vista).
    python tools/make_icon.py
"""
import math
import struct
import zlib
from pathlib import Path

SIZE = 256
BLUE = (37, 99, 235)
OUT = Path(__file__).resolve().parent.parent / "static"


def coverage(x, y):
    """Anti-aliased coverage (0..1) of the ring + tick at pixel (x, y)."""
    def sample(px, py):
        cx = cy = SIZE / 2
        d = math.hypot(px - cx, py - cy)
        if 0.30 * SIZE < d < 0.44 * SIZE:  # ring
            return 1.0
        # tick: two thick segments
        for (x1, y1, x2, y2) in ((0.33, 0.52, 0.45, 0.64), (0.45, 0.64, 0.68, 0.38)):
            x1, y1, x2, y2 = (v * SIZE for v in (x1, y1, x2, y2))
            vx, vy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / (vx * vx + vy * vy)))
            if math.hypot(px - (x1 + t * vx), py - (y1 + t * vy)) < 0.055 * SIZE:
                return 1.0
        return 0.0
    n = 3  # 3x3 supersampling
    return sum(sample(x + (i + 0.5) / n, y + (j + 0.5) / n) for i in range(n) for j in range(n)) / (n * n)


def png_bytes():
    rows = []
    for y in range(SIZE):
        row = bytearray([0])
        for x in range(SIZE):
            a = coverage(x, y)
            row += bytes((*BLUE, int(a * 255)))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def main():
    png = png_bytes()
    (OUT / "icon.png").write_bytes(png)
    ico = struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 6 + 16) + png
    (OUT / "icon.ico").write_bytes(ico)
    print("wrote", OUT / "icon.ico")


if __name__ == "__main__":
    main()
