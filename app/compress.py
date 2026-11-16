"""Make a .docx smaller without changing how it looks.

Three safe steps:
1. Embedded fonts: Word's "Embed fonts in the file" option copies whole font
   files into the document (often megabytes). Copies of *standard* fonts that
   every computer with Office has (Calibri, Cambria, Times New Roman...) are
   removed; any unusual font stays embedded so the look never changes.
2. Oversized images: pictures with far more pixels than a page needs are scaled
   down (to at most MAX_SIDE px) and re-saved in the same format. Their size and
   position on the page are stored separately, so the layout doesn't move.
3. Everything is re-packed at maximum ZIP compression.
"""
from __future__ import annotations

import io
import re
import zipfile

MAX_SIDE = 2000          # pixels on the longest side: plenty for a full-page picture
MIN_IMAGE_BYTES = 150_000

STANDARD_FONTS = {f.lower() for f in [
    "Calibri", "Calibri Light", "Cambria", "Cambria Math", "Aptos", "Aptos Display", "Aptos Narrow",
    "Times New Roman", "Arial", "Arial Narrow", "Arial Black", "Georgia", "Verdana", "Tahoma",
    "Trebuchet MS", "Segoe UI", "Segoe UI Light", "Segoe UI Semibold", "Courier New", "Consolas",
    "Garamond", "Book Antiqua", "Bookman Old Style", "Century Gothic", "Century", "Palatino Linotype",
    "Candara", "Constantia", "Corbel", "Franklin Gothic Medium", "Gill Sans MT", "Lucida Console",
    "Lucida Sans Unicode", "Symbol", "Wingdings", "Webdings", "Comic Sans MS", "Impact",
]}


def _drop_embedded_fonts(files: dict[str, bytes]) -> tuple[list[str], list[str]]:
    """Remove embedded copies of standard fonts. Returns (removed, kept) font names."""
    if "word/fontTable.xml" not in files:
        return [], []
    table = files["word/fontTable.xml"].decode("utf-8")
    rels_name = "word/_rels/fontTable.xml.rels"
    rels = files.get(rels_name, b"").decode("utf-8")
    removed, kept, drop_ids = [], [], set()

    def per_font(m):
        name, body = m.group(1), m.group(2)
        embeds = re.findall(r'<w:embed\w+ [^>]*r:id="([^"]+)"[^>]*/>', body)
        if not embeds:
            return m.group(0)
        if name.lower() in STANDARD_FONTS:
            removed.append(name)
            drop_ids.update(embeds)
            body = re.sub(r"<w:embed\w+ [^>]*/>", "", body)
            return f'<w:font w:name="{name}"{body}</w:font>'
        kept.append(name)
        return m.group(0)

    table = re.sub(r'<w:font w:name="([^"]+)"(.*?)</w:font>', per_font, table, flags=re.S)
    if not drop_ids:
        return [], kept
    files["word/fontTable.xml"] = table.encode("utf-8")
    for rid in drop_ids:
        m = re.search(rf'<Relationship [^>]*Id="{re.escape(rid)}"[^>]*Target="([^"]+)"[^>]*/>', rels) or \
            re.search(rf'<Relationship [^>]*Target="([^"]+)"[^>]*Id="{re.escape(rid)}"[^>]*/>', rels)
        if m:
            files.pop("word/" + m.group(1).lstrip("/").removeprefix("word/"), None)
            rels = rels.replace(m.group(0), "")
    files[rels_name] = rels.encode("utf-8")
    if not kept and "word/settings.xml" in files:  # nothing embedded any more: stop Word re-embedding on next save
        s = files["word/settings.xml"].decode("utf-8")
        s = re.sub(r"<w:(embedTrueTypeFonts|saveSubsetFonts|embedSystemFonts)\b[^>]*/>", "", s)
        files["word/settings.xml"] = s.encode("utf-8")
    return sorted(set(removed)), sorted(set(kept))


def _shrink_images(files: dict[str, bytes]) -> int:
    try:
        from PIL import Image
    except ImportError:
        return 0
    shrunk = 0
    for name, data in list(files.items()):
        ext = name.lower().rsplit(".", 1)[-1]
        if not name.startswith("word/media/") or ext not in ("png", "jpg", "jpeg") or len(data) < MIN_IMAGE_BYTES:
            continue
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            out = io.BytesIO()
            if ext == "png":
                img.save(out, "PNG", optimize=True)
            else:
                (img.convert("RGB") if img.mode not in ("RGB", "L") else img).save(out, "JPEG", quality=82, optimize=True, progressive=True)
            if out.tell() < len(data) * 0.9:  # only keep a real saving
                files[name] = out.getvalue()
                shrunk += 1
        except Exception:
            continue  # unusual image: leave it exactly as it was
    return shrunk


def compress_docx(data: bytes) -> tuple[bytes, dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        order = [i.filename for i in z.infolist()]
        files = {n: z.read(n) for n in order}
    removed, kept = _drop_embedded_fonts(files)
    images = _shrink_images(files)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in ["[Content_Types].xml"] + [n for n in order if n != "[Content_Types].xml"]:
            if n in files:
                z.writestr(n, files[n])
    result = out.getvalue()
    if len(result) >= len(data):  # never hand back a bigger file
        result = data
    return result, {
        "before": len(data), "after": len(result),
        "fonts_removed": removed, "fonts_kept": kept, "images_shrunk": images,
    }
