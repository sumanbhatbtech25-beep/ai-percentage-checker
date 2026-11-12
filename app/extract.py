"""Extract plain text from uploaded files (.txt, .md, .docx, .pdf)."""
import io

from docx import Document
from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages)
        # re-join lines broken mid-sentence by PDF layout
        lines, out = text.split("\n"), []
        for line in lines:
            line = line.strip()
            if not line:
                out.append("\n")
            elif out and out[-1] not in ("\n",) and not out[-1].rstrip().endswith((".", "!", "?", ":")):
                out[-1] = out[-1].rstrip("-") + ("" if out[-1].endswith("-") else " ") + line
            else:
                out.append(line)
        return "\n".join(x for x in out if x != "\n")
    if name.endswith((".txt", ".md")):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    raise ValueError("Unsupported file type. Upload .docx, .pdf or .txt")


def text_to_docx(text: str) -> bytes:
    doc = Document()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
