"""Write wording changes back into the ORIGINAL Word file.

Editors usually turn a report into plain text and give plain text back, which
loses the cover page, fonts, headings, tables and layout. Instead:

1. The text the user edited was extracted from the original .docx as its
   non-empty body paragraphs joined with newlines (see extract.docx_paragraphs).
2. We diff that original text against the edited text word by word, giving a
   small list of changes: (start, end, replacement) in the original text.
3. Each change is mapped to its paragraph and then to the exact *runs* (pieces
   of text with one formatting) it touches, and only those characters are
   rewritten. The edited word keeps the formatting of the text it replaces.

Everything else is untouched: paragraph styles, numbering, tables, images,
headers/footers, page setup, comments and all other runs.
"""
from __future__ import annotations

import difflib
import io
import re

from docx import Document
from docx.oxml.ns import qn
from docx.text.run import Run

TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
_TEXT_TAGS = {qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr"), qn("w:rPr")}


def docx_paragraphs(doc) -> list:
    """The paragraphs whose text the app works on (same rule the extractor uses)."""
    return [p for p in doc.paragraphs if p.text.strip()]


def _runs(paragraph) -> list[Run]:
    runs = []
    for item in paragraph.iter_inner_content():  # runs, including those inside hyperlinks
        runs.extend([item] if isinstance(item, Run) else item.runs)
    return runs


def _set_run_text(run: Run, text: str) -> None:
    """Change a run's text without destroying non-text content (images, fields...)."""
    if all(child.tag in _TEXT_TAGS for child in run._r):
        run.text = text  # plain text run: python-docx handles tabs / line breaks
        return
    ts = run._r.findall(qn("w:t"))
    if ts:
        ts[0].text = text
        ts[0].set(qn("xml:space"), "preserve")
        for t in ts[1:]:
            t.text = ""
    elif text:
        run._r.append(run._r.makeelement(qn("w:t"), {qn("xml:space"): "preserve"}))
        run._r.findall(qn("w:t"))[-1].text = text


def _apply_in_paragraph(paragraph, edits: list[tuple[int, int, str]]) -> None:
    """Apply (start, end, replacement) edits (paragraph-local offsets), last first."""
    runs = _runs(paragraph)
    for s, e, rep in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        texts = [r.text for r in runs]
        offs, pos = [], 0
        for t in texts:
            offs.append(pos)
            pos += len(t)
        textual = [i for i, t in enumerate(texts) if t]
        if not textual:
            continue
        # first run touched: the one containing `s`; for a pure insertion at a run
        # boundary, attach to the end of the previous run (keeps its formatting)
        first = None
        for i in textual:
            a, b = offs[i], offs[i] + len(texts[i])
            if a <= s < b or (s == e and a < s <= b):
                first = i
                break
        if first is None:
            first = textual[-1]
        for i in textual:
            a, b = offs[i], offs[i] + len(texts[i])
            lo, hi = max(s, a), min(e, b)
            if i == first:
                cut_hi = hi if hi > lo else max(s, a)
                new = texts[i][: max(s, a) - a] + rep + texts[i][max(cut_hi, max(s, a)) - a:]
                _set_run_text(runs[i], new)
            elif hi > lo:  # later runs: just delete the covered characters
                _set_run_text(runs[i], texts[i][: lo - a] + texts[i][hi - a:])


def text_edits(original: str, edited: str) -> list[tuple[int, int, str]]:
    """Word-level differences as (start, end, replacement) in `original`."""
    a, b = TOKEN_RE.findall(original), TOKEN_RE.findall(edited)
    offs, pos = [], 0
    for t in a:
        offs.append(pos)
        pos += len(t)
    offs.append(pos)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        s, e, rep = offs[i1], offs[i2], "".join(b[j1:j2])
        # shrink to the characters that really differ, so an unchanged paragraph
        # break next to an edited word never counts as part of the change
        old = original[s:e]
        p = 0
        while p < len(old) and p < len(rep) and old[p] == rep[p]:
            p += 1
        q = 0
        while q < len(old) - p and q < len(rep) - p and old[-1 - q] == rep[-1 - q]:
            q += 1
        s, e, rep = s + p, e - q, rep[p:len(rep) - q]
        if s != e or rep:
            out.append((s, e, rep))
    return out


def apply_edits_to_docx(data: bytes, edited_text: str) -> tuple[bytes, dict]:
    doc = Document(io.BytesIO(data))
    paras = docx_paragraphs(doc)
    texts = [p.text for p in paras]
    original = "\n".join(texts)
    edited = edited_text.replace("\r\n", "\n")
    edits = text_edits(original, edited)

    starts, pos = [], 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1  # +1 for the joining newline

    def locate(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    per_para: dict[int, list] = {}
    structural = 0
    for s, e, rep in edits:
        ps, pe = locate(s), locate(max(s, e - 1)) if e > s else locate(s)
        if s == e and s > 0 and s == starts[ps] and ps > 0:
            ps = pe = ps - 1  # insertion right after a newline: end of previous paragraph
            s = e = starts[ps] + len(texts[ps])
        if ps == pe:
            if "\n" in original[s:e]:
                structural += 1  # a paragraph break was removed (paragraphs merged): kept as two paragraphs
            local_s = s - starts[ps]
            local_e = min(e - starts[ps], len(texts[ps]))
            per_para.setdefault(ps, []).append((local_s, local_e, rep.replace("\n", " ")))
            structural += rep.count("\n")
            continue
        # the change spans paragraph breaks (paragraphs merged or split in free editing)
        structural += 1
        rep_parts = rep.split("\n")
        span = list(range(ps, pe + 1))
        if len(rep_parts) == len(span):  # same number of breaks: apply piece by piece
            for k, pi in enumerate(span):
                ls = s - starts[pi] if pi == ps else 0
                le = (e - starts[pi]) if pi == pe else len(texts[pi])
                per_para.setdefault(pi, []).append((max(0, ls), min(le, len(texts[pi])), rep_parts[k]))
            structural -= 1
        else:  # keep paragraphs, put the new wording in the first one
            for pi in span:
                ls = s - starts[pi] if pi == ps else 0
                le = (e - starts[pi]) if pi == pe else len(texts[pi])
                per_para.setdefault(pi, []).append((max(0, ls), min(le, len(texts[pi])), " ".join(rep_parts) if pi == ps else ""))

    for pi, pedits in per_para.items():
        _apply_in_paragraph(paras[pi], pedits)

    out = io.BytesIO()
    doc.save(out)
    result = [p.text for p in Document(io.BytesIO(out.getvalue())).paragraphs]

    def nonempty(lines):  # a paragraph whose words were all deleted stays in Word as a blank one
        return [x for x in lines if x.strip()]
    return out.getvalue(), {
        "changes": len(edits),
        "paragraphs_changed": len(per_para),
        "structure_changes": structural,  # paragraph merges/splits can't be mirrored exactly
        "exact": nonempty(result) == nonempty(edited.split("\n")),
    }
