"""Tests for the code where a bug costs the most. No GPU or AI models needed.

    .venv\\Scripts\\python -m unittest discover -s tests -v

Covers: writing edits into the user's own Word file, never cutting long reports,
host / cross-site protection, phone-mode address changes, saved-report storage
(encryption, deleted reports staying deleted, unsafe ids) and upload ids.
"""
import base64
import io
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

# saved reports go to a throwaway folder, never the user's real ones
os.environ["AIC_REPORTS_DIR"] = tempfile.mkdtemp(prefix="rp-tests-")

from docx import Document  # noqa: E402
from docx.shared import Inches  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import access, config, reports, security  # noqa: E402
from app.docx_edit import apply_edits_to_docx  # noqa: E402
from app.extract import extract_text  # noqa: E402
from app.main import app  # noqa: E402

ICON = Path(__file__).resolve().parent.parent / "static" / "icon.png"
SAME_SITE = {"Origin": "http://localhost:8000", "Sec-Fetch-Site": "same-origin"}


def make_docx(paragraphs: int = 40) -> bytes:
    """A small report with the things that must survive: mixed formatting, a table, an image."""
    d = Document()
    d.add_heading("Solar Tracker Report", 0)
    p = d.add_paragraph("The tracker uses ")
    p.add_run("two light sensors").bold = True
    p.add_run(" and a ")
    p.add_run("small servo motor").italic = True
    p.add_run(" to follow the sun across the sky during the day.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Part", "Cost"
    t.cell(1, 0).text, t.cell(1, 1).text = "Servo", "250"
    d.add_picture(str(ICON), width=Inches(1))
    for i in range(paragraphs):
        d.add_paragraph(f"Paragraph {i} explains how the panel angle changes the measured current and voltage readings.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def nonempty(texts):
    return [t for t in texts if t.strip()]


class DocxEditTests(unittest.TestCase):
    """Writing edits back into the original Word file must change only the wording."""

    def test_random_edits_are_exact_and_keep_structure(self):
        data = make_docx()
        original = extract_text("r.docx", data)
        words = list(re.finditer(r"[A-Za-z]{4,}", original))
        rnd = random.Random(7)
        for _ in range(60):
            text = original
            for m in sorted(rnd.sample(words, rnd.randint(1, 8)), key=lambda m: m.start(), reverse=True):
                rep = rnd.choice(["alpha", "", m.group(0) + " really", "a much longer phrase"])
                text = text[:m.start()] + rep + text[m.end():]
            out, stats = apply_edits_to_docx(data, text)
            d0, d1 = Document(io.BytesIO(data)), Document(io.BytesIO(out))
            self.assertTrue(stats["exact"])
            self.assertEqual(nonempty(p.text for p in d1.paragraphs), nonempty(text.split("\n")))
            self.assertEqual(len(d1.paragraphs), len(d0.paragraphs))
            self.assertEqual(len(d1.tables), 1)
            self.assertEqual(len(d1.inline_shapes), 1)
            self.assertEqual([c.text for r in d1.tables[0].rows for c in r.cells], ["Part", "Cost", "Servo", "250"])

    def test_formatting_and_untouched_paragraphs_survive(self):
        data = make_docx()
        text = extract_text("r.docx", data).replace("to follow the sun", "to track the sun", 1)
        out, _ = apply_edits_to_docx(data, text)
        d0, d1 = Document(io.BytesIO(data)), Document(io.BytesIO(out))
        runs = [(r.text, r.bold, r.italic) for r in d1.paragraphs[1].runs]
        self.assertIn(("two light sensors", True, None), runs)
        self.assertIn(("small servo motor", None, True), runs)
        self.assertIn("to track the sun", d1.paragraphs[1].text)
        changed = [i for i, (a, b) in enumerate(zip(d0.paragraphs, d1.paragraphs)) if a.text != b.text]
        self.assertEqual(changed, [1])
        for i, (a, b) in enumerate(zip(d0.paragraphs, d1.paragraphs)):
            if i != 1:
                self.assertEqual(a._p.xml, b._p.xml)  # byte-for-byte identical


class LengthLimitTests(unittest.TestCase):
    """Long reports are refused with a message, never silently cut (a cut used to delete their end)."""

    def setUp(self):
        self.c = TestClient(app, base_url="http://localhost:8000")

    def test_long_report_refused_not_cut(self):
        big = make_docx(paragraphs=1300)  # about 20,000+ words, over the limit
        self.assertGreater(len(extract_text("r.docx", big)), config.MAX_CHARS)
        r = self.c.post("/api/extract", files={"file": ("long.docx", big)}, headers=SAME_SITE)
        self.assertEqual(r.status_code, 413)
        self.assertIn("split it into parts", r.json()["detail"])
        self.assertEqual(self.c.post("/api/analyze", json={"text": "word " * 40_000}, headers=SAME_SITE).status_code, 413)

    def test_formatted_download_keeps_every_paragraph(self):
        data = make_docx(paragraphs=300)
        j = self.c.post("/api/extract", files={"file": ("ok.docx", data)}, headers=SAME_SITE).json()
        self.assertTrue(j["original_id"])
        edited = j["text"].replace("Paragraph 3 ", "Para 3 ", 1)
        r = self.c.post("/api/export-original", json={"text": edited, "original_id": j["original_id"], "filename": "ok.docx"}, headers=SAME_SITE)
        self.assertEqual(r.status_code, 200)
        d = Document(io.BytesIO(r.content))
        self.assertEqual(nonempty(p.text for p in d.paragraphs), nonempty(edited.split("\n")))

    def test_unknown_upload_id_is_refused(self):
        r = self.c.post("/api/export-original", json={"text": "x", "original_id": "nope", "filename": "a.docx"}, headers=SAME_SITE)
        self.assertEqual(r.status_code, 404)


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(app, base_url="http://localhost:8000")

    def test_unknown_host_is_refused(self):  # blocks DNS rebinding
        self.assertEqual(self.c.get("/api/reports", headers={"Host": "evil.example"}).status_code, 421)

    def test_cross_site_writes_are_blocked(self):
        self.assertEqual(self.c.post("/api/reports", json={"text": "x"}, headers={"Origin": "http://evil.example"}).status_code, 403)
        self.assertEqual(self.c.post("/api/reports", json={"text": "x"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)

    def test_security_headers(self):
        h = self.c.get("/").headers
        self.assertIn("script-src 'self'", h["content-security-policy"])
        self.assertEqual(h["x-frame-options"], "DENY")
        self.assertEqual(self.c.get("/api/reports").headers["cache-control"], "no-store")

    def test_phone_mode_follows_a_new_wifi_address(self):
        old_enabled, old_lan = access.enabled, security._lan_addresses
        try:
            access.enabled = lambda: True
            security._lan_addresses = lambda: {"192.168.1.20"}
            security._lan = (0.0, set())
            self.assertTrue(security.host_allowed("192.168.1.20"))
            security._lan_addresses = lambda: {"10.0.0.7"}  # laptop rejoined Wi-Fi with a new address
            security._lan = (0.0, security._lan[1])         # next refresh is due
            self.assertTrue(security.host_allowed("10.0.0.7"))
            self.assertFalse(security.host_allowed("evil.example"))
        finally:
            access.enabled, security._lan_addresses = old_enabled, old_lan
            security._lan = (0.0, set())


class ReportStorageTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(app, base_url="http://localhost:8000")

    def test_saved_reports_are_encrypted_on_disk(self):
        rid = reports.save_report("Confidential thesis text", ai_percent=10)["id"]
        raw = (reports.DIR / f"{rid}.json").read_bytes()
        if os.name == "nt":
            self.assertNotIn(b"Confidential", raw)
        self.assertEqual(reports.get_report(rid)["text"], "Confidential thesis text")
        reports.delete_report(rid)

    def test_deleted_report_does_not_come_back(self):
        rid = self.c.post("/api/reports", json={"text": "Thesis\nbody"}, headers=SAME_SITE).json()["id"]
        self.c.delete(f"/api/reports/{rid}", headers=SAME_SITE)
        r = self.c.post("/api/reports", json={"id": rid, "text": "Thesis\nedited body"}, headers=SAME_SITE)
        self.assertEqual(r.status_code, 410)
        self.assertNotIn(rid, [x["id"] for x in reports.list_reports()])

    def test_unsafe_report_ids_are_rejected(self):
        for bad in ("../../secrets", "ABCDEF123456", "x" * 12):
            with self.assertRaises(ValueError):
                reports.get_report(bad)

    def test_original_file_is_saved_with_the_report(self):
        data = make_docx()
        j = self.c.post("/api/extract", files={"file": ("r.docx", data)}, headers=SAME_SITE).json()
        saved = self.c.post("/api/reports", json={"text": j["text"], "original_id": j["original_id"], "original_name": "r.docx"}, headers=SAME_SITE).json()
        self.assertEqual(saved["original_name"], "r.docx")
        self.assertEqual(reports.get_original(saved["id"]), data)
        reports.delete_report(saved["id"])
        self.assertIsNone(reports.get_original(saved["id"]))


if __name__ == "__main__":
    unittest.main()
