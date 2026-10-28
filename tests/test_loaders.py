"""PDF, Markdown, and TXT loaders."""

from __future__ import annotations

import pytest

from ragship.ingest.loaders import DocumentError, load_document
from tests.pdfutil import make_pdf


def test_markdown_and_text():
    assert "hello" in load_document("note.md", b"# hello\n")
    assert load_document("note.txt", b"\xef\xbb\xbfplain text\n") == "plain text"


def test_pdf_roundtrip():
    text = load_document("manual.pdf", make_pdf("RagshipPdfToken ZEBRAUNIQUE sails."))
    assert "ZEBRAUNIQUE" in text


def test_bad_utf8_and_unsupported():
    with pytest.raises(DocumentError, match="UTF-8"):
        load_document("note.txt", b"\xff\xfe bad")
    with pytest.raises(DocumentError) as caught:
        load_document("sheet.docx", b"nope")
    assert caught.value.status_code == 415
