"""Read PDF, Markdown, and plain text into a single string."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".text"}


class DocumentError(ValueError):
    """The caller sent a document this service cannot index."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def safe_source_name(name: str) -> str:
    """Keep the basename. A missing suffix becomes .txt; any other suffix must be supported."""
    base = Path(name).name.strip()
    if base in {"", ".", ".."} or "\x00" in base:
        raise DocumentError("invalid source name")
    suffix = Path(base).suffix.lower()
    if not suffix:
        return f"{base}.txt"
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"unsupported file type: {suffix}. Use PDF, Markdown, or TXT.",
            status_code=415,
        )
    return base


def is_supported_name(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_SUFFIXES


def load_document(filename: str, data: bytes) -> str:
    name = safe_source_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"unsupported file type: {suffix or 'unknown'}. Use PDF, Markdown, or TXT.",
            status_code=415,
        )
    if not data or not data.strip():
        raise DocumentError("empty document")
    if suffix == ".pdf":
        text = _load_pdf(data)
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("document is not valid UTF-8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise DocumentError("document contains no text")
    return normalized


def _load_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentError("could not read PDF") from exc
    return "\n\n".join(pages)
