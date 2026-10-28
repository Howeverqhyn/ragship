"""Build a one-page PDF that pypdf can read back. Test helper only."""

from __future__ import annotations


def make_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 16 Tf 40 100 Td ({escaped}) Tj ET\n".encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    # Object 4 needs the "4 0 obj" wrapper. Rebuild it explicitly below.
    objects[3] = b"4 0 obj\n" + objects[3] + b"\nendobj\n"

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray()
    offsets: list[int] = []
    cursor = len(header)
    for obj in objects:
        offsets.append(cursor)
        body.extend(obj)
        cursor += len(obj)
    xref = bytearray(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    xref.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        xref.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{cursor}\n%%EOF\n"
    ).encode("ascii")
    return bytes(header + body + xref + trailer)
