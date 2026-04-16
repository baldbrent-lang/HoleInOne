"""QR code PNG generation."""
from __future__ import annotations

import io

import qrcode


def generate_qr_png(url: str) -> bytes:
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
