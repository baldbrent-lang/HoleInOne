"""Appearance-based golfer re-identification.

V0 ships a deterministic stub embedder so the data model + matcher are
exercised end-to-end. To plug in a real model:

  - Implement `embed_image_bytes` and `embed_clip_url` against your
    provider (CLIP via Replicate / fal.ai, OSNet self-hosted, etc.).
  - Set `EMBEDDING_PROVIDER=clip` (or whatever value you wire up) in
    Replit Secrets.

The stub mode short-circuits the matcher to round-robin assignment so
the demo flow still produces sensible per-golfer galleries.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable

from ..config import settings

EMBEDDING_DIM = 64  # production CLIP is 512; stub keeps payloads small


def is_stub() -> bool:
    return settings.embedding_provider == "stub"


def embed_image_bytes(data: bytes) -> list[float]:
    """Embed a selfie image. Returns a unit-length vector."""
    if is_stub():
        return _hash_embedding(data)
    raise NotImplementedError(
        f"embedding provider {settings.embedding_provider!r} not implemented"
    )


def embed_clip_url(url: str) -> list[float]:
    """Embed a tee-side clip frame. Returns a unit-length vector.

    Real implementation: download a representative frame from the clip,
    detect the golfer, crop, then run through the same model used for
    selfies.
    """
    if is_stub():
        return _hash_embedding(url.encode("utf-8"))
    raise NotImplementedError(
        f"embedding provider {settings.embedding_provider!r} not implemented"
    )


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _hash_embedding(data: bytes) -> list[float]:
    """Deterministic pseudo-embedding for stub mode."""
    h = hashlib.sha256(data).digest()
    raw = [(h[i % len(h)] - 128) / 128.0 for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]
