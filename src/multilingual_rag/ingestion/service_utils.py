"""Small utility functions used by ingestion modules."""

from __future__ import annotations

from hashlib import sha256


def checksum_text(text: str) -> str:
    """Return a stable SHA-256 checksum for text content."""
    return sha256(text.encode("utf-8")).hexdigest()

