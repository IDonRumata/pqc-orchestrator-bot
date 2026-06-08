"""Shared text helpers, chunking with overlap used by ingest and news monitor."""
from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks.

    The splitter prefers to cut on a paragraph or sentence boundary close to the
    target size so chunks stay readable, then steps back by the overlap amount.
    """
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Try to break on the nearest paragraph or sentence boundary.
            window = cleaned[start:end]
            boundary = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
            )
            if boundary > chunk_size // 2:
                end = start + boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks
