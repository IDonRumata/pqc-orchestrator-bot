"""Shared text helpers, chunking with overlap used by ingest and news monitor."""
from __future__ import annotations

import html
import re

# Telegram HTML tags we are willing to keep in model output.
_ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s", "code", "pre", "blockquote")


def to_telegram_html(text: str) -> str:
    """Convert stray Markdown into safe Telegram HTML.

    Models sometimes ignore the no-Markdown instruction and emit ``#`` headings,
    ``**bold**`` or ``---`` rules, which Telegram renders literally. This function
    converts the common cases to HTML and escapes every other ``<``, ``>`` and
    ``&`` so the message can never break Telegram HTML parsing, while preserving
    the small set of allowed tags the model is asked to use.
    """
    if not text:
        return ""

    # Protect already valid allowed tags by swapping them for private sentinels.
    def _protect(match: re.Match[str]) -> str:
        slash, name = match.group(1), match.group(2).lower()
        if name in _ALLOWED_TAGS:
            return f"\x00{slash}{name}\x01"
        return match.group(0)

    text = re.sub(r"<(/?)([a-zA-Z]+)>", _protect, text)

    # Markdown bold -> sentinel bold.
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"\x00b\x01{m.group(1)}\x00/b\x01", text, flags=re.S)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", lambda m: f"\x00b\x01{m.group(1)}\x00/b\x01", text, flags=re.S)

    # Markdown headings (#, ##, ###) -> bold line. Use [ \t] (not \s) around the
    # line anchors so trailing newlines are never swallowed and blank lines that
    # separate consecutive headings are preserved.
    text = re.sub(
        r"(?m)^[ \t]{0,3}#{1,6}[ \t]*(.+?)[ \t]*#*[ \t]*$",
        lambda m: f"\x00b\x01{m.group(1)}\x00/b\x01",
        text,
    )

    # Horizontal rules made of - * _ on their own line -> removed.
    text = re.sub(r"(?m)^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", text)

    # Markdown bullets * or + at line start -> a clean dash bullet.
    text = re.sub(r"(?m)^([ \t]*)[*+][ \t]+", r"\1- ", text)

    # Escape everything that is left, then restore the sentinels as real tags.
    text = html.escape(text, quote=False)
    text = text.replace("\x00", "<").replace("\x01", ">")

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
