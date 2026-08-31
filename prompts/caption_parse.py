"""Parse VLM caption output into long_caption, short_caption, and keywords."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FIELD_HEADER = re.compile(
    r"^(long_caption|short_caption|caption|keywords)\s*:\s*(.*)$",
    re.IGNORECASE,
)


@dataclass
class ParsedCaption:
    """Structured fields extracted from a VLM caption response."""

    long_caption: str = ""
    short_caption: str = ""
    keywords: str = ""
    raw: str = ""

    @property
    def clip_text(self) -> str:
        """Text for CLIP: short_caption + keywords, or a v1/malformed fallback."""
        short = self.short_caption.strip()
        keywords = self.keywords.strip()
        if short:
            return f"{short} {keywords}".strip()
        return self.bm25_caption or self.raw.strip()

    @property
    def bm25_caption(self) -> str:
        """Text for BM25: long_caption + keywords (full detail)."""
        long_ = self.long_caption.strip()
        keywords = self.keywords.strip()
        return f"{long_} {keywords}".strip()


def parse_caption_response(raw: str | None) -> ParsedCaption:
    """
    Parse ``long_caption:`` / ``short_caption:`` / ``keywords:`` (v2) or
    ``caption:`` / ``keywords:`` (v1). Malformed text becomes ``long_caption``.
    """
    text = (raw or "").strip()
    parsed = ParsedCaption(raw=text)
    if not text:
        return parsed

    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _FIELD_HEADER.match(line.strip())
        if match:
            current = match.group(1).lower()
            rest = match.group(2).strip()
            fields.setdefault(current, [])
            if rest:
                fields[current].append(rest)
            continue
        if current is not None:
            stripped = line.strip()
            if stripped:
                fields[current].append(stripped)

    def _join(name: str) -> str:
        return " ".join(fields.get(name, [])).strip()

    long_caption = _join("long_caption")
    short_caption = _join("short_caption")
    caption = _join("caption")
    keywords = _join("keywords")

    if long_caption or short_caption:
        parsed.long_caption = long_caption or caption
        parsed.short_caption = short_caption
        parsed.keywords = keywords
        return parsed

    if caption or keywords:
        logger.warning(
            "Caption response used v1 fields (caption/keywords); "
            "short_caption is empty so CLIP falls back to long text"
        )
        parsed.long_caption = caption or text
        parsed.keywords = keywords
        return parsed

    logger.warning(
        "Caption response had no labeled fields; treating entire text as long_caption"
    )
    parsed.long_caption = text
    return parsed
