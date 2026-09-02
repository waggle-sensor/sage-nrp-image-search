"""Shared caption-prompt catalog for weavloader and the benchmarking suite."""

from __future__ import annotations

import os
from pathlib import Path

from .caption_parse import ParsedCaption, parse_caption_response

DEFAULT_PROMPT_ID = "scientific_two_captions_v1"
_CATALOG_DIR = Path(__file__).resolve().parent

__all__ = [
    "DEFAULT_PROMPT_ID",
    "ParsedCaption",
    "get_prompt",
    "list_prompts",
    "load_caption_prompt",
    "parse_caption_response",
]


def list_prompts() -> list[str]:
    """Return prompt ids (file stems) available in this catalog."""
    return sorted(path.stem for path in _CATALOG_DIR.glob("*.txt"))


def get_prompt(prompt_id: str) -> str:
    """Load a prompt body by catalog id (filename without ``.txt``)."""
    path = _CATALOG_DIR / f"{prompt_id}.txt"
    if not path.is_file():
        available = ", ".join(list_prompts()) or "(none)"
        raise ValueError(
            f"Unknown prompt id {prompt_id!r}. Available: {available}"
        )
    return path.read_text(encoding="utf-8").strip()


def load_caption_prompt() -> str:
    """
    Resolve the caption prompt from the environment.

    ``CAPTION_MODEL_PROMPT`` overrides with a raw string. Otherwise
    ``CAPTION_PROMPT_ID`` selects a catalog file (default: ``scientific_two_captions_v1``).
    """
    raw = os.environ.get("CAPTION_MODEL_PROMPT")
    if raw:
        return raw
    prompt_id = os.environ.get("CAPTION_PROMPT_ID", DEFAULT_PROMPT_ID)
    return get_prompt(prompt_id)
