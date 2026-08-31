"""Make the repo-root ``prompts`` package importable, then re-export the parser."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_prompts_on_path() -> None:
    """Insert the directory that contains ``prompts/`` onto ``sys.path``."""
    here = Path(__file__).resolve()
    candidates = list(here.parents) + [
        Path("/app"),
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (root / "prompts" / "__init__.py").is_file():
            path = str(root)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    raise ImportError(
        "Could not find the prompts catalog. Stage prompts/ next to the app "
        "(Docker: /app/prompts) or run with the repo root on PYTHONPATH."
    )


ensure_prompts_on_path()

from prompts.caption_parse import ParsedCaption, parse_caption_response  # noqa: E402
from prompts import DEFAULT_PROMPT_ID, get_prompt, list_prompts, load_caption_prompt  # noqa: E402

__all__ = [
    "DEFAULT_PROMPT_ID",
    "ParsedCaption",
    "ensure_prompts_on_path",
    "get_prompt",
    "list_prompts",
    "load_caption_prompt",
    "parse_caption_response",
]
