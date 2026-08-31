# Caption prompt catalog

Versioned VLM prompts shared by weavloader ingest and the benchmarking suite.

## Files

| Id | File | Notes |
|----|------|--------|
| `scientific_detailed_v1` | `scientific_detailed_v1.txt` | Legacy two-field output (`caption:` + `keywords:`). |
| `scientific_two_captions_v1` | `scientific_two_captions_v1.txt` | **Default.** Emits `long_caption`, `short_caption` (≤55 tokens, visual only), and 15 `keywords`. |

Add a prompt by creating a new `.txt` file; the file stem is the prompt id.

## Selection

| Variable | Role |
|----------|------|
| `CAPTION_PROMPT_ID` | Catalog id (default `scientific_two_captions_v1`) |
| `CAPTION_MODEL_PROMPT` | Raw prompt override; if set, the catalog is ignored |

Python:

```python
from prompts import get_prompt, list_prompts, load_caption_prompt

load_caption_prompt()  # env-aware
get_prompt("scientific_detailed_v1")
```

Local benchmark runs already put the repo root on `PYTHONPATH` (`make run-local`). Weavloader and Docker images stage this directory as `/app/prompts`.

## How the v2 fields are used

- **CLIP** (`caption_vector`) embeds `short_caption` + `keywords` in one 77-token pass.
- **BM25** (`search_text`) indexes `long_caption` + `keywords` plus SAGE metadata.
- Milvus stores `long_caption` and `short_caption` as separate VARCHAR fields.

Existing collections must be re-captioned and re-embedded after switching to v2.
