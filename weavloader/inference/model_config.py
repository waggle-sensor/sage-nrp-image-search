'''This file contains the hyper parameters that can be changed to fine tune
the system. '''

import os

from .caption_parse import load_caption_prompt

nrp_enable_thinking = (
    os.environ.get("NRP_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
)
# Caption LLM image caps: LLM_MAX_IMAGE_* / LLM_IMAGE_BYTE_LIMITING (see inference/image_utils.py)
align_alpha = 0.7
clip_alpha = 0.7  # Used by fuse_embeddings() only; ingest stores caption/image vectors separately.
caption_model_prompt = load_caption_prompt()
