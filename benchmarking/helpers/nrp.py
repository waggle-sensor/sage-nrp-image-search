"""NRP managed-LLM fair-use and image-prep helpers for benchmark captioning."""

from __future__ import annotations

import base64
import logging
import os
from typing import Union
from PIL import Image
from imsearch_eval.framework.image_utils import prepare_llm_image_bytes

# https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/
# Per-user max concurrency for short requests (<35% of context length).
NRP_MAX_CONCURRENCY = {
    "kimi": 2,
    "glm-5": 2,
    "deepseek-v4-flash": 2,
    "minimax-m2": 8,
    "qwen3-small": 8,
    "gemma": 8,
    "gemma-small": 8,
    "qwen3": 16,
    "gpt-oss": 16,
    "qwen3-embedding": 16,
}

# Default when LLM_MODEL_PROVIDER=nrp and model is unknown.
NRP_DEFAULT_MAX_CONCURRENCY = 8

def generate_nrp_caption(
    client,
    image: Image.Image,
    prompt: str,
    model_name: Union[str, object] = "gemma",
    enable_thinking: bool = True,
) -> str:
    """Generate a caption via NRP with JPEG resize/quality prep."""
    model_str = model_name.value if hasattr(model_name, "value") else str(model_name)
    image_bytes, mime = prepare_llm_image_bytes(image)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        create_kwargs: dict = {
            "model": model_str,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        }
        if enable_thinking:
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        response = client.chat.completions.create(**create_kwargs)
        answer_str = response.choices[0].message.content
        logging.info("[NRP %s] Final Generated Description: %s", model_str, answer_str)
        return answer_str or ""
    except Exception as exc:
        logging.error(
            "[NRP] Error during %s inference via OpenAI client: %s",
            model_str,
            exc,
        )
        return ""

def nrp_max_concurrency(caption_model_name: str) -> int:
    """Return NRP fair-use max concurrent requests for a model id."""
    key = (caption_model_name or "").strip().lower()
    return NRP_MAX_CONCURRENCY.get(key, NRP_DEFAULT_MAX_CONCURRENCY)


def resolve_workers(llm_model_provider: str, caption_model_name: str) -> int:
    """
    Resolve WORKERS, clamping to NRP fair-use limits when captioning via NRP.

    Triton paths keep the higher default (16) for CLIP dynamic batching.
    NRP + gemma is capped at 8 concurrent requests per user.
    """
    provider = (llm_model_provider or "").strip().lower()
    if provider == "nrp":
        cap = nrp_max_concurrency(caption_model_name)
        default = cap
    else:
        cap = None
        default = 16

    workers = int(os.environ.get("WORKERS", default))
    if cap is not None and workers > cap:
        logging.warning(
            "WORKERS=%s exceeds NRP fair-use max concurrency %s for model %r; "
            "clamping to %s. See https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/",
            workers,
            cap,
            caption_model_name,
            cap,
        )
        workers = cap
    return workers
