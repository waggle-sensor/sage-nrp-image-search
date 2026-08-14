"""Ablation study helpers for benchmark index and query pipelines."""
import logging
import os
import re
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

def parse_bool_env(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def load_ablation_config() -> dict:
    """
    Load ablation settings from environment variables with validation.
    """
    enable_caption_generation = parse_bool_env("ENABLE_CAPTION_GENERATION", True)
    embed_image = parse_bool_env("EMBED_IMAGE", True)
    embed_caption = parse_bool_env("EMBED_CAPTION", True)
    index_clip_alpha = float(
        os.environ.get("INDEX_CLIP_ALPHA", 0.7)
    )
    enable_bm25 = parse_bool_env("ENABLE_BM25", True)
    query_alpha = float(os.environ.get("QUERY_ALPHA", 0.4))

    if not embed_image and not embed_caption:
        raise ValueError(
            "At least one of EMBED_IMAGE or EMBED_CAPTION must be true for indexing."
        )

    if not enable_caption_generation and embed_caption:
        logging.warning(
            "ENABLE_CAPTION_GENERATION=false with EMBED_CAPTION=true; "
            "disabling caption embedding for indexing."
        )
        embed_caption = False

    return {
        "enable_caption_generation": enable_caption_generation,
        "embed_image": embed_image,
        "embed_caption": embed_caption,
        "index_clip_alpha": index_clip_alpha,
        "enable_bm25": enable_bm25,
        "query_alpha": query_alpha,
    }


def resolve_query_alpha(ablation: dict) -> float:
    """Return hybrid query alpha; 1.0 when BM25 is disabled."""
    if not ablation["enable_bm25"]:
        return 1.0
    return ablation["query_alpha"]


def generate_index_caption(
    model_provider,
    image: Any,
    config,
    fallback_caption: str = "",
) -> str:
    """
    Generate a caption for indexing, or return empty string when captioning is disabled.
    """
    if not config.enable_caption_generation:
        return ""

    caption = model_provider.generate_caption(
        image,
        config.caption_model_prompt,
        model_name=config.caption_model_name,
        enable_thinking=config.nrp_enable_thinking,
    )
    if not caption:
        return fallback_caption
    return caption


def get_triton_model_utils(model_provider):
    """Resolve TritonModelUtils from a benchmark MixedModelProvider."""
    triton_provider = getattr(model_provider, "triton_model_provider", None)
    if triton_provider is None:
        raise ValueError("Model provider does not expose triton_model_provider")

    model_utils = getattr(triton_provider, "model_utils", None)
    if model_utils is None:
        raise ValueError("Triton model provider does not expose model_utils")
    return model_utils


def get_index_embedding(model_provider, caption: str, image: Any, config):
    """
    Build the index-time CLIP embedding according to ablation modality settings.

    Uses imsearch_eval's TritonModelUtils.get_clip_embeddings(), which accepts
    a configurable fusion alpha (unlike TritonModelProvider.get_embedding()).
    """
    model_utils = get_triton_model_utils(model_provider)

    if config.embed_image and config.embed_caption:
        return model_utils.get_clip_embeddings(
            caption, image=image, alpha=config.index_clip_alpha
        )

    if config.embed_image:
        return model_utils.get_clip_embeddings("", image=image, alpha=1.0)

    return model_utils.get_clip_embeddings(caption, image=None, alpha=0.0)


def _to_list(embedding) -> list:
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return list(embedding)


def get_index_embedding_pair(model_provider, caption: str, image: Any, config):
    """
    Build separate caption_vector and image_vector for Milvus indexing.

    Disabled modalities are stored as zero vectors; the corresponding hybrid
    search leg is omitted at query time via enable_image_vector / enable_caption_vector.
    """
    model_utils = get_triton_model_utils(model_provider)
    caption_vec, image_vec = model_utils.get_clip_embedding_pair(caption or "", image)
    if caption_vec is None or image_vec is None:
        return None, None

    dim = len(image_vec)
    if not config.embed_caption:
        caption_vec = np.zeros(dim, dtype=np.float32)
    if not config.embed_image:
        image_vec = np.zeros(dim, dtype=np.float32)
    return _to_list(caption_vec), _to_list(image_vec)


def build_search_text(
    caption: str,
    extra_fields: Optional[Iterable[Any]] = None,
    max_length: int = 65535,
) -> str:
    """Concatenate caption and optional metadata fields for BM25 search_text."""
    parts = [caption or ""]
    if extra_fields:
        parts.extend("" if field is None else str(field) for field in extra_fields)
    return " ".join(part for part in parts if part).strip()[:max_length]


def cache_index_image(image: Image.Image, image_id: Any, cache_dir: str) -> str:
    """Write a JPEG under cache_dir and return the path for Milvus `link`.

    Unused by Milvus query-time CLIP rerank (stored ``image_vector``s). Kept for
    callers that still want a local file path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(image_id) or "image")
    path = os.path.join(cache_dir, f"{safe_id}.jpg")
    image.convert("RGB").save(path, format="JPEG")
    return path


def milvus_index_payload(
    model_provider,
    caption: str,
    image: Image.Image,
    image_id: Any,
    config,
    extra_search_fields: Optional[Iterable[Any]] = None,
):
    """
    Build Milvus index fields: caption_vector, image_vector, search_text, link.

    ``link`` is left empty: CLIP rerank uses stored ``image_vector``s, so JPEG
    caching under IMAGE_CACHE_DIR is skipped to save ephemeral disk I/O.

    Raises ValueError if embeddings cannot be generated.
    """
    caption_vector, image_vector = get_index_embedding_pair(
        model_provider, caption, image, config
    )
    if caption_vector is None or image_vector is None:
        raise ValueError("Failed to generate CLIP embedding pair")
    _ = image_id  # reserved; link left empty (rerank uses image_vector)
    link = ""
    search_text = build_search_text(caption, extra_search_fields)
    return caption_vector, image_vector, search_text, link
