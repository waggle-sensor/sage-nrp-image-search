"""Shared MixedModelProvider for benchmark jobs (Triton CLIP + NRP/Triton caption)."""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image

from imsearch_eval.adapters import NRPModelProvider, TritonModelProvider
from imsearch_eval.framework import Config

from imsearch_eval.framework.image_utils import ensure_rgb

from helpers.nrp import generate_nrp_caption


class MixedModelProvider(NRPModelProvider):
    """
    Mixed model provider using NRPModelProvider and TritonModelProvider.

    TritonModelProvider handles CLIP embeddings. Captioning uses provider-
    agnostic LLM image prep (``prepare_llm_image`` / ``prepare_llm_image_bytes``)
    inside each caption backend (NRP gateway or Triton VLMs).
    """

    def __init__(
        self,
        api_key: str = os.environ.get("NRP_API_KEY"),
        base_url: str = "https://ellm.nrp-nautilus.io/v1",
        triton_model_provider: TritonModelProvider = None,
        config: Config = None,
        **client_kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, **client_kwargs)
        self.triton_model_provider = triton_model_provider
        self.config = config

        if self.config.llm_model_provider == "triton":
            self.model_utils = self.triton_model_provider.model_utils
        elif self.config.llm_model_provider == "nrp":
            self.config.is_nrp_key_set()
        else:
            raise ValueError(
                f"Invalid model provider: {self.config.llm_model_provider} not supported"
            )

    def get_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "clip",
    ):
        """Get embedding for text and/or image using the Triton model provider."""
        if image is not None:
            image = ensure_rgb(image)
        return self.triton_model_provider.get_embedding(text, image, model_name)

    def generate_caption(
        self,
        image: Image.Image,
        prompt: str,
        model_name: str = "gemma",
        enable_thinking: bool = True,
    ) -> str:
        if self.config.llm_model_provider == "triton":
            return self.triton_model_provider.generate_caption(
                image,
                prompt,
                model_name=model_name,
                enable_thinking=enable_thinking,
            )

        return generate_nrp_caption(
            self.client,
            image,
            prompt,
            model_name=model_name,
            enable_thinking=enable_thinking,
        )
