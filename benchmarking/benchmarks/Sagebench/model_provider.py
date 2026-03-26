from imsearch_eval.adapters import TritonModelProvider, NRPModelProvider
from imsearch_eval.framework import Config
import os
from PIL import Image
from typing import Optional


class MixedModelProvider(NRPModelProvider):
    """
    Mixed model provider using NRPModelProvider and TritonModelProvider.

    NRPModelProvider is used for caption generation and TritonModelProvider is
    used for embedding generation.
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
        """Get embedding for text and/or image using Triton model provider."""
        return self.triton_model_provider.get_embedding(text, image, model_name)
