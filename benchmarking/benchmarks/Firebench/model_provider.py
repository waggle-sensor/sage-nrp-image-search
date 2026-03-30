from imsearch_eval.adapters import TritonModelProvider, NRPModelProvider
from imsearch_eval.framework import Config
import os
from PIL import Image
from typing import Optional

class MixedModelProvider(NRPModelProvider):
    """
    Mixed model provider using NRPModelProvider and TritonModelProvider.

    NRPModelProvider is used for caption generation and TritonModelProvider is used for embedding generation.
    """

    def __init__(
        self,
        api_key: str = os.environ.get("NRP_API_KEY"),
        base_url: str = "https://ellm.nrp-nautilus.io/v1",
        triton_model_provider: TritonModelProvider = None,
        config: Config = None,
        **client_kwargs,
    ):
        """
        Initialize Mixed model provider.

        Args:
            api_key: NRP API token (defaults to environment variable "NRP_API_KEY").
            base_url: Envoy gateway URL. Defaults to the NRP-managed LLM endpoint.
            triton_model_provider: Triton model provider.
            config: Config object.
            **client_kwargs: Optional extra arguments passed to the NRPModelProvider.
        """      
        super().__init__(api_key=api_key, base_url=base_url, **client_kwargs)   
        self.triton_model_provider = triton_model_provider
        self.config = config
        
        # determine which model provider to use for caption generation
        if self.config.llm_model_provider == "triton":
            self.model_utils = self.triton_model_provider.model_utils
        elif self.config.llm_model_provider == "nrp":
            self.config.is_nrp_key_set()
        else:
            raise ValueError(f"Invalid model provider: {self.config.llm_model_provider} not supported")
    

    def get_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ):
        """
        Get embedding for text and/or image using triton model provider.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use ("clip", "colbert", "align")
            
        Returns:
            Embedding vector (numpy array)
        """
        return self.triton_model_provider.get_embedding(text, image, model_name)