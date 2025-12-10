"""
Triton-based adapters for benchmarking framework.

This module provides all Triton-related adapters:
- TritonModelUtils: Implementation of ModelUtils interface using Triton
- TritonModelProvider: ModelProvider implementation using TritonModelUtils
"""

import logging
import numpy as np
import tritonclient.grpc as TritonClient
from typing import Optional
from PIL import Image
import sys
import os

# Add framework to path
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../framework'))
if framework_path not in sys.path:
    sys.path.insert(0, framework_path)

from framework.interfaces import ModelProvider
from framework.model_utils import ModelUtils, fuse_embeddings


class TritonModelUtils(ModelUtils):
    """
    Triton-based implementation of ModelUtils.
    
    Provides embedding and caption generation using models served via Triton Inference Server.
    """
    
    def __init__(self, triton_client: TritonClient.InferenceServerClient):
        """
        Initialize Triton model utils.
        
        Args:
            triton_client: Triton inference server client
        """
        self.triton_client = triton_client
    
    def calculate_embedding(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ) -> Optional[np.ndarray]:
        """
        Calculate embedding for text and/or image using Triton.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use ("clip", "colbert", "align")
        
        Returns:
            Embedding vector (numpy array) or None on error
        """
        if model_name == "clip":
            return self.get_clip_embeddings(text, image)
        elif model_name == "colbert":
            return self.get_colbert_embedding(text)
        elif model_name == "align":
            return self.get_allign_embeddings(text, image)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def generate_caption(
        self,
        image: Image.Image,
        model_name: str = "gemma3"
    ) -> Optional[str]:
        """
        Generate a caption for an image using Triton.
        
        Args:
            image: PIL Image to caption
            model_name: Name of the model to use ("gemma3", "qwen2_5")
        
        Returns:
            Generated caption string or None on error
        """
        if model_name == "gemma3":
            return self.gemma3_run_model(image)
        elif model_name == "qwen2_5":
            return self.qwen2_5_run_model(image)
        else:
            raise ValueError(f"Unknown caption model name: {model_name}")
    
    def get_clip_embeddings(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        alpha: float = 0.7
    ) -> Optional[np.ndarray]:
        """
        Embed text and/or image using CLIP encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            alpha: Weight for fusing image and text embeddings (default: 0.7)
        
        Returns:
            Fused embedding vector (numpy array) or None on error
        """
        # Prepare inputs
        text_bytes = text.encode("utf-8")
        text_np = np.array([text_bytes], dtype="object")
        
        # Prepare image input
        if image is not None:
            image_np = np.array(image).astype(np.float32)
        else:
            image_np = np.zeros((1, 1, 3), dtype=np.float32)
        
        # Create Triton input objects
        inputs = [
            TritonClient.InferInput("text", [1], "BYTES"),
            TritonClient.InferInput("image", list(image_np.shape), "FP32")
        ]
        
        inputs[0].set_data_from_numpy(text_np)
        inputs[1].set_data_from_numpy(image_np)
        
        outputs = [
            TritonClient.InferRequestedOutput("text_embedding"),
            TritonClient.InferRequestedOutput("image_embedding")
        ]
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="clip", inputs=inputs, outputs=outputs)
            text_embedding = results.as_numpy("text_embedding")[0]
            image_embedding = results.as_numpy("image_embedding")[0]
        except Exception as e:
            logging.error(f"Error during CLIP inference: {str(e)}")
            return None
        
        # Fuse embeddings
        if image is not None:
            embedding = fuse_embeddings(image_embedding, text_embedding, alpha=alpha)
        else:
            embedding = text_embedding
        
        return embedding
    
    def get_colbert_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Embed text using ColBERT encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
        
        Returns:
            Token-level embeddings of shape [num_tokens, 128] or None on error
        """
        # Prepare input
        text_bytes = text.encode("utf-8")
        input_tensor = np.array([text_bytes], dtype="object")  # batch size = 1
        
        # Prepare inputs & outputs for Triton
        inputs = [
            TritonClient.InferInput("text", input_tensor.shape, "BYTES")
        ]
        outputs = [
            TritonClient.InferRequestedOutput("embedding"),
            TritonClient.InferRequestedOutput("token_lengths")
        ]
        
        # Add tensors
        inputs[0].set_data_from_numpy(input_tensor)
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="colbert", inputs=inputs, outputs=outputs)
            
            # Retrieve and reshape output
            emb_flat = results.as_numpy("embedding")  # shape: (1, max_len * 128)
            token_lengths = results.as_numpy("token_lengths")  # shape: (1,)
            num_tokens = token_lengths[0]
            
            # Reshape and unpad
            emb_3d = emb_flat.reshape(1, -1, 128)
            token_embeddings = emb_3d[0, :num_tokens, :]  # shape: [num_tokens, 128]
        except Exception as e:
            logging.error(f"Error during ColBERT inference: {str(e)}")
            return None
        
        return token_embeddings
    
    def get_allign_embeddings(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        alpha: float = 0.7
    ) -> Optional[np.ndarray]:
        """
        Embed text and/or image using ALIGN encoder served via Triton Inference Server.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            alpha: Weight for fusing image and text embeddings (default: 0.7)
        
        Returns:
            Fused embedding vector (numpy array) or None on error
        """
        # Prepare inputs
        text_bytes = text.encode("utf-8")
        text_np = np.array([text_bytes], dtype="object")
        
        # Prepare image input
        if image is not None:
            image_np = np.array(image).astype(np.float32)
        else:
            image_np = np.zeros((1, 1, 3), dtype=np.float32)
        
        # Create Triton input objects
        inputs = [
            TritonClient.InferInput("text", [1], "BYTES"),
            TritonClient.InferInput("image", list(image_np.shape), "FP32")
        ]
        
        inputs[0].set_data_from_numpy(text_np)
        inputs[1].set_data_from_numpy(image_np)
        
        outputs = [
            TritonClient.InferRequestedOutput("text_embedding"),
            TritonClient.InferRequestedOutput("image_embedding")
        ]
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="align", inputs=inputs, outputs=outputs)
            text_embedding = results.as_numpy("text_embedding")[0]
            image_embedding = results.as_numpy("image_embedding")[0]
        except Exception as e:
            logging.error(f"Error during ALIGN inference: {str(e)}")
            return None
        
        # Fuse embeddings
        if image is not None:
            embedding = fuse_embeddings(image_embedding, text_embedding, alpha=alpha)
        else:
            embedding = text_embedding
        
        return embedding
    
    def gemma3_run_model(self, image: Image.Image) -> Optional[str]:
        """
        Generate a caption for an image using Gemma3 model served via Triton.
        
        Args:
            image: PIL Image to caption
        
        Returns:
            Generated caption string or None on error
        """
        # Prepare image input
        image_np = np.array(image).astype(np.float32)
        
        # Create Triton input objects
        inputs = [
            TritonClient.InferInput("image", list(image_np.shape), "FP32")
        ]
        inputs[0].set_data_from_numpy(image_np)
        
        outputs = [
            TritonClient.InferRequestedOutput("caption")
        ]
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="gemma3", inputs=inputs, outputs=outputs)
            caption = results.as_numpy("caption")
            
            # Handle different return types
            if isinstance(caption, np.ndarray):
                if caption.dtype == object:
                    # If it's a bytes array, decode it
                    return caption[0].decode("utf-8") if len(caption) > 0 else ""
                else:
                    return str(caption[0]) if len(caption) > 0 else ""
            else:
                return str(caption) if caption else ""
        except Exception as e:
            logging.error(f"Error during Gemma3 inference: {str(e)}")
            return None
    
    def qwen2_5_run_model(self, image: Image.Image) -> Optional[str]:
        """
        Generate a caption for an image using Qwen2.5-VL model served via Triton.
        
        Args:
            image: PIL Image to caption
        
        Returns:
            Generated caption string or None on error
        """
        # Prepare image input
        image_np = np.array(image).astype(np.float32)
        
        # Create Triton input objects
        inputs = [
            TritonClient.InferInput("image", list(image_np.shape), "FP32")
        ]
        inputs[0].set_data_from_numpy(image_np)
        
        outputs = [
            TritonClient.InferRequestedOutput("caption")
        ]
        
        # Run inference
        try:
            results = self.triton_client.infer(model_name="qwen2_5", inputs=inputs, outputs=outputs)
            caption = results.as_numpy("caption")
            
            # Handle different return types
            if isinstance(caption, np.ndarray):
                if caption.dtype == object:
                    # If it's a bytes array, decode it
                    return caption[0].decode("utf-8") if len(caption) > 0 else ""
                else:
                    return str(caption[0]) if len(caption) > 0 else ""
            else:
                return str(caption) if caption else ""
        except Exception as e:
            logging.error(f"Error during Qwen2.5 inference: {str(e)}")
            return None


class TritonModelProvider(ModelProvider):
    """Triton model provider using TritonModelUtils."""
    
    def __init__(self, triton_client):
        """
        Initialize Triton model provider.
        
        Args:
            triton_client: Triton inference server client
        """
        self.triton_client = triton_client
        self.model_utils = TritonModelUtils(triton_client)
    
    def get_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        model_name: str = "clip"
    ):
        """
        Get embedding for text and/or image.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use ("clip", "colbert", "align")
            
        Returns:
            Embedding vector (numpy array)
        """
        return self.model_utils.calculate_embedding(text, image, model_name)
    
    def generate_caption(self, image: Image.Image, model_name: str = "gemma3") -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            model_name: Name of the model to use ("gemma3", "qwen2_5")
            
        Returns:
            Generated caption string
        """
        result = self.model_utils.generate_caption(image, model_name)
        return result if result else ""

