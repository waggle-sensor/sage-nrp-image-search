"""INQUIRE-specific configuration/hyperparameters."""

import os
import sys

# Add framework to path
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../framework'))
if framework_path not in sys.path:
    sys.path.insert(0, framework_path)

from weaviate.classes.config import VectorDistances, Configure
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy
from framework.interfaces import Config


class INQUIREConfig(Config):
    """Configuration for INQUIRE benchmark."""
    
    def __init__(self):
        """Initialize INQUIRE configuration."""
        # Weaviate HNSW hyperparameters
        self.hnsw_dist_metric = VectorDistances.COSINE
        self.hnsw_ef = -1
        self.hnsw_ef_construction = 100
        self.hnsw_maxConnections = 50
        self.hsnw_dynamicEfMax = 500
        self.hsnw_dynamicEfMin = 200
        self.hnsw_ef_factor = 20
        self.hsnw_filterStrategy = VectorFilterStrategy.ACORN
        self.hnsw_flatSearchCutoff = 40000
        self.hnsw_vector_cache_max_objects = 1e12
        self.hnsw_quantizer = Configure.VectorIndex.Quantizer.pq(
            training_limit=500000
        )
        
        # Model hyperparameters
        self.align_alpha = 0.7
        self.clip_alpha = 0.7
        
        # Query hyperparameters
        self.response_limit = 25
        self.query_alpha = 0.4
        self.hybrid_weight = 0.7
        self.colbert_weight = 0.3
        self.hybrid_colbert_blend_top_k = 25
        
        # Caption prompts
        self.qwen2_5_prompt = """
role:
You are a world-class Scientific Image Captioning Expert.

context:
You will be shown a scientific image captured by edge devices. Your goal is to analyze its content and significance in detail. 

task:
Generate exactly one scientifically detailed caption that accurately describes what is visible in the image and its scientific relevance. 
Make it as detailed as possible. Also extract text and numbers from the images.

constraints:
- Only return:
  1. A single caption.
  2. a list of 15 keywords relevant to the image.
- Do not include any additional text, explanations, or formatting.

format:
  caption: <your_scientific_caption_here>
  keywords: <keyword1>, <keyword2>, ...
"""
        self.gemma3_prompt = self.qwen2_5_prompt
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)
    
    def get_all(self) -> dict:
        """Get all configuration values."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

