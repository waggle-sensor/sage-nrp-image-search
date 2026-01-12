"""INQUIRE-specific configuration/hyperparameters."""

import os
from weaviate.classes.config import VectorDistances, Configure
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

from imsearch_eval.framework.interfaces import Config


class INQUIREConfig(Config):
    """Configuration for INQUIRE benchmark."""
    
    def __init__(self):
        """Initialize INQUIRE configuration."""
        # Environment variables
        self.INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")
        self.IMAGE_RESULTS_FILE = os.environ.get("IMAGE_RESULTS_FILE", "image_search_results.csv")
        self.QUERY_EVAL_METRICS_FILE = os.environ.get("QUERY_EVAL_METRICS_FILE", "query_eval_metrics.csv")
        self.WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "127.0.0.1")
        self.WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
        self.WEAVIATE_GRPC_PORT = os.environ.get("WEAVIATE_GRPC_PORT", "50051")
        self.TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
        self.TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
        self.COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "INQUIRE")
        self.QUERY_METHOD = os.environ.get("QUERY_METHOD", "clip_hybrid_query")
        self.TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        self.SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 0))
        self.WORKERS = int(os.environ.get("WORKERS", 0))
        self.IMAGE_BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", 100))
        self.UPLOAD_TO_S3 = os.environ.get("UPLOAD_TO_S3", "false").lower() == "true"
        self.S3_BUCKET = os.environ.get("S3_BUCKET", "sage_imsearch")
        self.S3_PREFIX = os.environ.get("S3_PREFIX", "dev-metrics")
        self.S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook")
        self.S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
        self.S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
        self.S3_SECURE = os.environ.get("S3_SECURE", "false").lower() == "true"

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

