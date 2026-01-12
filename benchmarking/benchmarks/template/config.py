"""Template for config.py"""

import os
from imsearch_eval.framework.interfaces import Config


class MyConfig(Config):
    """Configuration for MYBENCHMARK benchmark."""
    
    def __init__(self):
        """Initialize MYBENCHMARK configuration."""
        # TODO: Update with your parameters for the benchmark
        # Environment variables
        self.MYBENCHMARK_DATASET = os.environ.get("MYBENCHMARK_DATASET", "your-dataset/name")
        self.IMAGE_RESULTS_FILE = os.environ.get("IMAGE_RESULTS_FILE", "image_search_results.csv")
        self.QUERY_EVAL_METRICS_FILE = os.environ.get("QUERY_EVAL_METRICS_FILE", "query_eval_metrics.csv")
        self.WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "127.0.0.1")
        self.WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", "8080")
        self.WEAVIATE_GRPC_PORT = os.environ.get("WEAVIATE_GRPC_PORT", "50051")
        self.TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
        self.TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
        self.COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "MYBENCHMARK")
        self.QUERY_METHOD = os.environ.get("QUERY_METHOD", "clip_hybrid_query")
        self.TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        self.SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 0))
        self.WORKERS = int(os.environ.get("WORKERS", 5))
        self.IMAGE_BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", 100))
        self.UPLOAD_TO_S3 = os.environ.get("UPLOAD_TO_S3", "false").lower() == "true"
        self.S3_BUCKET = os.environ.get("S3_BUCKET", "sage_imsearch")
        self.S3_PREFIX = os.environ.get("S3_PREFIX", "dev-metrics")
        self.S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook")
        self.S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
        self.S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
        self.S3_SECURE = os.environ.get("S3_SECURE", "false").lower() == "true"
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)
    
    def get_all(self) -> dict:
        """Get all configuration values."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }