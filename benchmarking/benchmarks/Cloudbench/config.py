"""CloudBench-specific configuration/hyperparameters."""

import os
from weaviate.classes.config import VectorDistances, Configure
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

from imsearch_eval.framework.interfaces import Config
from helpers.ablation import load_ablation_config
from helpers.backend import apply_vector_db_config
from helpers.caption_parse import load_caption_prompt
from helpers.nrp import resolve_workers


class CloudBenchConfig(Config):
    """Configuration for CloudBench benchmark (cloud/atmospheric image retrieval)."""

    def __init__(self):
        """Initialize CloudBench configuration."""
        # dataset parameters
        self.cloudbench_dataset = os.environ.get(
            "CLOUDBENCH_DATASET", "sagecontinuum/CloudBench"
        )
        self.sample_size = int(os.environ.get("SAMPLE_SIZE", 0))
        self.seed = int(os.environ.get("SEED", 42))
        self._hf_token = os.environ.get("HF_TOKEN", "")
        # Upload parameters
        self._upload_to_s3 = os.environ.get("UPLOAD_TO_S3", "false").lower() == "true"
        self._s3_bucket = os.environ.get("S3_BUCKET", "sage_imsearch")
        self._s3_prefix = os.environ.get("S3_PREFIX", "dev-metrics/cloudbench")
        self._s3_endpoint = os.environ.get(
            "S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook"
        )
        self._s3_access_key = os.environ.get("S3_ACCESS_KEY", "")
        self._s3_secret_key = os.environ.get("S3_SECRET_KEY", "")
        self._s3_secure = os.environ.get("S3_SECURE", "false").lower() == "true"
        self._image_results_file = os.environ.get(
            "IMAGE_RESULTS_FILE", "image_search_results.csv"
        )
        self._query_eval_metrics_file = os.environ.get(
            "QUERY_EVAL_METRICS_FILE", "query_eval_metrics.csv"
        )
        self._config_values_file = os.environ.get(
            "CONFIG_VALUES_FILE", "config_values.csv"
        )

        # Collection
        self._collection_name = os.environ.get("COLLECTION_NAME", "CloudBench")

        # model provider parameters
        self.llm_model_provider = os.environ.get(
            "LLM_MODEL_PROVIDER", "triton"
        ).lower()
        self.caption_model_name = os.environ.get("CAPTION_MODEL_NAME", "gemma")
        self.nrp_enable_thinking = os.environ.get("NRP_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")

        # Triton parameters
        self._triton_host = os.environ.get("TRITON_HOST", "triton")
        self._triton_port = os.environ.get("TRITON_PORT", "8001")

        # Workers parameters
        self._workers = resolve_workers(self.llm_model_provider, self.caption_model_name)
        self._image_batch_size = int(os.environ.get("IMAGE_BATCH_SIZE", 32))
        self._query_batch_size = int(os.environ.get("QUERY_BATCH_SIZE", 16))

        # Logging parameters
        self._log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

        # Weaviate HNSW hyperparameters
        self.hnsw_dist_metric = getattr(
            VectorDistances, os.environ.get("HNSW_DIST_METRIC", "COSINE").upper()
        )
        self.hnsw_ef = int(os.environ.get("HNSW_EF", -1))
        self.hnsw_ef_construction = int(os.environ.get("HNSW_EF_CONSTRUCTION", 100))
        self.hnsw_maxConnections = int(os.environ.get("HNSW_MAX_CONNECTIONS", 50))
        self.hnsw_dynamicEfMax = int(os.environ.get("HNSW_DYNAMIC_EF_MAX", 500))
        self.hnsw_dynamicEfMin = int(os.environ.get("HNSW_DYNAMIC_EF_MIN", 200))
        self.hnsw_ef_factor = int(os.environ.get("HNSW_EF_FACTOR", 20))
        self.hnsw_filterStrategy = getattr(
            VectorFilterStrategy,
            os.environ.get("HNSW_FILTER_STRATEGY", "ACORN").upper(),
        )
        self.hnsw_flatSearchCutoff = int(
            os.environ.get("HNSW_FLAT_SEARCH_CUTOFF", 40000)
        )
        self.hnsw_vector_cache_max_objects = int(
            os.environ.get("HNSW_VECTOR_CACHE_MAX_OBJECTS", 1e12)
        )
        self.hnsw_quantizer = Configure.VectorIndex.Quantizer.pq(
            training_limit=int(
                os.environ.get("HNSW_QUANTIZER_TRAINING_LIMIT", 500000)
            )
        )

        # Ablation parameters
        ablation = load_ablation_config()
        self.enable_caption_generation = ablation["enable_caption_generation"]
        self.embed_image = ablation["embed_image"]
        self.embed_caption = ablation["embed_caption"]
        self.index_clip_alpha = ablation["index_clip_alpha"]
        self.enable_bm25 = ablation["enable_bm25"]
        self.skip_index = ablation["skip_index"]

        # Vector DB + query parameters
        apply_vector_db_config(self, ablation, query_properties=["long_caption"])
        self.response_limit = int(os.environ.get("RESPONSE_LIMIT", 25))
        self.caption_model_prompt = load_caption_prompt()

    @staticmethod
    def is_nrp_key_set():
        """Check if NRP API key is set."""
        if os.environ.get("NRP_API_KEY", "") == "":
            raise ValueError("NRP_API_KEY is not set")
