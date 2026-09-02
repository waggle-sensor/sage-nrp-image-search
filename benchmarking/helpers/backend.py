"""Vector database backend selection for benchmark runs."""

import logging
import os

from helpers.ablation import resolve_query_alpha


def parse_vector_db(name: str = None) -> str:
    """Return 'milvus' or 'weaviate'. Default is milvus."""
    value = (name or os.environ.get("VECTOR_DB", "milvus")).strip().lower()
    if value not in ("milvus", "weaviate"):
        raise ValueError(
            f"VECTOR_DB must be 'milvus' or 'weaviate', got {value!r}"
        )
    return value


def apply_vector_db_config(config, ablation: dict, query_properties=None):
    """
    Attach vector-DB connection and query settings onto a Config instance.

    Weaviate HNSW objects stay in each benchmark's config.py because they
    depend on the Weaviate SDK.
    """
    if query_properties is None:
        query_properties = ["long_caption"]

    config.vector_db = parse_vector_db()
    config._milvus_uri = os.environ.get(
        "MILVUS_URI", "https://milvus.nrp-nautilus.io:50051"
    )
    config._milvus_token = os.environ.get("MILVUS_TOKEN", "")
    config._milvus_db = os.environ.get(
        "MILVUS_DB", os.environ.get("MILVUS_DB_NAME", "image_search_svc")
    )
    config._image_cache_dir = os.environ.get(
        "IMAGE_CACHE_DIR", "/tmp/imsearch_images"
    )
    config._weaviate_host = os.environ.get("WEAVIATE_HOST", "127.0.0.1")
    config._weaviate_port = os.environ.get("WEAVIATE_PORT", "8080")
    config._weaviate_grpc_port = os.environ.get("WEAVIATE_GRPC_PORT", "50051")

    clip_alpha = float(os.environ.get("QUERY_CLIP_ALPHA", 0.7))
    retrieve_limit = ablation.get("retrieve_limit")
    rerank_fusion = ablation.get("rerank_fusion") or "clip"
    config.retrieve_limit = retrieve_limit
    config.rerank_fusion = rerank_fusion
    if config.vector_db == "milvus":
        config.query_method = os.environ.get(
            "QUERY_METHOD", "clip_hybrid_query_dual_index"
        )
        config.target_vector = os.environ.get("TARGET_VECTOR", "image_vector")
        config.advanced_query_parameters = {
            "query_alpha": resolve_query_alpha(ablation),
            "clip_alpha": clip_alpha,
            "enable_image_vector": ablation["embed_image"],
            "enable_caption_vector": ablation["embed_caption"],
            "enable_bm25": ablation["enable_bm25"],
            # CLIP rerank: query text embedding vs stored image_vector (no image I/O)
            "rerank": True,
            "rerank_fusion": rerank_fusion,
        }
        if retrieve_limit is not None:
            config.advanced_query_parameters["retrieve_limit"] = retrieve_limit
            if retrieve_limit > 64:
                config.advanced_query_parameters["dense_search_params"] = {
                    "metric_type": "COSINE",
                    "params": {"ef": max(64, retrieve_limit)},
                }
    else:
        config.query_method = os.environ.get("QUERY_METHOD", "clip_hybrid_query")
        config.target_vector = os.environ.get("TARGET_VECTOR", "clip")
        config.advanced_query_parameters = {
            "alpha": resolve_query_alpha(ablation),
            "query_properties": query_properties,
            "autocut_jumps": int(os.environ.get("AUTOCUT_JUMPS", 0)),
            "rerank_prop": os.environ.get("RERANK_PROP", "long_caption"),
            "clip_alpha": clip_alpha,
        }


def is_milvus(config) -> bool:
    return getattr(config, "vector_db", "milvus") == "milvus"


def _collection_exists(vector_db, collection_name: str) -> bool:
    milvus_client = getattr(vector_db, "milvus_client", None)
    if milvus_client is not None:
        return bool(milvus_client.has_collection(collection_name))
    weaviate_client = getattr(vector_db, "weaviate_client", None)
    if weaviate_client is not None:
        exists = getattr(weaviate_client.collections, "exists", None)
        if callable(exists):
            return bool(exists(collection_name))
        return collection_name in list(weaviate_client.collections.list_all())
    raise TypeError(
        f"Cannot check collection existence on {type(vector_db).__name__}"
    )


def reuse_existing_index(vector_db, collection_name: str) -> list:
    """SKIP_INDEX path: require the collection, do not drop or recaption.

    Returns an empty DLQ record list so callers can still write dlq_records.csv.
    """
    if not collection_name:
        raise ValueError("COLLECTION_NAME is required when SKIP_INDEX=true")
    if not _collection_exists(vector_db, collection_name):
        raise FileNotFoundError(
            f"SKIP_INDEX=true but collection {collection_name!r} does not exist. "
            "Point COLLECTION_NAME at the existing index, or unset SKIP_INDEX to ingest."
        )
    milvus_client = getattr(vector_db, "milvus_client", None)
    if milvus_client is not None:
        milvus_client.load_collection(collection_name)
    logging.info(
        "SKIP_INDEX=true: reusing collection %r (no drop, no recaption)",
        collection_name,
    )
    return []


def maybe_load_index(config, load_data_fn, data_loader, vector_db, hf_dataset):
    """Ingest unless SKIP_INDEX=true, in which case reuse the existing collection."""
    if getattr(config, "skip_index", False):
        return reuse_existing_index(vector_db, config._collection_name)
    return load_data_fn(data_loader, vector_db, hf_dataset)


def init_vector_db(config, triton_client):
    """
    Create the VectorDBAdapter and Query instance for this run.

    Returns:
        (vector_db, query_instance)
    """
    vector_db_name = parse_vector_db(getattr(config, "vector_db", None))
    collection_name = getattr(config, "_collection_name", None)

    if vector_db_name == "milvus":
        from imsearch_eval.adapters import MilvusAdapter, MilvusQuery

        logging.info(
            "Initializing Milvus client at %s (db=%s)",
            config._milvus_uri,
            config._milvus_db,
        )
        milvus_client = MilvusAdapter.init_client(
            uri=config._milvus_uri,
            token=config._milvus_token,
            db_name=config._milvus_db,
        )
        query_instance = MilvusQuery(
            milvus_client=milvus_client,
            collection_name=collection_name,
            triton_client=triton_client,
        )
        vector_db = MilvusAdapter(
            milvus_client=milvus_client,
            collection_name=collection_name,
            triton_client=triton_client,
            query_instance=query_instance,
        )
        return vector_db, query_instance

    from imsearch_eval.adapters import WeaviateAdapter, WeaviateQuery

    logging.info(
        "Initializing Weaviate client at %s:%s",
        config._weaviate_host,
        config._weaviate_port,
    )
    weaviate_client = WeaviateAdapter.init_client(
        host=config._weaviate_host,
        port=config._weaviate_port,
        grpc_port=config._weaviate_grpc_port,
    )
    query_instance = WeaviateQuery(
        weaviate_client=weaviate_client,
        triton_client=triton_client,
    )
    vector_db = WeaviateAdapter(
        weaviate_client=weaviate_client,
        triton_client=triton_client,
        query_instance=query_instance,
    )
    return vector_db, query_instance
