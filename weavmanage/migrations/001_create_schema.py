'''Create the Milvus hybrid search collection (dense CLIP + BM25 sparse).'''
import logging
import os

from pymilvus import DataType, Function, FunctionType

import HyperParameters as hp

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "HybridSearchExample")

SCALAR_VARCHAR_FIELDS = [
    "filename",
    "caption",
    "link",
    "vsn",
    "node",
    "zone",
    "task",
    "host",
    "job",
    "plugin",
    "camera",
    "project",
    "address",
]


def _field_type(client, field_name: str):
    """Return DataType for a field, or None if the collection/field is missing."""
    if not client.has_collection(COLLECTION_NAME):
        return None
    for field in client.describe_collection(COLLECTION_NAME).get("fields", []):
        if field.get("name") == field_name:
            return field.get("type")
    return None


def run(client):
    """Create the initial Milvus schema if it does not already exist.

    If an older VARCHAR ``timestamp`` collection is present, drop and recreate
    so TIMESTAMPTZ (Milvus 2.6.6+) can be used. Fresh re-ingest is expected.
    """
    existing_ts_type = _field_type(client, "timestamp")
    if existing_ts_type is not None:
        if existing_ts_type == DataType.TIMESTAMPTZ:
            logging.debug(
                f"Collection {COLLECTION_NAME} already exists with "
                "TIMESTAMPTZ timestamp, skipping create."
            )
            return
        logging.warning(
            f"Collection {COLLECTION_NAME} has timestamp type "
            f"{existing_ts_type}; dropping to recreate with TIMESTAMPTZ."
        )
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(enable_dynamic_field=hp.enable_dynamic_field)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(
        "vector", DataType.FLOAT_VECTOR, dim=hp.vector_dim
    )
    schema.add_field(
        "search_text",
        DataType.VARCHAR,
        max_length=hp.search_text_max_length,
        enable_analyzer=True,
        analyzer_params=hp.analyzer_params,
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    for name in SCALAR_VARCHAR_FIELDS:
        max_length = (
            hp.caption_max_length
            if name == "caption"
            else hp.scalar_varchar_max_length
        )
        schema.add_field(name, DataType.VARCHAR, max_length=max_length)

    # Stored as UTC absolute time; accept ISO 8601 with offset on insert.
    # https://milvus.io/docs/timestamptz-field.md
    schema.add_field("timestamp", DataType.TIMESTAMPTZ)

    schema.add_field("location_lat", DataType.FLOAT)
    schema.add_field("location_lon", DataType.FLOAT)

    schema.add_function(
        Function(
            name="search_text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["search_text"],
            output_field_names=["sparse"],
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=hp.dense_index_type,
        metric_type=hp.dense_metric_type,
        params={
            "M": hp.hnsw_M,
            "efConstruction": hp.hnsw_ef_construction,
        },
    )
    index_params.add_index(
        field_name="sparse",
        index_type=hp.sparse_index_type,
        metric_type=hp.sparse_metric_type,
        params={
            "inverted_index_algo": hp.bm25_inverted_index_algo,
            "bm25_k1": hp.bm25_k1,
            "bm25_b": hp.bm25_b,
        },
    )
    index_params.add_index(
        field_name="timestamp",
        index_type=hp.timestamptz_index_type,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(COLLECTION_NAME)
    logging.debug(
        f"Created and loaded collection {COLLECTION_NAME} "
        f"(HNSW M={hp.hnsw_M}, efC={hp.hnsw_ef_construction}; "
        f"BM25 k1={hp.bm25_k1}, b={hp.bm25_b}, algo={hp.bm25_inverted_index_algo}; "
        f"timestamp={hp.timestamptz_index_type})"
    )
