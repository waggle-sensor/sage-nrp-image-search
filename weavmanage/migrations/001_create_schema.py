'''Create the Milvus hybrid search collection (dense CLIP + BM25 sparse).'''
import logging
import os

from pymilvus import DataType, Function, FunctionType

import HyperParameters as hp

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "SageImageSearch")

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


def run(client):
    """Create the initial Milvus schema if it does not already exist."""
    if client.has_collection(COLLECTION_NAME):
        logging.debug(
            f"Collection {COLLECTION_NAME} already exists, skipping create."
        )
        return

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

    # WKT on insert (POINT(lon lat)); spatial filters via st_within / st_dwithin.
    # https://milvus.io/docs/geometry-field.md
    schema.add_field("location", DataType.GEOMETRY, nullable=True)

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
    index_params.add_index(
        field_name="location",
        index_type=hp.geometry_index_type,
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
        f"timestamp={hp.timestamptz_index_type}; "
        f"location={hp.geometry_index_type})"
    )
