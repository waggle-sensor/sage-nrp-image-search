'''Rename caption to long_caption and add short_caption.'''
import logging
import os

from pymilvus import DataType, Function, FunctionType

import HyperParameters as hp

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "SageImageSearch")

DENSE_VECTOR_FIELDS = ("caption_vector", "image_vector")
REQUIRED_CAPTION_FIELDS = ("long_caption", "short_caption")


def _field_names(client):
    desc = client.describe_collection(COLLECTION_NAME)
    return {field["name"] for field in desc.get("fields", [])}


def _varchar_max_length(name: str) -> int:
    if name == "long_caption":
        return hp.caption_max_length
    if name == "short_caption":
        return hp.short_caption_max_length
    return hp.scalar_varchar_max_length


def _create_collection(client):
    """Create dual-dense schema with long_caption + short_caption and load it."""
    schema = client.create_schema(enable_dynamic_field=hp.enable_dynamic_field)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    for field_name in DENSE_VECTOR_FIELDS:
        schema.add_field(field_name, DataType.FLOAT_VECTOR, dim=hp.vector_dim)
    schema.add_field(
        "search_text",
        DataType.VARCHAR,
        max_length=hp.search_text_max_length,
        enable_analyzer=True,
        analyzer_params=hp.analyzer_params,
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    for name in hp.SCALAR_VARCHAR_FIELDS:
        schema.add_field(name, DataType.VARCHAR, max_length=_varchar_max_length(name))

    schema.add_field("timestamp", DataType.TIMESTAMPTZ)
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
    for field_name in DENSE_VECTOR_FIELDS:
        index_params.add_index(
            field_name=field_name,
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
        f"(long_caption + short_caption; dense={','.join(DENSE_VECTOR_FIELDS)})"
    )


def run(client):
    """Drop collections that still use `caption` and recreate with split fields.

    Existing rows must be re-captioned and re-embedded; Milvus cannot rename
    VARCHAR fields in place.
    """
    if client.has_collection(COLLECTION_NAME):
        fields = _field_names(client)
        if all(name in fields for name in REQUIRED_CAPTION_FIELDS):
            logging.debug(
                f"Collection {COLLECTION_NAME} already has {REQUIRED_CAPTION_FIELDS}; skipping."
            )
            return

        logging.debug(
            f"Dropping {COLLECTION_NAME} to replace `caption` with "
            f"{REQUIRED_CAPTION_FIELDS} (fields={sorted(fields)})"
        )
        client.drop_collection(COLLECTION_NAME)

    _create_collection(client)
