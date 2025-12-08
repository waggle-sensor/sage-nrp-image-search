from weaviate.classes.config import Configure, Property, DataType, Multi2VecField
import HyperParameters as hp
import time
import logging

def run(client):
    """
    Create the initial schema after deleting the existing collection if it exists.
    This allows for reloading the schema without needing to restart the server.
    """

    collection_name = "INQUIRE"

    # Check if the collection exists
    if collection_name in client.collections.list_all():
        logging.debug(f"Collection '{collection_name}' exists. Deleting it first...")
        client.collections.delete(collection_name)

        # Ensure deletion before proceeding
        while collection_name in client.collections.list_all():
            time.sleep(1)  # Wait until it's fully deleted

    logging.debug(f"Creating collection '{collection_name}'...")

    # Create a schema to add images, audio, etc.
    client.collections.create(
        name=collection_name,
        description="A collection to test our set up using INQUIRE",
        properties=[
            Property(name="inat24_image_id", data_type=DataType.NUMBER),
            Property(name="inat24_file_name", data_type=DataType.TEXT),
            Property(name="query", data_type=DataType.TEXT),
            Property(name="query_id", data_type=DataType.NUMBER),
            Property(name="image", data_type=DataType.BLOB),
            Property(name="audio", data_type=DataType.BLOB),
            Property(name="video", data_type=DataType.BLOB),
            Property(name="caption", data_type=DataType.TEXT),  # Caption for keyword search
            Property(name="relevant", data_type=DataType.NUMBER),
            Property(name="clip_score", data_type=DataType.NUMBER),
            Property(name="supercategory", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="iconic_group", data_type=DataType.TEXT),
            Property(name="inat24_species_id", data_type=DataType.NUMBER),
            Property(name="inat24_species_name", data_type=DataType.TEXT),
            Property(name="location_uncertainty", data_type=DataType.NUMBER),
            Property(name="date", data_type=DataType.DATE),
            Property(name="location", data_type=DataType.GEO_COORDINATES)
        ],
        vectorizer_config=[
            # Configure.NamedVectors.multi2vec_bind(
            #     name="imagebind",
            #     vectorize_collection_name=False,
            #     # Define fields for vectorization
            #     image_fields=[
            #         Multi2VecField(name="image", weight=hp.imageWeight)
            #     ],
            #     text_fields=[
            #         Multi2VecField(name="caption", weight=hp.textWeight)
            #     ],
            #     audio_fields=[
            #         Multi2VecField(name="audio", weight=hp.audioWeight)
            #     ],
            #     video_fields=[
            #         Multi2VecField(name="video", weight=hp.videoWeight)
            #     ],
            #     vector_index_config=Configure.VectorIndex.hnsw(
            #         distance_metric=hp.hnsw_dist_metric,
            #         dynamic_ef_factor=hp.hnsw_ef_factor,
            #         dynamic_ef_max=hp.hsnw_dynamicEfMax,
            #         dynamic_ef_min=hp.hsnw_dynamicEfMin,
            #         ef=hp.hnsw_ef,
            #         ef_construction=hp.hnsw_ef_construction,
            #         filter_strategy=hp.hsnw_filterStrategy,
            #         flat_search_cutoff=hp.hnsw_flatSearchCutoff,
            #         max_connections=hp.hnsw_maxConnections,
            #         vector_cache_max_objects=int(hp.hnsw_vector_cache_max_objects),
            #         quantizer=hp.hnsw_quantizer,
            #     )
            # ),
            # Configure.NamedVectors.none(
            #     name="colbert",
            #     vector_index_config=Configure.VectorIndex.hnsw( #https://weaviate.io/developers/weaviate/concepts/vector-index , https://weaviate.io/developers/weaviate/config-refs/schema/vector-index
            #         distance_metric=hp.hnsw_dist_metric, #works well to compare images with different attributes such as brightness levels or sizes.
            #         dynamic_ef_factor=hp.hnsw_ef_factor,
            #         dynamic_ef_max=hp.hsnw_dynamicEfMax,
            #         dynamic_ef_min=hp.hsnw_dynamicEfMin,
            #         ef=hp.hnsw_ef,
            #         ef_construction=hp.hnsw_ef_construction,
            #         filter_strategy=hp.hsnw_filterStrategy,
            #         flat_search_cutoff=hp.hnsw_flatSearchCutoff,
            #         max_connections=hp.hnsw_maxConnections,
            #         vector_cache_max_objects=int(hp.hnsw_vector_cache_max_objects),
            #         quantizer=hp.hnsw_quantizer,
            #         multi_vector=Configure.VectorIndex.MultiVector.multi_vector()
            #     )
            # ),
            # Configure.NamedVectors.none(
            #     name="align",
            #     vector_index_config=Configure.VectorIndex.hnsw( #https://weaviate.io/developers/weaviate/concepts/vector-index , https://weaviate.io/developers/weaviate/config-refs/schema/vector-index
            #         distance_metric=hp.hnsw_dist_metric, #works well to compare images with different attributes such as brightness levels or sizes.
            #         dynamic_ef_factor=hp.hnsw_ef_factor,
            #         dynamic_ef_max=hp.hsnw_dynamicEfMax,
            #         dynamic_ef_min=hp.hsnw_dynamicEfMin,
            #         ef=hp.hnsw_ef,
            #         ef_construction=hp.hnsw_ef_construction,
            #         filter_strategy=hp.hsnw_filterStrategy,
            #         flat_search_cutoff=hp.hnsw_flatSearchCutoff,
            #         max_connections=hp.hnsw_maxConnections,
            #         vector_cache_max_objects=int(hp.hnsw_vector_cache_max_objects),
            #         quantizer=hp.hnsw_quantizer,
            #     )
            # ),
            Configure.NamedVectors.none(
                name="clip",
                vector_index_config=Configure.VectorIndex.hnsw( #https://weaviate.io/developers/weaviate/concepts/vector-index , https://weaviate.io/developers/weaviate/config-refs/schema/vector-index
                    distance_metric=hp.hnsw_dist_metric, #works well to compare images with different attributes such as brightness levels or sizes.
                    dynamic_ef_factor=hp.hnsw_ef_factor,
                    dynamic_ef_max=hp.hsnw_dynamicEfMax,
                    dynamic_ef_min=hp.hsnw_dynamicEfMin,
                    ef=hp.hnsw_ef,
                    ef_construction=hp.hnsw_ef_construction,
                    filter_strategy=hp.hsnw_filterStrategy,
                    flat_search_cutoff=hp.hnsw_flatSearchCutoff,
                    max_connections=hp.hnsw_maxConnections,
                    vector_cache_max_objects=int(hp.hnsw_vector_cache_max_objects),
                    quantizer=hp.hnsw_quantizer,
                )
            )
        ],
        reranker_config=Configure.Reranker.transformers()
    )

    logging.debug(f"Collection '{collection_name}' successfully created.")
