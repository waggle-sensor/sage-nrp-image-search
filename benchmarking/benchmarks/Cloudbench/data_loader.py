"""CloudBench-specific data loader for loading data into vector databases."""

import json
import os
import logging
from io import BytesIO, BufferedReader
from PIL import Image
import weaviate
from imsearch_eval.framework.interfaces import DataLoader
from helpers.ablation import (
    generate_index_caption,
    get_index_embedding,
    milvus_index_payload,
)
from helpers.dlq import soft_caption_dlq
from helpers.backend import is_milvus


class CloudBenchDataLoader(DataLoader):
    """Data loader for CloudBench dataset (cloud/atmospheric image retrieval)."""

    def process_item(self, item: dict, *, force_insert: bool = False) -> dict:
        """
        Process a single CloudBench dataset item.

        Args:
            item: Dictionary containing CloudBench dataset item with query_text,
                  query_id, image_id, relevance_label, image, and metadata.
            force_insert: Insert with summary/empty caption after DLQ retries are exhausted.
        Returns:
            Dictionary with 'properties' and 'vector' keys for Weaviate insertion,
            Milvus row dict, soft-DLQ sentinel, or None on hard failure.
        """
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}")

            if not isinstance(item.get("image"), Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(item.get('image'))}")

            image = item["image"]
            image_id = item.get("image_id", "")

            logging.debug(f"Processing item: {image_id}")

            # CloudBench fields (https://huggingface.co/datasets/sagecontinuum/CloudBench)
            query_text = item.get("query_text", "")
            query_id = item.get("query_id", "")
            relevance_label = item.get("relevance_label", 0)
            if hasattr(relevance_label, "item"):
                relevance_label = int(relevance_label.item())
            relevance_label = int(relevance_label)

            clip_score = float(item.get("clip_score", 0.0))
            license_ = item.get("license", "")
            doi = item.get("doi", "")
            summary = item.get("summary", "")
            cloud_coverage = item.get("cloud_coverage", "")
            confounder_type = item.get("confounder_type", "")
            lighting = item.get("lighting", "")
            viewpoint = item.get("viewpoint", "")
            occlusion_present = bool(item.get("occlusion_present", False))
            multiple_cloud_types = bool(item.get("multiple_cloud_types", False))
            horizon_visible = bool(item.get("horizon_visible", False))
            ground_visible = bool(item.get("ground_visible", False))
            sun_visible = bool(item.get("sun_visible", False))
            precipitation_visible = bool(item.get("precipitation_visible", False))
            overcast = bool(item.get("overcast", False))
            multiple_layers = bool(item.get("multiple_layers", False))
            storm_visible = bool(item.get("storm_visible", False))

            tags = item.get("tags", [])
            tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
            confidence = item.get("confidence", {})
            confidence_str = json.dumps(confidence) if isinstance(confidence, dict) else str(confidence)

            caption, caption_failed = generate_index_caption(
                self.model_provider,
                image,
                self.config,
                fallback_caption=summary or "",
            )
            if caption_failed and not force_insert:
                return soft_caption_dlq(image_id)

            if is_milvus(self.config):
                caption_vector, image_vector, search_text, link = milvus_index_payload(
                    self.model_provider, caption, image, image_id, self.config
                )
                return {
                    "image_id": image_id or "",
                    "query_text": query_text or "",
                    "query_id": str(query_id or ""),
                    "caption": caption or "",
                    "relevance_label": relevance_label,
                    "clip_score": clip_score,
                    "license": license_ or "",
                    "doi": doi or "",
                    "summary": summary or "",
                    "cloud_coverage": cloud_coverage or "",
                    "confounder_type": confounder_type or "",
                    "lighting": lighting or "",
                    "viewpoint": viewpoint or "",
                    "occlusion_present": occlusion_present,
                    "multiple_cloud_types": multiple_cloud_types,
                    "horizon_visible": horizon_visible,
                    "ground_visible": ground_visible,
                    "sun_visible": sun_visible,
                    "precipitation_visible": precipitation_visible,
                    "overcast": overcast,
                    "multiple_layers": multiple_layers,
                    "storm_visible": storm_visible,
                    "tags": tags_str,
                    "confidence": confidence_str,
                    "link": link,
                    "caption_vector": caption_vector,
                    "image_vector": image_vector,
                    "search_text": search_text,
                }

            # Convert image to BytesIO for encoding
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)

            # Encode image for Weaviate
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            clip_embedding = get_index_embedding(
                self.model_provider, caption, image, self.config
            )
            if clip_embedding is None:
                raise ValueError("Failed to generate CLIP embedding")

            properties = {
                "image_id": image_id,
                "query_text": query_text,
                "query_id": query_id,
                "image": encoded_image,
                "caption": caption,
                "relevance_label": relevance_label,
                "clip_score": clip_score,
                "license": license_,
                "doi": doi,
                "summary": summary,
                "cloud_coverage": cloud_coverage,
                "confounder_type": confounder_type,
                "lighting": lighting,
                "viewpoint": viewpoint,
                "occlusion_present": occlusion_present,
                "multiple_cloud_types": multiple_cloud_types,
                "horizon_visible": horizon_visible,
                "ground_visible": ground_visible,
                "sun_visible": sun_visible,
                "precipitation_visible": precipitation_visible,
                "overcast": overcast,
                "multiple_layers": multiple_layers,
                "storm_visible": storm_visible,
                "tags": tags_str,
                "confidence": confidence_str,
            }

            return {
                "properties": properties,
                "vector": {"clip": clip_embedding},
            }

        except Exception as e:
            logging.error(
                f"Error processing item {item.get('image_id', 'unknown')}: {e}"
            )
            return None

    def get_schema_config(self) -> dict:
        """
        Get schema configuration for CloudBench collection.
        """
        COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "CloudBench")
        if is_milvus(self.config):
            from imsearch_eval.adapters.milvus import build_benchmark_schema

            return build_benchmark_schema(
                name=COLLECTION_NAME,
                scalar_fields=[
                    {"field_name": "image_id", "datatype": "VARCHAR"},
                    {"field_name": "query_text", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "query_id", "datatype": "VARCHAR"},
                    {"field_name": "caption", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "relevance_label", "datatype": "INT64"},
                    {"field_name": "clip_score", "datatype": "FLOAT"},
                    {"field_name": "license", "datatype": "VARCHAR"},
                    {"field_name": "doi", "datatype": "VARCHAR"},
                    {"field_name": "summary", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "cloud_coverage", "datatype": "VARCHAR"},
                    {"field_name": "confounder_type", "datatype": "VARCHAR"},
                    {"field_name": "lighting", "datatype": "VARCHAR"},
                    {"field_name": "viewpoint", "datatype": "VARCHAR"},
                    {"field_name": "occlusion_present", "datatype": "BOOL"},
                    {"field_name": "multiple_cloud_types", "datatype": "BOOL"},
                    {"field_name": "horizon_visible", "datatype": "BOOL"},
                    {"field_name": "ground_visible", "datatype": "BOOL"},
                    {"field_name": "sun_visible", "datatype": "BOOL"},
                    {"field_name": "precipitation_visible", "datatype": "BOOL"},
                    {"field_name": "overcast", "datatype": "BOOL"},
                    {"field_name": "multiple_layers", "datatype": "BOOL"},
                    {"field_name": "storm_visible", "datatype": "BOOL"},
                    {"field_name": "tags", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "confidence", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "link", "datatype": "VARCHAR"},
                ],
            )

        from weaviate.classes.config import Configure, Property, DataType

        TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        return {
            "name": COLLECTION_NAME,
            "description": "CloudBench: cloud/atmospheric image retrieval benchmark (sagecontinuum/CloudBench)",
            "properties": [
                Property(name="image_id", data_type=DataType.TEXT),
                Property(name="query_text", data_type=DataType.TEXT),
                Property(name="query_id", data_type=DataType.TEXT),
                Property(name="image", data_type=DataType.BLOB),
                Property(name="caption", data_type=DataType.TEXT),
                Property(name="relevance_label", data_type=DataType.INT),
                Property(name="clip_score", data_type=DataType.NUMBER),
                Property(name="license", data_type=DataType.TEXT),
                Property(name="doi", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="cloud_coverage", data_type=DataType.TEXT),
                Property(name="confounder_type", data_type=DataType.TEXT),
                Property(name="lighting", data_type=DataType.TEXT),
                Property(name="viewpoint", data_type=DataType.TEXT),
                Property(name="occlusion_present", data_type=DataType.BOOL),
                Property(name="multiple_cloud_types", data_type=DataType.BOOL),
                Property(name="horizon_visible", data_type=DataType.BOOL),
                Property(name="ground_visible", data_type=DataType.BOOL),
                Property(name="sun_visible", data_type=DataType.BOOL),
                Property(name="precipitation_visible", data_type=DataType.BOOL),
                Property(name="overcast", data_type=DataType.BOOL),
                Property(name="multiple_layers", data_type=DataType.BOOL),
                Property(name="storm_visible", data_type=DataType.BOOL),
                Property(name="tags", data_type=DataType.TEXT),
                Property(name="confidence", data_type=DataType.TEXT),
            ],
            "vectorizer_config": [
                Configure.NamedVectors.none(
                    name=TARGET_VECTOR,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=self.config.hnsw_dist_metric,
                        dynamic_ef_factor=self.config.hnsw_ef_factor,
                        dynamic_ef_max=self.config.hnsw_dynamicEfMax,
                        dynamic_ef_min=self.config.hnsw_dynamicEfMin,
                        ef=self.config.hnsw_ef,
                        ef_construction=self.config.hnsw_ef_construction,
                        filter_strategy=self.config.hnsw_filterStrategy,
                        flat_search_cutoff=self.config.hnsw_flatSearchCutoff,
                        max_connections=self.config.hnsw_maxConnections,
                        vector_cache_max_objects=int(
                            self.config.hnsw_vector_cache_max_objects
                        ),
                        quantizer=self.config.hnsw_quantizer,
                    )
                )
            ],
            "reranker_config": Configure.Reranker.transformers(),
        }
