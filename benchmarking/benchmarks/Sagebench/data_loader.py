"""Sagebench benchmark data loader for loading data into vector databases."""

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

class SagebenchDataLoader(DataLoader):
    """Data loader for Sagebench dataset rows into Weaviate."""

    def process_item(self, item: dict, *, force_insert: bool = False) -> dict | None:
        """
        Process a single Sagebench dataset item.

        Returns a Weaviate insertion dict: {"properties": ..., "vector": ...}
        or None if processing fails.
        """
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}")
            if not isinstance(item.get("image"), Image.Image):
                raise TypeError(
                    f"Expected PIL.Image, got {type(item.get('image'))}"
                )

            image = item["image"]
            image_id = item.get("image_id", "")
            query_text = item.get("query_text", "")
            query_id = item.get("query_id", "")
            relevance_label = item.get("relevance_label", 0)

            # Handle potential numpy scalars from dataset backends.
            if hasattr(relevance_label, "item"):
                relevance_label = int(relevance_label.item())
            relevance_label = int(relevance_label)

            clip_score = float(item.get("clip_score", 0.0))
            license_ = item.get("license", "")
            doi = item.get("doi", "")
            summary = item.get("summary", "")

            # Taxonomy / taxonomy-like fields
            viewpoint = item.get("viewpoint", "")
            lighting = item.get("lighting", "")
            environment_type = item.get("environment_type", "")
            sky_condition = item.get("sky_condition", "")

            # Scene flags
            horizon_present = bool(item.get("horizon_present", False))
            ground_present = bool(item.get("ground_present", False))
            sky_dominates = bool(item.get("sky_dominates", False))
            vegetation_present = bool(item.get("vegetation_present", False))
            water_present = bool(item.get("water_present", False))
            buildings_present = bool(item.get("buildings_present", False))
            vehicle_present = bool(item.get("vehicle_present", False))
            person_present = bool(item.get("person_present", False))
            animal_present = bool(item.get("animal_present", False))
            night_scene = bool(item.get("night_scene", False))
            precipitation_visible = bool(item.get("precipitation_visible", False))
            multiple_objects = bool(item.get("multiple_objects", False))

            # Metadata
            vsn = item.get("vsn", "")
            zone = item.get("zone", "")
            host = item.get("host", "")
            job = item.get("job", "")
            plugin = item.get("plugin", "")
            camera = item.get("camera", "")
            project = item.get("project", "")
            address = item.get("address", "")

            tags = item.get("tags", [])
            tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
            confidence = item.get("confidence", {})
            confidence_str = (
                json.dumps(confidence)
                if isinstance(confidence, dict)
                else str(confidence)
            )

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
                    self.model_provider,
                    caption,
                    image,
                    image_id,
                    self.config,
                    extra_search_fields=[
                        camera, host, job, vsn, plugin, zone, project, address,
                    ],
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
                    "viewpoint": viewpoint or "",
                    "lighting": lighting or "",
                    "environment_type": environment_type or "",
                    "sky_condition": sky_condition or "",
                    "horizon_present": horizon_present,
                    "ground_present": ground_present,
                    "sky_dominates": sky_dominates,
                    "vegetation_present": vegetation_present,
                    "water_present": water_present,
                    "buildings_present": buildings_present,
                    "vehicle_present": vehicle_present,
                    "person_present": person_present,
                    "animal_present": animal_present,
                    "night_scene": night_scene,
                    "precipitation_visible": precipitation_visible,
                    "multiple_objects": multiple_objects,
                    "vsn": vsn or "",
                    "zone": zone or "",
                    "host": host or "",
                    "job": job or "",
                    "plugin": plugin or "",
                    "camera": camera or "",
                    "project": project or "",
                    "address": address or "",
                    "tags": tags_str,
                    "confidence": confidence_str,
                    "link": link,
                    "caption_vector": caption_vector,
                    "image_vector": image_vector,
                    "search_text": search_text,
                }

            # Encode image for Weaviate BLOB ingestion.
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)
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
                "viewpoint": viewpoint,
                "lighting": lighting,
                "environment_type": environment_type,
                "sky_condition": sky_condition,
                "horizon_present": horizon_present,
                "ground_present": ground_present,
                "sky_dominates": sky_dominates,
                "vegetation_present": vegetation_present,
                "water_present": water_present,
                "buildings_present": buildings_present,
                "vehicle_present": vehicle_present,
                "person_present": person_present,
                "animal_present": animal_present,
                "night_scene": night_scene,
                "precipitation_visible": precipitation_visible,
                "multiple_objects": multiple_objects,
                "vsn": vsn,
                "zone": zone,
                "host": host,
                "job": job,
                "plugin": plugin,
                "camera": camera,
                "project": project,
                "address": address,
                "tags": tags_str,
                "confidence": confidence_str,
            }

            return {
                "properties": properties,
                "vector": {"clip": clip_embedding},
            }
        except Exception as e:
            logging.error(
                f"Error processing Sagebench item {item.get('image_id', 'unknown')}: {e}"
            )
            return None

    def get_schema_config(self) -> dict:
        """Get schema configuration for the Sagebench collection."""
        collection_name = os.environ.get("COLLECTION_NAME", "Sagebench")
        if is_milvus(self.config):
            from imsearch_eval.adapters.milvus import build_benchmark_schema

            return build_benchmark_schema(
                name=collection_name,
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
                    {"field_name": "viewpoint", "datatype": "VARCHAR"},
                    {"field_name": "lighting", "datatype": "VARCHAR"},
                    {"field_name": "environment_type", "datatype": "VARCHAR"},
                    {"field_name": "sky_condition", "datatype": "VARCHAR"},
                    {"field_name": "horizon_present", "datatype": "BOOL"},
                    {"field_name": "ground_present", "datatype": "BOOL"},
                    {"field_name": "sky_dominates", "datatype": "BOOL"},
                    {"field_name": "vegetation_present", "datatype": "BOOL"},
                    {"field_name": "water_present", "datatype": "BOOL"},
                    {"field_name": "buildings_present", "datatype": "BOOL"},
                    {"field_name": "vehicle_present", "datatype": "BOOL"},
                    {"field_name": "person_present", "datatype": "BOOL"},
                    {"field_name": "animal_present", "datatype": "BOOL"},
                    {"field_name": "night_scene", "datatype": "BOOL"},
                    {"field_name": "precipitation_visible", "datatype": "BOOL"},
                    {"field_name": "multiple_objects", "datatype": "BOOL"},
                    {"field_name": "vsn", "datatype": "VARCHAR"},
                    {"field_name": "zone", "datatype": "VARCHAR"},
                    {"field_name": "host", "datatype": "VARCHAR"},
                    {"field_name": "job", "datatype": "VARCHAR"},
                    {"field_name": "plugin", "datatype": "VARCHAR"},
                    {"field_name": "camera", "datatype": "VARCHAR"},
                    {"field_name": "project", "datatype": "VARCHAR"},
                    {"field_name": "address", "datatype": "VARCHAR"},
                    {"field_name": "tags", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "confidence", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "link", "datatype": "VARCHAR"},
                ],
            )

        from weaviate.classes.config import Configure, Property, DataType

        target_vector = os.environ.get("TARGET_VECTOR", "clip")

        return {
            "name": collection_name,
            "description": "Sagebench: metadata-aware image retrieval on sagecontinuum/SageBench",
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
                Property(name="viewpoint", data_type=DataType.TEXT),
                Property(name="lighting", data_type=DataType.TEXT),
                Property(name="environment_type", data_type=DataType.TEXT),
                Property(name="sky_condition", data_type=DataType.TEXT),
                Property(name="horizon_present", data_type=DataType.BOOL),
                Property(name="ground_present", data_type=DataType.BOOL),
                Property(name="sky_dominates", data_type=DataType.BOOL),
                Property(name="vegetation_present", data_type=DataType.BOOL),
                Property(name="water_present", data_type=DataType.BOOL),
                Property(name="buildings_present", data_type=DataType.BOOL),
                Property(name="vehicle_present", data_type=DataType.BOOL),
                Property(name="person_present", data_type=DataType.BOOL),
                Property(name="animal_present", data_type=DataType.BOOL),
                Property(name="night_scene", data_type=DataType.BOOL),
                Property(name="precipitation_visible", data_type=DataType.BOOL),
                Property(name="multiple_objects", data_type=DataType.BOOL),
                Property(name="vsn", data_type=DataType.TEXT),
                Property(name="zone", data_type=DataType.TEXT),
                Property(name="host", data_type=DataType.TEXT),
                Property(name="job", data_type=DataType.TEXT),
                Property(name="plugin", data_type=DataType.TEXT),
                Property(name="camera", data_type=DataType.TEXT),
                Property(name="project", data_type=DataType.TEXT),
                Property(name="address", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT),
                Property(name="confidence", data_type=DataType.TEXT),
            ],
            "vectorizer_config": [
                Configure.NamedVectors.none(
                    name=target_vector,
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

