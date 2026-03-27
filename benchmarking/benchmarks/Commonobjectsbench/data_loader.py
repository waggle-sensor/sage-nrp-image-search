"""CommonObjectsBench-specific data loader for loading data into vector databases."""

import json
import os
import logging
from io import BytesIO, BufferedReader
from PIL import Image
import weaviate
from imsearch_eval.framework.interfaces import DataLoader

class CommonObjectsBenchDataLoader(DataLoader):
    """Data loader for CommonObjectsBench dataset (general object image retrieval)."""

    def process_item(self, item: dict) -> dict:
        """
        Process a single CommonObjectsBench dataset item.

        Args:
            item: Dictionary containing CommonObjectsBench dataset item with query_text,
                  query_id, image_id, relevance_label, image, and metadata.
        Returns:
            Dictionary with 'properties' and 'vector' keys for Weaviate insertion
        """
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}")

            if not isinstance(item.get("image"), Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(item.get('image'))}")

            image = item["image"]
            image_id = item.get("image_id", "")

            logging.debug(f"Processing item: {image_id}")

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
            viewpoint = item.get("viewpoint", "")
            lighting = item.get("lighting", "")
            environment_type = item.get("environment_type", "")

            urban_scene = bool(item.get("urban_scene", False))
            rural_scene = bool(item.get("rural_scene", False))
            outdoor_scene = bool(item.get("outdoor_scene", False))
            vehicle_present = bool(item.get("vehicle_present", False))
            person_present = bool(item.get("person_present", False))
            animal_present = bool(item.get("animal_present", False))
            food_present = bool(item.get("food_present", False))
            text_visible = bool(item.get("text_visible", False))
            multiple_objects = bool(item.get("multiple_objects", False))
            artificial_lighting = bool(item.get("artificial_lighting", False))
            occlusion_present = bool(item.get("occlusion_present", False))

            tags = item.get("tags", [])
            tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)
            confidence = item.get("confidence", {})
            confidence_str = (
                json.dumps(confidence) if isinstance(confidence, dict) else str(confidence)
            )

            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            caption = self.model_provider.generate_caption(
                image,
                self.config.gemma3_prompt,
                model_name=self.config.caption_model_name,
                enable_thinking=self.config.nrp_enable_thinking,
            )
            if not caption:
                caption = summary or ""

            clip_embedding = self.model_provider.get_embedding(
                caption, image=image, model_name="clip"
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
                "urban_scene": urban_scene,
                "rural_scene": rural_scene,
                "outdoor_scene": outdoor_scene,
                "vehicle_present": vehicle_present,
                "person_present": person_present,
                "animal_present": animal_present,
                "food_present": food_present,
                "text_visible": text_visible,
                "multiple_objects": multiple_objects,
                "artificial_lighting": artificial_lighting,
                "occlusion_present": occlusion_present,
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
        Get Weaviate schema configuration for CommonObjectsBench collection.

        Returns:
            Dictionary containing schema configuration
        """
        from weaviate.classes.config import Configure, Property, DataType

        TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "CommonObjectsBench")
        return {
            "name": COLLECTION_NAME,
            "description": "CommonObjectsBench: general object image retrieval (sagecontinuum/CommonObjectsBench)",
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
                Property(name="urban_scene", data_type=DataType.BOOL),
                Property(name="rural_scene", data_type=DataType.BOOL),
                Property(name="outdoor_scene", data_type=DataType.BOOL),
                Property(name="vehicle_present", data_type=DataType.BOOL),
                Property(name="person_present", data_type=DataType.BOOL),
                Property(name="animal_present", data_type=DataType.BOOL),
                Property(name="food_present", data_type=DataType.BOOL),
                Property(name="text_visible", data_type=DataType.BOOL),
                Property(name="multiple_objects", data_type=DataType.BOOL),
                Property(name="artificial_lighting", data_type=DataType.BOOL),
                Property(name="occlusion_present", data_type=DataType.BOOL),
                Property(name="tags", data_type=DataType.TEXT),
                Property(name="confidence", data_type=DataType.TEXT),
            ],
            "vector_config": [
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
