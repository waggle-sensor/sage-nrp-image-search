"""CloudBench-specific data loader for loading data into vector databases."""

import json
import os
import logging
from io import BytesIO, BufferedReader
from PIL import Image
import weaviate
from imsearch_eval.framework.interfaces import DataLoader


class CloudBenchDataLoader(DataLoader):
    """Data loader for CloudBench dataset (cloud/atmospheric image retrieval)."""

    def process_item(self, item: dict) -> dict:
        """
        Process a single CloudBench dataset item.

        Args:
            item: Dictionary containing CloudBench dataset item with query_text,
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

            # Convert image to BytesIO for encoding
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)

            # Encode image for Weaviate
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            # Generate caption using model provider
            caption = self.model_provider.generate_caption(
                image,
                self.config.gemma3_prompt,
                model_name=self.config._caption_model_name,
                enable_thinking=self.config._nrp_enable_thinking,
            )
            if not caption:
                caption = summary or ""  # Fallback to dataset summary

            # Generate CLIP embeddings
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
        Get Weaviate schema configuration for CloudBench collection.

        Returns:
            Dictionary containing schema configuration
        """
        from weaviate.classes.config import Configure, Property, DataType

        TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "CloudBench")
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
                        dynamic_ef_max=self.config.hsnw_dynamicEfMax,
                        dynamic_ef_min=self.config.hsnw_dynamicEfMin,
                        ef=self.config.hnsw_ef,
                        ef_construction=self.config.hnsw_ef_construction,
                        filter_strategy=self.config.hsnw_filterStrategy,
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
