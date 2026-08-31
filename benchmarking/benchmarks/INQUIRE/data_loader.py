"""INQUIRE-specific data loader for loading data into vector databases."""

import os
import logging
from dateutil.parser import parse
from io import BytesIO, BufferedReader
from PIL import Image
import weaviate
from weaviate.classes.data import GeoCoordinate
from imsearch_eval.framework.interfaces import DataLoader
from helpers.ablation import (
    generate_index_caption,
    get_index_embedding,
    milvus_index_payload,
)
from helpers.dlq import soft_caption_dlq
from helpers.backend import is_milvus


class INQUIREDataLoader(DataLoader):
    """Data loader for INQUIRE dataset."""

    def process_item(self, item: dict, *, force_insert: bool = False) -> dict:
        """Process a single INQUIRE dataset item."""
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}")

            if not isinstance(item.get("image"), Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(item.get('image'))}")

            image = item["image"]
            filename = item.get("inat24_file_name", "")

            logging.debug(f"Processing item: {filename}")

            query = item.get("query", "")
            query_id = item.get("query_id", 0)
            if hasattr(query_id, "item"):
                query_id = query_id.item()
            relevant = item.get("relevant", 0)
            if hasattr(relevant, "item"):
                relevant = int(relevant.item())
            relevant = int(relevant)
            clip_score = item.get("clip_score", 0.0)
            inat_id = item.get("inat24_image_id", 0)
            if hasattr(inat_id, "item"):
                inat_id = inat_id.item()
            supercategory = item.get("supercategory", "")
            category = item.get("category", "")
            iconic_group = item.get("iconic_group", "")
            species_id = item.get("inat24_species_id", 0)
            if hasattr(species_id, "item"):
                species_id = species_id.item()
            species_name = item.get("inat24_species_name", "")
            location_uncertainty = item.get("location_uncertainty", 0)
            lat = item.get("latitude", None)
            lon = item.get("longitude", None)
            raw_date = item.get("date", "")

            try:
                date_obj = parse(raw_date)
                date_rfc3339 = date_obj.isoformat()
            except Exception as e:
                logging.error(f"Error parsing date for image {filename}: {e}")
                date_rfc3339 = raw_date.replace(" ", "T") if raw_date else ""

            parsed, caption_failed = generate_index_caption(
                self.model_provider, image, self.config
            )
            if caption_failed and not force_insert:
                return soft_caption_dlq(inat_id, query_id)

            if is_milvus(self.config):
                from imsearch_eval.adapters.milvus import (
                    to_milvus_timestamptz,
                    to_milvus_wkt_point,
                )

                caption_vector, image_vector, search_text, link = milvus_index_payload(
                    self.model_provider,
                    parsed,
                    image,
                    filename or inat_id,
                    self.config,
                )
                location = None
                if lat not in (None, "") and lon not in (None, ""):
                    location = to_milvus_wkt_point(float(lon), float(lat))
                row = {
                    "inat24_image_id": int(inat_id) if inat_id not in (None, "") else 0,
                    "inat24_file_name": filename or "",
                    "query": query or "",
                    "query_id": int(query_id) if query_id not in (None, "") else 0,
                    "long_caption": parsed.long_caption or "",
                    "short_caption": parsed.short_caption or "",
                    "relevant": relevant,
                    "clip_score": float(clip_score or 0.0),
                    "supercategory": supercategory or "",
                    "category": category or "",
                    "iconic_group": iconic_group or "",
                    "inat24_species_id": int(species_id) if species_id not in (None, "") else 0,
                    "inat24_species_name": species_name or "",
                    "location_uncertainty": float(location_uncertainty or 0.0),
                    "date": to_milvus_timestamptz(date_rfc3339) or "1970-01-01T00:00:00Z",
                    "link": link,
                    "caption_vector": caption_vector,
                    "image_vector": image_vector,
                    "search_text": search_text,
                }
                if location is not None:
                    row["location"] = location
                return row

            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)

            clip_embedding = get_index_embedding(
                self.model_provider, parsed.clip_text, image, self.config
            )
            if clip_embedding is None:
                raise ValueError("Failed to generate CLIP embedding")

            properties = {
                "inat24_image_id": inat_id,
                "inat24_file_name": filename,
                "query": query,
                "query_id": query_id,
                "image": encoded_image,
                "long_caption": parsed.long_caption,
                "short_caption": parsed.short_caption,
                "relevant": relevant,
                "clip_score": clip_score,
                "supercategory": supercategory,
                "category": category,
                "iconic_group": iconic_group,
                "inat24_species_id": species_id,
                "inat24_species_name": species_name,
                "location_uncertainty": location_uncertainty,
                "date": date_rfc3339,
                "location": GeoCoordinate(latitude=float(lat), longitude=float(lon)) if lat and lon else None,
            }

            return {
                "properties": properties,
                "vector": {"clip": clip_embedding}
            }

        except Exception as e:
            logging.error(f"Error processing item {item.get('inat24_file_name', 'unknown')}: {e}")
            return None

    def get_schema_config(self) -> dict:
        """Get schema configuration for the INQUIRE collection."""
        COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "INQUIRE")
        if is_milvus(self.config):
            from imsearch_eval.adapters.milvus import build_benchmark_schema

            return build_benchmark_schema(
                name=COLLECTION_NAME,
                scalar_fields=[
                    {"field_name": "inat24_image_id", "datatype": "INT64"},
                    {"field_name": "inat24_file_name", "datatype": "VARCHAR"},
                    {"field_name": "query", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "query_id", "datatype": "INT64"},
                    {"field_name": "long_caption", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "short_caption", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "relevant", "datatype": "INT64"},
                    {"field_name": "clip_score", "datatype": "FLOAT"},
                    {"field_name": "supercategory", "datatype": "VARCHAR"},
                    {"field_name": "category", "datatype": "VARCHAR"},
                    {"field_name": "iconic_group", "datatype": "VARCHAR"},
                    {"field_name": "inat24_species_id", "datatype": "INT64"},
                    {"field_name": "inat24_species_name", "datatype": "VARCHAR"},
                    {"field_name": "location_uncertainty", "datatype": "FLOAT"},
                    {"field_name": "link", "datatype": "VARCHAR", "max_length": 2048},
                ],
                include_location=True,
                include_timestamp=True,
                timestamp_field="date",
            )

        from weaviate.classes.config import Configure, Property, DataType
        TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")
        return {
            "name": COLLECTION_NAME,
            "description": "A collection to test our set up using INQUIRE with Weaviate",
            "properties": [
                Property(name="inat24_image_id", data_type=DataType.NUMBER),
                Property(name="inat24_file_name", data_type=DataType.TEXT),
                Property(name="query", data_type=DataType.TEXT),
                Property(name="query_id", data_type=DataType.NUMBER),
                Property(name="image", data_type=DataType.BLOB),
                Property(name="audio", data_type=DataType.BLOB),
                Property(name="video", data_type=DataType.BLOB),
                Property(name="long_caption", data_type=DataType.TEXT),
                Property(name="short_caption", data_type=DataType.TEXT),
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
                        vector_cache_max_objects=int(self.config.hnsw_vector_cache_max_objects),
                        quantizer=self.config.hnsw_quantizer,
                    )
                )
            ],
            "reranker_config": Configure.Reranker.transformers()
        }
