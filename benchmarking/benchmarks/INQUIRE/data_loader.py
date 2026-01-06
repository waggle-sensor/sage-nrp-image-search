"""INQUIRE-specific data loader for loading data into vector databases."""

import os
import logging
import random
from dateutil.parser import parse
from io import BytesIO, BufferedReader
from PIL import Image
import weaviate
from weaviate.classes.data import GeoCoordinate

from imsearch_eval.framework.interfaces import DataLoader, ModelProvider


class INQUIREDataLoader(DataLoader):
    """Data loader for INQUIRE dataset."""
    
    def process_item(
        self,
        item: dict
    ) -> dict:
        """
        Process a single INQUIRE dataset item.
        
        Args:
            item: Dictionary containing INQUIRE dataset item
        Returns:
            Dictionary with 'properties' and 'vector' keys for Weaviate insertion
        """
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}")
            
            if not isinstance(item.get("image"), Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(item.get('image'))}")
            
            image = item["image"]
            filename = item.get("inat24_file_name", "")
            
            logging.debug(f"Processing item: {filename}")
            
            # Extract metadata
            query = item.get("query", "")
            query_id = item.get("query_id", 0)
            relevant = item.get("relevant", 0)
            clip_score = item.get("clip_score", 0.0)
            inat_id = item.get("inat24_image_id", 0)
            supercategory = item.get("supercategory", "")
            category = item.get("category", "")
            iconic_group = item.get("iconic_group", "")
            species_id = item.get("inat24_species_id", 0)
            species_name = item.get("inat24_species_name", "")
            location_uncertainty = item.get("location_uncertainty", 0)
            lat = item.get("latitude", None)
            lon = item.get("longitude", None)
            raw_date = item.get("date", "")
            
            # Parse date
            try:
                date_obj = parse(raw_date)
                date_rfc3339 = date_obj.isoformat()
            except Exception as e:
                logging.error(f"Error parsing date for image {filename}: {e}")
                date_rfc3339 = raw_date.replace(" ", "T") if raw_date else ""
            
            # Convert image to BytesIO for encoding
            image_stream = BytesIO()
            image.save(image_stream, format="JPEG")
            image_stream.seek(0)
            
            # Encode image for Weaviate
            buffered_stream = BufferedReader(image_stream)
            encoded_image = weaviate.util.image_encoder_b64(buffered_stream)
            
            # Generate caption using model provider
            caption = self.model_provider.generate_caption(image, self.config.gemma3_prompt, model_name="gemma3")
            
            if not caption:
                caption = ""  # Fallback if caption generation fails
            
            # Generate CLIP embeddings
            clip_embedding = self.model_provider.get_embedding(caption, image=image, model_name="clip")
            if clip_embedding is None:
                raise ValueError("Failed to generate CLIP embedding")
            
            # Construct properties and vector
            properties = {
                "inat24_image_id": inat_id,
                "inat24_file_name": filename,
                "query": query,
                "query_id": query_id,
                "image": encoded_image,
                "caption": caption,
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
        """
        Get Weaviate schema configuration for INQUIRE collection.
        
        Returns:
            Dictionary containing schema configuration
        """        
        from weaviate.classes.config import Configure, Property, DataType
        
        return {
            "name": "INQUIRE",
            "description": "A collection to test our set up using INQUIRE with Weaviate",
            "properties": [
                Property(name="inat24_image_id", data_type=DataType.NUMBER),
                Property(name="inat24_file_name", data_type=DataType.TEXT),
                Property(name="query", data_type=DataType.TEXT),
                Property(name="query_id", data_type=DataType.NUMBER),
                Property(name="image", data_type=DataType.BLOB),
                Property(name="audio", data_type=DataType.BLOB),
                Property(name="video", data_type=DataType.BLOB),
                Property(name="caption", data_type=DataType.TEXT),
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
                    name="clip",
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
                        vector_cache_max_objects=int(self.config.hnsw_vector_cache_max_objects),
                        quantizer=self.config.hnsw_quantizer,
                    )
                )
            ],
            "reranker_config": Configure.Reranker.transformers()
        }

