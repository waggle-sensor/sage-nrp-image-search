"""
Weaviate-based adapters for benchmarking framework.

This module provides all Weaviate-related adapters:
- WeaviateQuery: Query class for Weaviate with various search methods
- WeaviateAdapter: VectorDBAdapter implementation for Weaviate
"""

import sys
import os
import logging
import time
import weaviate
import pandas as pd
from typing import List, Dict, Any, Optional
from weaviate.classes.query import MetadataQuery, HybridFusion, Rerank

# Add framework to path
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../framework'))
if framework_path not in sys.path:
    sys.path.insert(0, framework_path)

from framework.interfaces import VectorDBAdapter, QueryResult, Query

# Import TritonModelUtils for embedding generation
from .triton import TritonModelUtils


class WeaviateQuery(Query):
    """
    Query class for Weaviate that provides various search methods.
    """
    
    def __init__(self, weaviate_client: weaviate.Client, triton_client=None, model_utils=None):
        """
        Initialize Weaviate query instance.
        
        Args:
            weaviate_client: Weaviate client connection
            triton_client: Optional Triton client for generating embeddings
            model_utils: Optional ModelUtils instance (if None and triton_client provided, creates TritonModelUtils)
        """
        self.weaviate_client = weaviate_client
        self.triton_client = triton_client
        
        # Create model_utils if triton_client is provided but model_utils is not
        if model_utils is None and triton_client is not None:
            self.model_utils = TritonModelUtils(triton_client)
        else:
            self.model_utils = model_utils
    
    def query(
        self,
        near_text: str,
        collection_name: str,
        limit: int = 25,
        query_method: str = "clip_hybrid_query",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a search query on Weaviate.
        
        This is the generic query method that routes to specific query methods
        based on the query_method parameter.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            query_method: Method name to call (e.g., "clip_hybrid_query", "hybrid_query", "colbert_query")
            **kwargs: Additional search parameters passed to the specific query method
        
        Returns:
            DataFrame with search results
        """
        # Route to the appropriate query method
        if query_method == "clip_hybrid_query":
            return self.clip_hybrid_query(near_text, collection_name, limit, **kwargs)
        elif query_method == "hybrid_query":
            return self.hybrid_query(near_text, collection_name, limit, **kwargs)
        elif query_method == "colbert_query":
            return self.colbert_query(near_text, collection_name, limit, **kwargs)
        else:
            # Default to clip_hybrid_query if method not recognized
            logging.warning(f"Unknown query method '{query_method}', defaulting to 'clip_hybrid_query'")
            return self.clip_hybrid_query(near_text, collection_name, limit, **kwargs)
    
    def get_location_coordinate(self, obj, coordinate_type: str) -> float:
        """
        Helper function to safely fetch latitude or longitude from the location property.
        
        Args:
            obj: Weaviate object
            coordinate_type: "latitude" or "longitude"
        
        Returns:
            Coordinate value as float, or 0.0 if not available
        """
        location = obj.properties.get("location", "")
        if location:
            try:
                if coordinate_type in ["latitude", "longitude"]:
                    return float(getattr(location, coordinate_type, 0.0))
            except (AttributeError, ValueError):
                logging.warning(f"Invalid {coordinate_type} value found for obj {obj.uuid}")
        return 0.0
    
    def _extract_object_data(self, obj) -> dict:
        """
        Extract common object data from Weaviate result.
        
        Args:
            obj: Weaviate object from query result
        
        Returns:
            Dictionary with extracted properties
        """
        return {
            "uuid": str(obj.uuid),
            "filename": obj.properties.get("filename", ""),
            "caption": obj.properties.get("caption", ""),
            "score": getattr(obj.metadata, "score", None),
            "explainScore": getattr(obj.metadata, "explain_score", None),
            "rerank_score": getattr(obj.metadata, "rerank_score", None),
            "distance": getattr(obj.metadata, "distance", None),
            "vsn": obj.properties.get("vsn", ""),
            "camera": obj.properties.get("camera", ""),
            "project": obj.properties.get("project", ""),
            "timestamp": obj.properties.get("timestamp", ""),
            "link": obj.properties.get("link", ""),
            "host": obj.properties.get("host", ""),
            "job": obj.properties.get("job", ""),
            "plugin": obj.properties.get("plugin", ""),
            "task": obj.properties.get("task", ""),
            "zone": obj.properties.get("zone", ""),
            "node": obj.properties.get("node", ""),
            "address": obj.properties.get("address", ""),
            "location_lat": self.get_location_coordinate(obj, "latitude"),
            "location_lon": self.get_location_coordinate(obj, "longitude"),
        }
    
    def hybrid_query(
        self,
        near_text: str,
        collection_name: str = "HybridSearchExample",
        limit: int = 25,
        alpha: float = 0.4,
        target_vector: str = "imagebind",
        query_properties: Optional[list] = None,
        autocut_jumps: int = 0
    ) -> pd.DataFrame:
        """
        Perform a hybrid vector and keyword search.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            alpha: Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
            target_vector: Name of the vector space to search in (default: "imagebind")
            query_properties: List of properties to search in keyword search
            autocut_jumps: Number of jumps for autocut (0 to disable)
        
        Returns:
            DataFrame with search results
        """
        collection = self.weaviate_client.collections.get(collection_name)
        
        if query_properties is None:
            query_properties = ["caption", "camera", "host", "job", "vsn", "plugin", "zone", "project", "address"]
        
        # Perform hybrid search
        res = collection.query.hybrid(
            query=near_text,
            target_vector=target_vector,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=query_properties,
            rerank=Rerank(
                prop="caption",
                query=near_text
            )
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)
    
    def colbert_query(
        self,
        near_text: str,
        collection_name: str = "HybridSearchExample",
        limit: int = 25,
        autocut_jumps: int = 0
    ) -> pd.DataFrame:
        """
        Perform a vector search using ColBERT embeddings.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            autocut_jumps: Number of jumps for autocut (0 to disable)
        
        Returns:
            DataFrame with search results
        """
        if not self.model_utils:
            raise ValueError("Model utils is required for ColBERT queries")
        
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Generate ColBERT embedding
        colbert_embedding = self.model_utils.get_colbert_embedding(near_text)
        if colbert_embedding is None:
            logging.error("Failed to generate ColBERT embedding")
            return pd.DataFrame()
        
        # For ColBERT, we need to use the mean of token embeddings for vector search
        # ColBERT returns token-level embeddings, so we average them
        if len(colbert_embedding.shape) > 1:
            colbert_vector = colbert_embedding.mean(axis=0)
        else:
            colbert_vector = colbert_embedding
        
        # Perform vector search
        res = collection.query.near_vector(
            near_vector=colbert_vector,
            target_vector="colbert",
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            rerank=Rerank(
                prop="caption",
                query=near_text
            )
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)
    
    def clip_hybrid_query(
        self,
        near_text: str,
        collection_name: str = "HybridSearchExample",
        limit: int = 25,
        alpha: float = 0.4,
        clip_alpha: float = 0.7,
        query_properties: Optional[list] = None,
        autocut_jumps: int = 0
    ) -> pd.DataFrame:
        """
        Perform a hybrid search using CLIP embeddings.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            alpha: Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
            clip_alpha: Weight for fusing CLIP image and text embeddings
            query_properties: List of properties to search in keyword search
            autocut_jumps: Number of jumps for autocut (0 to disable)
        
        Returns:
            DataFrame with search results
        """
        if not self.model_utils:
            raise ValueError("Model utils is required for CLIP hybrid queries")
        
        collection = self.weaviate_client.collections.get(collection_name)
        
        # Get CLIP embedding
        clip_embedding = self.model_utils.get_clip_embeddings(near_text, image=None, alpha=clip_alpha)
        if clip_embedding is None:
            logging.error("Failed to generate CLIP embedding")
            return pd.DataFrame()
        
        if query_properties is None:
            query_properties = ["caption", "camera", "host", "job", "vsn", "plugin", "zone", "project", "address"]
        
        # Perform hybrid search
        res = collection.query.hybrid(
            query=near_text,
            target_vector="clip",
            fusion_type=HybridFusion.RELATIVE_SCORE,
            auto_limit=autocut_jumps if autocut_jumps > 0 else None,
            limit=limit,
            alpha=alpha,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            query_properties=query_properties,
            vector=clip_embedding,
            rerank=Rerank(
                prop="caption",
                query=near_text
            )
        )
        
        # Extract results
        objects = []
        for obj in res.objects:
            obj_data = self._extract_object_data(obj)
            objects.append(obj_data)
        
        return pd.DataFrame(objects)


class WeaviateAdapter(VectorDBAdapter):
    """Weaviate adapter using framework WeaviateQuery implementation."""
    
    @classmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a Weaviate client connection.
        
        Args:
            **kwargs: Connection parameters:
                - host: Weaviate host (default: from WEAVIATE_HOST env or "127.0.0.1")
                - port: Weaviate REST port (default: from WEAVIATE_PORT env or "8080")
                - grpc_port: Weaviate GRPC port (default: from WEAVIATE_GRPC_PORT env or "50051")
        
        Returns:
            Weaviate client connection
        """
        host = kwargs.get("host", os.getenv("WEAVIATE_HOST", "127.0.0.1"))
        port = kwargs.get("port", os.getenv("WEAVIATE_PORT", "8080"))
        grpc_port = kwargs.get("grpc_port", os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        logging.debug(f"Attempting to connect to Weaviate at {host}:{port}")
        
        # Retry logic to connect to Weaviate
        while True:
            try:
                client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=grpc_port
                )
                logging.debug("Successfully connected to Weaviate")
                return client
            except weaviate.exceptions.WeaviateConnectionError as e:
                logging.error(f"Failed to connect to Weaviate: {e}")
                logging.debug("Retrying in 10 seconds...")
                time.sleep(10)
    
    def __init__(self, weaviate_client=None, triton_client=None, query_class=None, **client_kwargs):
        """
        Initialize Weaviate adapter.
        
        Args:
            weaviate_client: Pre-initialized Weaviate client (optional)
            triton_client: Pre-initialized Triton client (optional)
            query_class: Query class to use (defaults to WeaviateQuery from adapters)
            **client_kwargs: Additional parameters to pass to init_client if weaviate_client is None
        """
        if weaviate_client is None:
            weaviate_client = self.init_client(**client_kwargs)
        
        self.weaviate_client = weaviate_client
        self.triton_client = triton_client
        
        # Use WeaviateQuery by default, or allow custom query class
        if query_class is None:
            query_class = WeaviateQuery
        
        self.query_class = query_class
        self.query_instance = query_class(weaviate_client, triton_client)
    
    def search(
        self, 
        query: str, 
        collection_name: str = "INQUIRE",
        limit: int = 25,
        query_method: str = "clip_hybrid_query",
        **kwargs
    ) -> QueryResult:
        """
        Perform a search query on Weaviate.
        
        Args:
            query: Text query string
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            query_method: Method name to use (e.g., "clip_hybrid_query", "hybrid_query", "colbert_query")
            **kwargs: Additional search parameters
            
        Returns:
            QueryResult containing search results
        """
        # Use the generic query method from the Query interface
        df = self.query_instance.query(
            near_text=query,
            collection_name=collection_name,
            limit=limit,
            query_method=query_method,
            **kwargs
        )
        
        # Convert DataFrame to list of dicts for QueryResult
        results = df.to_dict('records')
        
        return QueryResult(results)
    
    def create_collection(
        self,
        collection_name: str,
        schema_config: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Create a Weaviate collection.
        
        Args:
            collection_name: Name of the collection to create
            schema_config: Dictionary containing schema configuration
            **kwargs: Additional parameters
            
        Returns:
            True if collection was created successfully
        """
        try:
            import time
            from weaviate.classes.config import Configure, Property
            
            # Delete existing collection if it exists
            if collection_name in self.weaviate_client.collections.list_all():
                logging.debug(f"Collection '{collection_name}' exists. Deleting it first...")
                self.weaviate_client.collections.delete(collection_name)
                
                # Wait until it's fully deleted
                while collection_name in self.weaviate_client.collections.list_all():
                    time.sleep(1)
            
            # Extract schema components
            description = schema_config.get("description", "")
            properties = schema_config.get("properties", [])
            vectorizer_config = schema_config.get("vectorizer_config", [])
            reranker_config = schema_config.get("reranker_config", None)
            
            # Create collection
            self.weaviate_client.collections.create(
                name=collection_name,
                description=description,
                properties=properties,
                vectorizer_config=vectorizer_config,
                reranker_config=reranker_config
            )
            
            logging.debug(f"Collection '{collection_name}' successfully created.")
            return True
            
        except Exception as e:
            logging.error(f"Error creating collection '{collection_name}': {e}")
            return False
    
    def delete_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """
        Delete a Weaviate collection.
        
        Args:
            collection_name: Name of the collection to delete
            **kwargs: Additional parameters
            
        Returns:
            True if collection was deleted successfully
        """
        try:
            if collection_name in self.weaviate_client.collections.list_all():
                self.weaviate_client.collections.delete(collection_name)
                logging.debug(f"Collection '{collection_name}' deleted.")
                return True
            else:
                logging.debug(f"Collection '{collection_name}' does not exist.")
                return False
        except Exception as e:
            logging.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def insert_data(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        **kwargs
    ) -> int:
        """
        Insert data into Weaviate collection.
        
        Args:
            collection_name: Name of the collection to insert into
            data: List of dictionaries with 'properties' and optionally 'vector' keys
            batch_size: Size of batches for insertion
            **kwargs: Additional parameters
            
        Returns:
            Number of items successfully inserted
        """
        import logging
        from itertools import islice
        
        try:
            collection = self.weaviate_client.collections.get(collection_name)
            inserted_count = 0
            
            # Helper to batch data
            def batched(iterable, n):
                it = iter(iterable)
                while batch := list(islice(it, n)):
                    yield batch
            
            # Insert in batches
            with collection.batch.fixed_size(batch_size=batch_size) as batch:
                for item in data:
                    if item is None:
                        continue
                    
                    properties = item.get("properties", {})
                    vector = item.get("vector", {})
                    
                    batch.add_object(properties=properties, vector=vector)
                    inserted_count += 1
                    
                    # Stop if too many errors
                    if batch.number_errors > 5:
                        logging.error("Batch import stopped due to excessive errors.")
                        break
            
            logging.debug(f"Inserted {inserted_count} items into '{collection_name}'.")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Error inserting data into '{collection_name}': {e}")
            return 0
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.weaviate_client:
            self.weaviate_client.close()

