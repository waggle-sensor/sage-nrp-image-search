"""Abstract interfaces for vector database and model providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image


class QueryResult:
    """Container for query results from a vector database."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize query result.
        
        Args:
            results: List of dictionaries containing result data
        """
        self.results = results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)


class VectorDBAdapter(ABC):
    """Abstract interface for vector database adapters."""
    
    @classmethod
    @abstractmethod
    def init_client(cls, **kwargs):
        """
        Initialize and return a client connection to the vector database.
        
        Args:
            **kwargs: Connection parameters (host, port, etc.)
            
        Returns:
            Client connection object
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        collection_name: str,
        limit: int = 25,
        **kwargs
    ) -> QueryResult:
        """
        Perform a search query on the vector database.
        
        Args:
            query: Text query string
            collection_name: Name of the collection/index to search
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            QueryResult containing search results
        """
        pass
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        schema_config: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Create a collection/index in the vector database.
        
        Args:
            collection_name: Name of the collection to create
            schema_config: Dictionary containing schema configuration
            **kwargs: Additional collection-specific parameters
            
        Returns:
            True if collection was created successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """
        Delete a collection/index from the vector database.
        
        Args:
            collection_name: Name of the collection to delete
            **kwargs: Additional parameters
            
        Returns:
            True if collection was deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def insert_data(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        **kwargs
    ) -> int:
        """
        Insert data into the vector database collection.
        
        Args:
            collection_name: Name of the collection to insert into
            data: List of dictionaries containing data to insert
                   Each dict should have 'properties' and optionally 'vector' keys
            batch_size: Size of batches for insertion
            **kwargs: Additional insertion parameters
            
        Returns:
            Number of items successfully inserted
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ModelProvider(ABC):
    """Abstract interface for model providers."""
    
    @abstractmethod
    def get_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        model_name: str = "default"
    ) -> Any:
        """
        Get embedding for text and/or image.
        
        Args:
            text: Text to embed
            image: Optional PIL Image to embed
            model_name: Name of the model to use
            
        Returns:
            Embedding vector (numpy array or similar)
        """
        pass
    
    @abstractmethod
    def generate_caption(self, image: Image.Image, model_name: str = "default") -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            model_name: Name of the model to use
            
        Returns:
            Generated caption string
        """
        pass


class DatasetLoader(ABC):
    """Abstract interface for dataset loaders."""
    
    @abstractmethod
    def load(self, split: str = "test", **kwargs) -> pd.DataFrame:
        """
        Load the dataset.
        
        Args:
            split: Dataset split to load (e.g., "test", "train", "val")
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            DataFrame containing the dataset
        """
        pass
    
    @abstractmethod
    def get_query_column(self) -> str:
        """
        Get the name of the column containing the query text.
        
        Returns:
            Column name for queries
        """
        pass
    
    @abstractmethod
    def get_query_id_column(self) -> str:
        """
        Get the name of the column containing the query ID.
        
        Returns:
            Column name for query IDs
        """
        pass
    
    @abstractmethod
    def get_relevance_column(self) -> str:
        """
        Get the name of the column containing relevance labels.
        
        Returns:
            Column name for relevance (1 = relevant, 0 = irrelevant)
        """
        pass
    
    def get_metadata_columns(self) -> List[str]:
        """
        Get optional metadata columns to include in evaluation stats.
        
        Returns:
            List of column names for metadata (e.g., ["category", "supercategory"])
        """
        return []


class DataLoader(ABC):
    """Abstract interface for loading data into vector databases."""
    
    @abstractmethod
    def process_item(
        self,
        item: Dict[str, Any],
        model_provider: ModelProvider
    ) -> Dict[str, Any]:
        """
        Process a single dataset item and prepare it for vector database insertion.
        
        Args:
            item: Dictionary containing raw dataset item
            model_provider: Model provider for generating embeddings/captions
            
        Returns:
            Dictionary with 'properties' and optionally 'vector' keys for insertion
        """
        pass
    
    @abstractmethod
    def get_schema_config(self) -> Dict[str, Any]:
        """
        Get the schema configuration for creating the collection.
        
        Returns:
            Dictionary containing schema configuration
        """
        pass
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        model_provider: ModelProvider
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of items. Default implementation processes items sequentially.
        Override for parallel processing.
        
        Args:
            batch: List of dataset items
            model_provider: Model provider for generating embeddings/captions
            
        Returns:
            List of processed items ready for insertion
        """
        return [self.process_item(item, model_provider) for item in batch]


class Config(ABC):
    """Abstract interface for configuration/hyperparameters."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        pass
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return {}


class Query(ABC):
    """Abstract interface for query classes used by vector database adapters."""
    
    @abstractmethod
    def query(
        self,
        near_text: str,
        collection_name: str,
        limit: int = 25,
        query_method: str = "default",
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform a search query on the vector database.
        
        Args:
            near_text: Text query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            query_method: Method/type of query to perform (implementation-specific, e.g., "hybrid", "vector", "keyword")
            **kwargs: Additional search parameters (implementation-specific)
        
        Returns:
            DataFrame with search results
        """
        pass

