# Template for dataset_loader.py
# Copy this file to dataset_loader.py and implement the DatasetLoader interface
#
# This template uses the abstract benchmarking framework:
# - Framework: Abstract interfaces (framework/interfaces.py)
#   - DatasetLoader: Interface for loading datasets
#   - Other interfaces: VectorDBAdapter, ModelProvider, Query, DataLoader, Config

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../framework'))

from framework.interfaces import DatasetLoader
import pandas as pd

class MyDatasetLoader(DatasetLoader):
    """
    DatasetLoader implementation for MYBENCHMARK.
    
    TODO: Replace MYBENCHMARK with your benchmark name
    TODO: Implement all required methods
    """
    
    def load(self, split="test", **kwargs) -> pd.DataFrame:
        """
        Load the dataset for the specified split.
        
        Args:
            split: Dataset split to load (e.g., "test", "train", "val")
            **kwargs: Additional arguments for dataset loading
        
        Returns:
            DataFrame with columns: query, query_id, relevant, and optional metadata
        """
        # TODO: Implement dataset loading
        # Example with HuggingFace:
        # dataset = load_dataset("your-dataset/name", split=split)
        # return dataset.to_pandas()
        
        # Example with local file:
        # return pd.read_csv(f"data/{split}.csv")
        
        raise NotImplementedError("Implement dataset loading logic")
    
    def get_query_column(self) -> str:
        """Return the column name containing query text."""
        return "query"  # TODO: Update with your column name
    
    def get_query_id_column(self) -> str:
        """Return the column name containing query IDs."""
        return "query_id"  # TODO: Update with your column name
    
    def get_relevance_column(self) -> str:
        """Return the column name containing relevance labels (1 for relevant, 0 for not)."""
        return "relevant"  # TODO: Update with your column name
    
    def get_metadata_columns(self) -> list:
        """
        Return list of optional metadata column names.
        
        These columns will be included in results but not used for evaluation.
        """
        return []  # TODO: Add metadata columns if available (e.g., ["category", "type"])

