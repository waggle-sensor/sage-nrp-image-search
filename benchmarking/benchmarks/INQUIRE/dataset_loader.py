"""INQUIRE dataset loader implementation."""

import os
import pandas as pd
from datasets import load_dataset

from imsearch_eval.framework.interfaces import DatasetLoader

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")


class INQUIREDatasetLoader(DatasetLoader):
    """Dataset loader for INQUIRE benchmark."""
    
    def __init__(self, dataset_name: str = None):
        """
        Initialize INQUIRE dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset name (defaults to INQUIRE_DATASET env var)
        """
        self.dataset_name = dataset_name or INQUIRE_DATASET
    
    def load(self, split: str = "test", **kwargs) -> pd.DataFrame:
        """
        Load INQUIRE dataset from HuggingFace.
        
        Args:
            split: Dataset split to load (e.g., "test", "train")
            **kwargs: Additional parameters (not used for INQUIRE)
            
        Returns:
            DataFrame containing the INQUIRE dataset
        """
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset.to_pandas()
    
    def get_query_column(self) -> str:
        """Get the name of the column containing the query text."""
        return "query"
    
    def get_query_id_column(self) -> str:
        """Get the name of the column containing the query ID."""
        return "query_id"
    
    def get_relevance_column(self) -> str:
        """Get the name of the column containing relevance labels."""
        return "relevant"
    
    def get_metadata_columns(self) -> list:
        """Get optional metadata columns to include in evaluation stats."""
        return ["category", "supercategory", "iconic_group"]

