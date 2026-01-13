# Template for benchmark_dataset.py
# Copy this file to benchmark_dataset.py and implement the BenchmarkDataset interface
#
# This template uses the abstract benchmarking framework:
# - Framework: Abstract interfaces (framework/interfaces.py)
#   - BenchmarkDataset: Interface for your benchmark dataset
#   - Other interfaces: VectorDBAdapter, ModelProvider, Query, DataLoader, Config

import os
import pandas as pd

from imsearch_eval.adapters.huggingface import HuggingFaceDataset

class MyBenchmarkDataset(HuggingFaceDataset):
    """
    Benchmark dataset class for MYBENCHMARK.
    
    TODO: Replace MYBENCHMARK with your benchmark name
    TODO: Implement all required methods
    """
    
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

