"""INQUIRE benchmark dataset implementation."""
from imsearch_eval.adapters.huggingface import HuggingFaceDataset

class INQUIRE(HuggingFaceDataset):
    """Benchmark dataset class for INQUIRE dataset."""
    
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

