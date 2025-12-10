"""Abstract benchmark evaluation framework."""

import os
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from sklearn.metrics import ndcg_score
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

from .interfaces import VectorDBAdapter, ModelProvider, QueryResult, DatasetLoader

QUERY_BATCH_SIZE = int(os.environ.get("QUERY_BATCH_SIZE", 100))


def batched(iterable, batch_size):
    """
    Yield successive batch_size chunks from iterable.
    Args:
        iterable: An iterable (e.g., list, DataFrame rows)
        batch_size: Size of each batch
    Yields:
        list: A batch of items from the iterable
    """
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def compute_ndcg(df: pd.DataFrame, relevance_col: str, sortby: str = "rerank_score") -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) using scikit-learn.
    Args:
        df: DataFrame containing search results with relevance labels
        relevance_col: Column name containing relevance labels
        sortby: Column to sort by (e.g., "rerank_score")
    Returns:
        float: NDCG score
    """
    if df.empty or len(df) < 2:
        return 0.0  # NDCG is not defined for a single document.

    # Ensure results are sorted (higher score = better ranking)
    df_sorted = df.sort_values(sortby, ascending=False)

    # Extract true relevance labels (1 = relevant, 0 = irrelevant)
    if relevance_col not in df_sorted.columns:
        return 0.0
    
    y_true = df_sorted[relevance_col].values.reshape(1, -1)  # Must be 2D array

    # Extract ranking scores (e.g., rerank_score or clip_score)
    y_score = df_sorted[sortby].values.reshape(1, -1)  # Must be 2D array

    # Compute NDCG using Scikit-Learn
    return ndcg_score(y_true, y_score)


class BenchmarkEvaluator:
    """Abstract evaluator for benchmarking vector database and model combinations."""
    
    def __init__(
        self, 
        vector_db: VectorDBAdapter,
        model_provider: ModelProvider,
        dataset_loader: DatasetLoader,
        collection_name: str = "default",
        query_method: str = "search",
        limit: int = 25,
        score_columns: List[str] = None
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            vector_db: Vector database adapter instance
            model_provider: Model provider instance
            dataset_loader: Dataset loader instance
            collection_name: Name of the collection to search
            query_method: Method name to use for querying (for logging/debugging)
            limit: Maximum number of results to return per query
            score_columns: List of column names to try for NDCG computation (in order of preference)
        """
        self.vector_db = vector_db
        self.model_provider = model_provider
        self.dataset_loader = dataset_loader
        self.collection_name = collection_name
        self.query_method = query_method
        self.limit = limit
        self.score_columns = score_columns or ["rerank_score", "clip_score", "score", "distance"]
    
    def evaluate_query(
        self, 
        query_row: pd.Series, 
        dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate a single query by comparing retrieved results to ground truth dataset.
        
        Args:
            query_row: Row from dataset containing query information
            dataset: Full dataset for ground truth comparison
            
        Returns:
            Tuple of (results_dataframe, query_statistics_dict)
        """
        query_col = self.dataset_loader.get_query_column()
        query_id_col = self.dataset_loader.get_query_id_column()
        relevance_col = self.dataset_loader.get_relevance_column()
        metadata_cols = self.dataset_loader.get_metadata_columns()
        
        query = str(query_row[query_col])
        query_id = query_row[query_id_col]

        # Log the query being evaluated
        logging.debug(f"Evaluating query {query_id}: {query}")

        # Run search query on vector database
        try:
            query_result = self.vector_db.search(
                query=query,
                collection_name=self.collection_name,
                limit=self.limit,
                query_method=self.query_method
            )
            results_df = query_result.to_dataframe()
        except Exception as e:
            logging.error(f"Error executing query {query_id}: {e}")
            results_df = pd.DataFrame()
        
        # Add query tracking columns
        results_df[f"queried_on_{query_id_col}"] = query_id
        results_df["queried_on_query"] = query

        # Check if no results were returned
        if results_df.empty:
            logging.debug(f"No results returned for query {query_id}")

            # Store per-query statistics with default values
            query_stats = {
                query_id_col: query_id,
                "query": query,
                "total_results": 0,
                "correctly_returned": 0,
                "incorrectly_returned": 0,
                "relevant_results": 0,
                "non_relevant_results": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "NDCG": 0.0,
            }
            
            # Add metadata columns
            for col in metadata_cols:
                query_stats[col] = query_row.get(col, "")

            return results_df, query_stats

        # Count total results returned
        total_results = len(results_df)

        # Check if result retrieval is correct and count relevant results
        correct_retrieval = 0
        relevant_results = 0
        for _, row in results_df.iterrows():
            # Check if this result belongs to the query
            if row.get(f"queried_on_{query_id_col}") == row.get(query_id_col):
                correct_retrieval += 1
            # Check relevance
            if row.get(relevance_col, 0):
                relevant_results += 1
        
        incorrect_retrieval = total_results - correct_retrieval
        non_relevant_results = total_results - relevant_results

        # Get number of relevant results in dataset for this query
        relevant_in_dataset = dataset[dataset[query_id_col] == query_id][relevance_col].sum()

        # Compute NDCG to evaluate ranking
        # Try different score columns, fallback to first available
        ndcg = 0.0
        secondary_ndcg = 0.0
        
        for col in self.score_columns:
            if col in results_df.columns:
                ndcg = compute_ndcg(results_df, relevance_col, sortby=col)
                break
        
        # Try to compute secondary NDCG with different score column
        for col in self.score_columns:
            if col in results_df.columns and col != (self.score_columns[0] if self.score_columns else None):
                secondary_ndcg = compute_ndcg(results_df, relevance_col, sortby=col)
                break

        # Store per-query statistics
        query_stats = {
            query_id_col: query_id,
            "query": query,
            "total_results": total_results,
            "correctly_returned": correct_retrieval,
            "incorrectly_returned": incorrect_retrieval,
            "relevant_results": relevant_results,
            "non_relevant_results": non_relevant_results,
            "accuracy": correct_retrieval / total_results if total_results else 0.0,
            "precision": relevant_results / total_results if total_results else 0.0,
            "recall": relevant_results / relevant_in_dataset if relevant_in_dataset else 0.0,
            "NDCG": ndcg,
        }
        
        # Add secondary NDCG if computed
        if secondary_ndcg > 0:
            query_stats["secondary_NDCG"] = secondary_ndcg
        
        # Add metadata columns
        for col in metadata_cols:
            query_stats[col] = query_row.get(col, "")

        return results_df, query_stats
    
    def evaluate_queries(self, dataset: pd.DataFrame = None, split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate unique queries in parallel.
        
        Args:
            dataset: Optional pre-loaded dataset. If None, will load using dataset_loader
            split: Dataset split to use if loading dataset
            
        Returns:
            Tuple of (all_results_dataframe, query_statistics_dataframe)
        """
        logging.debug("Starting Benchmark...")

        # Load dataset if not provided
        if dataset is None:
            dataset = self.dataset_loader.load(split=split)

        results = []
        query_stats = []

        # Convert dataset to Pandas DataFrame if it's not already
        if not isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_pandas()

        # Get unique queries along with their metadata
        query_col = self.dataset_loader.get_query_column()
        unique_queries = dataset.drop_duplicates(subset=[query_col])

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for batch in batched(unique_queries.iterrows(), QUERY_BATCH_SIZE):
                # Process in parallel
                futures = {
                    executor.submit(self.evaluate_query, query_row, dataset): query_row[query_col]
                    for _, query_row in batch
                }

                for future in futures:
                    df, stats = future.result()
                    results.append(df)
                    query_stats.append(stats)

        # Combine all results into a DataFrame
        all_results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        query_stats_df = pd.DataFrame(query_stats)

        return all_results_df, query_stats_df

