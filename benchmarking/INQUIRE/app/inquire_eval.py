'''This file contains the code to run generate the results of the Benchmark.'''

import os
import pandas as pd
import tritonclient.grpc as TritonClient
from query import Weav_query
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from itertools import islice
import logging

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")

# Batch size for parallel processing
QUERY_BATCH_SIZE = int(os.environ.get("QUERY_BATCH_SIZE", 100))

def load_inquire_dataset(split="test"):
    """ Load INQUIRE dataset from HuggingFace and return as pandas DataFrame. """
    dataset = load_dataset(INQUIRE_DATASET, split=split).to_pandas()
    return dataset

def compute_ndcg(df, sortby="rerank_score"):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) using scikit-learn.
    Args:
        df (pd.DataFrame): DataFrame containing Weaviate results
        sortby (str): Column to sort by (e.g., "rerank_score")
    Returns:
        float: NDCG score
    """
    if df.empty or len(df) < 2:
        return 0  # NDCG is not defined for a single document.

    # Ensure results are sorted (higher score = better ranking)
    df_sorted = df.sort_values(sortby, ascending=False)

    # Extract true relevance labels (1 = relevant, 0 = irrelevant)
    y_true = df_sorted["relevant"].values.reshape(1, -1)  # Must be 2D array

    # Extract ranking scores (e.g., rerank_score or clip_score)
    y_score = df_sorted[sortby].values.reshape(1, -1)  # Must be 2D array

    # Compute NDCG using Scikit-Learn
    return ndcg_score(y_true, y_score)

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

def evaluate_query(query_row, wq: Weav_query, dataset):
    """ Evaluates a single query by comparing retrieved results to ground truth dataset. """

    query = str(query_row["query"])
    query_id = query_row["query_id"]

    # Log the query being evaluated
    logging.debug(f"Evaluating query {query_id}: {query}")

    # Run search query on Weaviate
    weav_df = wq.clip_hybrid_query(query)
    weav_df["queried_on_query_id"] = query_id
    weav_df["queried_on_query"] = query

    # Check if no results were returned
    if weav_df.empty:
        logging.debug(f"No results returned for query {query_id}")

        # Store per-query statistics with default values
        query_stats = {
            "query_id": query_id,
            "query": query,
            "total_images": 0,
            "correctly_returned": 0,
            "incorrectly_returned": 0,
            "relevant_images": 0,
            "non_relevant_images": 0,
            "accuracy": 0,  
            "precision": 0,  
            "recall": 0,  
            "NDCG": 0,  
            "clip_NDCG": 0,
            "category": query_row["category"],
            "supercategory": query_row["supercategory"],
            "iconic_group": query_row["iconic_group"],
        }

        return weav_df, query_stats  # Return empty DataFrame and stats with default values

    # Count total images returned
    total_images = len(weav_df)

    # Check if image retrieval is correct and count relevant images
    correct_retrieval = 0
    relevant_images = 0
    for _, row in weav_df.iterrows():
        if row["queried_on_query_id"] == row["query_id"]:
            correct_retrieval += 1
            relevant_images += row["relevant"]
    incorrect_retrieval = total_images - correct_retrieval
    non_relevant_images = total_images  - relevant_images

    # get number of relevant images in dataset
    relevant_images_in_dataset = dataset[dataset["query_id"] == query_id]["relevant"].sum()

    # Comput NDCG to evaluate ranking
    ndcg = compute_ndcg(weav_df, sortby="rerank_score")
    clip_ndcg = compute_ndcg(weav_df, sortby="clip_score")

    # Store per-query statistics
    query_stats = {
        "query_id": query_id,
        "query": query,
        "total_images": total_images,
        "correctly_returned": correct_retrieval,
        "incorrectly_returned": incorrect_retrieval,
        "relevant_images": relevant_images,
        "non_relevant_images": non_relevant_images,
        "accuracy": correct_retrieval / total_images if total_images else 0, # not rank-aware metric
        "precision": relevant_images / total_images if total_images else 0, # not rank-aware metric
        "recall": relevant_images / relevant_images_in_dataset if relevant_images_in_dataset else 0, # not rank-aware metric
        "NDCG": ndcg, # https://www.aporia.com/learn/a-practical-guide-to-normalized-discounted-cumulative-gain-ndcg/
        "clip_NDCG": clip_ndcg,
        "category": query_row["category"],
        "supercategory": query_row["supercategory"],
        "iconic_group": query_row["iconic_group"],
    }

    return weav_df, query_stats

def evaluate_queries(weaviate_client, dataset):
    """ Evaluate unique queries in parallel using their full row data. """

    logging.debug("Starting INQUIRE Benchmark...")
    triton_client = TritonClient.InferenceServerClient(url="triton:8001")
    wq = Weav_query(weaviate_client, triton_client)

    results = []
    query_stats = []

    # Convert dataset to Pandas DataFrame if it's not already
    if not isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_pandas()

    # Get unique queries along with their metadata (e.g., query_id, category)
    unique_queries = dataset.drop_duplicates(subset=["query"])

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch in batched(unique_queries.iterrows(), QUERY_BATCH_SIZE):
            # Process in parallel
            futures = {
                executor.submit(evaluate_query, query_row, wq, dataset): query_row["query"]
                for _, query_row in batch
            }

            for future in futures:
                df, stats = future.result()
                results.append(df)
                query_stats.append(stats)

    # Combine all results into a DataFrame
    all_results_df = pd.concat(results, ignore_index=True)
    query_stats_df = pd.DataFrame(query_stats)

    return all_results_df, query_stats_df
