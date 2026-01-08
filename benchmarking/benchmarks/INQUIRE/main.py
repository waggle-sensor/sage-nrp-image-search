"""Main entry point for INQUIRE benchmarking using abstract framework."""

import os
import logging
import time
import tritonclient.grpc as TritonClient

from imsearch_eval import BenchmarkEvaluator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from dataset_loader import INQUIREDatasetLoader

# Environment variables
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")
IMAGE_RESULTS_FILE = os.environ.get("IMAGE_RESULTS_FILE", "image_search_results.csv")
QUERY_EVAL_METRICS_FILE = os.environ.get("QUERY_EVAL_METRICS_FILE", "query_eval_metrics.csv")
TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "INQUIRE")
QUERY_METHOD = os.environ.get("QUERY_METHOD", "clip_hybrid_query")
TARGET_VECTOR = os.environ.get("TARGET_VECTOR", "clip")


def main():
    """Run the INQUIRE benchmark evaluation."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Initialize clients
    logging.info("Initializing Weaviate client...")
    weaviate_client = WeaviateAdapter.init_client(
        host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
        port=os.getenv("WEAVIATE_PORT", "8080"),
        grpc_port=os.getenv("WEAVIATE_GRPC_PORT", "50051")
    )
    
    logging.info("Initializing Triton client...")
    triton_client = TritonClient.InferenceServerClient(url=f"{TRITON_HOST}:{TRITON_PORT}")

    # Create adapters
    logging.info("Creating adapters...")
    vector_db = WeaviateAdapter(
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )
    
    model_provider = TritonModelProvider(triton_client=triton_client)
    
    # Create dataset loader
    logging.info("Creating dataset loader...")
    dataset_loader = INQUIREDatasetLoader()

    # Create evaluator
    logging.info("Creating benchmark evaluator...")
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset_loader=dataset_loader,
        collection_name=COLLECTION_NAME,
        query_method=QUERY_METHOD,
        score_columns=["rerank_score", "clip_score", "score", "distance"],
        target_vector=TARGET_VECTOR
    )

    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        image_results, query_evaluation = evaluator.evaluate_queries(split="test")
    finally:
        # Clean up
        vector_db.close()

    # Save results (use /app/results if PVC is mounted, otherwise /app)
    results_dir = "/app/results" if os.path.exists("/app/results") else "/app"
    image_results_location = os.path.join(results_dir, IMAGE_RESULTS_FILE)
    query_evaluation_location = os.path.join(results_dir, QUERY_EVAL_METRICS_FILE)

    image_results.to_csv(image_results_location, index=False)
    query_evaluation.to_csv(query_evaluation_location, index=False)
    
    logging.info(f"Evaluation complete. Results saved to:")
    logging.info(f"  - {image_results_location}")
    logging.info(f"  - {query_evaluation_location}")

    # Keep the program running when the evaluation is done
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        exit()


if __name__ == "__main__":
    main()

