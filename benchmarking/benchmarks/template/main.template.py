# Template for main.py
# Copy this file to main.py and customize for your benchmark
#
# This template uses the abstract benchmarking framework:
# - Framework: Abstract interfaces and evaluation logic (framework/)
# - Adapters: Concrete implementations (adapters/)
#   - triton.py: TritonModelProvider, TritonModelUtils
#   - weaviate.py: WeaviateAdapter, WeaviateQuery

import os
import logging
from tritonclient.grpc import InferenceServerClient as TritonClient

from imsearch_eval import BenchmarkEvaluator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from benchmark_dataset import MyBenchmarkDataset  # TODO: Import your BenchmarkDataset

# Environment variables
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
TRITON_HOST = os.getenv("TRITON_HOST", "triton")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MYBENCHMARK")  # TODO: Update default
QUERY_METHOD = os.getenv("QUERY_METHOD", "clip_hybrid_query")  # TODO: Update default

# Results file names
RESULTS_FILE = os.getenv("RESULTS_FILE", "results.csv")  # TODO: Update
METRICS_FILE = os.getenv("METRICS_FILE", "metrics.csv")  # TODO: Update


def main():
    """Run the MYBENCHMARK evaluation."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting MYBENCHMARK evaluation...")
    
    # Initialize clients
    logging.info("Initializing Weaviate client...")
    weaviate_client = WeaviateAdapter.init_client(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    
    logging.info("Initializing Triton client...")
    triton_client = TritonClient(url=f"{TRITON_HOST}:{TRITON_PORT}")
    
    # Create adapters
    logging.info("Creating adapters...")
    vector_db = WeaviateAdapter(
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )
    
    model_provider = TritonModelProvider(triton_client=triton_client)
    
    # Create benchmark dataset class
    logging.info("Creating benchmark dataset class...")
    benchmark_dataset = MyBenchmarkDataset()  # TODO: Use your BenchmarkDataset
    
    # Create evaluator
    logging.info("Creating benchmark evaluator...")
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset=benchmark_dataset,
        collection_name=COLLECTION_NAME,
        query_method=QUERY_METHOD,
        score_columns=["rerank_score", "clip_score", "score", "distance"]  # TODO: Adjust as needed
    )
    
    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        image_results, query_evaluation = evaluator.evaluate_queries(split="test")
    finally:
        # Clean up
        logging.info("Closing connections...")
        vector_db.close()
    
    # Save results (use /app/results if PVC is mounted, otherwise /app)
    results_dir = "/app/results" if os.path.exists("/app/results") else "/app"
    results_path = os.path.join(results_dir, RESULTS_FILE)
    metrics_path = os.path.join(results_dir, METRICS_FILE)
    
    logging.info(f"Saving results to {results_path}...")
    image_results.to_csv(results_path, index=False)
    
    logging.info(f"Saving metrics to {metrics_path}...")
    query_evaluation.to_csv(metrics_path, index=False)
    
    logging.info("Evaluation complete!")
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

