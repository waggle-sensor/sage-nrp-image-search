"""script to run MYBENCHMARK: load data and evaluate queries."""

import os
import logging
import time
import sys
from pathlib import Path
import tritonclient.grpc as TritonClient
from datasets import Dataset

from imsearch_eval import BenchmarkEvaluator, VectorDBAdapter, BatchedIterator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider, WeaviateQuery
from benchmark_dataset import MyBenchmarkDataset  # TODO: Import your BenchmarkDataset
# from data_loader import MyDataLoader  # TODO: Import if you have a custom DataLoader
from config import MyConfig  # TODO: Set a Config class for your benchmark
from concurrent.futures import ThreadPoolExecutor, as_completed

config = MyConfig()

def load_data(data_loader, vector_db: VectorDBAdapter, hf_dataset: Dataset):
    """Load MYBENCHMARK dataset into vector database.
    
    TODO: Implement your data loading logic here.
    See benchmarks/INQUIRE/run_benchmark.py for a complete example.
    
    Args:
        data_loader: Your DataLoader instance
        vector_db: VectorDBAdapter instance
        hf_dataset: HuggingFace Dataset containing the dataset to load
    """
    try:
        # TODO: Create collection schema
        # logging.info("Creating collection schema...")
        # schema_config = data_loader.get_schema_config()
        # vector_db.create_collection(schema_config)

        # TODO: Process and insert data
        # logging.info("Processing and inserting data...")
        # results = data_loader.process_batch(batch_size=config._image_batch_size, dataset=hf_dataset, workers=config._workers)
        # inserted = vector_db.insert_data(config._collection_name, results, batch_size=config._image_batch_size)
        # logging.info(f"Inserted {inserted} items.")
        
        logging.info(f"Successfully loaded {config.mybenchmark_dataset} into Weaviate collection '{config._collection_name}'")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        vector_db.close()
        raise

def run_evaluation(evaluator: BenchmarkEvaluator, hf_dataset: Dataset):
    """Run the MYBENCHMARK benchmark evaluation.
    
    Args:
        evaluator: BenchmarkEvaluator instance
        hf_dataset: HuggingFace Dataset containing the dataset to evaluate
    
    Returns:
        Tuple of (image_results, query_evaluation) DataFrames
    """
    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        image_results, query_evaluation = evaluator.evaluate_queries(
            query_batch_size=config._query_batch_size,
            dataset=hf_dataset,
            workers=config._workers
        )
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        evaluator.vector_db.close()
        raise
    
    return image_results, query_evaluation

def upload_to_s3(local_file_path: str, s3_key: str):
    """Upload a file to S3-compatible storage using MinIO."""
    try:
        from minio import Minio
        from minio.error import S3Error
        
        if not config._s3_endpoint:
            raise ValueError("S3_ENDPOINT environment variable must be set")
        
        # Parse endpoint (remove http:// or https:// if present)
        endpoint = config._s3_endpoint.replace("http://", "").replace("https://", "")
        
        # Create MinIO client
        client = Minio(
            endpoint,
            access_key=config._s3_access_key,
            secret_key=config._s3_secret_key,
            secure=config._s3_secure
        )
        
        # Upload file
        logging.info(f"Uploading {local_file_path} to s3://{config._s3_bucket}/{s3_key}")
        client.fput_object(config._s3_bucket, s3_key, local_file_path)
        logging.info(f"Successfully uploaded to s3://{config._s3_bucket}/{s3_key}")
        
    except ImportError:
        logging.error("minio is not installed. Install it with: pip install minio")
        raise
    except S3Error as e:
        logging.error(f"Error uploading to S3: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error uploading to S3: {e}")
        raise

def main():
    """Main entry point for running the complete benchmark."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config._log_level, logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Step 0: load framework components
    logging.info("=" * 80)
    logging.info("Step 0: Setting up benchmark environment")
    logging.info("=" * 80)
    logging.info("Initializing Weaviate client...")
    weaviate_client = WeaviateAdapter.init_client(  # TODO: Update with your vector database client
        host=config._weaviate_host,
        port=config._weaviate_port,
        grpc_port=config._weaviate_grpc_port
    )
    
    logging.info("Initializing Triton client...")
    triton_client = TritonClient.InferenceServerClient(url=f"{config._triton_host}:{config._triton_port}")  # TODO: Update with your model provider client

    # Create query method
    query_method = WeaviateQuery(
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )

    # Create adapters
    logging.info("Creating adapters...")
    vector_db = WeaviateAdapter(  # TODO: Update with your vector database adapter
        weaviate_client=weaviate_client,
        triton_client=triton_client,
        query_method=query_method
    )

    model_provider = TritonModelProvider(triton_client=triton_client)  # TODO: Update with your model provider

    # Create benchmark dataset
    logging.info("Creating benchmark dataset class...")
    benchmark_dataset = MyBenchmarkDataset()  # TODO: Use your BenchmarkDataset
    hf_dataset = benchmark_dataset.load_as_dataset(split="test", sample_size=config.sample_size, seed=config.seed)  # TODO: Update parameters as needed

    # Create data loader
    logging.info("Creating data loader...")
    # TODO: Create your data loader if you have one
    # data_loader = MyDataLoader(
    #     config=config,
    #     model_provider=model_provider,
    #     dataset=benchmark_dataset,
    # )
    data_loader = None  # TODO: Replace with your data loader or None if not using one

    # Create evaluator
    logging.info("Creating benchmark evaluator...")
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset=benchmark_dataset,
        collection_name=config._collection_name,
        limit=config.response_limit,
        query_method=getattr(query_method, config.query_method),
        query_parameters=config.advanced_query_parameters,
        score_columns=["rerank_score", "clip_score"],  # TODO: Adjust as needed
        target_vector=config.target_vector
    )
    
    # Step 1: Load data
    logging.info("=" * 80)
    logging.info("Step 1: Loading data into vector database")
    logging.info("=" * 80)
    try:
        load_data(data_loader, vector_db, hf_dataset)
        logging.info("Data loading completed successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Run evaluation
    logging.info("=" * 80)
    logging.info("Step 2: Running benchmark evaluation")
    logging.info("=" * 80)
    try:
        image_results, query_evaluation = run_evaluation(evaluator, hf_dataset)
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        sys.exit(1)
    
    # Step 3: Save results locally
    logging.info("=" * 80)
    logging.info("Step 3: Saving results")
    logging.info("=" * 80)
    
    # Determine results directory (use /app/results if PVC is mounted, otherwise current directory)
    results_dir = Path("/app/results" if os.path.exists("/app/results") else ".")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    image_results_path = results_dir / config._image_results_file
    query_evaluation_path = results_dir / config._query_eval_metrics_file
    config_csv_path = results_dir / config._config_values_file
    
    image_results.to_csv(image_results_path, index=False)
    query_evaluation.to_csv(query_evaluation_path, index=False)
    
    config_csv_str = config.to_csv()
    with open(config_csv_path, "w") as f:
        f.write(config_csv_str)
    
    logging.info(f"Results saved locally to:")
    logging.info(f"  - {image_results_path}")
    logging.info(f"  - {query_evaluation_path}")
    logging.info(f"  - {config_csv_path}")
    
    # Step 4: Upload to S3 if enabled
    if config._upload_to_s3:
        if not config._s3_bucket:
            logging.warning("UPLOAD_TO_S3 is true but S3_BUCKET is not set. Skipping S3 upload.")
        elif not config._s3_endpoint:
            logging.warning("UPLOAD_TO_S3 is true but S3_ENDPOINT is not set. Skipping S3 upload.")
        elif not config._s3_access_key or not config._s3_secret_key:
            logging.warning("UPLOAD_TO_S3 is true but S3 credentials are not set. Skipping S3 upload.")
        else:
            logging.info("=" * 80)
            logging.info("Step 4: Uploading results to S3")
            logging.info("=" * 80)
            try:
                # Generate S3 keys with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                s3_key_image = f"{config._s3_prefix}/{timestamp}/{config._image_results_file}"
                s3_key_query = f"{config._s3_prefix}/{timestamp}/{config._query_eval_metrics_file}"
                s3_key_config = f"{config._s3_prefix}/{timestamp}/{config._config_values_file}"
                
                upload_to_s3(str(image_results_path), s3_key_image)
                upload_to_s3(str(query_evaluation_path), s3_key_query)
                upload_to_s3(str(config_csv_path), s3_key_config)
                
                logging.info("S3 upload completed successfully.")
            except Exception as e:
                logging.error(f"Error uploading to S3: {e}")
                logging.warning("Continuing despite S3 upload error...")
    else:
        logging.info("S3 upload is disabled (UPLOAD_TO_S3=false or not set).")
    
    vector_db.close()
    logging.info("=" * 80)
    logging.info("Benchmark run completed successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
