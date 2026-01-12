"""script to run MYBENCHMARK: load data and evaluate queries."""

import os
import logging
import time
import sys
from pathlib import Path
import tritonclient.grpc as TritonClient

from imsearch_eval import BenchmarkEvaluator, VectorDBAdapter, ModelProvider
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from benchmark_dataset import MyBenchmarkDataset  # TODO: Import your BenchmarkDataset
# from data_loader import MyDataLoader  # TODO: Import if you have a custom DataLoader
from config import MyConfig  # TODO:set a Config class for your benchmark

config = MyConfig()


def batched(iterable, batch_size):
    """Yield successive batch_size chunks from iterable."""
    from itertools import islice
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def load_data(vector_db: VectorDBAdapter, model_provider: ModelProvider):
    """Load MYBENCHMARK dataset into vector database.
    
    TODO: Implement your data loading logic here.
    See benchmarks/INQUIRE/run_benchmark.py for a complete example.
    """
    
    # TODO: Create your data loader if you have one
    # data_loader = MyDataLoader(config=MyConfig(), model_provider=model_provider)
    
    try:
        # TODO: Create collection schema
        # logging.info("Creating collection schema...")
        # schema_config = data_loader.get_schema_config()
        # vector_db.create_collection(schema_config)
        
        # TODO: Load dataset
        # logging.info(f"Loading dataset: {MYBENCHMARK_DATASET}")
        # from datasets import load_dataset
        # dataset = load_dataset(MYBENCHMARK_DATASET, split="test")
        
        # TODO: Sample if needed
        # if SAMPLE_SIZE > 0:
        #     import random
        #     sampled_indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
        #     dataset = dataset.select(sampled_indices)
        #     logging.info(f"Sampled {SAMPLE_SIZE} records from the dataset.")
        
        # TODO: Process and insert data
        # logging.info("Processing and inserting data...")
        # 
        # if WORKERS == -1:
        #     # Sequential processing
        #     logging.info("Processing sequentially...")
        #     all_processed = []
        #     for batch in batched(dataset, BATCH_SIZE):
        #         processed_batch = data_loader.process_batch(batch)
        #         all_processed.extend(processed_batch)
        #     
        #     # Insert all at once
        #     inserted = vector_db.insert_data(COLLECTION_NAME, all_processed, batch_size=BATCH_SIZE)
        #     logging.info(f"Inserted {inserted} items.")
        # else:
        #     # Parallel processing
        #     from concurrent.futures import ThreadPoolExecutor, as_completed
        #     num_workers = WORKERS if WORKERS > 0 else os.cpu_count()
        #     logging.info(f"Processing with {num_workers} parallel workers...")
        #     
        #     all_processed = []
        #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #         futures = {
        #             executor.submit(data_loader.process_batch, batch): batch
        #             for batch in batched(dataset, BATCH_SIZE)
        #         }
        #         
        #         for future in as_completed(futures):
        #             processed_batch = future.result()
        #             all_processed.extend(processed_batch)
        #     
        #     # Insert all at once
        #     inserted = vector_db.insert_data(COLLECTION_NAME, all_processed, batch_size=BATCH_SIZE)
        #     logging.info(f"Inserted {inserted} items.")
        
        logging.info(f"Successfully loaded {config.MYBENCHMARK_DATASET} into Weaviate collection '{config.COLLECTION_NAME}'")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    finally:
        # vector_db.close()
        pass


def upload_to_s3(local_file_path: str, s3_key: str):
    """Upload a file to S3-compatible storage using MinIO."""
    try:
        from minio import Minio
        from minio.error import S3Error
        
        if not config.S3_ENDPOINT:
            raise ValueError("S3_ENDPOINT environment variable must be set")
        
        # Parse endpoint (remove http:// or https:// if present)
        endpoint = config.S3_ENDPOINT.replace("http://", "").replace("https://", "")
        
        # Create MinIO client
        client = Minio(
            endpoint,
            access_key=config.S3_ACCESS_KEY,
            secret_key=config.S3_SECRET_KEY,
            secure=config.S3_SECURE
        )
        
        # Upload file
        logging.info(f"Uploading {local_file_path} to s3://{config.S3_BUCKET}/{s3_key}")
        client.fput_object(config.S3_BUCKET, s3_key, local_file_path)
        logging.info(f"Successfully uploaded to s3://{config.S3_BUCKET}/{s3_key}")
        
    except ImportError:
        logging.error("minio is not installed. Install it with: pip install minio")
        raise
    except S3Error as e:
        logging.error(f"Error uploading to S3: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error uploading to S3: {e}")
        raise


def run_evaluation(vector_db: VectorDBAdapter, model_provider: ModelProvider):
    """Run the MYBENCHMARK benchmark evaluation."""
    
    # Create benchmark dataset
    logging.info("Creating benchmark dataset class...")
    benchmark_dataset = MyBenchmarkDataset()  # TODO: Use your BenchmarkDataset

    # Create evaluator
    logging.info("Creating benchmark evaluator...")
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset=benchmark_dataset,
        collection_name=config.COLLECTION_NAME,
        query_method=config.QUERY_METHOD,
        score_columns=["rerank_score", "clip_score", "score", "distance"],  # TODO: Adjust as needed
        target_vector=config.TARGET_VECTOR
    )

    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        image_results, query_evaluation = evaluator.evaluate_queries(split="test")
    finally:
        # Clean up
        vector_db.close()
    
    return image_results, query_evaluation


def main():
    """Main entry point for running the complete benchmark."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Step 0: load framework components
    logging.info("=" * 80)
    logging.info("Step 0: Setting up benchmark environment")
    logging.info("=" * 80)
    logging.info("Initializing Weaviate client...")
    weaviate_client = WeaviateAdapter.init_client(  # TODO: Update with your vector database client
        host=config.WEAVIATE_HOST,
        port=config.WEAVIATE_PORT,
        grpc_port=config.WEAVIATE_GRPC_PORT
    )
    
    logging.info("Initializing Triton client...")
    triton_client = TritonClient.InferenceServerClient(url=f"{config.TRITON_HOST}:{config.TRITON_PORT}")  # TODO: Update with your model provider client

    # Create adapters
    logging.info("Creating adapters...")
    vector_db = WeaviateAdapter(  # TODO: Update with your vector database adapter
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )

    model_provider = TritonModelProvider(triton_client=triton_client)  # TODO: Update with your model provider
    
    # Step 1: Load data
    logging.info("=" * 80)
    logging.info("Step 1: Loading data into vector database")
    logging.info("=" * 80)
    try:
        load_data(vector_db, model_provider)  # Call the load_data function defined in this file
        logging.info("Data loading completed successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Run evaluation
    logging.info("=" * 80)
    logging.info("Step 2: Running benchmark evaluation")
    logging.info("=" * 80)
    try:
        image_results, query_evaluation = run_evaluation(vector_db, model_provider)
        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        sys.exit(1)
    
    # Step 3: Save results locally
    logging.info("=" * 80)
    logging.info("Step 3: Saving results")
    logging.info("=" * 80)
    
    # Determine results directory (use /app/results if mounted, otherwise current directory)
    results_dir = Path("/app/results" if os.path.exists("/app/results") else ".")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    image_results_path = results_dir / config.IMAGE_RESULTS_FILE
    query_evaluation_path = results_dir / config.QUERY_EVAL_METRICS_FILE
    
    image_results.to_csv(image_results_path, index=False)
    query_evaluation.to_csv(query_evaluation_path, index=False)
    
    logging.info(f"Results saved locally to:")
    logging.info(f"  - {image_results_path}")
    logging.info(f"  - {query_evaluation_path}")
    
    # Step 4: Upload to S3 if enabled
    if config.UPLOAD_TO_S3:
        if not config.S3_BUCKET:
            logging.warning("UPLOAD_TO_S3 is true but S3_BUCKET is not set. Skipping S3 upload.")
        elif not config.S3_ENDPOINT:
            logging.warning("UPLOAD_TO_S3 is true but S3_ENDPOINT is not set. Skipping S3 upload.")
        elif not config.S3_ACCESS_KEY or not config.S3_SECRET_KEY:
            logging.warning("UPLOAD_TO_S3 is true but S3 credentials are not set. Skipping S3 upload.")
        else:
            logging.info("=" * 80)
            logging.info("Step 4: Uploading results to S3")
            logging.info("=" * 80)
            try:
                # Generate S3 keys with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                s3_key_image = f"{config.S3_PREFIX}/{timestamp}/{config.IMAGE_RESULTS_FILE}"
                s3_key_query = f"{config.S3_PREFIX}/{timestamp}/{config.QUERY_EVAL_METRICS_FILE}"
                
                upload_to_s3(str(image_results_path), s3_key_image)
                upload_to_s3(str(query_evaluation_path), s3_key_query)
                
                logging.info("S3 upload completed successfully.")
            except Exception as e:
                logging.error(f"Error uploading to S3: {e}")
                logging.warning("Continuing despite S3 upload error...")
    else:
        logging.info("S3 upload is disabled (UPLOAD_TO_S3=false or not set).")
    
    logging.info("=" * 80)
    logging.info("Benchmark run completed successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

