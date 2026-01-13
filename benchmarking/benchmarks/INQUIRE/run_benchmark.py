"""run INQUIRE benchmark: load data and evaluate queries."""

import os
import logging
import time
import sys
from pathlib import Path
import tritonclient.grpc as TritonClient
from datasets import Dataset
from imsearch_eval import BenchmarkEvaluator, VectorDBAdapter, BatchedIterator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider, WeaviateQuery
from benchmark_dataset import INQUIRE
from config import INQUIREConfig
from data_loader import INQUIREDataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

config = INQUIREConfig()


def load_data(data_loader: INQUIREDataLoader, vector_db: VectorDBAdapter, hf_dataset: Dataset):
    """Load INQUIRE dataset into Weaviate for INQUIRE benchmark."""    
    try:
        # Create collection schema
        logging.info("Creating collection schema...")
        schema_config = data_loader.get_schema_config()
        vector_db.create_collection(schema_config)

        # Process and insert data
        logging.info("Processing and inserting data...")
        
        if config.workers == -1:
            # Sequential processing
            logging.info("Processing sequentially...")
            all_processed = []
            for batch in BatchedIterator(hf_dataset, config.image_batch_size):
                processed_batch = data_loader.process_batch(batch)
                all_processed.extend(processed_batch)
            
            # Insert all at once
            inserted = vector_db.insert_data(config.collection_name, all_processed, batch_size=config.image_batch_size)
            logging.info(f"Inserted {inserted} items.")
        else:
            # Parallel processing
            num_workers = config.workers if config.workers > 0 else os.cpu_count()
            logging.info(f"Processing with {num_workers} parallel workers...")
            
            all_processed = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(data_loader.process_batch, batch): batch
                    for batch in BatchedIterator(hf_dataset, config.image_batch_size)
                }
                
                for future in as_completed(futures):
                    processed_batch = future.result()
                    all_processed.extend(processed_batch)
            
            # Insert all at once
            inserted = vector_db.insert_data(config.collection_name, all_processed, batch_size=config.image_batch_size)
            logging.info(f"Inserted {inserted} items.")
        
        logging.info(f"Successfully loaded {config.inquire_dataset} into Weaviate collection '{config.collection_name}'")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        vector_db.close()
        raise

def run_evaluation(evaluator: BenchmarkEvaluator, hf_dataset: Dataset):
    """Run the INQUIRE benchmark evaluation."""
    # Run evaluation
    logging.info("Starting evaluation...")
    try:
        image_results, query_evaluation = evaluator.evaluate_queries(dataset=hf_dataset)
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
        
        if not config.s3_endpoint:
            raise ValueError("S3_ENDPOINT environment variable must be set")
        
        # Parse endpoint (remove http:// or https:// if present)
        endpoint = config.s3_endpoint.replace("http://", "").replace("https://", "")
        
        # Create MinIO client
        client = Minio(
            endpoint,
            access_key=config.s3_access_key,
            secret_key=config.s3_secret_key,
            secure=config.s3_secure
        )
        
        # Upload file
        logging.info(f"Uploading {local_file_path} to s3://{config.s3_bucket}/{s3_key}")
        client.fput_object(config.s3_bucket, s3_key, local_file_path)
        logging.info(f"Successfully uploaded to s3://{config.s3_bucket}/{s3_key}")
        
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
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Step 0: load framework components
    logging.info("=" * 80)
    logging.info("Step 0: Setting up benchmark environment")
    logging.info("=" * 80)
    logging.info("Initializing Weaviate client...")
    weaviate_client = WeaviateAdapter.init_client(
        host=config.weaviate_host,
        port=config.weaviate_port,
        grpc_port=config.weaviate_grpc_port
    )
    
    logging.info("Initializing Triton client...")
    triton_client = TritonClient.InferenceServerClient(url=f"{config.triton_host}:{config.triton_port}")

    # Create query method
    query_method = WeaviateQuery(
        weaviate_client=weaviate_client,
        triton_client=triton_client
    )

    # Create adapters
    logging.info("Creating adapters...")
    vector_db = WeaviateAdapter(
        weaviate_client=weaviate_client,
        triton_client=triton_client,
        query_method=query_method
    )

    model_provider = TritonModelProvider(triton_client=triton_client)

    # Create benchmark dataset
    logging.info("Creating benchmark dataset class...")
    benchmark_dataset = INQUIRE(dataset_name=config.inquire_dataset)
    hf_dataset = benchmark_dataset.load_as_dataset(split="test", sample_size=config.sample_size, seed=config.seed)

    # Create data loader
    logging.info("Creating data loader...")
    data_loader = INQUIREDataLoader(config=config, model_provider=model_provider)

    # Create evaluator
    logging.info("Creating benchmark evaluator...")
    evaluator = BenchmarkEvaluator(
        vector_db=vector_db,
        model_provider=model_provider,
        dataset=benchmark_dataset,
        collection_name=config.collection_name,
        limit=config.response_limit,
        query_method=getattr(query_method, config.query_method),
        query_parameters=config.advanced_query_parameters,
        score_columns=["rerank_score", "clip_score"],
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
    
    image_results_path = results_dir / config.image_results_file
    query_evaluation_path = results_dir / config.query_eval_metrics_file
    
    image_results.to_csv(image_results_path, index=False)
    query_evaluation.to_csv(query_evaluation_path, index=False)
    
    logging.info(f"Results saved locally to:")
    logging.info(f"  - {image_results_path}")
    logging.info(f"  - {query_evaluation_path}")
    
    # Step 4: Upload to S3 if enabled
    if config.upload_to_s3:
        if not config.s3_bucket:
            logging.warning("UPLOAD_TO_S3 is true but S3_BUCKET is not set. Skipping S3 upload.")
        elif not config.s3_endpoint:
            logging.warning("UPLOAD_TO_S3 is true but S3_ENDPOINT is not set. Skipping S3 upload.")
        elif not config.s3_access_key or not config.s3_secret_key:
            logging.warning("UPLOAD_TO_S3 is true but S3 credentials are not set. Skipping S3 upload.")
        else:
            logging.info("=" * 80)
            logging.info("Step 4: Uploading results to S3")
            logging.info("=" * 80)
            try:
                # Generate S3 keys with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                s3_key_image = f"{config.s3_prefix}/{timestamp}/{config.image_results_file}"
                s3_key_query = f"{config.s3_prefix}/{timestamp}/{config.query_eval_metrics_file}"
                
                upload_to_s3(str(image_results_path), s3_key_image)
                upload_to_s3(str(query_evaluation_path), s3_key_query)
                
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

