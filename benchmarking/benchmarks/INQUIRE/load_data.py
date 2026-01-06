"""Load INQUIRE dataset into vector database for INQUIRE benchmark."""

import os
import logging
import time
import random
from datasets import load_dataset
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.grpc as TritonClient
from config import INQUIREConfig
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
from data_loader import INQUIREDataLoader

# Environment variables
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 0))
WORKERS = int(os.environ.get("WORKERS", 0))
IMAGE_BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", 100))
TRITON_HOST = os.environ.get("TRITON_HOST", "triton")
TRITON_PORT = os.environ.get("TRITON_PORT", "8001")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "INQUIRE")


def batched(iterable, batch_size):
    """Yield successive batch_size chunks from iterable."""
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


def load_data():
    """Load INQUIRE dataset into Weaviate for INQUIRE benchmark."""
    
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
    vector_db = WeaviateAdapter(weaviate_client=weaviate_client, triton_client=triton_client)
    model_provider = TritonModelProvider(triton_client=triton_client)
    
    # Create data loader
    data_loader = INQUIREDataLoader(config=INQUIREConfig(), model_provider=model_provider)
    
    try:
        # Create collection schema
        logging.info("Creating collection schema...")
        schema_config = data_loader.get_schema_config()
        schema_config["name"] = COLLECTION_NAME
        vector_db.create_collection(COLLECTION_NAME, schema_config)
        
        # Load dataset
        logging.info(f"Loading dataset: {INQUIRE_DATASET}")
        dataset = load_dataset(INQUIRE_DATASET, split="test")
        
        # Sample if needed
        if SAMPLE_SIZE > 0:
            sampled_indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
            dataset = dataset.select(sampled_indices)
            logging.info(f"Sampled {SAMPLE_SIZE} records from the dataset.")
        
        # Process and insert data
        logging.info("Processing and inserting data...")
        
        if WORKERS == -1:
            # Sequential processing
            logging.info("Processing sequentially...")
            all_processed = []
            for batch in batched(dataset, IMAGE_BATCH_SIZE):
                processed_batch = data_loader.process_batch(batch)
                all_processed.extend(processed_batch)
            
            # Insert all at once
            inserted = vector_db.insert_data(COLLECTION_NAME, all_processed, batch_size=IMAGE_BATCH_SIZE)
            logging.info(f"Inserted {inserted} items.")
        else:
            # Parallel processing
            num_workers = WORKERS if WORKERS > 0 else os.cpu_count()
            logging.info(f"Processing with {num_workers} parallel workers...")
            
            all_processed = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(data_loader.process_batch, batch): batch
                    for batch in batched(dataset, IMAGE_BATCH_SIZE)
                }
                
                for future in as_completed(futures):
                    processed_batch = future.result()
                    all_processed.extend(processed_batch)
            
            # Insert all at once
            inserted = vector_db.insert_data(COLLECTION_NAME, all_processed, batch_size=IMAGE_BATCH_SIZE)
            logging.info(f"Inserted {inserted} items.")
        
        logging.info(f"Successfully loaded {INQUIRE_DATASET} into Weaviate collection '{COLLECTION_NAME}'")
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    finally:
        vector_db.close()


if __name__ == "__main__":
    load_data()
    
    # Keep the program running when the loading is done
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        exit()

