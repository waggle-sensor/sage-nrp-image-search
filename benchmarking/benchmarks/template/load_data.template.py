# Template for load_data.py
# Copy this file to load_data.py and customize for your benchmark
#
# This script loads data into the vector database.
# You can use the default DataLoader or implement a custom one.
#
# This template uses the abstract benchmarking framework:
# - Framework: Abstract interfaces (framework/)
# - Adapters: Concrete implementations (adapters/)
#   - triton.py: TritonModelProvider, TritonModelUtils
#   - weaviate.py: WeaviateAdapter, WeaviateQuery

import sys
import os
import logging

# Add framework and adapters to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../framework'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../adapters'))

from adapters import WeaviateAdapter, TritonModelProvider
from tritonclient.grpc import InferenceServerClient as TritonClient
# from data_loader import MyDataLoader  # TODO: Import if you have a custom DataLoader
# from config import MyConfig  # TODO: Import if you have a Config

# Environment variables
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
TRITON_HOST = os.getenv("TRITON_HOST", "triton")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "MYBENCHMARK")  # TODO: Update default

# Data loading parameters
BATCH_SIZE = int(os.getenv("IMAGE_BATCH_SIZE", "25"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "0"))  # 0 = all data
WORKERS = int(os.getenv("WORKERS", "5"))


def main():
    """Load data into the vector database."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting data loading...")
    
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
    
    # TODO: Implement data loading logic
    # Example:
    # 1. Load dataset using your DatasetLoader
    # 2. Create collection using vector_db.create_collection()
    # 3. Process and insert data using vector_db.insert_data()
    # 4. Or use a custom DataLoader if you have one
    
    logging.info("Data loading complete!")


if __name__ == "__main__":
    main()

