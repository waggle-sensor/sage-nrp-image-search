'''This file contains the code to interact with the weaviate client'''
import logging
import argparse
import os
import weaviate
import time

def initialize_weaviate_client():
    '''
    Intialize weaviate client based on arg or env var
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaviate_host",
        default=os.getenv("WEAVIATE_HOST","127.0.0.1"),
        help="Weaviate host IP.",
    )
    parser.add_argument(
        "--weaviate_port",
        default=os.getenv("WEAVIATE_PORT","8080"),
        help="Weaviate REST port.",
    )
    parser.add_argument(
        "--weaviate_grpc_port",
        default=os.getenv("WEAVIATE_GRPC_PORT","50051"),
        help="Weaviate GRPC port.",
    )
    args = parser.parse_args()

    weaviate_host = args.weaviate_host
    weaviate_port = args.weaviate_port
    weaviate_grpc_port = args.weaviate_grpc_port

    logging.debug(f"Attempting to connect to Weaviate at {weaviate_host}:{weaviate_port}")

    # Retry logic to connect to Weaviate
    while True:
        try:
            client = weaviate.connect_to_local(
                host=weaviate_host,
                port=weaviate_port,
                grpc_port=weaviate_grpc_port
            )
            logging.debug("Successfully connected to Weaviate")
            return client
        except weaviate.exceptions.WeaviateConnectionError as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            logging.debug("Retrying in 10 seconds...")
            time.sleep(10)