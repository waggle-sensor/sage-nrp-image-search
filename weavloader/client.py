'''This file contains the code to interact with the weaviate client'''
import logging
import argparse
import os
import weaviate
import time

def initialize_weaviate_client(weaviate_host: str, weaviate_port: int, weaviate_grpc_port: int):
    '''
    Intialize weaviate client

    Args:
        weaviate_host: Weaviate host IP
        weaviate_port: Weaviate REST port
        weaviate_grpc_port: Weaviate GRPC port
        
    Returns:
        weaviate.Client: Weaviate client
    '''
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