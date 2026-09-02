'''Milvus client initialization for weavmanage.'''
import logging
import os
import time

from pymilvus import MilvusClient


def initialize_milvus_client():
    '''
    Initialize Milvus client from env vars.
    Connects to the NRP-managed database (default: image_search_svc).
    '''
    uri = os.getenv("MILVUS_URI", "https://milvus.nrp-nautilus.io:50051")
    token = os.getenv("MILVUS_TOKEN", "")
    db_name = os.getenv("MILVUS_DB", "image_search_svc")

    logging.debug(f"Attempting to connect to Milvus at {uri} (db={db_name})")

    while True:
        try:
            kwargs = {"uri": uri, "db_name": db_name}
            if token:
                kwargs["token"] = token
            client = MilvusClient(**kwargs)
            # Verify connectivity
            client.list_collections()
            logging.debug(f"Successfully connected to Milvus db={db_name}")
            return client
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            logging.debug("Retrying in 10 seconds...")
            time.sleep(10)
