'''Milvus client initialization for weavloader.'''
import logging
import os
import time

from pymilvus import MilvusClient


def initialize_milvus_client(uri: str = None, token: str = None, db_name: str = None):
    '''
    Initialize Milvus client.

    Args:
        uri: Milvus URI (defaults to MILVUS_URI env)
        token: Auth token user:password (defaults to MILVUS_TOKEN env)
        db_name: Milvus database name (defaults to MILVUS_DB / image_search_svc)

    Returns:
        MilvusClient
    '''
    uri = uri or os.getenv("MILVUS_URI", "https://milvus.nrp-nautilus.io:50051")
    token = token if token is not None else os.getenv("MILVUS_TOKEN", "")
    db_name = db_name or os.getenv("MILVUS_DB", "image_search_svc")

    logging.debug(f"Attempting to connect to Milvus at {uri} (db={db_name})")

    while True:
        try:
            kwargs = {"uri": uri, "db_name": db_name}
            if token:
                kwargs["token"] = token
            client = MilvusClient(**kwargs)
            client.list_collections()
            logging.debug(f"Successfully connected to Milvus db={db_name}")
            return client
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            logging.debug("Retrying in 10 seconds...")
            time.sleep(10)
