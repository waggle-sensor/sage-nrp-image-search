'''Main File'''
#NOTE: This app will be deployed in our cloud under k8s namespace shared, 
# with a similiar set up as waggle-auth-app where updates are rolled out
# with python scripts aka Migrations.
# Additionally, the vectorize and reranker modules will be deployed on our 
# cloud with a machine with cuda and communication with our cloud k8s namespace shared.

import logging
from management import run_migrations
from client import initialize_weaviate_client

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Weaviate client connection
    client = initialize_weaviate_client()

    run_migrations(client)

    # close client, when done
    client.close() 