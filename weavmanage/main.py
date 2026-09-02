'''Main File'''
#NOTE: This app will be deployed in our cloud under k8s namespace shared,
# with a similiar set up as waggle-auth-app where updates are rolled out
# with python scripts aka Migrations.

import logging
from management import run_migrations
from client import initialize_milvus_client

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    client = initialize_milvus_client()

    run_migrations(client)

    client.close()
