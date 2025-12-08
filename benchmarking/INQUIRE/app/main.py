'''This file contains the code to run the Benchmark and save the results.'''

import os
from inquire_eval import evaluate_queries
from datasets import load_dataset
from client import initialize_weaviate_client
import logging
import time

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET", "sagecontinuum/INQUIRE-Benchmark-small")
IMAGE_RESULTS_FILE = os.environ.get("IMAGE_RESULTS_FILE", "image_search_results.csv")
QUERY_EVAL_METRICS_FILE = os.environ.get("QUERY_EVAL_METRICS_FILE", "query_eval_metrics.csv")

def load_inquire_dataset():
    """ Load INQUIRE dataset from HuggingFace and return it as a pandas DataFrame. """
    dataset = load_dataset(INQUIRE_DATASET, split="test").to_pandas()
    return dataset

if __name__ == "__main__":

   # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    # Load INQUIRE dataset
    inquire_dataset = load_inquire_dataset()

    # Connect to Weaviate and Evaluate search system
    with initialize_weaviate_client() as weaviate_client:
        image_results, query_evaluation = evaluate_queries(weaviate_client, inquire_dataset)

    # Save results
    image_results_location = os.path.join("/app", IMAGE_RESULTS_FILE)
    query_evaluation_location = os.path.join("/app", QUERY_EVAL_METRICS_FILE)

    image_results.to_csv(image_results_location, index=False)
    query_evaluation.to_csv(query_evaluation_location, index=False)
    logging.debug(f"Evaluation is done, INQUIRE results saved to {image_results_location} and {query_evaluation_location}")
    weaviate_client.close()

    # Keep the program running when the evaluation is done
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        exit()