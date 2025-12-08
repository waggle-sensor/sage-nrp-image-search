'''Main File'''
import logging
import os
import sys
from job_system import app as celery_app
import argparse

if __name__ == "__main__":
    # configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        default=os.getenv("LOG_LEVEL","INFO"),
        help="Log level.",
    )
    parser.add_argument(
        "--worker_type",
        default=None,
        help="Worker type to start (processor, moderator, cleaner).",
        choices=["processor", "moderator", "cleaner"],
    )
    args = parser.parse_args()

    # Configure logging
    LOG_LEVEL = args.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Check what type of worker we should start
    if args.worker_type == "processor":
        # Run as Celery processor worker
        logging.info("[MAIN] Starting Celery processor worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=image_processing',
            '--concurrency=6',
            f'-n processor@%h'
        ])
    elif args.worker_type == "moderator":
        # Run as Celery moderator worker
        logging.info("[MAIN] Starting Celery moderator worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=data_monitoring',
            '--concurrency=1',
            f'-n moderator@%h'
        ])
    elif args.worker_type == "cleaner":
        # Start the Celery cleanup worker
        logging.info("[MAIN] Starting Celery cleanup worker...")
        celery_app.worker_main([
            'worker',
            f'--loglevel={LOG_LEVEL.lower()}',
            '--queues=cleanup',
            '--concurrency=3',
            f'-n cleaner@%h'
        ])
    else: 
        logging.error("[MAIN] Invalid worker type, must be one of: processor, moderator, cleaner")
        sys.exit(1)