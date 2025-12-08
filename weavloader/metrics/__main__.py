#!/usr/bin/env python3
"""
Entry point for running the metrics package as a module.
"""

import logging
from .server import start_metrics_server

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    start_metrics_server()
