# src/preprocessing/__main__.py
"""Preprocessing pipeline entry point."""

import argparse
import logging
import os
import time
from datetime import datetime

from .prep import (
    load_raw_data,
    clean_data,
    feature_engineering,
    save_prepared_data,
)


def parse_args():
    """Parse command-line arguments for the preprocessing step.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing raw and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline step."
        )
    parser.add_argument("--raw-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for the preprocessing step.

    Creates a timestamped log file under ``artifacts/logs`` and enables console 
    logging.
    """
    os.makedirs("artifacts/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"artifacts/logs/prep_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    """Entry point for the preprocessing pipeline step.

    Orchestrates argument parsing, logging configuration, data loading, 
    cleaning, feature engineering, and artifact generation.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    start_time = time.time()
    logger.info("Starting preprocessing step")

    try:
        data = load_raw_data(args.raw_path)
        data = clean_data(data)
        data = feature_engineering(data)
        save_prepared_data(data, args.output_path)

    except Exception:
        logger.exception("Preprocessing failed")
        raise

    duration = time.time() - start_time
    logger.info("Preprocessing completed in %.2f seconds", duration)


if __name__ == "__main__":
    main()
