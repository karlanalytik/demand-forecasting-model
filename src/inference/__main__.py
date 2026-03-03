# src/inference/__main__.py
"""Inference pipeline entry point."""

import argparse
import logging
import time
import os
from datetime import datetime
from pathlib import Path

import joblib

from .inference import load_inference_data, feature_engineering


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference step."""
    parser = argparse.ArgumentParser(
        description="Run batch inference using a trained model."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to inference dataset",
    )

    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="data/raw",
        help="Directory containing test catalog data",
    )

    parser.add_argument(
        "--prep_data_dir",
        type=str,
        default="data/prep",
        help="Directory containing prepared training data",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="artifacts/xgboost_model.joblib",
        help="Path to trained model",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predictions/predictions.csv",
        help="Path to save predictions",
    )

    parser.add_argument(
        "--month",
        type=int,
        default=11,
        help="Target month for prediction",
    )

    parser.add_argument(
        "--date_block_num",
        type=int,
        default=34,
        help="Target date block number",
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for inference step.

    Creates a timestamped log file under ``artifacts/logs`` and enables console 
    logging.
    """
    os.makedirs("artifacts/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"artifacts/logs/inference_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    """Entry point for the inference pipeline.
    
    Orchestrates argument parsing, logging configuration, data loading, 
    cleaning, feature engineering, and artifact generation.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    start_time = time.time()
    logger.info("Starting inference pipeline")

    try:
        input_path = Path(args.input_path)
        raw_data_dir = Path(args.raw_data_dir)
        prep_data_dir = Path(args.prep_data_dir)
        model_path = Path(args.model_path)
        output_path = Path(args.output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = load_inference_data(
            input_path=input_path,
            raw_data_dir=raw_data_dir,
            prep_data_dir=prep_data_dir,
        )

        logger.info("Creating features for inference")
        df = feature_engineering(
            df,
            month=args.month,
            date_block_num=args.date_block_num,
        )

        feature_cols = [
            col for col in df.columns
            if col not in ["ID", "item_cnt_month"]
        ]

        X = df[feature_cols]  # noqa: N806  # pylint: disable=invalid-name
        # noqa justified because X/y is standard ML notation
        logger.info("Feature matrix shape: %s", X.shape)

        logger.info("Loading trained model from %s", args.model_path)
        model = joblib.load(model_path)

        logger.info("Running predictions")
        preds = model.predict(X)

        results = df.copy()
        results["item_cnt_month"] = preds

        results[["ID", "item_cnt_month"]].to_csv(output_path, index=False)
        logger.info("Predictions saved to %s", output_path)
        logger.info("Total predictions: %s", len(results))

        logger.info("Inference completed successfully")

    except Exception:
        logger.exception("Inference pipeline failed")
        raise

    duration = time.time() - start_time
    logger.info(
        "Inference pipeline completed successfully in %.2f seconds",
        duration,
    )


if __name__ == "__main__":
    main()
