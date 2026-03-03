# src/training/__main__.py
"""Training pipeline entry point."""

import argparse
import logging
import os
import time
from datetime import datetime

from .train import (
    load_data,
    train_val_split,
    train_model,
    evaluate,
    save_model
)


def parse_args():
    """Parse command-line arguments for the training step.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing raw and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Run the training pipeline step."
        )

    parser.add_argument(
        "--data_path",
        type = str,
        default="data/prep/sales_prep.csv",
        help="Path to processed monthly dataset"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="item_cnt_month",
        help="Target variable name"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="artifacts/xgboost_model.joblib",
        help="Path to save trained model"
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default="predictions/pred.csv",
        help="Path to save predictions"
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for the training step.

    Creates a timestamped log file under ``artifacts/logs`` and enables console 
    logging.
    """
    os.makedirs("artifacts/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"artifacts/logs/train_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    """Entry point for the training pipeline step.
    
    Orchestrates argument parsing, logging configuration, data loading, 
    time-based splitting, model training, evaluation, and artifact generation.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    start_time = time.time()
    logger.info("Starting training step")

    try:
        data = load_data(args.data_path)

        time_col = "date_block_num"
        data = data.sort_values(time_col).reset_index(drop=True)

        X_train, X_val, y_train, y_val = train_val_split(  # noqa: N806  # pylint: disable=invalid-name
            data,
            target=args.target,
            time_col=time_col
        )
        # noqa justified because X_train/X_val is standard ML notation

        model = train_model(X_train, y_train, X_val, y_val)
        evaluate(model, X_val, y_val)

        save_model(model, args.model_path)


    except Exception:
        logger.exception("Training failed")
        raise

    duration = time.time() - start_time
    logger.info("Training completed in %.2f seconds", duration)


if __name__ == "__main__":
    main()
