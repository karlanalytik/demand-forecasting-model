# src/train.py
"""Model training module.

This module loads prepared data, performs a time-based train/validation
split, trains a demand forecasting model, evaluates its performance,
and saves the trained model artifacts.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# =========================
# Logging configuration
# =========================
LOG_DIR = "artifacts/logs"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"train_{timestamp}.log")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================
# Arguments
# =========================
def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train demand forecasting model"
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


# =========================
# Functions
# =========================
def load_data(path: str) -> pd.DataFrame:
    """Load prepared dataset from disk.

    Parameters
    ----------
    path : str
        Relative path to the prepared CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    full_path = os.path.join(PROJECT_ROOT, path)
    logger.info("Loading data from %s", path)

    return pd.read_csv(full_path)


def train_val_split(
    data: pd.DataFrame,
    target: str,
    time_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and validation sets using time-based logic.

    The split is performed using the 80th percentile of the time column.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset, already sorted by time.
    target : str
        Target variable name.
    time_col : str, default="date"
        Time column used for splitting.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame
        Validation features.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    """
    split_date = data[time_col].quantile(0.8)
    logger.info("Using split date: %s", split_date)

    train = data[data[time_col] <= split_date]
    val = data[data[time_col] > split_date]

    X_train = train.drop(columns=[target])  # noqa: N806  # pylint: disable=invalid-name
    y_train = train[target]  # noqa: N806  # pylint: disable=invalid-name

    X_val = val.drop(columns=[target])  # noqa: N806  # pylint: disable=invalid-name
    y_val = val[target]  # noqa: N806  # pylint: disable=invalid-name
    # noqa justified because X_train/X_val is standard ML notation

    logger.info(
        "Train size: %d rows | Validation size: %d rows",
        len(train),
        len(val),
    )

    return X_train, X_val, y_train, y_val


def train_model(
    X_train: pd.DataFrame,  # noqa: N803  # pylint: disable=invalid-name
    y_train: pd.Series,
    X_val: pd.DataFrame,  # noqa: N803  # pylint: disable=invalid-name
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    # noqa justified because X_train/X_val is standard ML notation
    """Train an XGBoost regression model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target values.

    Returns
    -------
    xgb.XGBRegressor
        Trained XGBoost model.
    """
    logger.info("Training XGBoost model")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=55
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    return model


def evaluate(
    model: xgb.XGBRegressor,
    X_val: pd.DataFrame,  # noqa: N803  # pylint: disable=invalid-name
    y_val: pd.Series,
) -> tuple[np.ndarray, float]:
    # noqa justified because X_val is standard ML notation
    """Evaluate the model using RMSE.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target values.

    Returns
    -------
    preds : np.ndarray
        Model predictions.
    rmse : float
        Root Mean Squared Error on validation data.
    """
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    logger.info("Validation RMSE: %.4f", rmse)
    return preds, rmse


def save_model(model: xgb.XGBRegressor, path: str) -> None:
    """Save trained model to disk.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    path : str
        Output path for the serialized model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    logger.info("Model saved at %s", path)


def main() -> None:
    """Run the full training pipeline."""
    start_time = time.time()
    logger.info("Starting training pipeline")

    try:
        args = parse_args()

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
        logger.exception("Training pipeline failed")
        raise

    duration = time.time() - start_time
    logger.info(
        "Training pipeline completed successfully in %.2f seconds",
        duration,
    )


if __name__ == "__main__":
    main()
