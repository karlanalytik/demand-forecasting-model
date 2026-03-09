"""Model training module.

This module loads prepared data, performs a time-based train/validation
split, trains a demand forecasting model, evaluates its performance,
and saves the trained model artifacts.
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# =========================
# Logging configuration
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Argument parsing
# =========================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with SageMaker-compatible defaults."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "data/prep"),
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="sales_prep.csv",
        help="Training CSV filename.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="item_cnt_month",
        help="Target column name.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="date_block_num",
        help="Time column used for time-based split.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "artifacts"),
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "artifacts"),
        help="Directory where metrics/output files will be saved.",
    )

    return parser.parse_args()


# =========================
# Functions
# =========================
def load_data(path: str) -> pd.DataFrame:
    """Load prepared dataset from disk."""
    logger.info("Loading data from %s", path)
    return pd.read_csv(path)


def resolve_train_path(train_data_dir: str, train_file: str) -> str:
    """Resolve the CSV path from SageMaker channel or local fallback."""
    train_dir = Path(train_data_dir)

    if train_dir.is_file():
        return str(train_dir)

    if train_dir.exists():
        csv_files = sorted(train_dir.glob("*.csv"))
        if csv_files:
            logger.info("Using CSV found in train-data-dir: %s", csv_files[0])
            return str(csv_files[0])

    local_candidate = Path("data/prep") / train_file
    if local_candidate.exists():
        logger.info("Using local fallback CSV: %s", local_candidate)
        return str(local_candidate)

    raise FileNotFoundError(
        f"No CSV found in {train_data_dir} and fallback {local_candidate} does not exist."
    )


def train_val_split(
    data: pd.DataFrame,
    target: str,
    time_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and validation sets using time-based logic."""
    split_date = data[time_col].quantile(0.8)
    logger.info("Using split date: %s", split_date)

    train = data[data[time_col] <= split_date]
    val = data[data[time_col] > split_date]

    X_train = train.drop(columns=[target])  # noqa: N806
    y_train = train[target]  # noqa: N806

    X_val = val.drop(columns=[target])  # noqa: N806
    y_val = val[target]  # noqa: N806

    logger.info(
        "Train size: %d rows | Validation size: %d rows",
        len(train),
        len(val),
    )

    return X_train, X_val, y_train, y_val


def train_model(
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    X_val: pd.DataFrame,  # noqa: N803
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    """Train an XGBoost regression model."""
    logger.info("Training XGBoost model")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=55,
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
    X_val: pd.DataFrame,  # noqa: N803
    y_val: pd.Series,
) -> tuple[np.ndarray, float]:
    """Evaluate the model using RMSE."""
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    logger.info("Validation RMSE: %.4f", rmse)

    return preds, rmse


def save_model(model: xgb.XGBRegressor, path: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    logger.info("Model saved at %s", path)


def save_metrics(rmse: float, output_data_dir: str) -> None:
    """Save training metrics to disk."""
    os.makedirs(output_data_dir, exist_ok=True)
    metrics_path = os.path.join(output_data_dir, "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"rmse_validation={rmse:.6f}\n")

    logger.info("Metrics saved at %s", metrics_path)


def main() -> None:
    """Main training entrypoint compatible with SageMaker."""
    args = parse_args()

    logger.info("Arguments received:")
    for key, value in vars(args).items():
        logger.info("  %s=%s", key, value)

    train_path = resolve_train_path(args.train_data_dir, args.train_file)
    data = load_data(train_path)

    if args.target_col not in data.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    if args.time_col not in data.columns:
        raise ValueError(
            f"Time column '{args.time_col}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    X_train, X_val, y_train, y_val = train_val_split(
        data=data,
        target=args.target_col,
        time_col=args.time_col,
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    _, rmse = evaluate(
        model=model,
        X_val=X_val,
        y_val=y_val,
    )

    model_path = os.path.join(args.model_dir, "model.joblib")
    save_model(model, model_path)
    save_metrics(rmse, args.output_data_dir)


if __name__ == "__main__":
    main()
