"""Model training module for SageMaker pipeline.

This module loads prepared train and validation datasets from SageMaker
training channels, trains a demand forecasting model, evaluates its
performance, and saves the trained model artifact.
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with SageMaker-compatible defaults."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
        help="Directory containing validation data.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.csv",
        help="Training CSV filename.",
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default="validation.csv",
        help="Validation CSV filename.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="item_cnt_month",
        help="Target column name.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        help="Directory where metrics/output files will be saved.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.info("Ignoring unknown arguments from SageMaker/container: %s", unknown)

    return args


def load_data(path: str) -> pd.DataFrame:
    """Load prepared dataset from disk."""
    logger.info("Loading data from %s", path)
    return pd.read_csv(path)


def resolve_csv_path(data_dir: str, expected_file: str) -> str:
    """Resolve a CSV path from a SageMaker channel directory or local fallback."""
    data_path = Path(data_dir)

    if data_path.is_file():
        return str(data_path)

    if data_path.exists():
        expected_candidate = data_path / expected_file
        if expected_candidate.exists():
            logger.info("Using expected CSV file: %s", expected_candidate)
            return str(expected_candidate)

        csv_files = sorted(data_path.glob("*.csv"))
        if csv_files:
            logger.info("Using CSV found in %s: %s", data_dir, csv_files[0])
            return str(csv_files[0])

    local_candidate = Path("data/prep") / expected_file
    if local_candidate.exists():
        logger.info("Using local fallback CSV: %s", local_candidate)
        return str(local_candidate)

    raise FileNotFoundError(
        f"No CSV found in {data_dir} and fallback {local_candidate} does not exist."
    )


def split_features_target(
    data: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target])  # noqa: N806
    y = data[target]  # noqa: N806
    return X, y


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
        verbose=50,
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

    train_path = resolve_csv_path(args.train_data_dir, args.train_file)
    validation_path = resolve_csv_path(
        args.validation_data_dir,
        args.validation_file,
    )

    train_data = load_data(train_path)
    validation_data = load_data(validation_path)

    X_train, y_train = split_features_target(
        data=train_data,
        target=args.target_col,
    )
    X_val, y_val = split_features_target(
        data=validation_data,
        target=args.target_col,
    )

    logger.info(
        "Train size: %d rows | Validation size: %d rows",
        len(train_data),
        len(validation_data),
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
