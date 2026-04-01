"""Model evaluation script for SageMaker Processing step."""

import json
import logging
import os
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TARGET_COL = "item_cnt_month"

MODEL_DIR = "/opt/ml/processing/input/model"
TEST_DIR = "/opt/ml/processing/input/test"
EVALUATION_DIR = "/opt/ml/processing/output/evaluation"


def extract_model_artifacts(model_dir: str) -> None:
    base_path = Path(model_dir)
    tar_files = list(base_path.glob("*.tar.gz"))

    for tar_path in tar_files:
        logger.info("Extracting model artifact: %s", tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)


def find_model_file(model_dir: str) -> str:
    base_path = Path(model_dir)

    candidates = list(base_path.rglob("*.joblib")) + list(base_path.rglob("*.pkl"))

    if not candidates:
        raise FileNotFoundError(
            f"No .joblib or .pkl model file found in {model_dir}"
        )

    model_path = sorted(candidates)[0]
    logger.info("Resolved model file: %s", model_path)
    return str(model_path)


def find_test_file(test_dir: str) -> str:
    base_path = Path(test_dir)

    expected = base_path / "test.csv"
    if expected.exists():
        logger.info("Using expected test file: %s", expected)
        return str(expected)

    candidates = sorted(base_path.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {test_dir}")

    logger.info("Using fallback test file: %s", candidates[0])
    return str(candidates[0])


def load_model(model_dir: str):
    extract_model_artifacts(model_dir)
    model_path = find_model_file(model_dir)
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def load_test_data(test_dir: str) -> pd.DataFrame:
    test_path = find_test_file(test_dir)
    logger.info("Loading test data from %s", test_path)
    return pd.read_csv(test_path)


def evaluate_model(model, test_df: pd.DataFrame, target_col: str) -> float:
    if target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in test data. "
            f"Available columns: {list(test_df.columns)}"
        )

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    logger.info("Evaluation completed. RMSE = %.6f", rmse)
    return rmse


def save_evaluation(rmse: float, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    evaluation_path = os.path.join(output_dir, "evaluation.json")

    evaluation_content = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": 0.0
            }
        }
    }

    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(evaluation_content, file)

    logger.info("Evaluation report saved to %s", evaluation_path)


def main():
    """Main evaluation entrypoint."""
    logger.info("=== NUEVA VERSION DE EVALUATE.PY ===")
    logger.info("Starting evaluation step")

    # Ver qué archivos llegan del modelo
    logger.info("FILES IN MODEL DIR (ANTES):")
    for p in Path(MODEL_DIR).rglob("*"):
        logger.info(str(p))

    # Intentar extraer
    extract_model_artifacts(MODEL_DIR)

    # Ver después de extraer
    logger.info("FILES IN MODEL DIR (DESPUÉS):")
    for p in Path(MODEL_DIR).rglob("*"):
        logger.info(str(p))

    # Ver test también
    logger.info("FILES IN TEST DIR:")
    for p in Path(TEST_DIR).rglob("*"):
        logger.info(str(p))

    #  Flujo normal
    model = load_model(MODEL_DIR)
    test_df = load_test_data(TEST_DIR)
    rmse = evaluate_model(model, test_df, TARGET_COL)
    save_evaluation(rmse, EVALUATION_DIR)

    logger.info("Evaluation step completed successfully")


if __name__ == "__main__":
    main()