# src/inference.py
"""Model inference module.

This module loads a trained model and prepared reference data, performs
feature engineering for inference, generates predictions, and saves the
results to disk.
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = "data/raw"
PREP_DATA_PATH = "data/prep"


# =========================
# Arguments
# =========================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Run batch inference using a trained model"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to inference dataset"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="artifacts/model.joblib",
        help="Path to trained model"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predictions/predictions.csv",
        help="Path to save predictions"
    )

    return parser.parse_args()


# =========================
# Functions
# =========================
def load_inference_data(input_path: str) -> pd.DataFrame:
    """Load and assemble inference dataset.

    This function loads the inference input data and enriches it using
    historical prepared data and reference catalogs (items, categories,
    and shops).

    Parameters
    ----------
    input_path : Path
        Path to the inference input CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing features required for inference.
    """
    catalog_path = PROJECT_ROOT / RAW_DATA_PATH
    prepared_data_path = PROJECT_ROOT / PREP_DATA_PATH

    sales_pred = pd.read_csv(input_path)
    sales_hist = pd.read_csv(os.path.join(prepared_data_path, "sales_prep.csv"))
    items = pd.read_csv(os.path.join(catalog_path, "items.csv"))
    categories  = pd.read_csv(os.path.join(catalog_path, "item_categories.csv"))
    shops = pd.read_csv(os.path.join(catalog_path, "shops.csv"))

    df = pd.merge(sales_pred, items, how="left", on="item_id")
    df = pd.merge(df, categories, how="left", on="item_category_id")
    df = pd.merge(df, shops, how="left", on ="shop_id")

    cols = [
        col 
        for col in sales_hist.columns 
        if col not in ["date_block_num", "item_category_id", "item_cnt_month"]
    ]

    latest_sales = sales_hist[
        sales_hist["date_block_num"] == sales_hist["date_block_num"].max()
    ][cols]
    # TODO: Aquí definir month

    df = pd.merge(
        df, 
        latest_sales, 
        how="left", 
        on=["shop_id", "item_id"]
    )

    return df[["ID", "shop_id", "item_id", "item_category_id", "avg_price"]]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create features required for inference.

    This function adds time-related features required by the model.
    The values are currently fixed and should match the training logic.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with inference features.
    """
    df = df.copy()

    #df["year"] = df["date"].dt.year
    df["month"] = 11
    df["date_block_num"] = 34

    return df


def main() -> None:
    """Run the batch inference pipeline."""
    args = parse_args()

    input_path = PROJECT_ROOT / args.input_path
    model_path = PROJECT_ROOT / args.model_path
    output_path = PROJECT_ROOT / args.output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading inference data...")
    df = load_inference_data(input_path)

    print(f"Rows loaded: {len(df)}")

    print("Creating features...")
    df = feature_engineering(df)

    feature_cols = [
        col for col in df.columns
        if col not in ["ID", "item_cnt_month"]
    ]

    X = df[feature_cols]

    print("Loading trained model...")
    model = joblib.load(model_path)

    print("Running predictions...")
    preds = model.predict(X)

    results = df.copy()
    results["item_cnt_month"] = preds

    print(f"Saving predictions to {output_path}")
    results[["ID", "item_cnt_month"]].to_csv(output_path, index=False)

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
