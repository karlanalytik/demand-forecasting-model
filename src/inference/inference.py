# src/inference.py
"""Model inference module.

This module loads a trained model and prepared reference data, performs
feature engineering for inference, generates predictions, and saves the
results to disk.
"""

import logging
from pathlib import Path

import pandas as pd

# =========================
# Logging configuration
# =========================
logger = logging.getLogger(__name__)


# =========================
# Functions
# =========================
def load_inference_data(
    input_path: Path,
    raw_data_dir: Path,
    prep_data_dir: Path,
) -> pd.DataFrame:
    """Load and assemble inference dataset.

    This function loads the inference input data and enriches it using
    historical prepared data and reference catalogs (items, categories,
    and shops).

    Parameters
    ----------
    input_path : Path
        Path to inference input CSV.
    raw_data_dir : Path
        Directory containing raw catalog data.
    prep_data_dir : Path
        Directory containing prepared training data.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with historical features.
    """
    logger.info("Loading inference input data")

    sales_pred = pd.read_csv(input_path)
    sales_hist = pd.read_csv(prep_data_dir / "sales_prep.csv")
    items = pd.read_csv(raw_data_dir / "items.csv")
    categories = pd.read_csv(raw_data_dir / "item_categories.csv")
    shops = pd.read_csv(raw_data_dir / "shops.csv")

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

    df = pd.merge(
        df,
        latest_sales,
        how="left",
        on=["shop_id", "item_id"]
    )

    logger.info("Inference dataset assembled: %s rows", len(df))
    return df[["ID", "shop_id", "item_id", "item_category_id", "avg_price"]]


def feature_engineering(
    df: pd.DataFrame,
    month: int,
    date_block_num: int,
) -> pd.DataFrame:
    """Create features required for inference.
    
    This function adds time-related features required by the model.
    The values are currently fixed and should match the training logic.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    month : int
        Target month for prediction.
    date_block_num : int
        Corresponding date block number.

    Returns
    -------
    pd.DataFrame
        DataFrame with inference features.
    """
    logger.info("Running feature engineering for inference")

    df = df.copy()

    #df["year"] = df["date"].dt.year
    df["month"] = month
    df["date_block_num"] = date_block_num

    return df
