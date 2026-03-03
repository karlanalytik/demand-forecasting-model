# src/preprocessing/prep.py
"""Data preparation utilities.

This module provides reusable functions for loading raw data, performing 
cleaning, generating features, and saving the prepared dataset.
"""

import logging
import os

import pandas as pd

# =========================
# Logging configuration
# =========================
logger = logging.getLogger(__name__)


# =========================
# Functions
# =========================
def load_raw_data(path: str) -> pd.DataFrame:
    """Load and merge raw data from multiple CSV files.

    Parameters
    ----------
    path : str
        Directory path containing the raw CSV files.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing sales, items, categories, and shops data.
    """
    logger.info("Loading raw data from %s", path)

    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    categories  = pd.read_csv(os.path.join(path, "item_categories.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))

    df = pd.merge(sales, items, how="left", on="item_id")
    df = pd.merge(df, categories, how="left", on="item_category_id")
    df = pd.merge(df, shops, how="left", on="shop_id")

    logger.info("Raw data loaded: %d rows", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning.

    This function converts date columns, removes duplicates, and
    resets the DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning")

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d.%m.%Y",
        errors="coerce"
    )

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    if before != after:
        logger.info("Removed %d duplicate rows", before - after)
    # Remove negative sales (common in this dataset)
    #if "item_cnt_day" in df.columns:
    #    df = df[df["item_cnt_day"] >= 0]
    df.reset_index(drop=True, inplace=True)

    logger.info("Data cleaning completed")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregated features for training.

    Daily sales are aggregated into monthly metrics per shop and item,
    and time-based features are created.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly aggregated features.
    """
    logger.info("Starting feature engineering")

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby(
            [
                "month",
                "date_block_num",
                "shop_id",
                "item_id",
                "item_category_id"
            ],
            as_index = False
        )
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            avg_price=("item_price", "mean")
            )
            .sort_values(by="date_block_num")
    )

    logger.info("Feature engineering completed")
    return monthly


def save_prepared_data(df: pd.DataFrame, path: str) -> None:
    """Save the prepared dataset to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame to be saved.
    path : str
        Output directory path.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, "sales_prep.csv")

    df.to_csv(output_path, index=False)
    logger.info("Prepared data saved to %s", output_path)
