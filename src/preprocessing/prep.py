"""SageMaker preprocessing script for the demand forecasting pipeline."""

import logging
import os

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TARGET_COL = "item_cnt_month"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load and merge raw data from multiple CSV files."""
    logger.info("Loading raw data from %s", path)

    required_files = [
        "sales_train.csv",
        "items.csv",
        "item_categories.csv",
        "shops.csv",
    ]

    missing_files = [
        file_name for file_name in required_files
        if not os.path.exists(os.path.join(path, file_name))
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {path}: {missing_files}"
        )

    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    categories = pd.read_csv(os.path.join(path, "item_categories.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))

    d = pd.merge(sales, items, how="left", on="item_id")
    d = pd.merge(d, categories, how="left", on="item_category_id")
    d = pd.merge(d, shops, how="left", on="shop_id")

    logger.info("Raw data loaded: %d rows, %d columns", d.shape[0], d.shape[1])
    return d


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning."""
    logger.info("Starting data cleaning")

    d = df.copy()

    d["date"] = pd.to_datetime(
        d["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    invalid_dates = d["date"].isna().sum()
    if invalid_dates > 0:
        logger.warning("Dropping %d rows with invalid dates", invalid_dates)
        d = d.dropna(subset=["date"])

    before = len(d)
    d = d.drop_duplicates()
    after = len(d)

    if before != after:
        logger.info("Removed %d duplicate rows", before - after)

    d = d.reset_index(drop=True)

    logger.info("Data cleaning completed: %d rows", len(d))
    return d


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly aggregated features."""
    logger.info("Starting feature engineering")

    d = df.copy()

    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month

    monthly = (
        d.groupby(
            [
                "month",
                "date_block_num",
                "shop_id",
                "item_id",
                "item_category_id",
            ],
            as_index=False,
        )
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            avg_price=("item_price", "mean"),
        )
        .sort_values(by="date_block_num")
        .reset_index(drop=True)
    )

    logger.info(
        "Feature engineering completed: %d rows, %d columns",
        monthly.shape[0],
        monthly.shape[1],
    )
    return monthly


def split_data(df: pd.DataFrame):
    """Split data into train, validation, and test by time."""
    logger.info("Splitting dataset into train, validation, and test")

    max_block = df["date_block_num"].max()

    train_df = df[df["date_block_num"] < max_block - 1].copy()
    validation_df = df[df["date_block_num"] == max_block - 1].copy()
    test_df = df[df["date_block_num"] == max_block].copy()

    logger.info(
        "Split sizes | train: %d | validation: %d | test: %d",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    return train_df, validation_df, test_df


def build_inference_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create inference dataset without target column."""
    d = df.copy()

    if target_col in d.columns:
        d = d.drop(columns=[target_col])

    return d


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_path: str,
    target_col: str = TARGET_COL,
) -> None:
    """Save SageMaker datasets to output paths."""
    train_path = os.path.join(base_path, "train")
    validation_path = os.path.join(base_path, "validation")
    test_path = os.path.join(base_path, "test")
    test_inference_path = os.path.join(base_path, "test_inference")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(test_inference_path, exist_ok=True)

    train_file = os.path.join(train_path, "train.csv")
    validation_file = os.path.join(validation_path, "validation.csv")
    test_file = os.path.join(test_path, "test.csv")
    test_inference_file = os.path.join(
        test_inference_path,
        "test_inference.csv",
    )

    test_inference_df = build_inference_dataset(test_df, target_col)

    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)
    test_df.to_csv(test_file, index=False)
    test_inference_df.to_csv(test_inference_file, index=False)

    logger.info("Saved train dataset to %s", train_file)
    logger.info("Saved validation dataset to %s", validation_file)
    logger.info("Saved test dataset to %s", test_file)
    logger.info("Saved test inference dataset to %s", test_inference_file)


def main() -> None:
    """Main preprocessing entrypoint for SageMaker Processing."""
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"

    logger.info("Starting preprocessing job")
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    raw_df = load_raw_data(input_dir)
    clean_df = clean_data(raw_df)
    features_df = feature_engineering(clean_df)

    train_df, validation_df, test_df = split_data(features_df)
    save_splits(train_df, validation_df, test_df, output_dir)

    logger.info("Preprocessing job completed successfully")


if __name__ == "__main__":
    main()