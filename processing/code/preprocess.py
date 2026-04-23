"""SageMaker Processing entrypoint for data preprocessing."""

import argparse
import logging
import os

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
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
            f"Faltan archivos requeridos en {path}: {missing_files}"
        )

    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    categories = pd.read_csv(os.path.join(path, "item_categories.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))

    df = pd.merge(sales, items, how="left", on="item_id")
    df = pd.merge(df, categories, how="left", on="item_category_id")
    df = pd.merge(df, shops, how="left", on="shop_id")

    logger.info("Raw data loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning")

    d = df.copy()

    d["date"] = pd.to_datetime(
        d["date"],
        format="%d.%m.%Y",
        errors="coerce"
    )

    null_dates = d["date"].isna().sum()
    if null_dates > 0:
        logger.warning(
            "Se encontraron %d fechas inválidas; serán eliminadas",
            null_dates
        )
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
            as_index=False
        )
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            avg_price=("item_price", "mean")
        )
        .sort_values(by="date_block_num")
        .reset_index(drop=True)
    )

    logger.info(
        "Feature engineering completed: %d rows, %d columns",
        monthly.shape[0],
        monthly.shape[1]
    )
    return monthly


def split_data(df: pd.DataFrame):
    """Split data by date_block_num for time-series workflow."""
    max_block = df["date_block_num"].max()

    train_df = df[df["date_block_num"] < max_block - 1].copy()
    validation_df = df[df["date_block_num"] == max_block - 1].copy()
    test_df = df[df["date_block_num"] == max_block].copy()

    logger.info(
        "Split completed | train: %d rows | validation: %d rows | test: %d rows",
        len(train_df), len(validation_df), len(test_df)
    )

    return train_df, validation_df, test_df


def save_split(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    logger.info("Saved file to %s", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Directorio local de entrada montado por SageMaker Processing"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    train_dir = "/opt/ml/processing/output/train"
    validation_dir = "/opt/ml/processing/output/validation"
    test_dir = "/opt/ml/processing/output/test"

    logger.info("Starting preprocessing job")
    logger.info("Input dir: %s", input_dir)

    raw_df = load_raw_data(input_dir)
    clean_df = clean_data(raw_df)
    features_df = feature_engineering(clean_df)

    train_df, validation_df, test_df = split_data(features_df)

    save_split(train_df, train_dir, "train.csv")
    save_split(validation_df, validation_dir, "validation.csv")
    save_split(test_df, test_dir, "test.csv")

    logger.info("Preprocessing job completed successfully")


if __name__ == "__main__":
    main()
