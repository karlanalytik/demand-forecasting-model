"""
prep.py

Script de preparación de datos.
Lee datos crudos desde data/raw, realiza limpieza y transformaciones básicas,
y guarda los datos preparados en data/prep.
"""

import pandas as pd
import numpy as np
import os


# =========================
# Paths
# =========================
RAW_DATA_PATH = 'data/raw'
PREP_DATA_PATH = 'data/prep'


# =========================
# Functions
# =========================
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from several CSV files and integrate a unique dataframe.
    """
    sales = pd.read_csv(os.path.join(path, 'sales_train.csv'))
    items = pd.read_csv(os.path.join(path, 'items.csv'))
    cats  = pd.read_csv(os.path.join(path, 'item_categories.csv'))
    shops = pd.read_csv(os.path.join(path, 'shops.csv'))

    df = pd.merge(sales, items, how = 'left', on = 'item_id')
    df = pd.merge(df, cats, how = 'left', on = 'item_category_id')
    df = pd.merge(df, shops, how = 'left', on = 'shop_id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning.
    """
    df = df.copy()

    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'], 
                                format = '%d.%m.%Y', 
                                errors = 'coerce')
    
    # Drop duplicates
    df.drop_duplicates(inplace = True)

    # Remove negative sales (common in this dataset)
    #if 'item_cnt_day' in df.columns:
    #    df = df[df['item_cnt_day'] >= 0]

    # Clean dataset index
    df.reset_index(drop = True, inplace = True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for training.
    """
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    monthly = (
        df.groupby([
            'month', 'date_block_num', 'shop_id', 'item_id', 'item_category_id'
            ], as_index = False)
            .agg(
                item_cnt_month = ('item_cnt_day', 'sum'),
                avg_price = ('item_price', 'mean'),
                )
                )
    
    monthly.sort_values(by = 'date_block_num', inplace = True)

    return monthly


def save_prepared_data(df: pd.DataFrame, path: str) -> None:
    """
    Save prepared data to CSV.
    """
    os.makedirs(path, exist_ok = True)
    output_path = os.path.join(path, 'sales_prep.csv')
    df.to_csv(output_path, index = False)


# =========================
# Main
# =========================
def main():
    print('Loading raw data...')
    df = load_raw_data(RAW_DATA_PATH)

    print('Cleaning data...')
    df = clean_data(df)

    print('Creating features...')
    df = feature_engineering(df)

    print('Saving prepared data...')
    save_prepared_data(df, PREP_DATA_PATH)

    print('Data preparation completed successfully.')


if __name__ == '__main__':
    main()
