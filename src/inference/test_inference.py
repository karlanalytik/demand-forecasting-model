# src/inference/test/test_inference.py
from pathlib import Path

import pandas as pd

from inference import load_inference_data, feature_engineering


def _write_min_catalogs(raw_dir: Path):
    """Helper: escribe items, categorías y shops."""
    items = pd.DataFrame(
        {
            "item_id": [10, 11, 99],
            "item_name": ["A", "B", "Z"],
            "item_category_id": [5, 6, 9],
        }
    )
    categories = pd.DataFrame(
        {"item_category_id": [5, 6, 9], "item_category_name": ["cat5", "cat6", "cat9"]}
    )
    shops = pd.DataFrame({"shop_id": [1, 2], "shop_name": ["shop1", "shop2"]})

    items.to_csv(raw_dir / "items.csv", index=False)
    categories.to_csv(raw_dir / "item_categories.csv", index=False)
    shops.to_csv(raw_dir / "shops.csv", index=False)


def _write_sales_hist(prep_dir: Path):
    """Helper: escribe sales_prep.csv con dos date_block_num para probar 'latest'."""
    sales_hist = pd.DataFrame(
        {
            "month": [1, 2, 2],
            "date_block_num": [0, 1, 1],
            "shop_id": [1, 1, 2],
            "item_id": [10, 10, 11],
            "item_category_id": [5, 5, 6],
            "item_cnt_month": [5, 7, 3],
            "avg_price": [100.0, 120.0, 200.0],
        }
    )
    sales_hist.to_csv(prep_dir / "sales_prep.csv", index=False)


def test_load_inference_data_regresa_columnas_y_filas(tmp_path):
    # input pred
    sales_pred = pd.DataFrame(
        {"ID": [0, 1], "shop_id": [1, 1], "item_id": [10, 11]}
    )
    input_path = tmp_path / "test.csv"
    sales_pred.to_csv(input_path, index=False)

    raw_dir = tmp_path / "raw"
    prep_dir = tmp_path / "prep"
    raw_dir.mkdir()
    prep_dir.mkdir()

    _write_min_catalogs(raw_dir)
    _write_sales_hist(prep_dir)

    out = load_inference_data(
        input_path=Path(input_path),
        raw_data_dir=Path(raw_dir),
        prep_data_dir=Path(prep_dir),
    )

    assert out.shape[0] == 2
    assert list(out.columns) == ["ID", "shop_id", "item_id", "item_category_id", "avg_price"]


def test_load_inference_data_usa_avg_price_del_ultimo_date_block(tmp_path):
    # Predicción para shop_id=1, item_id=10 (tiene histórico en date_block 0 y 1)
    sales_pred = pd.DataFrame({"ID": [0], "shop_id": [1], "item_id": [10]})
    input_path = tmp_path / "test.csv"
    sales_pred.to_csv(input_path, index=False)

    raw_dir = tmp_path / "raw"
    prep_dir = tmp_path / "prep"
    raw_dir.mkdir()
    prep_dir.mkdir()

    _write_min_catalogs(raw_dir)
    _write_sales_hist(prep_dir)

    out = load_inference_data(Path(input_path), Path(raw_dir), Path(prep_dir))

    # Debe traer el avg_price del último date_block_num (=1): 120.0
    assert out.loc[0, "avg_price"] == 120.0


def test_load_inference_data_deja_nan_si_no_hay_historico(tmp_path):
    # Predicción para un item/shop que NO aparece en el último bloque histórico
    sales_pred = pd.DataFrame({"ID": [0], "shop_id": [2], "item_id": [99]})
    input_path = tmp_path / "test.csv"
    sales_pred.to_csv(input_path, index=False)

    raw_dir = tmp_path / "raw"
    prep_dir = tmp_path / "prep"
    raw_dir.mkdir()
    prep_dir.mkdir()

    _write_min_catalogs(raw_dir)
    _write_sales_hist(prep_dir)

    out = load_inference_data(Path(input_path), Path(raw_dir), Path(prep_dir))

    assert pd.isna(out.loc[0, "avg_price"])


def test_feature_engineering_agrega_month_y_date_block_num():
    df = pd.DataFrame(
        {
            "ID": [0],
            "shop_id": [1],
            "item_id": [10],
            "item_category_id": [5],
            "avg_price": [120.0],
        }
    )

    out = feature_engineering(df, month=3, date_block_num=34)

    assert out.loc[0, "month"] == 3
    assert out.loc[0, "date_block_num"] == 34


def test_feature_engineering_no_modifica_df_original():
    df = pd.DataFrame(
        {
            "ID": [0],
            "shop_id": [1],
            "item_id": [10],
            "item_category_id": [5],
            "avg_price": [120.0],
        }
    )

    _ = feature_engineering(df, month=3, date_block_num=34)

    assert "month" not in df.columns
    assert "date_block_num" not in df.columns
