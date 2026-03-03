# src/preprocessing/test/test_prep.py
import pandas as pd

from preprocessing.prep import (
    load_raw_data,
    clean_data,
    feature_engineering,
    save_prepared_data,
)


def test_load_raw_data_carga_y_mergea_csvs(tmp_path):
    # Archivos mínimos requeridos
    sales = pd.DataFrame(
        {
            "date": ["01.01.2013", "02.01.2013"],
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 11],
            "item_price": [100.0, 200.0],
            "item_cnt_day": [1, 2],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [10, 11],
            "item_name": ["A", "B"],
            "item_category_id": [5, 6],
        }
    )
    categories = pd.DataFrame(
        {"item_category_id": [5, 6], "item_category_name": ["cat5", "cat6"]}
    )
    shops = pd.DataFrame({"shop_id": [1], "shop_name": ["shop1"]})

    sales.to_csv(tmp_path / "sales_train.csv", index=False)
    items.to_csv(tmp_path / "items.csv", index=False)
    categories.to_csv(tmp_path / "item_categories.csv", index=False)
    shops.to_csv(tmp_path / "shops.csv", index=False)

    df = load_raw_data(str(tmp_path))

    assert df.shape[0] == 2
    assert "item_name" in df.columns
    assert "item_category_name" in df.columns
    assert "shop_name" in df.columns


def test_clean_data_convierte_date_a_datetime_y_coerce_invalidos():
    df = pd.DataFrame(
        {
            "date": ["01.01.2013", "mal-formato"],
            "shop_id": [1, 1],
            "item_id": [10, 10],
        }
    )

    out = clean_data(df)

    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].isna().sum() == 1  # "mal-formato" -> NaT


def test_clean_data_elimina_duplicados():
    df = pd.DataFrame(
        {
            "date": ["01.01.2013", "01.01.2013"],
            "shop_id": [1, 1],
            "item_id": [10, 10],
        }
    )

    out = clean_data(df)

    assert out.shape[0] == 1


def test_clean_data_resetea_indice():
    df = pd.DataFrame(
        {
            "date": ["01.01.2013", "02.01.2013"],
            "shop_id": [1, 1],
            "item_id": [10, 11],
        }
    )

    out = clean_data(df)

    assert list(out.index) == list(range(len(out)))  # 0..n-1


def test_feature_engineering_agrega_por_mes_y_calcula_sum_y_mean():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["01.01.2013", "02.01.2013"], format="%d.%m.%Y"),
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_category_id": [5, 5],
            "item_cnt_day": [2, 3],
            "item_price": [100.0, 200.0],
        }
    )

    monthly = feature_engineering(df)

    assert monthly.shape[0] == 1
    assert monthly.loc[0, "item_cnt_month"] == 5   # 2 + 3
    assert monthly.loc[0, "avg_price"] == 150.0    # mean(100, 200)
    assert monthly.loc[0, "month"] == 1


def test_save_prepared_data_guarda_archivo_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    save_prepared_data(df, str(tmp_path))

    out_file = tmp_path / "sales_prep.csv"
    assert out_file.exists()

    loaded = pd.read_csv(out_file)
    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["a", "b"]
