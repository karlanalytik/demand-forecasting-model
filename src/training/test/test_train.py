# src/training/test/test_train.py
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import training.train as t


def test_load_data_lee_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)

    out = t.load_data(str(p))

    assert out.shape == (2, 2)
    assert list(out.columns) == ["a", "b"]


def test_train_val_split_separa_y_remueve_target():
    data = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "f1": [10, 11, 12, 13, 14],
            "f2": [0, 1, 0, 1, 0],
            "y": [0, 1, 0, 1, 0],
        }
    )

    X_train, X_val, y_train, y_val = t.train_val_split(data, target="y", time_col="time")

    assert "y" not in X_train.columns
    assert "y" not in X_val.columns
    assert len(X_train) + len(X_val) == len(data)
    assert len(y_train) + len(y_val) == len(data)


def test_train_model_entrena_xgboost_real_rapido(monkeypatch):
    """
    Probamos train_model() entrenando un XGBRegressor REAL,
    pero lo hacemos rápido: pocos árboles + sin verbose.
    """

    class SmallXGB(xgb.XGBRegressor):
        def __init__(self, **kwargs):
            super().__init__(
                n_estimators=20,
                learning_rate=0.1,
                max_depth=3,
                subsample=1.0,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                random_state=55,
            )

        # Para evitar spam por verbose=50 desde train_model()
        def fit(self, X, y, eval_set=None, verbose=None, **fit_kwargs):
            return super().fit(X, y, eval_set=eval_set, verbose=False, **fit_kwargs)

    # Hacemos que t.train_model use SmallXGB (pero sigue siendo XGBoost real)
    monkeypatch.setattr(t.xgb, "XGBRegressor", SmallXGB)

    # Dataset mini (suficiente para que entrene y prediga)
    X_train = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [0, 1, 0, 1, 0]})
    y_train = pd.Series([1.0, 2.0, 1.5, 2.5, 2.0])

    X_val = pd.DataFrame({"f1": [6, 7], "f2": [1, 0]})
    y_val = pd.Series([3.0, 3.5])

    model = t.train_model(X_train, y_train, X_val, y_val)

    # Checks simples de “sí entrenó y sirve”
    preds = model.predict(X_val)
    assert isinstance(model, xgb.XGBRegressor)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)


def test_evaluate_regresa_preds_y_rmse_con_modelo_real(monkeypatch):
    """
    Entrenamos un modelo REAL pequeñito y evaluamos.
    """
    class SmallXGB(xgb.XGBRegressor):
        def __init__(self, **kwargs):
            super().__init__(
                n_estimators=10,
                learning_rate=0.2,
                max_depth=2,
                subsample=1.0,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                random_state=55,
            )

        def fit(self, X, y, eval_set=None, verbose=None, **fit_kwargs):
            return super().fit(X, y, eval_set=eval_set, verbose=False, **fit_kwargs)

    monkeypatch.setattr(t.xgb, "XGBRegressor", SmallXGB)

    X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [0, 1, 0, 1]})
    y_train = pd.Series([1.0, 2.0, 1.5, 2.5])

    X_val = pd.DataFrame({"f1": [5, 6], "f2": [0, 1]})
    y_val = pd.Series([2.8, 3.2])

    model = t.train_model(X_train, y_train, X_val, y_val)

    preds, rmse = t.evaluate(model, X_val, y_val)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert isinstance(rmse, float)
    assert rmse >= 0.0


def test_save_model_guarda_archivo(tmp_path):
    model = xgb.XGBRegressor(n_estimators=1, objective="reg:squarederror", random_state=55)

    out_path = tmp_path / "models" / "model.joblib"
    t.save_model(model, str(out_path))

    assert out_path.exists()
    assert os.path.getsize(out_path) > 0
