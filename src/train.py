"""
train.py

Training script for Demand Forecasting model.
"""


from sklearn.metrics import mean_squared_error
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import joblib
import os


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================
# Arguments
# =========================
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description = 'Train demand forecasting model'
        )

    parser.add_argument(
        '--data_path',
        type = str,
        default = 'data/prep/sales_prep.csv',
        help = 'Path to processed monthly dataset'
    )

    parser.add_argument(
        '--target',
        type = str,
        default = 'item_cnt_month',
        help = 'Target variable name'
    )

    parser.add_argument(
        '--model_path',
        type = str,
        default = 'artifacts/xgboost_model.joblib',
        help = 'Path to save trained model'
    )

    parser.add_argument(
        '--pred_path',
        type = str,
        default = 'predictions/pred.csv',
        help = 'Path to save predictions'
    )

    return parser.parse_args()


# =========================
# Load data
# =========================
def load_data(path):
    print(f'Loading data from {path}')
    df = pd.read_csv(os.path.join(PROJECT_ROOT, path))
    return df


# =====================================
# Train / validation split (time-based)
# =====================================
def train_val_split(data, target, time_col = 'date'):
    """
    Assumes data is already sorted by time
    """
    split_date = data[time_col].quantile(0.8)

    train = data[data[time_col] <= split_date]
    val = data[data[time_col] > split_date]

    X_train = train.drop(columns = [target])
    y_train = train[target]

    X_val = val.drop(columns = [target])
    y_val = val[target]

    return X_train, X_val, y_train, y_val


# =========================
# Model training
# =========================
def train_model(X_train, y_train, X_val, y_val):

    model = xgb.XGBRegressor(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 8,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'reg:squarederror',
        random_state = 55
    )

    model.fit(
        X_train,
        y_train,
        eval_set = [(X_val, y_val)],
        verbose = 50
    )

    return model


# =========================
# Evaluation
# =========================
def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f'Validation RMSE: {rmse:.4f}')
    return preds, rmse


# =========================
# Save artifacts
# =========================
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    joblib.dump(model, path)
    print(f'Model saved at {path}')


# =========================
# Main
# =========================
def main():

    args = parse_args()

    print('Loading prepared data...')
    data = load_data(args.data_path)

    time_col = 'date_block_num'
    target = args.target

    data = data.sort_values(time_col).reset_index(drop = True)

    print('Training model...')
    X_train, X_val, y_train, y_val = train_val_split(
        data,
        target = target,
        time_col = time_col
    )

    model = train_model(X_train, y_train, X_val, y_val)

    print('Saving model...')
    save_model(model, args.model_path)

    print("Training completed successfully.")


if __name__ == '__main__':
    main()
