# forecasting_utils.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from tensorflow.keras.models import load_model
import torch

# Nếu dùng mô hình PyTorch (TiDE)
def load_tide_model(model_path):
    from tide_model import TiDEModel  # Đảm bảo bạn có file định nghĩa TiDEModel
    model = TiDEModel.load(model_path)
    model.eval()
    return model

def run_forecast(model_name: str, df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Thêm các đặc trưng
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    target_cols = ['min_temp', 'mean_temp', 'max_temp']
    for col in target_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_rolling3_hist'] = df[col].shift(1).rolling(window=3).mean()

    df.dropna(inplace=True)
    last_known = df.iloc[[-1]].copy()

    feature_list_path = f"models/{model_name}/feature_list.pkl"
    if not os.path.exists(feature_list_path):
        raise ValueError(f"Không tìm thấy feature_list.pkl cho mô hình {model_name}")
    features = joblib.load(feature_list_path)

    future_predictions = []
    for i in range(7):
        current_date = last_known.index[0]
        next_date = current_date + timedelta(days=1)
        predicted_values = {}

        for target in target_cols:
            model_path = f"models/{model_name}/{target}_model.pkl"
            x_scaler_path = f"models/{model_name}/{target}_X_scaler.pkl"
            y_scaler_path = f"models/{model_name}/{target}_y_scaler.pkl"

            if not all(map(os.path.exists, [model_path, x_scaler_path, y_scaler_path])):
                continue  # Bỏ qua nếu thiếu model hoặc scaler

            model = joblib.load(model_path)
            x_scaler = joblib.load(x_scaler_path)
            y_scaler = joblib.load(y_scaler_path)

            X_input = last_known[features].values
            X_scaled = x_scaler.transform(X_input)
            y_pred_scaled = model.predict(X_scaled)
            y_pred_K = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
            predicted_values[target] = y_pred_K - 273.15

        future_predictions.append({'date': next_date, **predicted_values})

        new_row = last_known.copy()
        new_row.index = [next_date]
        new_row['day_of_year'] = next_date.dayofyear
        new_row['month'] = next_date.month
        new_row['day_sin'] = np.sin(2 * np.pi * new_row['day_of_year'] / 365)
        new_row['day_cos'] = np.cos(2 * np.pi * new_row['day_of_year'] / 365)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)

        for target in target_cols:
            new_row[f'{target}_lag1'] = predicted_values[target]

        for target in target_cols:
            val1 = last_known[f'{target}_lag1'].iloc[0]
            val2 = last_known[target].iloc[0]
            val3 = predicted_values[target]
            new_row[f'{target}_rolling3_hist'] = (val1 + val2 + val3) / 3

        last_known = new_row

    return pd.DataFrame(future_predictions).set_index('date')
