import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
import json


def XGB_MT1R1(df):
    if len(df.columns) < 2:
        raise ValueError("Input CSV must have at least two columns (e.g., DateTime and Target).")
    target = df.columns[1]
    first_col_name = df.columns[0]

    try:
        df[first_col_name] = pd.to_datetime(df[first_col_name])
        datetime_available = True
    except Exception:
        datetime_available = False

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    feature_cols = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found besides the target.")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target]
    y_test = test_df[target]

    model = xgb.XGBRegressor(n_estimators=1000, eval_metric="rmse", use_label_encoder=False)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    evals_result = model.evals_result()
    actual_epochs = len(evals_result['validation_0']['rmse'])
    train_rmse_list = evals_result['validation_0']['rmse']
    test_rmse_list = evals_result['validation_1']['rmse']

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_loss = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_loss = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)

    loss_curve = json.dumps([
        {"x": list(range(1, actual_epochs + 1)), "y": train_rmse_list, "type": "scatter", "mode": "lines+markers", "name": "Train RMSE"},
        {"x": list(range(1, actual_epochs + 1)), "y": test_rmse_list, "type": "scatter", "mode": "lines+markers", "name": "Test RMSE"}
    ])

    if datetime_available:
        x_axis = df[first_col_name].iloc[split_index:].dt.strftime('%Y-%m-%d %H:%M').tolist()
    else:
        x_axis = list(test_df.index)

    performance_plot = json.dumps([
        {"x": x_axis, "y": y_test.tolist(), "type": "scatter", "mode": "lines+markers", "name": "Actual"},
        {"x": x_axis, "y": y_test_pred.tolist(), "type": "scatter", "mode": "lines+markers", "name": "Predicted"}
    ])

    output_df = pd.DataFrame({
        first_col_name if datetime_available else "DateTime": x_axis,
        target: y_test.tolist(),
        "pred": y_test_pred.tolist()
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "loss_curve": loss_curve,
        "performance_plot": performance_plot,
        "anomaly_csv": csv_output,
        "target_column_used": target
    }


def LSTM_FINAL(df, look_back=6):
    if len(df.columns) < 2:
        raise ValueError("Input CSV must have at least two columns (e.g., DateTime and Target).")
    target = df.columns[1]
    first_col_name = df.columns[0]

    datetime_available = False
    try:
        df[first_col_name] = pd.to_datetime(df[first_col_name])
        datetime_available = True
    except Exception:
        pass

    data = df[[target]].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, Y_train, epochs=40, batch_size=64, validation_data=(X_test, Y_test), verbose=0)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict_inv = scaler.inverse_transform(train_predict)
    Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
    test_predict_inv = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    train_rmse = np.sqrt(mean_squared_error(Y_train_inv, train_predict_inv))
    test_rmse = np.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))
    train_r2 = r2_score(Y_train_inv, train_predict_inv)
    test_r2 = r2_score(Y_test_inv, test_predict_inv)

    loss_curve = json.dumps([
        {"x": list(range(1, len(history.history['loss']) + 1)), "y": list(np.sqrt(history.history['loss'])), "type": "scatter", "mode": "lines+markers", "name": "Train RMSE"},
        {"x": list(range(1, len(history.history['val_loss']) + 1)), "y": list(np.sqrt(history.history['val_loss'])), "type": "scatter", "mode": "lines+markers", "name": "Validation RMSE"}
    ])

    date_index_start = train_size + look_back
    if datetime_available and date_index_start < len(df):
        test_dates = df[first_col_name].iloc[date_index_start:].dt.strftime('%Y-%m-%d %H:%M').tolist()
        x_axis = test_dates[:len(Y_test_inv)]
    else:
        x_axis = list(range(len(Y_test_inv)))

    performance_plot = json.dumps([
        {"x": x_axis, "y": Y_test_inv.flatten().tolist(), "type": "scatter", "mode": "lines+markers", "name": "Actual"},
        {"x": x_axis, "y": test_predict_inv.flatten().tolist(), "type": "scatter", "mode": "lines+markers", "name": "Predicted"}
    ])

    output_df = pd.DataFrame({
        first_col_name if datetime_available else "DateTime": x_axis,
        target: Y_test_inv.flatten(),
        "pred": test_predict_inv.flatten()
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)

    return {
        "train_loss": train_rmse,
        "test_loss": test_rmse,
        "train_accuracy": train_r2,
        "test_accuracy": test_r2,
        "loss_curve": loss_curve,
        "performance_plot": performance_plot,
        "anomaly_csv": csv_output,
        "target_column_used": target
    }
