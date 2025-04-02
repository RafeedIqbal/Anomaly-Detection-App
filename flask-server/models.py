import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
import matplotlib
matplotlib.use('Agg')


def XGB_MT1R1(df, target='Toronto'):
    # Ensure target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV file")
    
    # Create train-test split (80% train, 20% test)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    # Drop non-numeric columns from features (e.g., DateTime)
    X_train = train_df.drop(columns=[target]).select_dtypes(include=['number'])
    X_test = test_df.drop(columns=[target]).select_dtypes(include=['number'])
    y_train = train_df[target]
    y_test = test_df[target]

    # Train XGBoost model with evaluation results recorded
    model = xgb.XGBRegressor(n_estimators=100, eval_metric="rmse", use_label_encoder=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    
    # Retrieve evaluation results for loss curve plotting
    evals_result = model.evals_result()
    train_rmse_list = evals_result['validation_0']['rmse']
    test_rmse_list = evals_result['validation_1']['rmse']

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics: RMSE and R² score
    train_loss = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_loss = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)

    # Generate training loss curve plot (training and test RMSE over epochs)
    plt.figure(figsize=(10,6))
    epochs = range(1, len(train_rmse_list) + 1)
    plt.plot(epochs, train_rmse_list, label="Train Loss")
    plt.plot(epochs, test_rmse_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training Loss Curve")
    plt.legend()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    loss_curve_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()

    # Generate performance plot (actual vs. predicted on test set)
    plt.figure(figsize=(10,6))
    plt.plot(test_df.index, y_test, label="Actual")
    plt.plot(test_df.index, y_test_pred, label="Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Target Value")
    plt.title("Test Performance")
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    performance_plot_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()

    # Create output dataframe for anomaly detection:
    # Use the DateTime column from test_df if it exists; otherwise, generate one.
    if 'DateTime' in test_df.columns:
        date_column = test_df['DateTime']
    else:
        date_column = pd.Series(test_df.index, name="DateTime")
    
    output_df = pd.DataFrame({
        "DateTime": date_column,
        target: y_test,
        "pred": y_test_pred,
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)

    # Return all results including the CSV output for anomaly detection
    result = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "loss_curve": loss_curve_base64,
        "performance_plot": performance_plot_base64,
        "anomaly_csv": csv_output
    }
    return result


def LSTM_FINAL(df, target='Toronto'):
    # Check that the target column exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the data")
    
    # Use only the target column for prediction (preserve DateTime if available)
    data = df[[target]].copy()
    
    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Split data: 80% train, 20% test
    train_size = int(len(data_scaled) * 0.8)
    if train_size < 2 or (len(data_scaled) - train_size) < 2:
        raise ValueError("Not enough data available for a proper train/test split.")
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]
    
    # Create dataset without a look-back (predict next value)
    def create_dataset(dataset):
        X, Y = [], []
        for i in range(len(dataset) - 1):
            X.append(dataset[i, 0])
            Y.append(dataset[i + 1, 0])
        return np.array(X), np.array(Y)
    
    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Not enough samples in training or testing set after dataset creation.")
    
    # Reshape input to be [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model and capture the training loss history
    history = model.fit(X_train, Y_train, epochs=50, batch_size=200, verbose=0)
    
    # Make predictions on test set
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions and true values back to original scale
    test_predict_inv = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    # Calculate performance metrics (RMSE and R² score)
    test_rmse = np.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))
    test_r2 = r2_score(Y_test_inv, test_predict_inv)
    
    # Create training loss curve plot (MSE converted to RMSE per epoch)
    plt.figure(figsize=(10,6))
    train_loss_rmse = np.sqrt(np.array(history.history['loss']))
    plt.plot(train_loss_rmse, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training Loss Curve')
    plt.legend()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    train_loss_curve = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()
    
    # Create test performance plot (actual vs. predicted)
    plt.figure(figsize=(10,6))
    plt.plot(Y_test_inv, label='Actual')
    plt.plot(test_predict_inv, label='Predicted')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Target Value')
    plt.title('Test Performance')
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    test_predictions_plot = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()
    
    # Create output CSV for anomaly detection:
    # If a DateTime column exists in the original df, use it for the test set.
    if 'DateTime' in df.columns:
        date_series = df['DateTime'].reset_index(drop=True)
        test_dates = date_series[train_size:].reset_index(drop=True)
        # Adjust dates to align with predictions (since create_dataset shifts by one)
        test_dates = test_dates.iloc[1:].reset_index(drop=True)
    else:
        test_dates = pd.Series(range(len(Y_test_inv)), name="DateTime")
    
    output_df = pd.DataFrame({
        "DateTime": test_dates,
        target: Y_test_inv.flatten(),
        "pred": test_predict_inv.flatten()
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)
    
    # Return all results including the CSV output for anomaly detection
    result = {
        "train_loss": None,  # (Training loss for LSTM was not computed separately)
        "test_loss": test_rmse,
        "train_accuracy": None,  # R² on training data not computed here
        "test_accuracy": test_r2,
        "train_loss_curve": train_loss_curve,
        "test_predictions_plot": test_predictions_plot,
        "anomaly_csv": csv_output
    }
    return result
