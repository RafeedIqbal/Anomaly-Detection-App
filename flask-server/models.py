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


import urllib.request
api_url = 'https://raw.githubusercontent.com/tanmayyb/ele70_bv03/refs/heads/main/api/datasets.py'
exec(urllib.request.urlopen(api_url).read())

def load_dataset_from_file(file):
   # returns target_name, dataset, dt 
   return DatasetPreprocessor.load_dataset_from_file(file)

def train_xgboost(target_name, dataset, dt, plot=False):

    (X_train, X_test, y_train, y_test), (train_idx, test_idx) = create_train_test_split(dataset, target=target_name, dt=dt)
    y_test_numpy = y_test.to_numpy()

    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import pandas as pd


    booster = xgb.XGBRegressor(n_estimators=1000)
    booster.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        # eval_metric='rmse',
        verbose=False,
    )
    # pred = booster.predict(X_test)
    y_train_pred = booster.predict(X_train)
    y_test_pred = booster.predict(X_test)

    
    evals_result = booster.evals_result()
    train_loss_list = evals_result['validation_0']['rmse']  # training loss
    test_loss_list = evals_result['validation_1']['rmse']  # test loss

    train_loss_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_loss_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # model output dataframe
    # to be used for anomaly detection
    output_df = pd.concat([dt, y_test, pd.Series(y_test_pred, index=y_test.index, name='pred')],axis=1).dropna()
   

    mse = mean_squared_error(y_true=y_test,
                    y_pred=y_test_pred)
    mae = mean_absolute_error(y_true=y_test,
                    y_pred=y_test_pred)
    def mean_absolute_percentage_error(y_true, y_pred):
        """Calculates MAPE given y_true and y_pred"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    
    train_accuracy = mean_absolute_percentage_error(y_train, y_train_pred)
    test_accuracy = mean_absolute_percentage_error(y_test, y_test_pred)

    # Generate LOSS curve plots
    plt.figure(figsize=(10,6))
    epochs = range(1, len(train_loss_list) + 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training Loss Curve")
    plt.legend()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    loss_curve_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()

    # Generate PERF plots (actual vs. predicted on test set)
    plt.figure(figsize=(10,6))
    plt.plot(output_df.index, y_test, label="Actual")
    plt.plot(output_df.index, y_test_pred, label="Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Target Value")
    plt.title("Test Performance")
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    performance_plot_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()
        
    return {
        "train_loss": train_loss_rmse, # train_loss_rmse
        "test_loss": test_loss_rmse, # test_loss_rmse
        "train_accuracy": train_accuracy, # train_mean_absolute_percentage_error
        "test_accuracy": test_accuracy, # test_mean_absolute_percentage_error
        "loss_curve": loss_curve_base64, # train_test_loss_curves
        "performance_plot": performance_plot_base64, # performance_plot
        "anomaly_csv": output_df.to_csv(index=False) # forecast_model_output
    }


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
