import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Import for date formatting
import io
import base64
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Use specific Keras imports if needed, otherwise just keras.api is fine
# from tensorflow import keras # Or from keras.api import ... if using specific backend
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting


# Function modified: Refined plotting
def XGB_MT1R1(df):
    # --- Data Preparation (Same as before) ---
    if len(df.columns) < 2:
        raise ValueError("Input CSV must have at least two columns (e.g., DateTime and Target).")
    target = df.columns[1]
    first_col_name = df.columns[0]
    print(f"Inferred target column for XGBoost: '{target}'")

    # Ensure the first column is parsed as datetime for plotting
    try:
        # Attempt to convert, handling potential errors
        df[first_col_name] = pd.to_datetime(df[first_col_name])
        datetime_available = True
    except Exception as e:
        print(f"Warning: Could not parse first column ('{first_col_name}') as DateTime: {e}. Performance plot will use index.")
        datetime_available = False

    if target not in df.columns:
        raise ValueError(f"Inferred target column '{target}' somehow not found in DataFrame.")

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    # Prepare features (only numeric, excluding target)
    # Use all other numeric columns as features
    feature_cols = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols:
         # Handle case where only DateTime and Target exist (or other non-numerics)
         # Fallback: Maybe use lagged target? For now, raise error if no numeric features.
         raise ValueError("No numeric feature columns found besides the target. XGBoost requires numeric features.")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target]
    y_test = test_df[target]


    # --- Model Training (Same as before) ---
    model = xgb.XGBRegressor(n_estimators=1000, eval_metric="rmse", use_label_encoder=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False # Keep verbose False for cleaner logs in API
    )

    # Retrieve evaluation results
    evals_result = model.evals_result()
    # Adjust epochs based on early stopping if it occurred
    actual_epochs = len(evals_result['validation_0']['rmse'])
    train_rmse_list = evals_result['validation_0']['rmse']
    test_rmse_list = evals_result['validation_1']['rmse']


    # --- Predictions and Metrics (Same as before) ---
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_loss = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_loss = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)

    # --- Plotting ---

    # 1. Training Loss Curve (RMSE vs. Epochs)
    plt.figure(figsize=(10, 6))
    epochs = range(1, actual_epochs + 1)
    plt.plot(epochs, train_rmse_list, label="Train RMSE")
    plt.plot(epochs, test_rmse_list, label="Test RMSE")
    plt.xlabel("Boosting Round (Epoch)")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    buf1.seek(0)
    loss_curve_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()

    # 2. Performance Plot (Actual vs. Predicted over Time)
    plt.figure(figsize=(12, 6)) # Wider plot for time series
    # Use DateTime if available, otherwise index
    if datetime_available and first_col_name in test_df.columns:
        x_axis = test_df[first_col_name]
        x_label = "DateTime"
        # Format the x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
        plt.gcf().autofmt_xdate() # Rotate date labels
    else:
        x_axis = test_df.index
        x_label = "Test Sample Index"

    plt.plot(x_axis, y_test, label="Actual", marker='.', linestyle='-', markersize=4, alpha=0.8)
    plt.plot(x_axis, y_test_pred, label="Predicted", marker='.', linestyle='--', markersize=4, alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(f"{target} Value")
    plt.title(f"XGBoost: Actual vs. Predicted {target} (Test Set)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    performance_plot_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()


    # --- Output CSV Preparation (Same as before, but ensure DateTime column handling is consistent) ---
    if datetime_available and first_col_name in test_df.columns:
        date_column_for_output = test_df[first_col_name]
        output_col_name = first_col_name
    else:
        date_column_for_output = pd.Series(test_df.index, name="DateTime") # Fallback name
        output_col_name = "DateTime"

    output_df = pd.DataFrame({
        output_col_name: date_column_for_output,
        target: y_test.values, # Ensure using numpy array or list here
        "pred": y_test_pred,
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)


    # --- Return Results ---
    result = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "loss_curve": loss_curve_base64,         # Train/Test RMSE vs Epoch
        "performance_plot": performance_plot_base64, # Actual vs Pred over Time
        "anomaly_csv": csv_output,
        "target_column_used": target
    }
    return result


# Function modified: Refined plotting, added validation loss
def LSTM_FINAL(df, look_back=6):
    # --- Data Preparation ---
    if len(df.columns) < 2:
        raise ValueError("Input CSV must have at least two columns (e.g., DateTime and Target).")
    target = df.columns[1]
    first_col_name = df.columns[0]
    print(f"Inferred target column for LSTM: '{target}'")

    # Ensure the first column is parsed as datetime for plotting/output
    datetime_available = False
    if first_col_name in df.columns:
        try:
            df[first_col_name] = pd.to_datetime(df[first_col_name])
            datetime_available = True
        except Exception as e:
            print(f"Warning: Could not parse first column ('{first_col_name}') as DateTime: {e}. Performance plot/output will use index.")

    if target not in df.columns:
        raise ValueError(f"Inferred target column '{target}' somehow not found in DataFrame.")

    # Use only the target column for LSTM input
    data = df[[target]].copy()

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Split data
    train_size = int(len(data_scaled) * 0.8)
    if train_size < look_back + 1 or (len(data_scaled) - train_size) < look_back + 1:
        raise ValueError("Not enough data available for a proper train/test split with the given look_back.")
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # Create dataset with look-back structure
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        # Stop look_back steps before the end to have a corresponding Y
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Not enough samples in training or testing set after dataset creation.")

    # Reshape input for LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    # --- Model Building & Training ---
    model = Sequential()
    # Consider adding more units or layers if needed, but 25 is often a starting point
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1))) # Increased units slightly
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model, including validation data to plot validation loss
    # Use a portion of the training data for validation if test set is strictly for final eval,
    # or use the prepared X_test, Y_test for validation during training. Using X_test here.
    print(f"Training LSTM... X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    history = model.fit(
        X_train, Y_train,
        epochs=40,         # Consider making this configurable or adding early stopping
        batch_size=64,     # Smaller batch size might help convergence
        validation_data=(X_test, Y_test), # Provide validation data
        verbose=0          # Keep verbose 0 for API
    )


    # --- Predictions and Metrics ---
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predict_inv = scaler.inverse_transform(train_predict)
    Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
    test_predict_inv = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate metrics on inverse-transformed data
    train_rmse = np.sqrt(mean_squared_error(Y_train_inv, train_predict_inv))
    test_rmse = np.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))
    train_r2 = r2_score(Y_train_inv, train_predict_inv)
    test_r2 = r2_score(Y_test_inv, test_predict_inv)


    # --- Plotting ---

    # 1. Training & Validation Loss Curve (RMSE vs. Epochs)
    plt.figure(figsize=(10, 6))
    # Get loss history (convert MSE to RMSE)
    train_loss_rmse = np.sqrt(history.history['loss'])
    val_loss_rmse = np.sqrt(history.history['val_loss'])
    epochs = range(1, len(train_loss_rmse) + 1)

    plt.plot(epochs, train_loss_rmse, label='Train RMSE')
    plt.plot(epochs, val_loss_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('LSTM Training & Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    train_loss_curve_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()

    # 2. Test Performance Plot (Actual vs. Predicted over Time)
    plt.figure(figsize=(12, 6)) # Wider plot
    # Get the corresponding dates for the test set predictions (Y_test_inv)
    output_col_name = "DateTime" # Default name if parsing fails
    if datetime_available and first_col_name in df.columns:
        # Dates start after train_size AND after the look_back period used for the first prediction
        date_index_start = train_size + look_back
        if date_index_start < len(df):
             test_dates = df[first_col_name].iloc[date_index_start:].reset_index(drop=True)
             output_col_name = first_col_name
             # Ensure lengths match (Y_test_inv might be shorter if create_dataset loop stops early)
             if len(test_dates) > len(Y_test_inv):
                 test_dates = test_dates[:len(Y_test_inv)]
             elif len(test_dates) < len(Y_test_inv):
                  print(f"Warning: Fewer dates ({len(test_dates)}) than test predictions ({len(Y_test_inv)}). Truncating predictions for plot.")
                  Y_test_inv = Y_test_inv[:len(test_dates)]
                  test_predict_inv = test_predict_inv[:len(test_dates)]

             x_axis = test_dates
             x_label = "DateTime"
             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
             plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
             plt.gcf().autofmt_xdate() # Rotate date labels
        else:
             print("Warning: Cannot determine test dates due to index calculation. Using index for x-axis.")
             x_axis = np.arange(len(Y_test_inv)) # Use range based on prediction length
             x_label = "Test Sample Index"
             datetime_available = False # Force fallback for CSV output too
    else:
        x_axis = np.arange(len(Y_test_inv))
        x_label = "Test Sample Index"

    plt.plot(x_axis, Y_test_inv, label='Actual', marker='.', linestyle='-', markersize=4, alpha=0.8)
    plt.plot(x_axis, test_predict_inv, label='Predicted', marker='.', linestyle='--', markersize=4, alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(f"{target} Value")
    plt.title(f'LSTM: Actual vs. Predicted {target} (Test Set)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    test_predictions_plot_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()


    # --- Output CSV Preparation ---
    # Reuse the date logic from plotting if available
    if datetime_available and 'test_dates' in locals() and len(test_dates) == len(Y_test_inv):
        date_column_for_output = test_dates
    else:
        # Fallback to index-based "DateTime"
        print(f"Warning: Using index for DateTime column in output CSV.")
        date_column_for_output = pd.Series(range(len(Y_test_inv)), name="DateTime")
        output_col_name = "DateTime" # Ensure correct name


    output_df = pd.DataFrame({
        output_col_name: date_column_for_output,
        target: Y_test_inv.flatten(),
        "pred": test_predict_inv.flatten()
    })
    output_df["error"] = (output_df[target] - output_df["pred"]).abs()
    csv_output = output_df.to_csv(index=False)


    # --- Return Results ---
    result = {
        "train_loss": train_rmse,
        "test_loss": test_rmse,
        "train_accuracy": train_r2,
        "test_accuracy": test_r2,
        "loss_curve": train_loss_curve_base64,         # Train/Validation RMSE vs Epoch
        "performance_plot": test_predictions_plot_base64, # Actual vs Pred over Time (Test)
        "anomaly_csv": csv_output,
        "target_column_used": target
    }
    return result