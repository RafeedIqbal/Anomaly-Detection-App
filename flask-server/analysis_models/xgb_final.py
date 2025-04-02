# Load the datasets API
import urllib.request
api_url = 'https://raw.githubusercontent.com/tanmayyb/ele70_bv03/refs/heads/main/api/datasets.py'
exec(urllib.request.urlopen(api_url).read())

# Load the IESO dataset
ieso = IESODataset('zonal') # option 1
# ieso = IESODataset('fsa') # option 2

target_options = ieso.get_target_options() # returns list of the target options
available_dates = ieso.get_dates() # returns list of available dates (str)
## send these options to the GUI to show to user
## target and date options depend on dataset type (zonal or fsa)
# print(target_options)
# print(available_dates)


# this is the user input for the target
target_val = 4
user_selected_start_date = '2010'
user_selected_end_date = '2020'

ieso.set_target(target_val)
ieso.load_dataset(start_date=user_selected_start_date, end_date=user_selected_end_date, download=True)
climate = ClimateDataset(ieso)
climate.load_dataset(sample_num=5, download=True)
preprocessor = DatasetPreprocessor(ieso, climate)
target_name, dataset, dt = preprocessor.preprocess()


# Create train and test sets
(X_train, X_test, y_train, y_test), (train_idx, test_idx) = create_train_test_split(dataset, target=target_name, dt=dt)
y_test_numpy = y_test.to_numpy()




RUN_WITH_CALLBACK = False
# Train the model and predict the test set
import numpy as np # linear algebra
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

if RUN_WITH_CALLBACK:
    class PlotCallback(xgb.callback.TrainingCallback):
        def __init__(self, update_callback):
            self.update_callback = update_callback

        def before_training(self, model):
            return model

        def after_training(self, model):
            return model

        def after_iteration(self, model, epoch, evals_log):
            # Adjusting for the correct structure:
            train_loss = evals_log.get('train', {}).get('rmse', [None])[epoch]
            val_loss = evals_log.get('eval', {}).get('rmse', [None])[epoch]

            if train_loss is not None and val_loss is not None:
                self.update_callback(train_loss, val_loss)
            # Returning False indicates training should continue
            return False

    class XGBoostTrainer:
        def __init__(self, update_callback):
            """
            Args:
                update_callback: Function that takes (train_loss, val_loss) as arguments.
            """
            self.update_callback = update_callback

        def train(self, X_train, y_train, X_test, y_test):
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
            evals = [(dtrain, "train"), (dtest, "eval")]

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                callbacks=[PlotCallback(self.update_callback)]
            )
            return booster

    # Dummy update function for demonstration:
    def update_plot(train_loss, val_loss):
        print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")

    trainer = XGBoostTrainer(update_plot)
    booster = trainer.train(X_train, y_train, X_test, y_test)

    dtest = xgb.DMatrix(X_test)
    pred = booster.predict(dtest)

else:
    # Train without callback (NO PLOTTING OF TRAINING PROGRESS)
    booster = xgb.XGBRegressor(n_estimators=1000)
    booster.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            # early_stopping_rounds=50,
        verbose=False)
    pred = booster.predict(X_test)



# model output dataframe
# to be used for anomaly detection
output_df = pd.concat([dt, y_test, pd.Series(pred, index=y_test.index, name='pred')],axis=1).dropna()



# calculate MSE, MAE, MAPE
mse = mean_squared_error(y_true=y_test,
                   y_pred=pred)
mae = mean_absolute_error(y_true=y_test,
                   y_pred=pred)
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"RMSE: {np.sqrt(mse)}")
print(f"MAE: {mae}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, pred)}")




# plot prediction
from plotly import graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=test_idx,
    y=y_test.to_numpy(),
    name='Actual',
    line_color='blue')
)

fig.add_trace(go.Scattergl(
    x=test_idx,
    y=pred,
    name='Predicted',
    line_color='red')
)


# Set the theme to 'plotly_white'
fig.update_layout(
    title=f"Time Series Forecasting for {target_name} with XGBoostRegressor",
    xaxis_title="t (1 unit = 1 hour)",
    yaxis_title="Energy Demand",
    template="plotly_white",
    xaxis = dict( rangeslider=dict(
      visible=True
    ))
)
fig.show()
# Save the plot to a file
import os
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig.write_html(f'{output_dir}/xgb_mt1r1_pred.html', auto_open=False)