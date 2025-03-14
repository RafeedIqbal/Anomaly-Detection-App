import re
import urllib.request
from dataset_loader import *

ieso_dataset = load_ieso_dataset(2010, 2020, join=True).iloc[:-1]
weather_dataset = load_climate_dataset(2010, 2020, join=True).iloc[1:]

# LOAD DATASETS AND PREPROCESS

dataset = pd.merge(ieso_dataset, weather_dataset, on="DateTime")

def preprocess(dataset:pd.DataFrame, split_datetime=True) -> pd.DataFrame:
  df = dataset.copy()
  ieso_cols = ['Toronto']
  climate_cols = [
       'Temp (°C)',
      #  'Temp Flag',
       'Dew Point Temp (°C)',
      #  'Dew Point Temp Flag',
       'Rel Hum (%)',
      #  'Rel Hum Flag',
       'Precip. Amount (mm)',
      #  'Precip. Amount Flag',
      #  'Wind Dir (10s deg)',
      #  'Wind Dir Flag',
      #  'Wind Spd (km/h)',
      #  'Wind Spd Flag',
      #  'Visibility (km)',
      #  'Visibility Flag',
       'Stn Press (kPa)',
      #  'Stn Press Flag',
       'Hmdx',
      #  'Hmdx Flag', 'Wind Chill', 'Wind Chill Flag', 'Weather'
      ]
  if split_datetime:
    df['Y'] = df['DateTime'].dt.year
    df['M'] = df['DateTime'].dt.month
    df['D'] = df['DateTime'].dt.day
    df['H'] = df['DateTime'].dt.hour
    cols = ['Y', 'M', 'D', 'H']
  else:
    cols = ['DateTime']

  # delete leap day
  df = df[~((df.DateTime.dt.month == 2) & (df.DateTime.dt.day == 29))]
  dt = df['DateTime'] # store datettime

  cols += ieso_cols+climate_cols

  df = df[cols]

  # make columns names better
  df.columns = df.columns.str.replace('.', '')
  df.columns = df.columns.str.replace(' ', '')
  df.columns = df.columns.str.replace(r"\(.*?\)", "", regex=True)

  nans = df.isna().sum().to_dict()

  # dirty mean imputation
  data = df.fillna(df.mean())

  # # dirty -1 imputation
  # data = df.fillna(pd.Series(index=df.columns, data=[-1.0]*len(df.columns)))

  data = data.reset_index()

  return df, nans, dt

df, nans, dt = preprocess(dataset)

df

def create_train_test_split(dataset:pd.DataFrame, target:str='target', split_coeff:float=0.8, dt=None) -> tuple:
  training_cutoff = int(split_coeff*len(dataset))
  train = dataset.iloc[:training_cutoff]
  test = dataset.iloc[training_cutoff:]

  X_train = train.drop(columns=[target])
  y_train = train[target]
  X_test = test.drop(columns=[target])
  y_test = test[target]

  (train_idx, test_idx) = None, None
  if dt is not None:
    train_idx = dt[:training_cutoff]
    test_idx = dt[training_cutoff:]

  return (X_train, X_test, y_train, y_test), (train_idx, test_idx)

target = 'Toronto'
(X_train, X_test, y_train, y_test), (train_idx, test_idx) = create_train_test_split(df, target=target, dt=dt)
y_test_numpy = y_test.to_numpy()

# XGBoost Regression Model for Time Series Forecasting

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

# https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost#Create-XGBoost-Model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        # early_stopping_rounds=50,
       verbose=False)

pred = reg.predict(X_test)

# Results and Analysis

#@markdown plot prediction
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
    title=f"Time Series Forecasting for {target} with XGBoostRegressor",
    xaxis_title="t (1 unit = 1 hour)",
    yaxis_title="Energy Demand",
    template="plotly_white",
    xaxis = dict( rangeslider=dict(
      visible=True
    ))
)
fig.show()
# fig.write_html(f'mcr4_xgb_mt1r1_pred.html')