import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from dataset_loader import *

ieso_dataset = load_ieso_dataset(2018, 2020, join=True).iloc[:-1]
weather_dataset = load_climate_dataset(2018, 2020, join=True).iloc[1:]

# LOAD DATASETS AND PREPROCESS

#@markdown merge and preprocess

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

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


data = df[['Toronto']]


scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

train_size = int(len(data_scaled) * 0.80)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(X_train, Y_train, epochs=50, batch_size=200, verbose=0)


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((len(train_predict), len(data.columns) - 1))), axis=1))[:, 0]
Y_train = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((len(Y_train), len(data.columns) - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((len(test_predict), len(data.columns) - 1))), axis=1))[:, 0]
Y_test = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((len(Y_test), len(data.columns) - 1))), axis=1))[:, 0]

# prompt: write the same code but without any look_back parameters

import numpy as np
def create_dataset(dataset):
    X, Y = [], []
    for i in range(len(dataset) - 1):
        a = dataset[i, 0]
        X.append(a)
        Y.append(dataset[i + 1, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train_data)
X_test, Y_test = create_dataset(test_data)

X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=50, batch_size=200, verbose=0)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((len(train_predict), len(data.columns) - 1))), axis=1))[:, 0]
Y_train = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((len(Y_train), len(data.columns) - 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((len(test_predict), len(data.columns) - 1))), axis=1))[:, 0]
Y_test = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((len(Y_test), len(data.columns) - 1))), axis=1))[:, 0]

print(len(Y_test))

#@markdown plot prediction
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=dt[train_size + look_back + 1:][:len(Y_test)],
    y=Y_test,
    name='Actual',
    line_color='blue')
)

fig.add_trace(go.Scattergl(
    x=dt[train_size + look_back + 1:][:len(Y_test)],
    y=test_predict,
    name='Predicted',
    line_color='red')
)

# Set the theme to 'plotly_white'
fig.update_layout(
    title=f"Time Series Forecasting for with LSTM",
    xaxis_title="Date", # Change the x-axis title
    yaxis_title="Energy Demand",
    template="plotly_white",
    xaxis = dict(
      rangeslider=dict(
          visible=True
      ),
      tickformat="%Y-%m-%d"
    )
)
fig.show()

tmp = pd.concat([dt, pd.Series(Y_test, name='Y_test'), pd.Series(test_predict, name='pred')], axis=1).dropna()

tmp

Y_test

# prompt: how many rows is Y_test have

print(len(Y_test))
print(len(test_predict))
print(len(df))

# prompt: make the filtered_dt start the row with 0

# Filter the data based on the specified date range
start_date = '2018-10-20 00:00:00'
end_date = '2020-12-31 23:00:00'
mask = (dt >= start_date) & (dt <= end_date)

# Apply the mask to the relevant variables
filtered_dt = dt[mask].reset_index(drop=True) # Reset index here
filtered_Y_test = Y_test[(dt[train_size + look_back + 1:] >= start_date) & (dt[train_size + look_back + 1:] <= end_date)]
filtered_test_predict = test_predict[(dt[train_size + look_back + 1:] >= start_date) & (dt[train_size + look_back + 1:] <= end_date)]


# Create the filtered plot
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=filtered_dt[:len(filtered_Y_test)],
    y=filtered_Y_test,
    name='Actual',
    line_color='blue')
)

fig.add_trace(go.Scattergl(
    x=filtered_dt[:len(filtered_Y_test)],
    y=filtered_test_predict,
    name='Predicted',
    line_color='red')
)

# Set the theme to 'plotly_white'
fig.update_layout(
    title=f"Time Series Forecasting for with LSTM (2018-10-20 to 2020-12-31)",
    xaxis_title="Date", # Change the x-axis title
    yaxis_title="Energy Demand",
    template="plotly_white",
    xaxis = dict(
      rangeslider=dict(
          visible=True
      ),
      tickformat="%Y-%m-%d"
    )
)
fig.show()


