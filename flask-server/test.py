import pandas as pd
from models import XGB_MT1R1, LSTM_FINAL

# Load your CSV file into a DataFrame
df = pd.read_csv('flask-server/asd.csv')

# If your CSV does not have a column named 'Toronto', specify your target column
target_column = 'Toronto'  # Change this to your target column name

# Test the XGB model function
xgb_result = XGB_MT1R1(df, target=target_column)
print("XGB_MT1R1 Results:", xgb_result)

# Test the LSTM model function
lstm_result = LSTM_FINAL(df, target=target_column)
print("LSTM_FINAL Results:", lstm_result)
