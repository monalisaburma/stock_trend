import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "AAPL")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

AAPL_data = yf.download(stock, start, end)
AAPL_data.reset_index(inplace=True)

# Load the model
model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(AAPL_data)

# Calculate moving averages
AAPL_data['MA_for_100_days'] = AAPL_data['Adj Close'].rolling(100).mean()
AAPL_data['MA_for_200_days'] = AAPL_data['Adj Close'].rolling(200).mean()
AAPL_data['MA_for_250_days'] = AAPL_data['Adj Close'].rolling(250).mean()

# Filter out rows with NaN values in moving averages
AAPL_data_filtered = AAPL_data.dropna(subset=['MA_for_100_days', 'MA_for_200_days', 'MA_for_250_days'])

# Plotting function
def plot_graph(figsize, values, labels, title):
    fig = plt.figure(figsize=figsize)
    for value, label in zip(values, labels):
        plt.plot(value, label=label)
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    st.pyplot(fig)

# Plot moving averages

st.subheader("MA for 250 days")
plot_graph((15, 5), [AAPL_data_filtered.set_index('Date')['MA_for_250_days']], ['MA for 250 Days'], 'MA for 250 Days of Google data')

st.subheader('Original Close Price and MA for 250 days')
plot_graph((15, 6), [AAPL_data_filtered.set_index('Date')['Adj Close'], AAPL_data_filtered.set_index('Date')['MA_for_250_days']],
           ['Adj Close', 'MA for 250 Days'], 'MA for 250 Days of AAPL Data')

st.subheader('Original Close Price and MA for 200 days')
plot_graph((15, 6), [AAPL_data_filtered.set_index('Date')['Adj Close'], AAPL_data_filtered.set_index('Date')['MA_for_200_days']],
           ['Adj Close', 'MA for 200 Days'], 'MA for 200 Days of AAPL Data')

st.subheader('Original Close Price and MA for 100 days')
plot_graph((15, 6), [AAPL_data_filtered.set_index('Date')['Adj Close'], AAPL_data_filtered.set_index('Date')['MA_for_100_days']],
           ['Adj Close', 'MA for 100 Days'], 'MA for 100 Days of AAPL Data')

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
plot_graph((15, 5), [AAPL_data_filtered.set_index('Date')['Adj Close'], AAPL_data_filtered.set_index('Date')['MA_for_100_days'], AAPL_data_filtered.set_index('Date')['MA_for_250_days']],
           ['Adj Close', 'MA for 100 Days', 'MA for 250 Days'], 'Adjusted Close and Moving Averages')

# Scaling data for model
splitting_len = int(len(AAPL_data) * 0.7)
x_test = AAPL_data[['Close']].iloc[splitting_len:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame({
    'Date': AAPL_data['Date'].iloc[splitting_len + 100:].values,
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
})
ploting_data.set_index('Date', inplace=True)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Function to plot predictions
def plot_prediction_graph(figsize, plot_data, title):
    fig = plt.figure(figsize=figsize)
    plt.plot(plot_data.index, plot_data['original_test_data'], label='Original Test Data', color='blue')
    plt.plot(plot_data.index, plot_data['predictions'], label='Predicted Test Data', color='orange')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

st.subheader('Test Data with Predictions')
plot_prediction_graph((15, 6), ploting_data, 'Test Data with Predictions')

# Ensure the indices are properly set for concatenation
Adj_close_price = AAPL_data_filtered[['Adj Close']].set_index(AAPL_data_filtered['Date'])

# Plot the whole data including predictions
whole_data = pd.concat([Adj_close_price[:splitting_len + 100], ploting_data], axis=0)

fig = plt.figure(figsize=(15, 6))
plt.plot(whole_data.index, whole_data['Adj Close'], label='Training Data')
plt.plot(whole_data.index, whole_data['original_test_data'], label='Original Test Data')
plt.plot(whole_data.index, whole_data['predictions'], label='Predicted Test Data')
plt.xlabel("Years")
plt.ylabel("Whole Data")
plt.title("Whole Data of AAPL data")
plt.legend()
st.pyplot(fig)
