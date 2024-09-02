import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Download stock data
start = '2020-01-01'
end = '2024-01-01'
st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker", 'TSLA')
df = yf.download(user_input, start=start, end=end)

# Describing data
st.subheader('Data from 2020 to 2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100 Day MA', color='red')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.plot(ma100, label='100 Day MA', color='red')
plt.plot(ma200, label='200 Day MA', color='green')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data
x_train = []
y_train = []
time_steps = 100

for i in range(time_steps, len(data_training_array)):
    x_train.append(data_training_array[i-time_steps:i, 0])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
y_train = y_train.reshape(-1, 1)  # Ensure y_train is 2D

# Load pre-trained model
model = load_model('keras_model.h5')

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(time_steps, len(input_data)):
    x_test.append(input_data[i-time_steps:i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_test = y_test.reshape(-1, 1)  # Ensure y_test is 2D

# Predicting values
y_predicted = model.predict(x_test)

# Scale back to original values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting predictions vs original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
