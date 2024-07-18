import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)
    data['SettlementDate'] = pd.to_datetime(data['SettlementDate'])
    data['StartTime'] = pd.to_datetime(data['StartTime'])
    data.set_index('StartTime', inplace=True)
    data = data.sort_index()
    return data

@st.cache_resource
def load_model_and_scaler(model_path, scaler_data):
    model = load_model(model_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(scaler_data)
    return model, scaler

model_file_name = 'LSTM_price.h5'

if not os.path.exists(model_file_name):
    st.error(f"Failed to find the model file: {model_file_name}.")
    st.stop()

test_file_path = 'test_1.csv'
test_data = load_and_preprocess_data(test_file_path)
test_price_data = test_data[['SystemSellPrice']].values

combined_price_data = test_price_data

model, scaler = load_model_and_scaler(model_file_name, combined_price_data)

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10

st.title('System Sell / Buy Price Prediction')

data_percentage = st.slider('Select the percentage of test data to use', 1, 100, 50)
test_data_length = int(len(test_price_data) * (data_percentage / 100))

if st.button('Predict'):
    if test_data_length <= time_step:
        st.error("The selected percentage of test data is too low to create sequences for prediction.")
    else:
        test_scaled_data = scaler.transform(test_price_data[:test_data_length])
        X_test, y_test = create_dataset(test_scaled_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_predictions = model.predict(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        
        test_series_adjusted = pd.Series(test_price_data.flatten()[time_step + 1:test_data_length], 
                                         index=test_data.index[time_step + 1:test_data_length])
        st.write(f'Predicted System Sell Prices for {data_percentage}% of test data:')

        plt.figure(figsize=(14, 8))
        plt.plot(test_series_adjusted.index, test_series_adjusted, label='Actual Test Data', color='blue')
        plt.plot(test_series_adjusted.index, test_predictions[:len(test_series_adjusted)], label='Predicted Test Data', color='green')
        plt.xlabel('Date')
        plt.ylabel('System Sell Price')
        plt.legend()
        st.pyplot(plt)

if st.checkbox('Show Scatter Plot'):
    if test_data_length <= time_step:
        st.error("The selected percentage of test data is too low to create sequences for prediction.")
    else:
        test_scaled_data = scaler.transform(test_price_data[:test_data_length])
        X_test, y_test = create_dataset(test_scaled_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_predictions = model.predict(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        
        test_series_adjusted = pd.Series(test_price_data.flatten()[time_step + 1:test_data_length], 
                                         index=test_data.index[time_step + 1:test_data_length])
        plt.figure(figsize=(8, 8))
        plt.scatter(test_series_adjusted, test_predictions[:len(test_series_adjusted)], color='purple')
        plt.plot([test_series_adjusted.min(), test_series_adjusted.max()], 
                 [test_series_adjusted.min(), test_series_adjusted.max()], color='red', linewidth=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices Scatter Plot')
        st.pyplot(plt)
