import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pycoingecko import CoinGeckoAPI

st.title("📈 Price Forecasting Dashboard (Stocks & Crypto, LSTM)")

# User input
mode = st.selectbox("Select Asset Type", ["Stock", "Crypto"])
if mode == "Stock":
    symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL")
else:
    symbol = st.text_input("Enter Crypto ID (e.g., bitcoin, ethereum)", value="bitcoin")

def fetch_data(symbol, mode):
    if mode == "Stock":
        df = yf.download(symbol, period="1y")
        df = df[['Close']]
    else:
        cg = CoinGeckoAPI()
        data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=365)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.resample('1D').mean().dropna()
    return df

def add_indicators(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if st.button("Forecast"):
    with st.spinner("Fetching data and training model..."):
        df = fetch_data(symbol, mode)
        if df.empty:
            st.error("Failed to fetch data. Check the input.")
            st.stop()

        df = add_indicators(df)
        st.line_chart(df[['Close', 'MA10', 'MA50']])

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])

        time_steps = 30
        X, y = create_sequences(scaled_data, time_steps)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build and train model
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # Predict
        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(y_test)

        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual_prices, label='Actual')
        ax.plot(predicted_prices, label='Predicted')
        ax.set_title(f"Price Prediction for {symbol}")
        ax.legend()
        st.pyplot(fig)

        # Multi-step forecast (next 7 days)
        future_prices = []
        last_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
        for _ in range(7):
            next_pred = model.predict(last_seq)[0][0]
            future_prices.append(next_pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)
        forecasted = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

        st.success(f"Next day predicted price: ${forecasted[0][0]:.2f}")
        st.subheader("7-Day Forecast")
        st.line_chart(pd.Series(forecasted.flatten(), name="Forecast"))
