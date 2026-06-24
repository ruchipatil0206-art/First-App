import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Stock Price Forecast using ARIMA",
    layout="wide"
)

st.title("📈 Stock Price Forecast (ARIMA)")
st.write("Yahoo Finance Data + ARIMA Forecast")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    try:
        # Download last 5 years data
        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"]

        st.subheader("Historical Stock Price")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices,
                mode="lines",
                name="Close Price"
            )
        )

        fig.update_layout(
            title=f"{ticker} - Last 5 Years",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("ARIMA Model Forecast")

        # ARIMA Model
        model = ARIMA(close_prices, order=(5, 1, 0))
        fitted_model = model.fit()

        # Forecast until June 2027
        target_date = pd.Timestamp("2027-06-30")

        last_date = close_prices.index[-1]

        months_diff = (
            (target_date.year - last_date.year) * 12
            + (target_date.month - last_date.month)
        )

        trading_days = max(months_diff * 21, 1)

        forecast = fitted_model.forecast(
            steps=trading_days
        )

        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=trading_days
        )

        forecast_series = pd.Series(
            forecast.values,
            index=future_dates
        )

        june_2027 = forecast_series[
            forecast_series.index.month == 6
        ]

        predicted_price = float(june_2027.iloc[-1])

        st.success(
            f"Predicted Price for June 2027: ${predicted_price:.2f}"
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices,
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            title=f"{ticker} Forecast Until June 2027",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")
