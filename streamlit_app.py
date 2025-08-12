import streamlit as st
import pandas as pd
import sys
from pipeline_module import gather_ticker_data, create_prediction_pipeline, encode_year

# Inject encode_year for joblib unpickling
sys.modules["__main__"].encode_year = encode_year  # type: ignore

# Load pipeline (fresh each time to avoid reusing fine-tuned model)
def load_pipeline():
    return create_prediction_pipeline()

pipeline = load_pipeline()

# Page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown(
    """
    Enter an **NSE ticker symbol** below to view the 7-day forecast of its stock prices.  
    Example: `RELIANCE.NS`, `TATASTEEL.NS`
    """
)

# Ticker input
ticker = st.text_input(
    "Enter NSE Ticker Symbol",
    placeholder="e.g., RELIANCE.NS, TATASTEEL.NS"
)

# Function to shift weekend dates to weekdays without overlap
def shift_weekends_to_weekdays_no_overlap(dates):
    shifted_dates = []
    existing_dates = set(dates)

    for d in dates:
        if d.weekday() == 5:  # Saturday → Monday
            new_date = d + pd.Timedelta(days=2)
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        elif d.weekday() == 6:  # Sunday → Tuesday
            new_date = d + pd.Timedelta(days=2)
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        else:
            shifted_dates.append(d)
    return pd.to_datetime(shifted_dates)

# Forecast button
if st.button("🔍 Get 7-Day Forecast", use_container_width=True):
    if not ticker:
        st.warning("⚠️ Please enter a valid NSE ticker symbol.")
    else:
        try:
            # Step 1: Fetch historical data
            with st.spinner(f"📥 Fetching historical data for **{ticker}**..."):
                historical_df = gather_ticker_data(ticker_symbol=ticker)

            # Step 2: Fine-tune model
            with st.spinner("🛠 Fine-tuning model on historical data..."):
                pipeline.fit(historical_df)

            # Step 3: Predict next 7 days
            with st.spinner("📊 Generating 7-day forecast..."):
                forecast_df = pipeline.predict(historical_df, forecast_horizon=7)

            # Adjust forecast dates
            forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            forecast_df['Date'] = shift_weekends_to_weekdays_no_overlap(forecast_df['Date'])
            forecast_df = forecast_df.sort_values('Date').reset_index(drop=True)
            forecast_df['Weekday'] = forecast_df['Date'].dt.day_name()

            # Display forecast table
            display_df = forecast_df.set_index(forecast_df['Date'].dt.date)
            display_df['Weekday'] = forecast_df['Weekday'].values
            st.subheader("📊 Predicted Prices")
            st.dataframe(display_df[['Adj Close', 'Weekday']], use_container_width=True)

            # Prepare data for chart (last 30 days + forecast)
            last_30_df = historical_df.tail(30)
            historical_plot = last_30_df['Adj Close'].rename('Historical Price')
            forecast_plot = forecast_df.set_index('Date')['Adj Close'].rename('Forecasted Price')
            chart_data = pd.concat([historical_plot, forecast_plot], axis=1)

            # Display chart
            st.subheader("📈 Historical Data vs Forecast")
            st.line_chart(chart_data, color=["#007bff", "#ff7f0e"])

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

