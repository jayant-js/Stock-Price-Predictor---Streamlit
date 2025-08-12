import streamlit as st
import pandas as pd
import sys
from pipeline_module import gather_ticker_data, create_prediction_pipeline, encode_year

# Inject encode_year for joblib unpickling
sys.modules["__main__"].encode_year = encode_year  # type: ignore

# Cache pipeline so it’s created only once
@st.cache_resource
def load_pipeline():
    return create_prediction_pipeline()

pipeline = load_pipeline()

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")
st.title('📈 Stock Price Predictor')

ticker = st.text_input(
    'Enter a valid NSE Ticker Symbol',
    placeholder='e.g., RELIANCE.NS, TATASTEEL.NS'
)

def shift_weekends_to_weekdays_no_overlap(dates):
    shifted_dates = []
    existing_dates = set(dates)
    
    for d in dates:
        if d.weekday() == 5:  # Saturday
            new_date = d + pd.Timedelta(days=2)  # Monday
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        elif d.weekday() == 6:  # Sunday
            new_date = d + pd.Timedelta(days=2)  # Tuesday
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        else:
            shifted_dates.append(d)
    return pd.to_datetime(shifted_dates)

if st.button('Get 7-Day Forecast'):
    if not ticker:
        st.warning('Please enter a ticker symbol.')
    else:
        with st.spinner(f'Fetching forecast for {ticker}...'):
            try:
                # Historical data (full set)
                historical_df = gather_ticker_data(ticker_symbol=ticker)

                # Forecast
                forecast_df = pipeline.predict(historical_df, forecast_horizon=7)

                # Clean up forecast dates
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                forecast_df['Date'] = shift_weekends_to_weekdays_no_overlap(forecast_df['Date'])
                forecast_df['Weekday'] = forecast_df['Date'].dt.day_name()

                # Display forecast
                display_df = forecast_df.set_index(forecast_df['Date'].dt.date)
                display_df['Weekday'] = forecast_df['Weekday'].values
                st.subheader('Predicted Prices')
                st.dataframe(display_df[['Adj Close', 'Weekday']], use_container_width=True)

                # Last 30 days historical data
                last_30_df = historical_df.tail(30)

                # Plot
                historical_plot = last_30_df['Adj Close'].rename('Historical Price')
                forecast_plot = forecast_df.set_index('Date')['Adj Close'].rename('Forecasted Price')
                chart_data = pd.concat([historical_plot, forecast_plot], axis=1)

                st.subheader('Historical Data vs. Forecast')
                st.line_chart(chart_data, color=["#007bff", "#ff7f0e"])

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")