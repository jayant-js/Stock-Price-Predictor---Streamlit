import streamlit as st
import requests
import pandas as pd
import json
from pipeline_module import create_pipeline

st.title("📈 Stock Price Prediction App")
ticker = st.text_input("Give the stock's Ticker Symbol").upper()
if st.button('Predict'):
    if ticker:
        with st.spinner('Training the model...'):
            try:
                pipeline = create_pipeline(ticker)
                pipeline.fit(X=pd.DataFrame())
                st.success(f'Model Trained successfully for the ticker: {ticker}')
            except Exception as e:
                st.error(f'Training Failed: {e}')
                st.stop()
        with st.spinner('Fetching Forecast...'):
            try:
                forecast_df = pipeline.named_steps['model_training_and_prediction'].predict(n=7)
                forecast_df.reset_index(inplace=True)
                st.subheader('Forecast for next 7 days:')
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                st.dataframe(forecast_df.set_index('Date'))
            except Exception as e:
                st.error(f'Prediction failed: {e}')
    else:
        st.warning('Please enter a valid stock ticker')
