from darts.models import TSMixerModel
from darts import TimeSeries
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
import yfinance as yf
import pandas as pd
import ta.trend, ta.momentum
from joblib import load
from pathlib import Path

def gather_ticker_data(ticker_symbol:str):
    data = yf.download(ticker_symbol, period='5y', auto_adjust=False)
    if data is None:
        raise ValueError(f'No data found for {ticker_symbol}')
    data.columns = data.columns.droplevel('Ticker')
    return data

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        ticker_data = X.copy()
        ticker_data = ticker_data.dropna()
        ticker_data['RSI 14'] = ta.momentum.RSIIndicator(close=ticker_data['Adj Close'].squeeze(), window=14).rsi()
        macd = ta.trend.MACD(close=ticker_data['Adj Close'].squeeze())
        ticker_data['macd'] = macd.macd()
        ticker_data['macd signal'] = macd.macd_signal()

        ticker_data['20MA'] = ticker_data['Adj Close'].rolling(window=20).mean()
        ticker_data['50MA'] = ticker_data['Adj Close'].rolling(window=50).mean()
        ticker_data['200MA'] = ticker_data['Adj Close'].rolling(window=200).mean()

        ticker_data['20EMA'] = ticker_data['Adj Close'].ewm(span=20, adjust = False).mean()
        ticker_data['50EMA'] = ticker_data['Adj Close'].ewm(span=50, adjust = False).mean()
        ticker_data['200EMA'] = ticker_data['Adj Close'].ewm(span=200, adjust = False).mean()

        ticker_data['Close lag 1'] = ticker_data['Adj Close'].shift(1)
        ticker_data['Close lag 2'] = ticker_data['Adj Close'].shift(2)

        ticker_data = ticker_data.asfreq('D')
        ticker_data.interpolate(method='linear', inplace=True)
        ticker_data.bfill(inplace=True)
        return ticker_data
    
BASE_DIR = Path(__file__).parent

def encode_year(idx: pd.DatetimeIndex):
    return (idx.year - 2000)/50

class DartsPredictionTransformer(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.covariates_scaler = load(f'{BASE_DIR}/scaler_covariates.joblib')
        self.target_scaler = load(f'{BASE_DIR}/scaler_target.joblib')
        self.model = TSMixerModel.load(f'{BASE_DIR}/my_ts_mixer_model.pt')
        self.model.to_cpu()

    def fit(self, X, y=None):
        return self
    
    def predict(self, X, forecast_horizon: int) -> pd.DataFrame:
        target_series = TimeSeries.from_dataframe(X, value_cols=['Adj Close'], freq='D')
        past_covariates_series = TimeSeries.from_dataframe(X.drop(columns=['Adj Close']), freq='D')

        target_scaled = self.target_scaler.transform(target_series)
        past_covariates_scaled = self.covariates_scaler.transform(past_covariates_series)

        prediction = self.model.predict(n=forecast_horizon, series = target_scaled, past_covariates = past_covariates_scaled)
        prediction_unscaled = self.target_scaler.inverse_transform(prediction)
        return prediction_unscaled.to_dataframe().reset_index()
    
def create_prediction_pipeline():
    prediction_pipeline = Pipeline([
        ('feature-engineering', FeatureEngineeringTransformer()),
        ('prediction', DartsPredictionTransformer())
    ])
    return prediction_pipeline  