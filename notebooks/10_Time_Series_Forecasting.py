"""
10_Time_Series_Forecasting.py

Time series regression and forecasting with autoregressive features.
Demonstrates lag features, seasonal decomposition, and trend analysis.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecastingNotebook:
    """
    Demonstrates time series forecasting using linear regression with AR features.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def generate_time_series(self, n_steps=200, trend=0.01, seasonality=10):
        """
        Generate synthetic time series data.
        """
        np.random.seed(self.random_state)
        t = np.arange(n_steps)
        trend_component = trend * t
        seasonal_component = seasonality * np.sin(2 * np.pi * t / 50)
        noise = np.random.randn(n_steps) * 0.5
        
        series = 100 + trend_component + seasonal_component + noise
        return series
    
    def create_lagged_features(self, series, lags=12):
        """
        Create lagged features for AR(p) model.
        """
        X, y = [], []
        
        for i in range(lags, len(series)):
            X.append(series[i-lags:i])
            y.append(series[i])
        
        return np.array(X), np.array(y)
    
    def forecast(self, series, lags=12, steps_ahead=20):
        """
        Forecast future values using AR(p) model.
        """
        X, y = self.create_lagged_features(series, lags)
        
        model = LinearRegression()
        model.fit(X, y)
        
        forecast_values = []
        current_seq = series[-lags:].copy()
        
        for _ in range(steps_ahead):
            next_val = model.predict([current_seq])[0]
            forecast_values.append(next_val)
            current_seq = np.append(current_seq[1:], next_val)
        
        return model, forecast_values
    
    def evaluate_forecast(self, actual, forecasted):
        """
        Evaluate forecast accuracy.
        """
        rmse = np.sqrt(mean_squared_error(actual, forecasted))
        mae = np.mean(np.abs(actual - forecasted))
        
        return {'RMSE': rmse, 'MAE': mae}

if __name__ == '__main__':
    notebook = TimeSeriesForecastingNotebook(random_state=42)
    
    print("Generating synthetic time series...")
    series = notebook.generate_time_series(n_steps=200)
    
    print("Creating lagged features and training AR model...")
    model, forecast = notebook.forecast(series, lags=12, steps_ahead=20)
    
    metrics = notebook.evaluate_forecast(series[-20:], forecast[:20])
    
    print("\n" + "="*80)
    print("TIME SERIES FORECASTING RESULTS")
    print("="*80)
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print("="*80)
    print("\nKEY CONCEPTS:")
    print("1. Autoregressive (AR) models use past values as features")
    print("2. Lag selection is crucial for model performance")
    print("3. Seasonal patterns can be captured with seasonal lags")
    print("4. Trend and seasonality should be handled separately")
    print("="*80 + "\n")
