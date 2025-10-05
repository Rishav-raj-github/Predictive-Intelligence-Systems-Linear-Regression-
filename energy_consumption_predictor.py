#!/usr/bin/env python3
"""
EnergyOptim Pro - Multi-variate Energy Consumption Modeling
Advanced Linear Regression with Temporal and Weather Features

This project predicts building energy consumption using multiple linear regression
with advanced feature engineering including:
- Temporal features (hour, day, month, seasonality)
- Weather variables (temperature, humidity, wind speed)
- Interaction terms between weather and time
- Polynomial features for non-linear relationships

Difficulty: Advanced-Medium
Techniques: Multiple Linear Regression, Feature Engineering, Cross-Validation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class EnergyConsumptionPredictor:
    """
    Advanced energy consumption predictor using multi-variate linear regression
    with temporal and weather features.
    """
    
    def __init__(self, use_polynomial=True, degree=2, regularization='ridge', alpha=1.0):
        """
        Initialize the predictor with configurable options.
        
        Parameters:
        -----------
        use_polynomial : bool
            Whether to use polynomial features
        degree : int
            Degree of polynomial features
        regularization : str
            Type of regularization ('none', 'ridge', 'lasso')
        alpha : float
            Regularization strength
        """
        self.use_polynomial = use_polynomial
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        
        # Initialize model based on regularization type
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.scaler = StandardScaler()
        self.poly = None
        if use_polynomial:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    def generate_synthetic_data(self, n_samples=1000, start_date='2024-01-01'):
        """
        Generate synthetic energy consumption data with realistic patterns.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        start_date : str
            Starting date for time series
        
        Returns:
        --------
        DataFrame with features and target variable
        """
        np.random.seed(42)
        
        # Generate datetime index
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(hours=i) for i in range(n_samples)]
        
        # Extract temporal features
        hours = np.array([d.hour for d in dates])
        days = np.array([d.day for d in dates])
        months = np.array([d.month for d in dates])
        day_of_week = np.array([d.weekday() for d in dates])
        
        # Generate weather features with realistic patterns
        # Temperature varies with time of day and season
        temp_base = 15 + 10 * np.sin(2 * np.pi * months / 12)  # Seasonal variation
        temp_daily = 5 * np.sin(2 * np.pi * hours / 24)  # Daily variation
        temperature = temp_base + temp_daily + np.random.normal(0, 2, n_samples)
        
        # Humidity (inverse correlation with temperature)
        humidity = 70 - 0.5 * temperature + np.random.normal(0, 5, n_samples)
        humidity = np.clip(humidity, 20, 100)
        
        # Wind speed
        wind_speed = np.abs(np.random.normal(15, 5, n_samples))
        
        # Solar radiation (depends on hour and season)
        solar_rad = np.maximum(0, 800 * np.sin(np.pi * hours / 12) * 
                              (1 + 0.3 * np.sin(2 * np.pi * months / 12))) + \
                   np.random.normal(0, 50, n_samples)
        
        # Generate energy consumption (target variable)
        # Base load
        base_load = 50
        
        # Heating/cooling load (higher at temperature extremes)
        hvac_load = 2 * (temperature - 20)**2
        
        # Time-of-day pattern (higher during business hours)
        hourly_pattern = 30 * np.exp(-((hours - 14)**2) / 20)
        
        # Weekend effect (lower on weekends)
        weekend_effect = -15 * (day_of_week >= 5)
        
        # Seasonal effect
        seasonal_effect = 20 * np.sin(2 * np.pi * months / 12)
        
        # Wind cooling effect
        wind_effect = -0.3 * wind_speed
        
        # Solar heating effect
        solar_effect = 0.01 * solar_rad
        
        # Combined energy consumption with noise
        energy_consumption = (base_load + hvac_load + hourly_pattern + 
                            weekend_effect + seasonal_effect + wind_effect +
                            solar_effect + np.random.normal(0, 5, n_samples))
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': dates,
            'hour': hours,
            'day': days,
            'month': months,
            'day_of_week': day_of_week,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_rad,
            'is_weekend': (day_of_week >= 5).astype(int),
            'energy_consumption': energy_consumption
        })
        
        return df
    
    def engineer_features(self, df):
        """
        Create advanced engineered features.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe with raw features
        
        Returns:
        --------
        DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Cyclical encoding for temporal features
        df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng['hour'] / 24)
        df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng['hour'] / 24)
        df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['month'] / 12)
        df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['month'] / 12)
        
        # Interaction terms
        df_eng['temp_hour'] = df_eng['temperature'] * df_eng['hour']
        df_eng['temp_squared'] = df_eng['temperature'] ** 2
        df_eng['humidity_temp'] = df_eng['humidity'] * df_eng['temperature']
        df_eng['wind_temp'] = df_eng['wind_speed'] * df_eng['temperature']
        
        # Comfort index (feels-like temperature)
        df_eng['heat_index'] = df_eng['temperature'] + 0.5 * (df_eng['humidity'] - 50) / 10
        
        # Wind chill effect
        df_eng['wind_chill'] = df_eng['temperature'] - 0.5 * df_eng['wind_speed']
        
        return df_eng
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for modeling.
        
        Parameters:
        -----------
        df : DataFrame
            Dataframe with all features
        
        Returns:
        --------
        X : array
            Feature matrix
        y : array
            Target variable
        """
        # Select feature columns
        feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                       'temperature', 'humidity', 'wind_speed', 'solar_radiation',
                       'is_weekend', 'temp_hour', 'temp_squared', 'humidity_temp',
                       'wind_temp', 'heat_index', 'wind_chill']
        
        X = df[feature_cols].values
        y = df['energy_consumption'].values
        
        return X, y
    
    def fit(self, X_train, y_train):
        """
        Train the model with optional polynomial features and scaling.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Apply polynomial features if enabled
        if self.use_polynomial:
            X_scaled = self.poly.fit_transform(X_scaled)
        
        # Fit model
        self.model.fit(X_scaled, y_train)
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        """
        # Scale features
        X_scaled = self.scaler.transform(X_test)
        
        # Apply polynomial features if enabled
        if self.use_polynomial:
            X_scaled = self.poly.transform(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        """
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics, y_pred


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("EnergyOptim Pro - Advanced Energy Consumption Prediction")
    print("Multi-variate Linear Regression with Feature Engineering")
    print("=" * 70)
    
    # Initialize predictor
    predictor = EnergyConsumptionPredictor(
        use_polynomial=True,
        degree=2,
        regularization='ridge',
        alpha=0.1
    )
    
    # Generate synthetic data
    print("\n[1] Generating synthetic energy consumption data...")
    df = predictor.generate_synthetic_data(n_samples=2000)
    print(f"   Generated {len(df)} samples")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Engineer features
    print("\n[2] Engineering advanced features...")
    df_eng = predictor.engineer_features(df)
    print(f"   Created {len(df_eng.columns) - len(df.columns)} new features")
    
    # Prepare data
    print("\n[3] Preparing feature matrix...")
    X, y = predictor.prepare_features(df_eng)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target variable shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    print("\n[4] Training Ridge regression model...")
    predictor.fit(X_train, y_train)
    print("   Model training completed!")
    
    # Evaluate
    print("\n[5] Evaluating model performance...")
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)
    for metric, value in metrics.items():
        print(f"   {metric:10s}: {value:10.4f}")
    
    # Cross-validation
    print("\n[6] Performing 5-fold cross-validation...")
    X_scaled = predictor.scaler.transform(X_train)
    if predictor.use_polynomial:
        X_scaled = predictor.poly.transform(X_scaled)
    
    cv_scores = cross_val_score(predictor.model, X_scaled, y_train, 
                               cv=5, scoring='r2')
    print(f"   CV R² scores: {cv_scores}")
    print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("\nKey Features:")
    print("  - Multi-variate regression with 15+ engineered features")
    print("  - Temporal encoding (cyclical hour/month transformations)")
    print("  - Weather interactions and polynomial terms")
    print("  - Ridge regularization to prevent overfitting")
    print("  - Cross-validation for robust performance estimation")
    print("=" * 70)


if __name__ == "__main__":
    main()
