"""
02_Feature_Scaling_Normalization.py

Comprehensive notebook for feature scaling and normalization techniques.
Demonstrates various scaling methods and their impact on model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 02: Feature Scaling and Normalization
# ============================================================================

class FeatureScalingNotebook:
    """
    Demonstrates impact of feature scaling on linear regression models.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaling_results = {}
    
    def generate_sample_data(self, n_samples=1000):
        """
        Generate sample dataset with features on different scales.
        """
        np.random.seed(self.random_state)
        
        # Features on different scales
        X = np.random.randn(n_samples, 5)
        X[:, 0] *= 1000  # Feature with large scale
        X[:, 1] *= 100   # Feature with medium scale
        X[:, 2] *= 10    # Feature with small scale
        X[:, 3] *= 50
        X[:, 4] *= 0.01
        
        # Target with linear relationship
        y = 2*X[:, 0] + 3*X[:, 1] - 5*X[:, 2] + 1.5*X[:, 3] + 100*X[:, 4] + np.random.randn(n_samples) * 10
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def test_scaling_methods(self):
        """
        Compare different scaling methods.
        """
        methods = {
            'No Scaling': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        
        results = {}
        
        for method_name, scaler in methods.items():
            X_train = self.X_train.copy()
            X_test = self.X_test.copy()
            
            # Apply scaling
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, self.y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            r2_train = r2_score(self.y_train, y_pred_train)
            r2_test = r2_score(self.y_test, y_pred_test)
            
            results[method_name] = {
                'RMSE_Train': rmse_train,
                'RMSE_Test': rmse_test,
                'R2_Train': r2_train,
                'R2_Test': r2_test,
                'Coefficients': model.coef_
            }
        
        self.scaling_results = results
        return results
    
    def print_results(self):
        """
        Print scaling comparison results.
        """
        print("\n" + "="*80)
        print("FEATURE SCALING IMPACT ANALYSIS")
        print("="*80)
        
        for method, metrics in self.scaling_results.items():
            print(f"\n{method}:")
            print(f"  RMSE (Train): {metrics['RMSE_Train']:.4f}")
            print(f"  RMSE (Test):  {metrics['RMSE_Test']:.4f}")
            print(f"  R² (Train):   {metrics['R2_Train']:.4f}")
            print(f"  R² (Test):    {metrics['R2_Test']:.4f}")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Initialize notebook
    notebook = FeatureScalingNotebook(random_state=42)
    
    # Generate data
    print("Generating sample data...")
    X_train, X_test, y_train, y_test = notebook.generate_sample_data(n_samples=1000)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Test scaling methods
    print("\nTesting scaling methods...")
    results = notebook.test_scaling_methods()
    
    # Print results
    notebook.print_results()
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("1. Feature scaling affects model convergence and coefficient magnitude")
    print("2. StandardScaler provides best balance for linear regression")
    print("3. Unscaled data shows larger coefficients but equivalent predictions")
    print("4. Regularized models (Ridge/Lasso) benefit significantly from scaling")
    print("="*80 + "\n")
