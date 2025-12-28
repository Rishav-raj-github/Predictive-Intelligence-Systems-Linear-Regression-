"""
04_Regularization_Techniques.py

Comprehensive notebook for regularization techniques in linear regression.
Demonstrates Ridge, Lasso, and ElasticNet regularization with tuning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 04: Regularization Techniques
# ============================================================================

class RegularizationNotebook:
    """
    Demonstrates regularization techniques for preventing overfitting.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.results = {}
    
    def generate_data(self, n_samples=200, n_features=100, sparsity=0.9):
        """
        Generate high-dimensional dataset with sparse true coefficients.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        
        # Create sparse coefficients
        true_coef = np.zeros(n_features)
        n_true = max(1, int(n_features * (1 - sparsity)))
        true_coef[:n_true] = np.random.randn(n_true) * 5
        
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def tune_ridge(self, X_train, X_test, y_train, y_test):
        """
        Tune Ridge regression with GridSearchCV.
        """
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge = Ridge()
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_ridge = grid_search.best_estimator_
        y_pred_test = best_ridge.predict(X_test)
        
        return {
            'model': best_ridge,
            'best_alpha': grid_search.best_params_['alpha'],
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R2': r2_score(y_test, y_pred_test)
        }
    
    def tune_lasso(self, X_train, X_test, y_train, y_test):
        """
        Tune Lasso regression with GridSearchCV.
        """
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
        lasso = Lasso(max_iter=10000)
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_lasso = grid_search.best_estimator_
        y_pred_test = best_lasso.predict(X_test)
        
        n_selected = np.sum(best_lasso.coef_ != 0)
        
        return {
            'model': best_lasso,
            'best_alpha': grid_search.best_params_['alpha'],
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R2': r2_score(y_test, y_pred_test),
            'n_selected_features': n_selected
        }
    
    def tune_elasticnet(self, X_train, X_test, y_train, y_test):
        """
        Tune ElasticNet regression.
        """
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
        elasticnet = ElasticNet(max_iter=10000)
        grid_search = GridSearchCV(elasticnet, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_en = grid_search.best_estimator_
        y_pred_test = best_en.predict(X_test)
        
        return {
            'model': best_en,
            'best_alpha': grid_search.best_params_['alpha'],
            'best_l1_ratio': grid_search.best_params_['l1_ratio'],
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R2': r2_score(y_test, y_pred_test)
        }
    
    def print_results(self):
        """
        Print regularization comparison.
        """
        print("\n" + "="*80)
        print("REGULARIZATION TECHNIQUES COMPARISON")
        print("="*80)
        
        for method, metrics in self.results.items():
            print(f"\n{method}:")
            for key, value in metrics.items():
                if key != 'model':
                    print(f"  {key}: {value}")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    notebook = RegularizationNotebook(random_state=42)
    
    print("Generating high-dimensional sparse data...")
    X_train, X_test, y_train, y_test = notebook.generate_data(
        n_samples=200, n_features=100, sparsity=0.9
    )
    print(f"Dataset shape: {X_train.shape} (sparse, 90% zero coefficients)")
    
    print("\nTuning regularization models...")
    notebook.results['Ridge'] = notebook.tune_ridge(X_train, X_test, y_train, y_test)
    notebook.results['Lasso'] = notebook.tune_lasso(X_train, X_test, y_train, y_test)
    notebook.results['ElasticNet'] = notebook.tune_elasticnet(
        X_train, X_test, y_train, y_test
    )
    
    notebook.print_results()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Ridge: Better for correlated features, all features retained")
    print("2. Lasso: Feature selection via sparsity, better interpretability")
    print("3. ElasticNet: Combines Ridge and Lasso benefits")
    print("4. GridSearchCV ensures optimal hyperparameter selection")
    print("="*80 + "\n")
