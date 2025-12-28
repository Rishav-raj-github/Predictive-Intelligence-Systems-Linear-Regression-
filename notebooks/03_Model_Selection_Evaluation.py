"""
03_Model_Selection_Evaluation.py

Comprehensive notebook for model selection and performance evaluation.
Compares multiple linear regression variants and evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 03: Model Selection and Evaluation
# ============================================================================

class ModelSelectionNotebook:
    """
    Demonstrates model selection and comprehensive evaluation strategies.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def generate_dataset(self, n_samples=500, n_features=10):
        """
        Generate synthetic regression dataset.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple model variants.
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0),
            'Ridge (alpha=10.0)': Ridge(alpha=10.0),
            'Lasso (alpha=0.1)': Lasso(alpha=0.1),
            'Lasso (alpha=1.0)': Lasso(alpha=1.0)
        }
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            results = {
                'RMSE_Train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE_Train': mean_absolute_error(y_train, y_pred_train),
                'MAE_Test': mean_absolute_error(y_test, y_pred_test),
                'R2_Train': r2_score(y_train, y_pred_train),
                'R2_Test': r2_score(y_test, y_pred_test),
                'CV_Score': cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='r2').mean()
            }
            
            self.models[name] = model
            self.results[name] = results
    
    def print_comparison(self):
        """
        Print model comparison results.
        """
        print("\n" + "="*100)
        print("MODEL SELECTION AND EVALUATION RESULTS")
        print("="*100)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE (Train):      {metrics['RMSE_Train']:.4f}")
            print(f"  RMSE (Test):       {metrics['RMSE_Test']:.4f}")
            print(f"  MAE (Train):       {metrics['MAE_Train']:.4f}")
            print(f"  MAE (Test):        {metrics['MAE_Test']:.4f}")
            print(f"  R² (Train):        {metrics['R2_Train']:.4f}")
            print(f"  R² (Test):         {metrics['R2_Test']:.4f}")
            print(f"  5-Fold CV (R²):   {metrics['CV_Score']:.4f}")
    
    def identify_best_model(self):
        """
        Identify best model based on test R2 score.
        """
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['R2_Test'])
        return best_model[0], best_model[1]['R2_Test']

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Initialize notebook
    notebook = ModelSelectionNotebook(random_state=42)
    
    # Generate data
    print("Generating synthetic dataset...")
    X_train, X_test, y_train, y_test = notebook.generate_dataset(
        n_samples=500, n_features=10
    )
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train and evaluate models
    print("\nTraining models...")
    notebook.train_models(X_train, X_test, y_train, y_test)
    
    # Print results
    notebook.print_comparison()
    
    # Identify best model
    best_name, best_r2 = notebook.identify_best_model()
    
    print("\n" + "="*100)
    print(f"BEST MODEL: {best_name} with R² = {best_r2:.4f}")
    print("="*100)
    print("\nKEY RECOMMENDATIONS:")
    print("1. Use cross-validation for robust model evaluation")
    print("2. Monitor both train and test metrics for overfitting")
    print("3. Regularization (Ridge/Lasso) reduces variance")
    print("4. R² and RMSE provide complementary perspectives")
    print("="*100 + "\n")
