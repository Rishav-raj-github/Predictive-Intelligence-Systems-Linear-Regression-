"""
05_Multicollinearity_Diagnostics.py

Comprehensive notebook for detecting and handling multicollinearity.
Demonstrates VIF analysis, correlation analysis, and diagnostic methods.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 05: Multicollinearity Detection and Diagnosis
# ============================================================================

class MulticollinearityDiagnosticsNotebook:
    """
    Demonstrates detection and handling of multicollinearity.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.diagnostics = {}
    
    def generate_multicollinear_data(self, n_samples=200, n_features=10):
        """
        Generate data with multicollinearity.
        """
        np.random.seed(self.random_state)
        
        # Create base features
        X_base = np.random.randn(n_samples, n_features // 2)
        
        # Create highly correlated features
        X_corr = X_base + np.random.randn(n_samples, n_features // 2) * 0.1
        
        X = np.hstack([X_base, X_corr])
        
        y = X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 3] + np.random.randn(n_samples) * 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def calculate_correlation_matrix(self, X):
        """
        Calculate and analyze correlation matrix.
        """
        corr_matrix = np.corrcoef(X.T)
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr.append((i, j, corr_matrix[i, j]))
        
        return corr_matrix, high_corr
    
    def calculate_condition_number(self, X):
        """
        Calculate condition number as multicollinearity measure.
        """
        cov_matrix = np.cov(X.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        return condition_number
    
    def estimate_vif(self, X):
        """
        Estimate Variance Inflation Factor for each feature.
        Simplified VIF calculation using R-squared values.
        """
        vif_values = {}
        n_features = X.shape[1]
        
        for i in range(n_features):
            # Use other features to predict feature i
            X_others = np.delete(X, i, axis=1)
            y_feature = X[:, i]
            
            model = LinearRegression()
            model.fit(X_others, y_feature)
            y_pred = model.predict(X_others)
            
            # Calculate R-squared
            ss_res = np.sum((y_feature - y_pred) ** 2)
            ss_tot = np.sum((y_feature - np.mean(y_feature)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # VIF = 1 / (1 - R^2)
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            vif_values[f'Feature_{i}'] = vif
        
        return vif_values
    
    def print_diagnostics(self, X_train):
        """
        Print multicollinearity diagnostics.
        """
        print("\n" + "="*80)
        print("MULTICOLLINEARITY DIAGNOSTICS")
        print("="*80)
        
        # Correlation analysis
        corr_matrix, high_corr = self.calculate_correlation_matrix(X_train)
        print("\nHigh Correlations (|r| > 0.8):")
        if high_corr:
            for i, j, corr in high_corr:
                print(f"  Features {i} <-> {j}: r = {corr:.4f}")
        else:
            print("  None detected")
        
        # Condition number
        cond_num = self.calculate_condition_number(X_train)
        print(f"\nCondition Number: {cond_num:.2f}")
        if cond_num > 30:
            print("  => Severe multicollinearity detected")
        elif cond_num > 10:
            print("  => Moderate multicollinearity")
        
        # VIF values
        vif_values = self.estimate_vif(X_train)
        print("\nVariance Inflation Factors (VIF):")
        for feature, vif in vif_values.items():
            if np.isinf(vif):
                print(f"  {feature}: Inf (perfect multicollinearity)")
            else:
                print(f"  {feature}: {vif:.2f}")
                if vif > 10:
                    print(f"    => High multicollinearity")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    notebook = MulticollinearityDiagnosticsNotebook(random_state=42)
    
    print("Generating data with multicollinearity...")
    X_train, X_test, y_train, y_test = notebook.generate_multicollinear_data(
        n_samples=200, n_features=10
    )
    print(f"Dataset shape: {X_train.shape}")
    
    notebook.print_diagnostics(X_train)
    
    print("\n" + "="*80)
    print("REMEDIES FOR MULTICOLLINEARITY:")
    print("="*80)
    print("1. Remove highly correlated features")
    print("2. Combine correlated features (PCA, feature engineering)")
    print("3. Use regularization (Ridge, Lasso)")
    print("4. Collect more data to increase precision")
    print("5. Use domain knowledge to select features")
    print("="*80 + "\n")
