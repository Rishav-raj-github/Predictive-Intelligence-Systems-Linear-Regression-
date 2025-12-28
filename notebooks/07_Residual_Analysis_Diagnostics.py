"""
07_Residual_Analysis_Diagnostics.py

Comprehensive notebook for residual analysis and model diagnostics.
Demonstrates residual plots, normality tests, and heteroscedasticity detection.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 07: Residual Analysis and Diagnostics
# ============================================================================

class ResidualAnalysisNotebook:
    """
    Demonstrates comprehensive residual analysis for model diagnostics.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.residuals = None
        self.fitted_values = None
    
    def generate_data(self, n_samples=200, n_features=10):
        """
        Generate synthetic regression data.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        coef = np.random.randn(n_features)
        y = X @ coef + np.random.randn(n_samples) * 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit_and_analyze(self, X_train, X_test, y_train, y_test):
        """
        Fit model and compute residuals.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test
        
        self.residuals = residuals_test
        self.fitted_values = y_pred_test
        
        return model, residuals_train, residuals_test, y_pred_train, y_pred_test
    
    def test_normality(self, residuals):
        """
        Test normality of residuals using Shapiro-Wilk test.
        """
        if len(residuals) > 5000:
            sample_residuals = np.random.choice(residuals, 5000, replace=False)
        else:
            sample_residuals = residuals
        
        statistic, pvalue = stats.shapiro(sample_residuals)
        return {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'pvalue': pvalue,
            'normal': pvalue > 0.05
        }
    
    def test_heteroscedasticity(self, residuals, fitted_values):
        """
        Test for heteroscedasticity using Breusch-Pagan test approximation.
        """
        # Approximate Breusch-Pagan test
        residuals_squared = residuals ** 2
        
        # Fit auxiliary regression
        X_aux = fitted_values.reshape(-1, 1)
        model_aux = LinearRegression()
        model_aux.fit(X_aux, residuals_squared)
        
        predictions_aux = model_aux.predict(X_aux)
        ss_total = np.sum((residuals_squared - np.mean(residuals_squared)) ** 2)
        ss_residual = np.sum((residuals_squared - predictions_aux) ** 2)
        
        if ss_total > 0:
            r_squared = 1 - (ss_residual / ss_total)
        else:
            r_squared = 0
        
        lm_statistic = len(residuals) * r_squared
        pvalue = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        return {
            'test': 'Breusch-Pagan (approx)',
            'statistic': lm_statistic,
            'pvalue': pvalue,
            'homoscedastic': pvalue > 0.05
        }
    
    def compute_diagnostics(self, residuals, fitted_values):
        """
        Compute comprehensive diagnostic statistics.
        """
        diagnostics = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'autocorrelation_lag1': np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        }
        
        return diagnostics
    
    def print_diagnostics_report(self):
        """
        Print comprehensive diagnostics report.
        """
        print("\n" + "="*80)
        print("RESIDUAL ANALYSIS AND DIAGNOSTICS REPORT")
        print("="*80)
        
        # Residual statistics
        print("\nRESIDUAL SUMMARY STATISTICS:")
        diag = self.compute_diagnostics(self.residuals, self.fitted_values)
        for key, value in diag.items():
            print(f"  {key}: {value:.6f}")
        
        # Normality test
        normality = self.test_normality(self.residuals)
        print(f"\n{normality['test'].upper()}:")
        print(f"  p-value: {normality['pvalue']:.6f}")
        print(f"  Result: {'Residuals are normally distributed' if normality['normal'] else 'Residuals deviate from normality'}")
        
        # Heteroscedasticity test
        hetero = self.test_heteroscedasticity(self.residuals, self.fitted_values)
        print(f"\n{hetero['test'].upper()}:")
        print(f"  p-value: {hetero['pvalue']:.6f}")
        print(f"  Result: {'Homoscedasticity confirmed' if hetero['homoscedastic'] else 'Heteroscedasticity detected'}")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    notebook = ResidualAnalysisNotebook(random_state=42)
    
    print("Generating synthetic data...")
    X_train, X_test, y_train, y_test = notebook.generate_data(
        n_samples=200, n_features=10
    )
    
    print("Fitting model and analyzing residuals...")
    model, res_train, res_test, y_pred_train, y_pred_test = notebook.fit_and_analyze(
        X_train, X_test, y_train, y_test
    )
    
    # Print diagnostic report
    notebook.print_diagnostics_report()
    
    print("\n" + "="*80)
    print("MODEL ASSUMPTIONS CHECKLIST:")
    print("="*80)
    print("1. Linearity: Check scatter plot of residuals vs fitted values")
    print("2. Normality: Assessed via Shapiro-Wilk test above")
    print("3. Homoscedasticity: Assessed via Breusch-Pagan test above")
    print("4. Independence: Check autocorrelation of residuals")
    print("5. No Multicollinearity: Covered in previous notebooks")
    print("="*80 + "\n")
