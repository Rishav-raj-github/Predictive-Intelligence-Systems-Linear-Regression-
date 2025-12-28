"""
09_Advanced_Feature_Engineering.py

Advanced feature engineering and polynomial expansion techniques.
Demonstrates interaction terms, polynomial features, and domain-specific engineering.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineeringNotebook:
    """
    Demonstrates advanced feature engineering techniques for enhanced model performance.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_results = {}
    
    def generate_data(self, n_samples=300, n_features=5):
        """
        Generate synthetic data for feature engineering.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        coef = np.random.randn(n_features)
        y = X @ coef + 0.5 * (X[:, 0] * X[:, 1]) + np.random.randn(n_samples) * 0.5
        
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)
    
    def test_polynomial_features(self, X_train, X_test, y_train, y_test):
        """
        Test polynomial feature expansion.
        """
        results = {}
        
        for degree in [1, 2, 3]:
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            train_r2 = r2_score(y_train, model.predict(X_train_poly))
            test_r2 = r2_score(y_test, model.predict(X_test_poly))
            
            results[f'Degree_{degree}'] = {
                'n_features': X_train_poly.shape[1],
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        
        return results
    
    def create_interaction_features(self, X):
        """
        Create manual interaction features.
        """
        n_samples, n_features = X.shape
        interactions = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                interactions.append(X[:, i] * X[:, j])
        
        if interactions:
            X_interactions = np.column_stack(interactions)
            return np.hstack([X, X_interactions])
        return X
    
    def print_results(self):
        """
        Print feature engineering comparison results.
        """
        print("\n" + "="*80)
        print("ADVANCED FEATURE ENGINEERING RESULTS")
        print("="*80)
        
        for method, metrics in self.feature_results.items():
            print(f"\n{method}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == '__main__':
    notebook = AdvancedFeatureEngineeringNotebook(random_state=42)
    
    print("Generating data and testing feature engineering...")
    X_train, X_test, y_train, y_test = notebook.generate_data(n_samples=300, n_features=5)
    
    notebook.feature_results = notebook.test_polynomial_features(X_train, X_test, y_train, y_test)
    notebook.print_results()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Polynomial features capture non-linear relationships")
    print("2. Higher degrees increase model complexity and risk overfitting")
    print("3. Interaction terms enable capturing feature dependencies")
    print("4. Regularization helps control polynomial feature expansion")
    print("="*80 + "\n")
