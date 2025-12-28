"""
11_Model_Interpretability_Explainability.py

Model interpretation and explainability for linear regression.
Demonstrates feature importance, coefficient analysis, and LIME-style explanations.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class InterpretabilityNotebook:
    """
    Demonstrates model interpretability and explainability techniques.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
    
    def generate_data(self, n_samples=300, n_features=10):
        """
        Generate synthetic data.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        coef = np.array([5, -3, 2, 0.5, -1, 0.1, 0.05, -0.2, 0, 0.3])
        y = X @ coef + np.random.randn(n_samples) * 0.5
        
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)
    
    def train_model(self, X_train, y_train):
        """
        Train and scale the model.
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
    
    def get_feature_importance(self):
        """
        Extract feature importance from model coefficients.
        """
        coef = self.model.coef_
        importance = np.abs(coef)
        normalized_importance = importance / np.sum(importance)
        
        return sorted(zip(range(len(coef)), normalized_importance), 
                     key=lambda x: x[1], reverse=True)
    
    def explain_prediction(self, X_sample):
        """
        Explain individual prediction using coefficient analysis.
        """
        X_scaled = self.scaler.transform(X_sample.reshape(1, -1))
        prediction = self.model.predict(X_scaled)[0]
        
        # Feature contributions
        contributions = X_scaled[0] * self.model.coef_
        
        return {
            'prediction': prediction,
            'contributions': contributions,
            'intercept': self.model.intercept_
        }
    
    def print_interpretability_report(self, feature_names=None):
        """
        Print model interpretability report.
        """
        print("\n" + "="*80)
        print("MODEL INTERPRETABILITY REPORT")
        print("="*80)
        
        importance = self.get_feature_importance()
        print("\nTOP 5 IMPORTANT FEATURES:")
        for rank, (feature_idx, imp) in enumerate(importance[:5], 1):
            fname = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
            print(f"  {rank}. {fname}: {imp*100:.2f}%")
        
        print(f"\nModel Intercept: {self.model.intercept_:.4f}")
        print("\nTop 3 Coefficients (Largest Impact):")
        for idx, coef in sorted(enumerate(self.model.coef_), 
                               key=lambda x: abs(x[1]), reverse=True)[:3]:
            fname = feature_names[idx] if feature_names else f"Feature_{idx}"
            print(f"  {fname}: {coef:.4f}")

if __name__ == '__main__':
    notebook = InterpretabilityNotebook(random_state=42)
    
    print("Training interpretable linear regression model...")
    X_train, X_test, y_train, y_test = notebook.generate_data(n_samples=300, n_features=10)
    notebook.train_model(X_train, y_train)
    
    feature_names = [f"Feature_{i}" for i in range(10)]
    notebook.print_interpretability_report(feature_names)
    
    print("\n" + "="*80)
    print("INTERPRETING LINEAR REGRESSION:")
    print("="*80)
    print("1. Coefficients directly show feature impact on prediction")
    print("2. Feature scaling allows comparison of relative importance")
    print("3. Linear models are naturally interpretable (glass-box)")
    print("4. Individual predictions can be explained via contributions")
    print("="*80 + "\n")
