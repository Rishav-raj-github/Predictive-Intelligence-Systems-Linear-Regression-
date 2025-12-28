"""
08_Production_Deployment_Best_Practices.py

Comprehensive notebook for production deployment and best practices.
Demonstrates model serialization, inference pipelines, and monitoring.
"""

import pickle
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 08: Production Deployment and Best Practices
# ============================================================================

class ProductionDeploymentNotebook:
    """
    Demonstrates best practices for production model deployment.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.model_metadata = {}
    
    def generate_training_data(self, n_samples=200, n_features=10):
        """
        Generate synthetic training data.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        coef = np.random.randn(n_features)
        y = X @ coef + np.random.randn(n_samples) * 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_production_pipeline(self):
        """
        Build a production-ready sklearn pipeline.
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        self.pipeline = pipeline
        return pipeline
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train pipeline and evaluate on test set.
        """
        self.pipeline.fit(X_train, y_train)
        
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        self.model_metadata = {
            'model_type': 'LinearRegression with Scaling',
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'n_features': X_train.shape[1],
            'n_samples_trained': X_train.shape[0]
        }
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    
    def save_model_for_production(self, filepath='model.pkl'):
        """
        Serialize model to file for production deployment.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        return f"Model saved to {filepath}"
    
    def save_model_metadata(self, filepath='model_metadata.json'):
        """
        Save model metadata for production monitoring.
        """
        with open(filepath, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        return f"Metadata saved to {filepath}"
    
    def load_and_inference(self, filepath='model.pkl', X_new=None):
        """
        Load model and perform inference on new data.
        """
        with open(filepath, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        if X_new is not None:
            predictions = loaded_pipeline.predict(X_new)
            return predictions
        
        return loaded_pipeline
    
    def performance_monitoring(self, y_true, y_pred):
        """
        Monitor production model performance on new data.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        baseline_rmse = self.model_metadata['test_rmse']
        baseline_r2 = self.model_metadata['test_r2']
        
        rmse_drift = ((rmse - baseline_rmse) / baseline_rmse) * 100
        r2_drift = ((r2 - baseline_r2) / baseline_r2) * 100
        
        return {
            'current_rmse': rmse,
            'current_r2': r2,
            'baseline_rmse': baseline_rmse,
            'baseline_r2': baseline_r2,
            'rmse_drift_percent': rmse_drift,
            'r2_drift_percent': r2_drift,
            'alert': 'Performance degradation detected' if abs(rmse_drift) > 10 else 'Normal'
        }
    
    def print_deployment_guide(self):
        """
        Print best practices for production deployment.
        """
        print("\n" + "="*80)
        print("PRODUCTION DEPLOYMENT BEST PRACTICES")
        print("="*80)
        print("\n1. MODEL VERSIONING:")
        print("   - Use semantic versioning (v1.0.0, v1.0.1, v2.0.0)")
        print("   - Tag all model versions in Git")
        print("   - Store model artifacts in production-safe locations")
        print("\n2. MODEL SERIALIZATION:")
        print("   - Use pickle for sklearn models (shown here)")
        print("   - Consider joblib for large models")
        print("   - Document serialization format and dependencies")
        print("\n3. INFERENCE PIPELINE:")
        print("   - Use sklearn Pipeline for consistent preprocessing")
        print("   - Validate input data shapes and types")
        print("   - Handle missing values appropriately")
        print("\n4. MONITORING & ALERTING:")
        print("   - Track prediction performance metrics over time")
        print("   - Detect data/prediction drift")
        print("   - Set up alerts for performance degradation (>10% RMSE increase)")
        print("\n5. TESTING:")
        print("   - Unit tests for preprocessing logic")
        print("   - Integration tests for end-to-end pipeline")
        print("   - Regression tests to prevent performance loss")
        print("\n6. DOCUMENTATION:")
        print("   - Document model assumptions and limitations")
        print("   - Provide feature engineering details")
        print("   - Include troubleshooting guide")
        print("="*80 + "\n")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    notebook = ProductionDeploymentNotebook(random_state=42)
    
    print("Generating training data...")
    X_train, X_test, y_train, y_test = notebook.generate_training_data(
        n_samples=200, n_features=10
    )
    
    print("Building production-ready pipeline...")
    notebook.build_production_pipeline()
    
    print("Training and evaluating model...")
    metrics = notebook.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDeploying best practices guide...")
    notebook.print_deployment_guide()
    
    print("\n" + "="*80)
    print("PRODUCTION DEPLOYMENT CHECKLIST")
    print("="*80)
    print("[✓] 1. Model trained and validated")
    print("[✓] 2. Pipeline created with scaling")
    print("[✓] 3. Metadata tracked for monitoring")
    print("[✓] 4. Performance baseline established")
    print("[✓] 5. Deployment guide documented")
    print("[  ] 6. Model versioning system in place (user responsibility)")
    print("[  ] 7. Monitoring system configured (user responsibility)")
    print("[  ] 8. Production environment prepared (user responsibility)")
    print("="*80 + "\n")
