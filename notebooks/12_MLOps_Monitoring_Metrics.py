"""
12_MLOps_Monitoring_Metrics.py

MLOps practices and continuous monitoring for production models.
Demonstrates metrics tracking, performance monitoring, and drift detection.
"""

import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class MLOpsMonitoringNotebook:
    """
    Demonstrates MLOps practices and monitoring for continuous model improvement.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_versions = {}
        self.metrics_history = []
    
    def simulate_model_predictions(self, n_days=30):
        """
        Simulate predictions over time with potential drift.
        """
        np.random.seed(self.random_state)
        predictions_log = []
        
        for day in range(n_days):
            date = (datetime.now() - timedelta(days=n_days-day)).isoformat()
            n_predictions = np.random.randint(50, 150)
            
            # Simulate drift over time
            drift_factor = 0.01 * day
            y_true = np.random.randn(n_predictions) * 10 + drift_factor
            y_pred = y_true + np.random.randn(n_predictions) * 0.5
            
            predictions_log.append({
                'date': date,
                'n_predictions': n_predictions,
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist()
            })
        
        return predictions_log
    
    def compute_monitoring_metrics(self, y_true, y_pred):
        """
        Compute comprehensive monitoring metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(np.abs(y_true), np.abs(y_pred))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'n_samples': len(y_true)
        }
    
    def detect_performance_drift(self, baseline_metrics, current_metrics, threshold=0.1):
        """
        Detect performance drift compared to baseline.
        """
        rmse_drift = (current_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']
        r2_drift = (current_metrics['r2'] - baseline_metrics['r2']) / baseline_metrics['r2']
        
        drift_detected = abs(rmse_drift) > threshold or abs(r2_drift) > threshold
        
        return {
            'drift_detected': drift_detected,
            'rmse_drift_percent': rmse_drift * 100,
            'r2_drift_percent': r2_drift * 100,
            'recommendation': 'Retrain model' if drift_detected else 'Continue monitoring'
        }
    
    def generate_monitoring_report(self, predictions_log):
        """
        Generate comprehensive MLOps monitoring report.
        """
        print("\n" + "="*80)
        print("MLOPS MONITORING AND METRICS REPORT")
        print("="*80)
        
        # Compute metrics for all predictions
        all_y_true = []
        all_y_pred = []
        
        for day_log in predictions_log:
            all_y_true.extend(day_log['y_true'])
            all_y_pred.extend(day_log['y_pred'])
        
        baseline_metrics = self.compute_monitoring_metrics(np.array(all_y_true[:500]), 
                                                          np.array(all_y_pred[:500]))
        current_metrics = self.compute_monitoring_metrics(np.array(all_y_true), 
                                                         np.array(all_y_pred))
        
        print("\nBASELINE METRICS (First 500 predictions):")
        for key, value in baseline_metrics.items():
            if key != 'n_samples':
                print(f"  {key}: {value:.4f}")
        
        print("\nCURRENT METRICS (All predictions):")
        for key, value in current_metrics.items():
            if key != 'n_samples':
                print(f"  {key}: {value:.4f}")
        
        drift_report = self.detect_performance_drift(baseline_metrics, current_metrics)
        
        print("\nDRIFT DETECTION:")
        print(f"  Drift Detected: {drift_report['drift_detected']}")
        print(f"  RMSE Drift: {drift_report['rmse_drift_percent']:.2f}%")
        print(f"  R² Drift: {drift_report['r2_drift_percent']:.2f}%")
        print(f"  Recommendation: {drift_report['recommendation']}")

if __name__ == '__main__':
    notebook = MLOpsMonitoringNotebook(random_state=42)
    
    print("Simulating production predictions over 30 days...")
    predictions_log = notebook.simulate_model_predictions(n_days=30)
    
    notebook.generate_monitoring_report(predictions_log)
    
    print("\n" + "="*80)
    print("MLOPS BEST PRACTICES:")
    print("="*80)
    print("1. Continuous metric tracking and dashboards")
    print("2. Automated drift detection and alerting")
    print("3. Version control for models and experiments")
    print("4. Regular retraining triggers based on performance")
    print("5. A/B testing for model updates")
    print("6. Rollback mechanisms for production failures")
    print("="*80 + "\n")
