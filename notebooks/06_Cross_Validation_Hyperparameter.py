"""
06_Cross_Validation_Hyperparameter.py

Comprehensive notebook for cross-validation and hyperparameter tuning.
Demonstrates k-fold CV, stratified CV, and hyperparameter optimization.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NOTEBOOK 06: Cross-Validation and Hyperparameter Tuning
# ============================================================================

class CrossValidationNotebook:
    """
    Demonstrates cross-validation strategies and hyperparameter optimization.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cv_results = {}
    
    def generate_data(self, n_samples=300, n_features=15):
        """
        Generate synthetic dataset.
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        coef = np.random.randn(n_features)
        y = X @ coef + np.random.randn(n_samples) * 0.3
        return X, y
    
    def perform_kfold_cv(self, X, y, k=5):
        """
        Perform k-fold cross-validation.
        """
        kfold = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            fold_scores.append(score)
        
        return {
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        }
    
    def validate_alphas(self, X, y, alphas=[0.01, 0.1, 1.0, 10.0, 100.0]):
        """
        Validate different alpha values.
        """
        results = {}
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            
            # Use cross_validate for detailed metrics
            scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error'}
            cv_results = cross_validate(
                model, X, y, cv=5, scoring=scoring, return_train_score=True
            )
            
            results[alpha] = {
                'train_r2_mean': np.mean(cv_results['train_r2']),
                'test_r2_mean': np.mean(cv_results['test_r2']),
                'train_r2_std': np.std(cv_results['train_r2']),
                'test_r2_std': np.std(cv_results['test_r2'])
            }
        
        return results
    
    def find_optimal_alpha(self, X, y):
        """
        Find optimal alpha value.
        """
        alphas = np.logspace(-3, 3, 20)
        validation_scores = []
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            cv_scores = cross_validate(
                model, X, y, cv=5, scoring='r2'
            )['test_r2']
            validation_scores.append(np.mean(cv_scores))
        
        optimal_idx = np.argmax(validation_scores)
        return alphas[optimal_idx], validation_scores
    
    def print_results(self):
        """
        Print cross-validation results.
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        
        for k_folds, results in self.cv_results.items():
            print(f"\n{k_folds}-Fold CV:")
            print(f"  Mean R²: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    notebook = CrossValidationNotebook(random_state=42)
    
    print("Generating synthetic dataset...")
    X, y = notebook.generate_data(n_samples=300, n_features=15)
    
    # K-fold CV
    print("\nPerforming 5-fold cross-validation...")
    notebook.cv_results['5-Fold'] = notebook.perform_kfold_cv(X, y, k=5)
    
    print("Validating alpha values...")
    alpha_results = notebook.validate_alphas(X, y)
    
    print("\nAlpha Validation Results:")
    for alpha, metrics in alpha_results.items():
        print(f"\n  Alpha = {alpha:.2f}:")
        print(f"    Train R²: {metrics['train_r2_mean']:.4f} (+/- {metrics['train_r2_std']:.4f})")
        print(f"    Test R²:  {metrics['test_r2_mean']:.4f} (+/- {metrics['test_r2_std']:.4f})")
    
    optimal_alpha, scores = notebook.find_optimal_alpha(X, y)
    
    notebook.print_results()
    
    print("\n" + "="*80)
    print(f"OPTIMAL ALPHA: {optimal_alpha:.4f}")
    print("="*80)
    print("\nKEY INSIGHTS:")
    print("1. Cross-validation provides unbiased performance estimation")
    print("2. k-fold CV is more stable than train-test split")
    print("3. Standard deviation shows model stability")
    print("4. Hyperparameter tuning via CV prevents overfitting")
    print("="*80 + "\n")
