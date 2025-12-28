"""01_Data_Exploration_Analysis.py - EDA for Linear Regression"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class DataExplorationNotebook:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target = target_col
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
    
    def get_basic_statistics(self) -> Dict:
        """Get basic statistics of the dataset."""
        stats_dict = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_mean': float(self.y.mean()),
            'target_std': float(self.y.std()),
        }
        return stats_dict
    
    def check_feature_distribution(self) -> Dict:
        """Analyze feature distributions."""
        distributions = {}
        for col in self.X.select_dtypes(include=[np.number]).columns:
            distributions[col] = {
                'skewness': float(stats.skew(self.X[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.X[col].dropna())),
                'min': float(self.X[col].min()),
                'max': float(self.X[col].max()),
            }
        return distributions
    
    def correlation_analysis(self) -> np.ndarray:
        """Analyze correlations with target."""
        numeric_df = pd.concat([self.X.select_dtypes(include=[np.number]), self.y], axis=1)
        return numeric_df.corr()[self.target].sort_values(ascending=False).to_dict()
    
    def detect_outliers(self, method='iqr') -> Dict:
        """Detect outliers using IQR or Z-score."""
        outliers = {}
        for col in self.X.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1, Q3 = self.X[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outlier_mask = (self.X[col] < Q1 - 1.5*IQR) | (self.X[col] > Q3 + 1.5*IQR)
            else:
                z_scores = np.abs(stats.zscore(self.X[col].dropna()))
                outlier_mask = z_scores > 3
            outliers[col] = int(outlier_mask.sum())
        return outliers

def main():
    print("Data Exploration & Analysis for Linear Regression")

if __name__ == "__main__":
    from typing import Dict
    main()
