"""
Feature Engineering & Selection Module
=======================================

Prevents overfitting and underfitting by:
1. Feature selection (mutual information)
2. Feature importance ranking
3. Correlation analysis
4. Dimensionality reduction (optional PCA)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from typing import Tuple, List


class FeatureEngineer:
    """
    Professional feature engineering pipeline
    """
    
    def __init__(self, n_features_to_select: int = 15):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        n_features_to_select : int
            Number of top features to keep (prevents overfitting)
        """
        self.n_features_to_select = n_features_to_select
        self.selected_features = None
        self.feature_scores = None
        
    def select_best_features(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            method: str = 'mutual_info') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using mutual information
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target labels
        method : str
            'mutual_info' or 'correlation'
            
        Returns:
        --------
        X_selected : DataFrame
            Selected features
        selected_names : list
            Names of selected features
        """
        if method == 'mutual_info':
            # Mutual Information (best for classification)
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Rank features
            feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': mi_scores
            }).sort_values('score', ascending=False)
            
            # Select top N
            top_features = feature_scores.head(self.n_features_to_select)['feature'].tolist()
            
            self.selected_features = top_features
            self.feature_scores = feature_scores
            
            print(f"\n📊 Feature Selection (Mutual Information):")
            print(f"   Selected {len(top_features)} / {len(X.columns)} features")
            print(f"\n   Top 10 Features:")
            for idx, row in feature_scores.head(10).iterrows():
                print(f"      {idx+1:2d}. {row['feature']:<30} (score: {row['score']:.4f})")
            
            return X[top_features], top_features
        
        elif method == 'correlation':
            # Correlation-based selection
            correlation = X.apply(lambda x: x.corr(y)).abs()
            top_features = correlation.nlargest(self.n_features_to_select).index.tolist()
            
            self.selected_features = top_features
            return X[top_features], top_features
    
    def remove_correlated_features(self, 
                                   X: pd.DataFrame, 
                                   threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features (prevent redundancy)
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        threshold : float
            Correlation threshold (> threshold removed)
            
        Returns:
        --------
        X_reduced : DataFrame
            Features with correlations removed
        """
        corr_matrix = X.corr().abs()
        
        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"\n🔍 Correlation Analysis:")
        print(f"   Removed {len(to_drop)} highly correlated features (>{threshold})")
        if to_drop:
            print(f"   Dropped: {', '.join(to_drop[:5])}")
        
        return X.drop(columns=to_drop)
